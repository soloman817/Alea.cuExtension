module Alea.CUDA.Extension.Scan

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.Timing
open Alea.CUDA.Extension.Reduce

type IScan<'T> =
    abstract Scan : int * DevicePtr<'T> -> DevicePtr<'T>
    abstract Scan : 'T[] -> 'T[]

/// Multiscan function for a warps in the block.
let [<ReflectedDefinition>] inline multiScan (init: unit -> 'T) (op:'T -> 'T -> 'T) numWarps logNumWarps tid (x:'T) (totalRef : 'T ref) =
    let warp = tid / WARP_SIZE
    let lane = tid &&& (WARP_SIZE - 1)
    let warpStride = WARP_SIZE + WARP_SIZE / 2
    let sharedSize = numWarps * warpStride
    let shared = __shared__<'T>(sharedSize).Ptr(0)
    let warpShared = (shared + warp * warpStride).Volatile()      
    let s = warpShared + (lane + WARP_SIZE / 2)

    let warp = tid / WARP_SIZE
    let lane = (WARP_SIZE - 1) &&& tid

    let warpShared = shared.Volatile()
    let s = warpShared + (lane + WARP_SIZE / 2)
    warpShared.[lane] <- init()
    s.[0] <- x

    // Run inclusive scan on each warp's data.
    let mutable warpScan = x
    for i = 0 to LOG_WARP_SIZE - 1 do
        let offset = 1 <<< i
        warpScan <- op warpScan s.[-offset]   
        if i < LOG_WARP_SIZE - 1 then s.[0] <- warpScan
        
    let totalsShared = __shared__<'T>(2*numWarps).Ptr(0).Volatile() 

    if lane = WARP_SIZE - 1 then
        totalsShared.[numWarps + warp] <- warpScan  

    // Synchronize to make all the totals available to the reduction code.
    __syncthreads()

    if tid < numWarps then
        // Grab the block total for the tid'th block. This is the last element
        // in the block's scanned sequence. This operation avoids bank conflicts.
        let total = totalsShared.[numWarps + tid]
        totalsShared.[tid] <- init()
        let s = (totalsShared + numWarps + tid).Volatile()  

        let mutable totalsScan = total

        for i = 0 to logNumWarps - 1 do
            let offset = 1 <<< i
            totalsScan <- op totalsScan s.[-offset]
            s.[0] <- totalsScan

        // Shift by one to store exclusive scan, overwriting element at index numWarps but that thoes not matter.
        totalsShared.[tid + 1] <- totalsScan 

    // Synchronize to make the block scan available to all warps.
    __syncthreads()

    // The total is the last element.
    totalRef := totalsShared.[2 * numWarps - 1]

    // Add the block scan to the inclusive scan for the block.
    op warpScan totalsShared.[warp]
        
let inline scanReduceKernel (plan:Plan) (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) (transfExpr:Expr<'T -> 'T>) =
    let numThreads = plan.numThreadsReduction
    let numWarps = plan.numWarpsReduction
    let logNumWarps = log2 numWarps
    <@ fun numRanges (dRangeTotals:DevicePtr<'T>) ->
        let init = %initExpr
        let op = %opExpr
        let transf = %transfExpr

        let tid = threadIdx.x
        let x = if tid < numRanges then dRangeTotals.[tid] else init()
            
        let total:ref<'T> = ref (init())
        let scan = multiScan init op numWarps logNumWarps tid x total

        // Store the value from the inclusive scan shifted by one for the exclusive scan.
        // TODO check if the last scan is the total so second next line is not necessary
        if tid < numRanges then dRangeTotals.[tid + 1] <- scan

        // Have the first thread in the block set the scan total.
        if tid = 0 then 
            dRangeTotals.[0] <- init()  
            //dRangeTotals.[numRanges] <- !total
    @>

let inline scanDownSweepKernel (plan:Plan) (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) (transfExpr:Expr<'T -> 'T>) =
    let numWarps = plan.numWarps
    let numValues = plan.numValues
    let valuesPerThread = plan.valuesPerThread
    let valuesPerWarp = plan.valuesPerWarp 
    let logNumWarps = log2 numWarps
    let size = numWarps * valuesPerThread * (WARP_SIZE + 1)
    <@ fun (dValuesIn:DevicePtr<'T>) (dValuesOut:DevicePtr<'T>) (dRangeTotals:DevicePtr<'T>) (dRanges:DevicePtr<int>) (inclusive:int) ->
        let init = %initExpr
        let op = %opExpr
        let transf = %transfExpr

        let block = blockIdx.x
        let tid = threadIdx.x
        let warp = tid / WARP_SIZE
        let lane = (WARP_SIZE - 1) &&& tid
        let index = valuesPerWarp * warp + lane

        let mutable blockScan = dRangeTotals.[block]
        let mutable rangeX = dRanges.[block]
        let rangeY = dRanges.[block + 1]
            
        let shared = __shared__<'T>(size).Ptr(0).Volatile()

        // Use a stride of 33 slots per warp per value to allow conflict-free transposes from strided to thread order.
        let warpShared = shared + warp * valuesPerThread * (WARP_SIZE + 1)
        let threadShared = warpShared + lane

        // Transpose values into thread order.
        let mutable offset = valuesPerThread * lane
        offset <- offset + offset / WARP_SIZE

        while rangeX < rangeY do

            for i = 0 to valuesPerThread - 1 do
                let source = rangeX + index + i * WARP_SIZE
                let x = transf dValuesIn.[source]
                threadShared.[i * (WARP_SIZE + 1)] <- x

            // Transpose into thread order by reading from transposeValues.
            // Compute the exclusive or inclusive scan of the thread values and their sum.
            let threadScan = __local__<'T>(valuesPerThread)
            let mutable scan = init()

            for i = 0 to valuesPerThread - 1 do 
                let x = warpShared.[offset + i]
                threadScan.[i] <- scan
                if (inclusive <> 0) then threadScan.[i] <- op threadScan.[i] x
                scan <- op scan x               
 
            // Multiscan for each thread's scan offset within the block. 
            // Subtract scan value to make it an exclusive scan.
            let localTotal:ref<'T> = ref (init())
            let localScan = multiScan init op numWarps logNumWarps tid scan localTotal
            // TODO how to fix that?
            let scanOffset = localScan + blockScan - scan
                
            // Add the scan offset to each exclusive scan and put the values back into the shared memory they came out of.
            for i = 0 to valuesPerThread - 1 do
                let x = threadScan.[i] + scanOffset
                warpShared.[offset + i] <- x
                
            // Store the scan back to global memory.
            for i = 0 to valuesPerThread - 1 do
                let x = threadShared.[i * (WARP_SIZE + 1)]
                let target = rangeX + index + i * WARP_SIZE
                dValuesOut.[target] <- x

            // Grab the last element of totals_shared, which was set in Multiscan.
            // This is the total for all the values encountered in this pass.
            blockScan <- blockScan + !localTotal

            rangeX <- rangeX + numValues
    @>

type Api<'T> =
    abstract Invoke : 'T[] * int -> 'T[]
    abstract Invoke : 'T[] * int * TimingCollectFunc -> 'T[]

/// <summary>
/// Global scan algorithm template. 
/// </summary>
let inline scan (plan:Plan) (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (transf:Expr<'T -> 'T>) = cuda {
    let! kernel1 = reduceUpSweepKernel plan init op transf |> defineKernelFunc
    let! kernel2 = scanReduceKernel plan init op transf |> defineKernelFunc
    let! kernel3 = scanDownSweepKernel plan init op transf |> defineKernelFunc

    let invoke (m:Module) (tc:TimingCollectFunc option) (values:'T[]) (inclusive:int) =
        let numSm = m.Worker.Device.Attribute(DeviceAttribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        let ranges = plan.blockRanges numSm values.Length
        let numRanges = ranges.Length - 1

        // The arrays dValuesIn and dValuesOut should be multiply of plan.numValues, so that we  do not need to do the range checks in the kernel.
        //let size = int(ceil(float(values.Length) / float(plan.numValues))) * plan.numValues
        // in fact, this padding should work, but we use numValues + 1 for now, to see
        // if it still has problem, if still has problem, then means we have other problems
        //let padding = values.Length + plan.numValues - ranges.[ranges.Length - 1]
        let padding = plan.numValues + 1
        let size = values.Length + padding
        use dValuesIn = m.Worker.Malloc<'T>(size)
        use dValuesOut = m.Worker.Malloc<'T>(size)
        use dRanges = m.Worker.Malloc(ranges)
        use dRangeTotals = m.Worker.Malloc<'T>(Array.zeroCreate (numRanges + 1))  

        printfn "====> size = %A, ranges = %A" size ranges

        // Scatter relevant data into padded array dValuesIn.
        dValuesIn.Scatter(values)
        if padding > 0 then
            DevicePtrUtil.Scatter(m.Worker, Array.zeroCreate<'T>(padding), dValuesIn.Ptr + values.Length, padding)

        printfn "0) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())

        //let lp = LaunchParam (numRanges-1, plan.numThreads) |> setDiagnoser (diagnose "upSweep")
        let lp = LaunchParam (numRanges, plan.numThreads)
        let lp = match tc with Some(tc) -> lp |> Engine.setDiagnoser ((tcToDiag tc) "upsweep") | None -> lp
        kernel1.Launch m lp dValuesIn.Ptr dRanges.Ptr dRangeTotals.Ptr

        printfn "1) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())

        let lp = LaunchParam(1, plan.numThreadsReduction)
        let lp = match tc with Some(tc) -> lp |> Engine.setDiagnoser ((tcToDiag tc) "reduce") | None -> lp
        kernel2.Launch m lp numRanges dRangeTotals.Ptr

        printfn "2) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())

        let lp = LaunchParam(numRanges, plan.numThreads)
        let lp = match tc with Some(tc) -> lp |> Engine.setDiagnoser ((tcToDiag tc) "downsweep") | None -> lp
        kernel3.Launch m lp dValuesIn.Ptr dValuesOut.Ptr dRangeTotals.Ptr dRanges.Ptr inclusive

        printfn "3) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())

        // Gather only the relevant data from GPU.
        let hValuesOut = Array.zeroCreate values.Length
        DevicePtrUtil.Gather(m.Worker, dValuesOut.Ptr, hValuesOut, values.Length)
        hValuesOut

    return PFunc(fun (m:Module) ->
        let invoke = invoke m
        { new Api<'T> with
            member this.Invoke(values, inclusive) =
                invoke None values inclusive
            member this.Invoke(values, inclusive, tc) =
                invoke (Some tc) values inclusive
        } ) }

/// Specialized version for sum without expression splicing to check performance impact.
module Sum =  

    /// Multiscan function for a warps in the block with shared memory passed in from caller.
    let [<ReflectedDefinition>] inline multiScan numWarps logNumWarps tid (x:'T) (totalRef : 'T ref) =
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)
        let warpStride = WARP_SIZE + WARP_SIZE / 2
        let sharedSize = numWarps * warpStride
        let shared = __shared__<'T>(sharedSize).Ptr(0)
        let warpShared = (shared + warp * warpStride).Volatile()
        let s = warpShared + (lane + WARP_SIZE / 2)
        warpShared.[lane] <- 0G
        s.[0] <- x

        // Run inclusive scan on each warp's data.
        let mutable sum = x

        for i = 0 to LOG_WARP_SIZE - 1 do
            let offset = 1 <<< i
            sum <- sum + s.[-offset]   
            if i < LOG_WARP_SIZE - 1 then s.[0] <- sum
        
        let totalsShared = __shared__<'T>(2*numWarps).Ptr(0).Volatile() 

        if lane = WARP_SIZE - 1 then
            totalsShared.[numWarps + warp] <- sum  

        // Synchronize to make all the totals available to the reduction code.
        __syncthreads()

        if tid < numWarps then
            // Grab the block total for the tid'th block. This is the last element
            // in the block's scanned sequence. This operation avoids bank conflicts.
            let total = totalsShared.[numWarps + tid]
            totalsShared.[tid] <- 0G
            let s = (totalsShared + numWarps + tid).Volatile()  

            let mutable totalsSum = total

            for i = 0 to logNumWarps - 1 do
                let offset = 1 <<< i
                totalsSum <- totalsSum + s.[-offset]
                s.[0] <- totalsSum

            // Subtract total from totalsSum for an exclusive scan.
            totalsShared.[tid] <- totalsSum - total

        // Synchronize to make the block scan available to all warps.
        __syncthreads()

        // The total is the last element.
        totalRef := totalsShared.[2 * numWarps - 1]

        // Add the block scan to the inclusive sum for the block.
        sum + totalsShared.[warp]

    let inline upSweepKernel (plan:Plan) =
        let numThreads = plan.numThreads
        let numWarps = plan.numWarps
        let logNumWarps = log2 numWarps
        <@ fun (dValues:DevicePtr<'T>) (dRanges:DevicePtr<int>) (dRangeTotals:DevicePtr<'T>) ->
            let block = blockIdx.x
            let tid = threadIdx.x
            let rangeX = dRanges.[block]
            let rangeY = dRanges.[block + 1]

            // Loop through all elements in the interval, adding up values.
            // There is no need to synchronize until we perform the multiscan.
            let mutable sum = 0G
            let mutable index = rangeX + tid
            while index < rangeY do
                sum <- sum + dValues.[index]
                index <- index + numThreads

            // A full multiscan is unnecessary here - we really only need the total.
            // But this is easy and won't slow us down since this kernel is already bandwidth limited.
            let total:ref<'T> = ref 0G
            let blockSum = multiScan numWarps logNumWarps tid sum total

            if tid = 0 then dRangeTotals.[block] <- !total
        @>

    let inline reduceKernel (plan:Plan) =
        let numThreads = plan.numThreadsReduction
        let numWarps = plan.numWarpsReduction
        let logNumWarps = log2 numWarps
        <@ fun numRanges (dRangeTotals:DevicePtr<'T>) ->
            let tid = threadIdx.x
            let x = if tid < numRanges then dRangeTotals.[tid] else 0G
            
            let total:ref<'T> = ref 0G
            let sum = multiScan numWarps logNumWarps tid x total

            // Subtract the value from the inclusive scan for the exclusive scan.
            if tid < numRanges then dRangeTotals.[tid] <- sum - x

            // Have the first thread in the block set the scan total.
            if tid = 0 then dRangeTotals.[numRanges] <- !total
        @>

    let inline downSweepKernel (plan:Plan) =
        let numWarps = plan.numWarps
        let numValues = plan.numValues
        let valuesPerThread = plan.valuesPerThread
        let valuesPerWarp = plan.valuesPerWarp 
        let logNumWarps = log2 numWarps
        let size = numWarps * valuesPerThread * (WARP_SIZE + 1)
        <@ fun (dValuesIn:DevicePtr<'T>) (dValuesOut:DevicePtr<'T>) (dRangeTotals:DevicePtr<'T>) (dRanges:DevicePtr<int>) (inclusive:int) ->
            let block = blockIdx.x
            let tid = threadIdx.x
            let warp = tid / WARP_SIZE
            let lane = (WARP_SIZE - 1) &&& tid
            let index = valuesPerWarp * warp + lane

            let mutable blockScan = dRangeTotals.[block]
            let mutable rangeX = dRanges.[block]
            let rangeY = dRanges.[block + 1]
            
            let shared = __shared__<'T>(size).Ptr(0).Volatile()

            // Use a stride of 33 slots per warp per value to allow conflict-free transposes from strided to thread order.
            let warpShared = shared + warp * valuesPerThread * (WARP_SIZE + 1)
            let threadShared = warpShared + lane

            // Transpose values into thread order.
            let mutable offset = valuesPerThread * lane
            offset <- offset + offset / WARP_SIZE

            while rangeX < rangeY do

                for i = 0 to valuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let x = dValuesIn.[source]
                    threadShared.[i * (WARP_SIZE + 1)] <- x

                // Transpose into thread order by reading from transposeValues.
                // Compute the exclusive or inclusive scan of the thread values and their sum.
                let scan = __local__<'T>(valuesPerThread)
                let mutable sum = 0G

                for i = 0 to valuesPerThread - 1 do 
                    let x = warpShared.[offset + i]
                    scan.[i] <- sum
                    if (inclusive <> 0) then scan.[i] <- scan.[i] + x
                    sum <- sum + x               
 
                // Multiscan for each thread's scan offset within the block. 
                // Subtract sum to make it an exclusive scan.
                let localTotal:ref<'T> = ref 0G
                let localScan = multiScan numWarps logNumWarps tid sum localTotal
                let scanOffset = localScan + blockScan - sum
                
                // Add the scan offset to each exclusive scan and put the values back into the shared memory they came out of.
                for i = 0 to valuesPerThread - 1 do
                    let x = scan.[i] + scanOffset
                    warpShared.[offset + i] <- x
                
                // Store the scan back to global memory.
                for i = 0 to valuesPerThread - 1 do
                    let x = threadShared.[i * (WARP_SIZE + 1)]
                    let target = rangeX + index + i * WARP_SIZE
                    dValuesOut.[target] <- x

                // Grab the last element of totals_shared, which was set in Multiscan.
                // This is the total for all the values encountered in this pass.
                blockScan <- blockScan + !localTotal

                rangeX <- rangeX + numValues
        @>

    type Api<'T> =
        abstract Invoke : 'T[] * int -> 'T[]
        abstract Invoke : 'T[] * int * TimingCollectFunc -> 'T[]

    /// <summary>
    /// Global scan algorithm template. 
    /// </summary>
    let inline scan (plan:Plan) = cuda {
        let! kernel1 = upSweepKernel plan |> defineKernelFunc
        let! kernel2 = reduceKernel plan |> defineKernelFunc
        let! kernel3 = downSweepKernel plan |> defineKernelFunc

        let invoke (m:Module) (tc:TimingCollectFunc option) (values:'T[]) (inclusive:int) =
            let numSm = m.Worker.Device.Attribute(DeviceAttribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            let ranges = plan.blockRanges numSm values.Length
            let numRanges = ranges.Length - 1

            // The arrays dValuesIn and dValuesOut should be multiply of plan.numValues, so that we  do not need to do the range checks in the kernel.
            //let size = int(ceil(float(values.Length) / float(plan.numValues))) * plan.numValues
            // in fact, this padding should work, but we use numValues + 1 for now, to see
            // if it still has problem, if still has problem, then means we have other problems
            //let padding = values.Length + plan.numValues - ranges.[ranges.Length - 1]
            let padding = plan.numValues + 1
            let size = values.Length + padding
            use dValuesIn = m.Worker.Malloc<'T>(size)
            use dValuesOut = m.Worker.Malloc<'T>(size)
            use dRanges = m.Worker.Malloc(ranges)
            use dRangeTotals = m.Worker.Malloc<'T>(Array.zeroCreate (numRanges + 1))  

            printfn "====> size = %A, ranges = %A" size ranges

            // Scatter relevant data into padded array dValuesIn.
            dValuesIn.Scatter(values)
            if padding > 0 then
                DevicePtrUtil.Scatter(m.Worker, Array.zeroCreate<'T>(padding), dValuesIn.Ptr + values.Length, padding)

            printfn "0) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())

            //let lp = LaunchParam (numRanges-1, plan.numThreads) |> setDiagnoser (diagnose "upSweep")
            let lp = LaunchParam (numRanges, plan.numThreads)
            let lp = match tc with Some(tc) -> lp |> Engine.setDiagnoser ((tcToDiag tc) "upsweep") | None -> lp
            kernel1.Launch m lp dValuesIn.Ptr dRanges.Ptr dRangeTotals.Ptr

            printfn "1) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())

            let lp = LaunchParam(1, plan.numThreadsReduction)
            let lp = match tc with Some(tc) -> lp |> Engine.setDiagnoser ((tcToDiag tc) "reduce") | None -> lp
            kernel2.Launch m lp numRanges dRangeTotals.Ptr

            printfn "2) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())

            let lp = LaunchParam(numRanges, plan.numThreads)
            let lp = match tc with Some(tc) -> lp |> Engine.setDiagnoser ((tcToDiag tc) "downsweep") | None -> lp
            kernel3.Launch m lp dValuesIn.Ptr dValuesOut.Ptr dRangeTotals.Ptr dRanges.Ptr inclusive

            printfn "3) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())

            // Gather only the relevant data from GPU.
            let hValuesOut = Array.zeroCreate values.Length
            DevicePtrUtil.Gather(m.Worker, dValuesOut.Ptr, hValuesOut, values.Length)
            hValuesOut

        return PFunc(fun (m:Module) ->
            let invoke = invoke m
            { new Api<'T> with
                member this.Invoke(values, inclusive) =
                    invoke None values inclusive
                member this.Invoke(values, inclusive, tc) =
                    invoke (Some tc) values inclusive
            } ) }

