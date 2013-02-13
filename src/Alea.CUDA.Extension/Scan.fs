module Alea.CUDA.Extension.Scan

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.Timing
open Alea.CUDA.Extension.Reduce

type IScan<'T when 'T : unmanaged> =
    abstract Scatter : 'T[] -> int -> DeviceMemory<'T> -> unit 
    abstract Gather : int -> DevicePtr<'T> -> 'T[]
    abstract Scan : int * DevicePtr<'T> * DevicePtr<'T> * bool -> unit 
    abstract Scan : 'T[] * bool -> 'T[]
    abstract Scan : 'T[] * bool * TimingCollectFunc -> 'T[]

module Generic = 
    /// Multi-scan function for all warps in the block.
    let [<ReflectedDefinition>] inline multiScan (init: unit -> 'T) (op:'T -> 'T -> 'T) numWarps logNumWarps tid (x:'T) (totalRef : 'T ref) =
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)
        let warpStride = WARP_SIZE + WARP_SIZE / 2
    
        // Allocate shared memory.
        let shared = __shared__<'T>(numWarps * warpStride).Ptr(0)
        let totalsShared = __shared__<'T>(2 * numWarps).Ptr(0).Volatile() 

        let warpShared = (shared + warp * warpStride).Volatile()
        let s = (warpShared + lane + WARP_SIZE / 2).Volatile()
        warpShared.[lane] <- init()
        s.[0] <- x

        // Run inclusive scan on each warp's data.
        let mutable scan = x
        for i = 0 to LOG_WARP_SIZE - 1 do
            let offset = 1 <<< i
            scan <- op scan s.[-offset]   
            if i < LOG_WARP_SIZE - 1 then s.[0] <- scan
        
        if lane = WARP_SIZE - 1 then
            totalsShared.[numWarps + warp] <- scan  

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

            // Store totalsScan shifted by one to the right for an exclusive scan.
            //if 0 < tid && tid < numWarps - 1 then totalsShared.[tid + 1] <- totalsScan 
            //if tid = 0 then totalsShared.[tid] <- init()
            totalsShared.[tid + 1] <- totalsScan

        // Synchronize to make the block scan available to all warps.
        __syncthreads()

        // The total is the last element.
        totalRef := totalsShared.[2 * numWarps - 1]

        // Add the block scan to the inclusive scan for the block.
        op scan totalsShared.[warp]

    /// Multi-scan function for all warps in the block.
    let [<ReflectedDefinition>] inline multiScanExcl (init: unit -> 'T) (op:'T -> 'T -> 'T) numWarps logNumWarps tid (x:'T) (totalRef : 'T ref) =
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)
        let warpStride = WARP_SIZE + WARP_SIZE / 2
    
        // Allocate shared memory.
        let shared = __shared__<'T>(numWarps * warpStride).Ptr(0)
        let totalsShared = __shared__<'T>(2 * numWarps).Ptr(0).Volatile() 
        let exclScan = __shared__<'T>(numWarps * WARP_SIZE + 1).Ptr(0)

        let warpShared = (shared + warp * warpStride).Volatile()
        let s = (warpShared + lane + WARP_SIZE / 2).Volatile()
        warpShared.[lane] <- init()
        s.[0] <- x

        // Run inclusive scan on each warp's data.
        let mutable scan = x
        for i = 0 to LOG_WARP_SIZE - 1 do
            let offset = 1 <<< i
            scan <- op scan s.[-offset]   
            if i < LOG_WARP_SIZE - 1 then s.[0] <- scan
        
        if lane = WARP_SIZE - 1 then
            totalsShared.[numWarps + warp] <- scan  

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

            // Store totalsScan shifted by one to the right for an exclusive scan.
            //if 0 < tid && tid < numWarps - 1 then totalsShared.[tid + 1] <- totalsScan 
            //if tid = 0 then totalsShared.[tid] <- init()
            totalsShared.[tid + 1] <- totalsScan

        // Synchronize to make the block scan available to all warps.
        __syncthreads()

        // The total is the last element.
        totalRef := totalsShared.[2 * numWarps - 1]

        // Add the block scan to the inclusive scan for the block.
        if tid = 0 then exclScan.[tid] <- init()
        exclScan.[tid + 1] <- op totalsShared.[warp] scan

        // Synchronize to make the exclusive scan available to all threads.
        __syncthreads()

        exclScan.[tid]

    /// Exclusive scan of range totals.        
    let inline scanReduceKernel (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) (transfExpr:Expr<'T -> 'T>) (plan:Plan)  =
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
            dRangeTotals.[tid] <- multiScanExcl init op numWarps logNumWarps tid x total
        @>

    let inline scanDownSweepKernel (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) (transfExpr:Expr<'T -> 'T>) (plan:Plan) =
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
            let index = warp * valuesPerWarp + lane

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
 
                // Exclusive multi-scan for each thread's scan offset within the block. 
                let localTotal:ref<'T> = ref (init())
                let localScan = multiScanExcl init op numWarps logNumWarps tid scan localTotal
                let scanOffset = op blockScan localScan  
                
                // Apply the scan offset to each exclusive scan and put the values back into the shared memory they came out of.
                for i = 0 to valuesPerThread - 1 do
                    let x = op threadScan.[i] scanOffset
                    warpShared.[offset + i] <- x
                
                // Store the scan back to global memory.
                for i = 0 to valuesPerThread - 1 do
                    let x = threadShared.[i * (WARP_SIZE + 1)]
                    let target = rangeX + index + i * WARP_SIZE
                    dValuesOut.[target] <- x

                // Grab the last element of totals_shared, which was set in Multiscan.
                // This is the total for all the values encountered in this pass.
                blockScan <- op blockScan !localTotal

                rangeX <- rangeX + numValues
        @>

/// Specialized version for sum without expression splicing and slightly more efficient implementation based on inclusive multiscan.
module Sum =  

    /// Multiscan function for warps in the block.
    let [<ReflectedDefinition>] inline multiScan numWarps logNumWarps tid (x:'T) (totalRef : 'T ref) =
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)
        let warpStride = WARP_SIZE + WARP_SIZE / 2

        // Allocate shared memory
        let shared = __shared__<'T>(numWarps * warpStride).Ptr(0)
        let totalsShared = __shared__<'T>(2*numWarps).Ptr(0).Volatile() 

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

    /// Exclusive scan of range totals.
    let inline scanReduceKernel (plan:Plan) =
        let numThreads = plan.numThreadsReduction
        let numWarps = plan.numWarpsReduction
        let logNumWarps = log2 numWarps
        <@ fun numRanges (dRangeTotals:DevicePtr<'T>) ->
            let tid = threadIdx.x
            let x = if tid < numRanges then dRangeTotals.[tid] else 0G
            
            let total:ref<'T> = ref 0G
            let sum = multiScan numWarps logNumWarps tid x total

            // Subtract the value from the inclusive scan for the exclusive scan.
            if tid < numRanges then dRangeTotals.[tid + 1] <- sum

            // Have the first thread in the block set the scan total.
            if tid = 0 then dRangeTotals.[0] <- 0G
        @>

    let inline scanDownSweepKernel (plan:Plan) =
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


/// The length of the arrays dValuesIn and dValuesOut must be multiply of plan.numValues,
/// so that we  do not need to do the range checks in the kernel.
let scanPadding (plan:Plan) n =
    // let padding = (divup n plan.numValues) * plan.numValues
    // try with a safe value first
    plan.numValues + 1

/// Scatter relevant data into padded array dValues.
let scatter (m:Module) (values:'T[]) padding (dValues:DeviceMemory<'T>)=
    dValues.Scatter(values)
    if padding > 0 then
        DevicePtrUtil.Scatter(m.Worker, Array.zeroCreate<'T>(padding), dValues.Ptr + values.Length, padding)

// Gather only the relevant data from GPU.
let gather (m:Module) numValues (dValues:DevicePtr<'T>) =
    let hValues = Array.zeroCreate numValues
    DevicePtrUtil.Gather(m.Worker, dValues, hValues, numValues)
    hValues
    
/// Scan builder to unify scan cuda monad with a function taking the kernel1, kernel2, kernel3 as args.
let inline scanBuilder (plan:Plan) 
                       (kernelExpr1:Plan -> Expr<DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<'T> -> unit>)
                       (kernelExpr2:Plan -> Expr<int -> DevicePtr<'T> -> unit>)
                       (kernelExpr3:Plan -> Expr<DevicePtr<'T> -> DevicePtr<'T> -> DevicePtr<'T> -> DevicePtr<int> -> int -> unit>) = cuda {
    let! kernel1 = kernelExpr1 plan |> defineKernelFunc
    let! kernel2 = kernelExpr2 plan |> defineKernelFunc
    let! kernel3 = kernelExpr3 plan |> defineKernelFunc

    let launch (m:Module) (tc:TimingCollectFunc option) numValues (dValuesIn:DevicePtr<'T>) (dValuesOut:DevicePtr<'T>) (inclusive:bool) =
        let numSm = m.Worker.Device.Attribute(DeviceAttribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        let ranges = plan.blockRanges numSm numValues
        let numRanges = ranges.Length - 1
        let inclusive = if inclusive then 1 else 0
        use dRanges = m.Worker.Malloc(ranges)
        use dRangeTotals = m.Worker.Malloc<'T>(Array.zeroCreate (numRanges + 1))  

        printfn "====> ranges = %A" ranges
        printfn "0) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())

        //let lp = LaunchParam (numRanges-1, plan.numThreads) |> setDiagnoser (diagnose "upSweep")
        let lp = LaunchParam (numRanges, plan.numThreads)
        let lp = match tc with Some(tc) -> lp |> Engine.setDiagnoser ((tcToDiag tc) "upsweep") | None -> lp
        kernel1.Launch m lp dValuesIn dRanges.Ptr dRangeTotals.Ptr

        printfn "1) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())

        let lp = LaunchParam(1, plan.numThreadsReduction)
        let lp = match tc with Some(tc) -> lp |> Engine.setDiagnoser ((tcToDiag tc) "reduce") | None -> lp
        kernel2.Launch m lp numRanges dRangeTotals.Ptr

        printfn "2) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())

        let lp = LaunchParam(numRanges, plan.numThreads)
        let lp = match tc with Some(tc) -> lp |> Engine.setDiagnoser ((tcToDiag tc) "downsweep") | None -> lp
        kernel3.Launch m lp dValuesIn dValuesOut dRangeTotals.Ptr dRanges.Ptr inclusive

        printfn "3) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())

    return PFunc(fun (m:Module) ->
        let launch = launch m
        { new IScan<'T> with
            member this.Scatter values padding dValues = scatter m values padding dValues
            member this.Gather numValues dValuesOut = gather m numValues dValuesOut 
            member this.Scan(values, inclusive) =
                let padding = scanPadding plan values.Length
                let size = values.Length + padding
                use dValuesIn = m.Worker.Malloc<'T>(size)
                use dValuesOut = m.Worker.Malloc<'T>(size)                 
                this.Scatter values padding dValuesIn
                launch None values.Length dValuesIn.Ptr dValuesOut.Ptr inclusive
                this.Gather values.Length dValuesOut.Ptr
            member this.Scan(values, inclusive, tc) =
                let padding = scanPadding plan values.Length
                let size = values.Length + padding
                use dValuesIn = m.Worker.Malloc<'T>(size)
                use dValuesOut = m.Worker.Malloc<'T>(size)                 
                launch (Some tc) values.Length dValuesIn.Ptr dValuesOut.Ptr inclusive
                this.Gather values.Length dValuesOut.Ptr
            member this.Scan(numValues, dValuesIn, dValuesOut, inclusive) =
                launch None numValues dValuesIn dValuesOut inclusive
        } ) }

/// <summary>
/// Global scan algorithm template. 
/// </summary>
let inline scan (plan:Plan) = 
    scanBuilder plan Sum.reduceUpSweepKernel Sum.scanReduceKernel Sum.scanDownSweepKernel

/// <summary>
/// Global scan algorithm template. 
/// </summary>
let inline genericScan (plan:Plan) (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (transf:Expr<'T -> 'T>)  =
    scanBuilder plan (Generic.reduceUpSweepKernel init op transf) 
                     (Generic.scanReduceKernel init op transf) 
                     (Generic.scanDownSweepKernel init op transf)

