module Alea.CUDA.Extension.Scan

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.Timing
open Alea.CUDA.Extension.Reduce

// IScan represents a scanner. It is created by given the number of scanning values.
// Then it calcuate the ranges, which is an integer array of size numRanges + 1.
// The member NumRangeTotals gives you a hint on the minumal size requirement on rangeTotals.
// The member Scan takes this signature:
// Scan : hint -> ranges -> rangeTotals -> values -> results -> inclusive -> unit
type IScan<'T when 'T : unmanaged> =
    abstract Ranges : int[]
    abstract NumRangeTotals : int
    abstract Scan : ActionHint -> DevicePtr<int> -> DevicePtr<'T> -> DevicePtr<'T> -> DevicePtr<'T> -> bool -> unit

module Generic = 
    /// Multi-scan function for all warps in the block.
    let [<ReflectedDefinition>] multiScan (init:unit -> 'T) (op:'T -> 'T -> 'T) numWarps logNumWarps tid (x:'T) (totalRef:'T ref) =
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)
        let warpStride = WARP_SIZE + WARP_SIZE / 2 + 1
    
        // Allocate shared memory.
        let shared = __shared__<'T>(numWarps * warpStride).Ptr(0).Volatile()
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
    let [<ReflectedDefinition>] multiScanExcl (init: unit -> 'T) (op:'T -> 'T -> 'T) numWarps logNumWarps tid (x:'T) (totalRef : 'T ref) =
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)
        let warpStride = WARP_SIZE + WARP_SIZE / 2 + 1
    
        // Allocate shared memory.
        let shared = __shared__<'T>(numWarps * warpStride).Ptr(0).Volatile()
        let totalsShared = __shared__<'T>(2 * numWarps).Ptr(0).Volatile() 
        let exclScan = __shared__<'T>(numWarps * WARP_SIZE + 1).Ptr(0).Volatile()

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
    let scanReduceKernel (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) (transfExpr:Expr<'T -> 'T>) (plan:Plan)  =
        let numThreads = plan.NumThreadsReduction
        let numWarps = plan.NumWarpsReduction
        let logNumWarps = log2 numWarps
        <@ fun numRanges (dRangeTotals:DevicePtr<'T>) ->
            let init = %initExpr
            let op = %opExpr
            let transf = %transfExpr

            let tid = threadIdx.x
            let x = if tid < numRanges then dRangeTotals.[tid] else init()
            let total:ref<'T> = ref (init())
            let sum = multiScan init op numWarps logNumWarps tid x total
            // Shift the value from the inclusive scan for the exclusive scan.
            if tid < numRanges then dRangeTotals.[tid + 1] <- sum
            // Have the first thread in the block set the scan total.
            if tid = 0 then dRangeTotals.[0] <- init() @>

    let scanDownSweepKernel (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) (transfExpr:Expr<'T -> 'T>) (plan:Plan) =
        let numWarps = plan.NumWarps
        let numValues = plan.NumValues
        let valuesPerThread = plan.ValuesPerThread
        let valuesPerWarp = plan.ValuesPerWarp 
        let logNumWarps = log2 numWarps
        let size = numWarps * valuesPerThread * (WARP_SIZE + 1)
        <@ fun (N:int) (dValuesIn:DevicePtr<'T>) (dValuesOut:DevicePtr<'T>) (dRangeTotals:DevicePtr<'T>) (dRanges:DevicePtr<int>) (inclusive:int) ->
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
                    let x = if source < N then transf dValuesIn.[source] else (init())
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
                    if target < N then dValuesOut.[target] <- x

                // Grab the last element of totals_shared, which was set in Multiscan.
                // This is the total for all the values encountered in this pass.
                blockScan <- op blockScan !localTotal

                rangeX <- rangeX + numValues @>

/// Specialized version for sum without expression splicing and slightly more efficient implementation based on inclusive multiscan.
module Sum =  
    /// Multiscan function for warps in the block.
    let [<ReflectedDefinition>] inline multiScan numWarps logNumWarps tid (x:'T) (totalRef:'T ref) =
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)
        let warpStride = WARP_SIZE + WARP_SIZE / 2 + 1

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
        let numThreads = plan.NumThreadsReduction
        let numWarps = plan.NumWarpsReduction
        let logNumWarps = log2 numWarps
        <@ fun numRanges (dRangeTotals:DevicePtr<'T>) ->
            let tid = threadIdx.x
            let x = if tid < numRanges then dRangeTotals.[tid] else 0G
            let total:ref<'T> = ref 0G
            let sum = multiScan numWarps logNumWarps tid x total
            // Shift the value from the inclusive scan for the exclusive scan.
            if tid < numRanges then dRangeTotals.[tid + 1] <- sum
            // Have the first thread in the block set the scan total.
            if tid = 0 then dRangeTotals.[0] <- 0G @>

    let inline scanDownSweepKernel (plan:Plan) =
        let numWarps = plan.NumWarps
        let numValues = plan.NumValues
        let valuesPerThread = plan.ValuesPerThread
        let valuesPerWarp = plan.ValuesPerWarp 
        let logNumWarps = log2 numWarps
        let size = numWarps * valuesPerThread * (WARP_SIZE + 1)
        <@ fun (N:int) (dValuesIn:DevicePtr<'T>) (dValuesOut:DevicePtr<'T>) (dRangeTotals:DevicePtr<'T>) (dRanges:DevicePtr<int>) (inclusive:int) ->
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
                    let x = if source < N then dValuesIn.[source] else 0G
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
                    if target < N then dValuesOut.[target] <- x

                // Grab the last element of totals_shared, which was set in Multiscan.
                // This is the total for all the values encountered in this pass.
                blockScan <- blockScan + !localTotal

                rangeX <- rangeX + numValues @>

// UpsweepKernel values ranges rangeTotals
type UpsweepKernel<'T> = Reduce.UpsweepKernel<'T>
// ReduceKernel numRanges rangeTotals
type ReduceKernel<'T> = Reduce.ReduceKernel<'T>
// DownsweepKernel numValues -> values -> results -> rangeTotals -> ranges -> inclusive
type DownsweepKernel<'T> = int -> DevicePtr<'T> -> DevicePtr<'T> -> DevicePtr<'T> -> DevicePtr<int> -> int -> unit

/// Scan builder to unify scan cuda monad with a function taking the kernel1, kernel2, kernel3 as args.
let build (upsweep:Plan -> Expr<UpsweepKernel<'T>>) (reduce:Plan -> Expr<ReduceKernel<'T>>) (downsweep:Plan -> Expr<DownsweepKernel<'T>>) = cuda {
    let plan = if sizeof<'T> > 4 then plan64 else plan32
    let! upsweep = upsweep plan |> defineKernelFuncWithName "scan_upsweep"
    let! reduce = reduce plan |> defineKernelFuncWithName "scan_reduce"
    let! downsweep = downsweep plan |> defineKernelFuncWithName "scan_downsweep"

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let upsweep = upsweep.Apply m
        let reduce = reduce.Apply m
        let downsweep = downsweep.Apply m
        let numSm = m.Worker.Device.NumSm

        // factory to create scanner (IScan)
        fun (n:int) ->
            let ranges = plan.BlockRanges numSm n
            let numRanges = ranges.Length - 1

            let lpUpsweep = LaunchParam(numRanges, plan.NumThreads)
            let lpReduce = LaunchParam(1, plan.NumThreadsReduction)
            let lpDownsweep = LaunchParam(numRanges, plan.NumThreads)

            let launch (hint:ActionHint) (ranges:DevicePtr<int>) (rangeTotals:DevicePtr<'T>) (input:DevicePtr<'T>) (output:DevicePtr<'T>) (inclusive:bool) =
                let inclusive = if inclusive then 1 else 0
                let lpUpsweep = lpUpsweep |> hint.ModifyLaunchParam
                let lpReduce = lpReduce |> hint.ModifyLaunchParam
                let lpDownsweep = lpDownsweep |> hint.ModifyLaunchParam

                fun () ->
                    upsweep.Launch lpUpsweep input ranges rangeTotals
                    reduce.Launch lpReduce numRanges rangeTotals
                    downsweep.Launch lpDownsweep n input output rangeTotals ranges inclusive
                |> worker.Eval // the three kernels should be launched together without interrupt.

            { new IScan<'T> with
                member this.Ranges = ranges
                member this.NumRangeTotals = numRanges + 1
                member this.Scan lphint ranges rangeTotals input output inclusive = launch lphint ranges rangeTotals input output inclusive
            } ) }

/// <summary>
/// Global scan algorithm template. 
/// </summary>
let generic (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (transf:Expr<'T -> 'T>) =
    let upsweep = Generic.reduceUpSweepKernel init op transf
    let reduce = Generic.scanReduceKernel init op transf
    let downsweep = Generic.scanDownSweepKernel init op transf
    build upsweep reduce downsweep

/// <summary>
/// Global scan algorithm template. 
/// </summary>
let inline sum () = 
    let upsweep = Sum.reduceUpSweepKernel
    let reduce = Sum.scanReduceKernel
    let downsweep = Sum.scanDownSweepKernel
    build upsweep reduce downsweep

