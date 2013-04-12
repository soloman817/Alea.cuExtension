module Alea.CUDA.Extension.Reduce

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Util

[<Struct;Align(8)>]
type Range =
    val x : int
    val y : int

    [<ReflectedDefinition>]
    new (x, y) = { x = x; y = y }

    override this.ToString() = sprintf "(%d,%d)" this.x this.y

// IReduce represents a reducer. It is created by given the number of reducing values.
// Then it calcuate the ranges, which is an integer array of size numRanges + 1.
// The member NumRangeTotals gives you a hint on the minumal size requirement on rangeTotals.
// The member Reduce takes this signature:
// Reduce : hint -> ranges -> rangeTotals -> values -> unit
// After calling this function, the result is stored in rangeTotals[0], you should decide
// a way to extract that value, because it will dispear one you run Reduce for another input.
type IReduce<'T when 'T:unmanaged> =
    abstract Ranges : Range[]
    abstract NumRangeTotals : int
    abstract Reduce : ActionHint -> DevicePtr<Range> -> DevicePtr<'T> -> DevicePtr<'T> -> unit

type Plan =
    { NumThreads:int; ValuesPerThread:int; NumThreadsReduction:int; BlockPerSm:int} 
    member this.ValuesPerWarp = this.ValuesPerThread * WARP_SIZE
    member this.NumWarps = this.NumThreads / WARP_SIZE
    member this.NumWarpsReduction = this.NumThreadsReduction / WARP_SIZE
    member this.NumValues = this.NumThreads * this.ValuesPerThread

    /// <summary>
    /// Finds the ranges for each block to process. 
    /// Note that each range must begin a multiple of the block size.
    /// It returns a sequence of length 1 + effective num blocks (which is equal to min numRanges numBricks).
    /// The range pairs can be obtained by blockRanges numRanges count |> Seq.pairwise
    /// </summary>
    /// <param name="numSm">number of SM</param>
    /// <param name="count">length of the array</param>    
    member this.BlockRanges numSm count =
        let numBlocks = min (this.BlockPerSm * numSm) this.NumThreadsReduction
        let blockSize = this.NumThreads * this.ValuesPerThread     
        let numBricks = divup count blockSize
        let numBlocks = min numBlocks numBricks 

        let brickDivQuot = numBricks / numBlocks 
        let brickDivRem = numBricks % numBlocks

        let ranges = [| 1..numBlocks |] |> Array.scan (fun s i -> 
            let bricks = if (i-1) < brickDivRem then brickDivQuot + 1 else brickDivQuot
            min (s + bricks * blockSize) count) 0
           
        ranges |> Seq.pairwise |> Seq.map (fun (x, y) -> Range(x, y)) |> Array.ofSeq

type Planner =
    | Default
    | Specific of Plan
    | ByWorker of DeviceWorker

module Generic = 
    /// Multi-reduce function for all warps in the block.
    let [<ReflectedDefinition>] multiReduce (init: unit -> 'T) (op:'T -> 'T -> 'T) numWarps logNumWarps tid (x:'T) =
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)
        let warpStride = WARP_SIZE + WARP_SIZE / 2 + 1
        let sharedSize = numWarps * warpStride
        let shared = __shared__<'T>(sharedSize).Ptr(0).Volatile()
        let warpShared = shared + warp * warpStride     
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

        // Last line of warp stores the warp scan.
        if lane = WARP_SIZE - 1 then totalsShared.[numWarps + warp] <- warpScan  

        // Synchronize to make all the totals available to the reduction code.
        __syncthreads()

        // Run an exclusive scan for the warp scans. 
        if tid < numWarps then
            // Grab the block total for the tid'th block. This is the last element
            // in the block's scanned sequence. This operation avoids bank conflicts.
            let total = totalsShared.[numWarps + tid]
            totalsShared.[tid] <- init()
            let s = totalsShared + numWarps + tid

            let mutable totalsScan = total
            for i = 0 to logNumWarps - 1 do
                let offset = 1 <<< i
                totalsScan <- op totalsScan s.[-offset]
                s.[0] <- totalsScan

        // Synchronize to make the block scan available to all warps.
        __syncthreads()

        // The total is the last element.
        totalsShared.[2 * numWarps - 1]

    /// Reduces ranges and store reduced values in array of the range totals.         
    let reduceUpSweepKernel (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) (transfExpr:Expr<'T -> 'T>) (plan:Plan) =
        let numThreads = plan.NumThreads
        let numWarps = plan.NumWarps
        let logNumWarps = log2 numWarps
        <@ fun (dValues:DevicePtr<'T>) (dRanges:DevicePtr<Range>) (dRangeTotals:DevicePtr<'T>) ->
            let init = %initExpr
            let op = %opExpr
            let transf = %transfExpr

            // Each block is processing a range.
            let rangeId = blockIdx.x
            let tid = threadIdx.x
            let range = dRanges.[rangeId]

            // Loop through all elements in the interval, adding up values.
            // There is no need to synchronize until we perform the multireduce.
            let mutable reduced = init()
            let mutable index = range.x + tid
            while index < range.y do              
                reduced <- op reduced (transf dValues.[index]) 
                index <- index + numThreads

            // Get the total.
            let total = multiReduce init op numWarps logNumWarps tid reduced 

            if tid = 0 then dRangeTotals.[rangeId] <- total @>

    /// Reduces range totals to a single total, which is written back to the first element in the range totals input array.
    let reduceRangeTotalsKernel (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) (plan:Plan) =
        let numThreads = plan.NumThreadsReduction
        let numWarps = plan.NumWarpsReduction
        let logNumWarps = log2 numWarps
        <@ fun numRanges (dRangeTotals:DevicePtr<'T>) ->
            let init = %initExpr
            let op = %opExpr

            let tid = threadIdx.x
            let x = if tid < numRanges then dRangeTotals.[tid] else (init())
            let total = multiReduce init op numWarps logNumWarps tid x

            // Have the first thread in the block set the total and store it in the first element of the input array.
            if tid = 0 then dRangeTotals.[0] <- total @>

/// Specialized version for sum without expression splicing to check performance impact.
module Sum =   
    /// Multi-reduce function for a warps in the block.
    let [<ReflectedDefinition>] inline multiReduce numWarps logNumWarps tid (x:'T) =
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)
        let warpStride = WARP_SIZE + WARP_SIZE / 2 + 1
        let sharedSize = numWarps * warpStride
        let shared = __shared__<'T>(sharedSize).Ptr(0).Volatile()
        let warpShared = shared + warp * warpStride     
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
            let s = totalsShared + numWarps + tid

            let mutable totalsSum = total
            for i = 0 to logNumWarps - 1 do
                let offset = 1 <<< i
                totalsSum <- totalsSum + s.[-offset]
                s.[0] <- totalsSum

        // Synchronize to make the block scan available to all warps.
        __syncthreads()

        // The total is the last element.
        totalsShared.[2 * numWarps - 1]

    /// Reduces ranges and store reduced values in array of the range totals.    
    let inline reduceUpSweepKernel (plan:Plan) =
        let numThreads = plan.NumThreads
        let numWarps = plan.NumWarps
        let logNumWarps = log2 numWarps
        <@ fun (dValues:DevicePtr<'T>) (dRanges:DevicePtr<Range>) (dRangeTotals:DevicePtr<'T>) ->
            let block = blockIdx.x
            let tid = threadIdx.x
            let range = dRanges.[block]

            // Loop through all elements in the interval, adding up values.
            // There is no need to synchronize until we perform the multiscan.
            let mutable sum = 0G
            let mutable index = range.x + tid
            while index < range.y do              
                sum <- sum + dValues.[index] 
                index <- index + numThreads

            // A full multiscan is unnecessary here - we really only need the total.
            let total = multiReduce numWarps logNumWarps tid sum 

            if tid = 0 then dRangeTotals.[block] <- total @>

    /// Reduces range totals to a single total, which is written back to the first element in the range totals input array.
    let inline reduceRangeTotalsKernel (plan:Plan) =
        let numThreads = plan.NumThreadsReduction
        let numWarps = plan.NumWarpsReduction
        let logNumWarps = log2 numWarps
        <@ fun numRanges (dRangeTotals:DevicePtr<'T>) ->
            let tid = threadIdx.x

            let tid = threadIdx.x
            let x = if tid < numRanges then dRangeTotals.[tid] else 0G
            let total = multiReduce numWarps logNumWarps tid x

            // Have the first thread in the block set the range total.
            if tid = 0 then dRangeTotals.[0] <- total @>

// UpsweepKernel values ranges rangeTotals
type UpsweepKernel<'T> = DevicePtr<'T> -> DevicePtr<Range> -> DevicePtr<'T> -> unit
// ReduceKernel numRanges rangeTotals
type ReduceKernel<'T> = int -> DevicePtr<'T> -> unit

let plan32 : Plan = { NumThreads = 1024; ValuesPerThread = 4; NumThreadsReduction = 256; BlockPerSm = 1 }
let plan64 : Plan = { NumThreads = 512; ValuesPerThread = 4; NumThreadsReduction = 256; BlockPerSm = 1 }

let build (planner:Planner) (upsweep:Plan -> Expr<UpsweepKernel<'T>>) (reduce:Plan -> Expr<ReduceKernel<'T>>) = cuda {
    let plan =
        match planner with
        | Planner.Default -> if sizeof<'T> > 4 then plan64 else plan32
        | Planner.Specific(plan) -> plan
        | Planner.ByWorker(worker) -> failwith "TODO"

    let! upsweep = upsweep plan |> defineKernelFuncWithName "reduce_upsweep"
    let! reduce = reduce plan |> defineKernelFuncWithName "reduce_reduce"

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let upsweep = upsweep.Apply m
        let reduce = reduce.Apply m
        let numSm = worker.Device.NumSm

        // factory to create reducer (IReduce)
        fun (n:int) ->
            let ranges = plan.BlockRanges numSm n
            let numRanges = ranges.Length

            let lpUpsweep = LaunchParam(numRanges, plan.NumThreads)
            let lpReduce = LaunchParam(1, plan.NumThreadsReduction)

            let launch (hint:ActionHint) (ranges:DevicePtr<Range>) (rangeTotals:DevicePtr<'T>) (values:DevicePtr<'T>) =
                let lpUpsweep = lpUpsweep |> hint.ModifyLaunchParam
                let lpReduce = lpReduce |> hint.ModifyLaunchParam

                fun () ->
                    
                    // Launch range reduction kernel to calculate the totals per range.
                    upsweep.Launch lpUpsweep values ranges rangeTotals

                    // Need to aggregate the block sums as well.
                    if numRanges > 1 then reduce.Launch lpReduce numRanges rangeTotals
                |> worker.Eval // the two kernels should be launched together without interrupt.

            { new IReduce<'T> with
                member this.Ranges = ranges
                member this.NumRangeTotals = numRanges
                member this.Reduce lphint ranges rangeTotals values = launch lphint ranges rangeTotals values
            } ) }

let generic planner init op transf =
    let upsweep = Generic.reduceUpSweepKernel init op transf
    let reduce = Generic.reduceRangeTotalsKernel init op
    build planner upsweep reduce

let inline sum planner =
    let upsweep = Sum.reduceUpSweepKernel
    let reduce = Sum.reduceRangeTotalsKernel
    build planner upsweep reduce

