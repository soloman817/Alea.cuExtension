module Alea.CUDA.Extension.Reduce

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Util

type IReduce<'T when 'T:unmanaged> =
    abstract Ranges : int[] // Ranges is a host array of int, which will be scattered later with blob
    abstract NumRangeTotals : int // NumRangeTotals is a length of rangeTotals, which will be used to calc blob size later
    // lphint -> ranges -> rangeTotals -> values -> unit (result is rangeTotals.[0], you should create dscalar for it)
    abstract Reduce : LPHint -> DevicePtr<int> -> DevicePtr<'T> -> DevicePtr<'T> -> unit

type Plan =
    {numThreads:int; valuesPerThread:int; numThreadsReduction:int; blockPerSm:int} 
    member this.valuesPerWarp = this.valuesPerThread * WARP_SIZE
    member this.numWarps = this.numThreads / WARP_SIZE
    member this.numWarpsReduction = this.numThreadsReduction / WARP_SIZE
    member this.numValues = this.numThreads * this.valuesPerThread

    /// <summary>
    /// Finds the ranges for each block to process. 
    /// Note that each range must begin a multiple of the block size.
    /// It returns a sequence of length 1 + effective num blocks (which is equal to min numRanges numBricks).
    /// The range pairs can be obtained by blockRanges numRanges count |> Seq.pairwise
    /// </summary>
    /// <param name="numSm">number of SM</param>
    /// <param name="count">length of the array</param>    
    member this.blockRanges numSm count =
        let numBlocks = this.blockPerSm * numSm 
        let blockSize = this.numThreads * this.valuesPerThread     
        let numBricks = divup count blockSize
        let numBlocks = min numBlocks numBricks 

        let brickDivQuot = numBricks / numBlocks 
        let brickDivRem = numBricks % numBlocks
        
        let ranges = [| 1..numBlocks |] |> Array.scan (fun s i -> 
            let bricks = if (i-1) < brickDivRem then brickDivQuot + 1 else brickDivQuot
            min (s + bricks * blockSize) count) 0
           
        ranges

/// The standard thread plan for 32 bit values.
let plan32 = {numThreads = 1024; valuesPerThread = 4; numThreadsReduction = 256; blockPerSm = 2}
    
/// The thread plan for 64 bit values such as float.
let plan64 = {numThreads = 512; valuesPerThread = 4; numThreadsReduction = 256; blockPerSm = 2}

module Generic = 
    /// Multi-reduce function for all warps in the block.
    let [<ReflectedDefinition>] multiReduce (init: unit -> 'T) (op:'T -> 'T -> 'T) numWarps logNumWarps tid (x:'T) =
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)

        let warpSharedStride = WARP_SIZE + WARP_SIZE / 2 + 1
        let warpSharedSize = numWarps * warpSharedStride
        let totalSharedSize = 2 * numWarps

        let shared = __shared__<'T>(warpSharedSize + totalSharedSize).Ptr(0).Volatile()
        let warpShared = shared + warp * warpSharedStride
        let totalShared = shared + warpSharedSize

        // inclusive scan on warps
        warpShared.[lane] <- init() // zero set the first half warp for 0, will be used in scan
        let s0 = warpShared + WARP_SIZE / 2 + lane
        s0.[0] <- x
        for i = 0 to LOG_WARP_SIZE - 1 do
            let offset = 1 <<< i
            s0.[0] <- op s0.[0] s0.[-offset]

        // Synchronize to make all the totals available to the reduction code.
        __syncthreads()

        // now reduce the warps totoal
        if tid < numWarps then
            let s1 = totalShared + numWarps + tid
            let total = warpShared.[warpSharedStride * tid + WARP_SIZE / 2 + WARP_SIZE - 1]
            totalShared.[tid] <- init()
            s1.[0] <- total

            for i = 0 to logNumWarps - 1 do
                let offset = 1 <<< i
                s1.[0] <- op s1.[0] s1.[-offset]

        // Synchronize to make the block scan available to all warps.
        __syncthreads()

        // The total is the last elements
        totalShared.[totalSharedSize - 1]

    /// Reduces ranges and store reduced values in array of the range totals.         
    let reduceUpSweepKernel (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) (transfExpr:Expr<'T -> 'T>) (plan:Plan) =
        let numThreads = plan.numThreads
        let numWarps = plan.numWarps
        let logNumWarps = log2 numWarps
        <@ fun (dValues:DevicePtr<'T>) (dRanges:DevicePtr<int>) (dRangeTotals:DevicePtr<'T>) ->
            let init = %initExpr
            let op = %opExpr
            let transf = %transfExpr

            // Each block is processing a range.
            let range = blockIdx.x
            let tid = threadIdx.x
            let rangeX = dRanges.[range]
            let rangeY = dRanges.[range + 1]

            // Loop through all elements in the interval, adding up values.
            // There is no need to synchronize until we perform the multireduce.
            let mutable reduced = init()
            let mutable index = rangeX + tid
            while index < rangeY do              
                reduced <- op reduced (transf dValues.[index]) 
                index <- index + numThreads

            // Get the total.
            let total = multiReduce init op numWarps logNumWarps tid reduced 

            if tid = 0 then dRangeTotals.[range] <- total
        @>

    /// Reduces range totals to a single total, which is written back to the first element in the range totals input array.
    let reduceRangeTotalsKernel (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) (plan:Plan) =
        let numThreads = plan.numThreadsReduction
        let numWarps = plan.numWarpsReduction
        let logNumWarps = log2 numWarps
        <@ fun numRanges (dRangeTotals:DevicePtr<'T>) ->
            let init = %initExpr
            let op = %opExpr

            let tid = threadIdx.x
            let x = if tid < numRanges then dRangeTotals.[tid] else init()          
            let total = multiReduce init op numWarps logNumWarps tid x 

            // Have the first thread in the block set the total and store it in the first element of the input array.
            if tid = 0 then dRangeTotals.[0] <- total
        @>

/// Specialized version for sum without expression splicing to check performance impact.
module Sum =   
        
    /// Multi-reduce function for a warps in the block.
    let [<ReflectedDefinition>] inline multiReduce numWarps logNumWarps tid (x:'T) =
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)

        let warpSharedStride = WARP_SIZE + WARP_SIZE / 2 + 1
        let warpSharedSize = numWarps * warpSharedStride
        let totalSharedSize = 2 * numWarps

        let shared = __shared__<'T>(warpSharedSize + totalSharedSize).Ptr(0).Volatile()
        let warpShared = shared + warp * warpSharedStride
        let totalShared = shared + warpSharedSize

        // inclusive scan on warps
        warpShared.[lane] <- 0G // zero set the first half warp for 0, will be used in scan
        let s0 = warpShared + WARP_SIZE / 2 + lane
        s0.[0] <- x
        for i = 0 to LOG_WARP_SIZE - 1 do
            let offset = 1 <<< i
            s0.[0] <- s0.[0] + s0.[-offset]

        // Synchronize to make all the totals available to the reduction code.
        __syncthreads()

        // now reduce the warps totoal
        if tid < numWarps then
            let s1 = totalShared + numWarps + tid
            let total = warpShared.[warpSharedStride * tid + WARP_SIZE / 2 + WARP_SIZE - 1]
            totalShared.[tid] <- 0G
            s1.[0] <- total

            for i = 0 to logNumWarps - 1 do
                let offset = 1 <<< i
                s1.[0] <- s1.[0] + s1.[-offset]

        // Synchronize to make the block scan available to all warps.
        __syncthreads()

        // The total is the last elements
        totalShared.[totalSharedSize - 1]

    /// Reduces ranges and store reduced values in array of the range totals.    
    let inline reduceUpSweepKernel (plan:Plan) =
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
            let total = multiReduce numWarps logNumWarps tid sum 

            if tid = 0 then dRangeTotals.[block] <- total
        @>

    /// Reduces range totals to a single total, which is written back to the first element in the range totals input array.
    let inline reduceRangeTotalsKernel (plan:Plan) =
        let numThreads = plan.numThreadsReduction
        let numWarps = plan.numWarpsReduction
        let logNumWarps = log2 numWarps
        <@ fun numRanges (dRangeTotals:DevicePtr<'T>) ->
            let tid = threadIdx.x
            let x = if tid < numRanges then dRangeTotals.[tid] else 0G         
            let total = multiReduce numWarps logNumWarps tid x 

            // Have the first thread in the block set the range total.
            if tid = 0 then dRangeTotals.[0] <- total
        @>

let reduceBuilder (kernelExpr1:Plan -> Expr<DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<'T> -> unit>)
                  (kernelExpr2:Plan -> Expr<int -> DevicePtr<'T> -> unit>) = cuda {
    let plan = if sizeof<'T> >= 8 then plan64 else plan32

    let! upsweep = kernelExpr1 plan |> defineKernelFuncWithName "upsweep"
    let! reduce = kernelExpr2 plan |> defineKernelFuncWithName "reduce"

    return PFunc(fun (m:Module) ->
        let upsweep = upsweep.Apply m
        let reduce = reduce.Apply m
        let worker = m.Worker
        let numSm = m.Worker.Device.NumSm

        // factory to create reducer (IReduce)
        fun (n:int) ->
            let ranges = plan.blockRanges numSm n
            let numRanges = ranges.Length - 1

            let lpUpsweep = LaunchParam(numRanges, plan.numThreads)
            let lpReduce = LaunchParam(1, plan.numThreadsReduction)

            let launch (lphint:LPHint) (ranges:DevicePtr<int>) (rangeTotals:DevicePtr<'T>) (values:DevicePtr<'T>) =
                let lpUpsweep = lpUpsweep |> lphint.Modify
                let lpReduce = lpReduce |> lphint.Modify

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

