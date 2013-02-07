module Alea.CUDA.Extension.Reduce

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension.Common

type IReduce<'T> =
    abstract Reduce : int * DevicePtr<'T> -> 'T
    abstract Reduce : 'T[] -> 'T

type Plan =
    {numThreads:int; valuesPerThread:int; numThreadsReduction:int; blockPerSm:int} 
    member this.valuesPerWarp = this.valuesPerThread * WARP_SIZE
    member this.numWarps = this.numThreads / WARP_SIZE
    member this.numWarpsReduction = this.numThreadsReduction / WARP_SIZE

    /// <summary>
    /// Finds the ranges for each block to process. 
    /// Note that each range must begin a multiple of the block size.
    /// It returns a sequence of length 1 + effective num blocks (which is equal to min numBlocks numBricks).
    /// The range pairs can be obtained by blockRanges numBlocks count |> Seq.pairwise
    /// </summary>
    /// <param name="numBlocks">maximal num blocks to be launched, equal to blocks per SM multiplied with number of SMs</param>
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
let plan32 = {numThreads = 1024; valuesPerThread = 4; numThreadsReduction = 256; blockPerSm = 1}
    
/// The thread plan for 64 bit values such as float.
let plan64 = {numThreads = 512; valuesPerThread = 4; numThreadsReduction = 256; blockPerSm = 1}

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

        // unroll
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

            // unroll
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
        <@ fun (dValues:DevicePtr<'T>) (dRanges:DevicePtr<int>) (dBlockTotals:DevicePtr<'T>) ->
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
            let total:ref<'T> = ref 0G
            let blockSum = multiScan numWarps logNumWarps tid sum total

            if tid = 0 then dBlockTotals.[block] <- !total
        @>

    let inline reduceKernel (plan:Plan) =
        let numThreads = plan.numThreadsReduction
        let numWarps = plan.numWarpsReduction
        let logNumWarps = log2 numWarps
        <@ fun numBlocks (dBlockTotals:DevicePtr<'T>) ->
            let tid = threadIdx.x
            let x = if tid < numBlocks then dBlockTotals.[tid] else 0G
            
            let total:ref<'T> = ref 0G
            let sum = multiScan numWarps logNumWarps tid x total

            // Subtract the value from the inclusive scan for the exclusive scan.
            if tid < numBlocks then dBlockTotals.[tid] <- sum - x

            // Have the first thread in the block set the scan total.
            if tid = 0 then dBlockTotals.[numBlocks] <- !total
        @>

    let inline reduce (plan:Plan) = cuda {
        let! upSweep = upSweepKernel plan |> defineKernelFuncWithName "upSweep"
        let! reduce = reduceKernel plan |> defineKernelFuncWithName "reduce"

        let launch (m:Module) (n:int) (values:DevicePtr<'T>) =
            let numSm = m.Worker.Device.Attribute(DeviceAttribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            let ranges = plan.blockRanges numSm n
            let numBlocks = ranges.Length - 1
            use dRanges = m.Worker.Malloc(ranges)
            use dBlockTotals = m.Worker.Malloc<'T>(Array.zeroCreate (numBlocks + 1))  
            
            // Launch block reduction kernel to calculate the totals per range
            let lp = LaunchParam(numBlocks, plan.numThreads)
            upSweep.Launch m lp values dRanges.Ptr dBlockTotals.Ptr

            if numBlocks = 1 then
                // We are finished 
                let blockTotals = dBlockTotals.ToHost()
                blockTotals.[0]
            else
                // Need to aggregate the block sums as well
                let lp = LaunchParam(1, plan.numThreadsReduction)
                reduce.Launch m lp numBlocks dBlockTotals.Ptr
                let blockTotals = dBlockTotals.ToHost()
                blockTotals.[blockTotals.Length - 1]

        return PFunc(fun (m:Module) ->
            let launch = launch m
            { new IReduce<'T> with
                member this.Reduce (n, values) = 
                    launch n values 
                member this.Reduce values =  
                    let dValues = m.Worker.Malloc(values)
                    launch values.Length dValues.Ptr
            } ) }

module Generic =   
        
    /// Multiscan function for a warps in the block with shared memory passed in from caller.
    let [<ReflectedDefinition>] inline multiScan (init: unit -> 'T) (op:'T -> 'T -> 'T) numWarps logNumWarps tid (x:'T) (totalRef : 'T ref) =
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)
        let warpStride = WARP_SIZE + WARP_SIZE / 2
        let sharedSize = numWarps * warpStride
        let shared = __shared__<'T>(sharedSize).Ptr(0)
        let warpShared = (shared + warp * warpStride).Volatile()      
        let s = warpShared + (lane + WARP_SIZE / 2)

        warpShared.[lane] <- init()  
        s.[0] <- x

        // Run inclusive scan on each warp's data.
        let mutable sum = x

        // unroll
        for i = 0 to LOG_WARP_SIZE - 1 do
            let offset = 1 <<< i
            sum <- op sum s.[-offset]   
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
            totalsShared.[tid] <- init()
            let s = (totalsShared + numWarps + tid).Volatile()  

            let mutable totalsSum = total

            // unroll
            for i = 0 to logNumWarps - 1 do
                let offset = 1 <<< i
                totalsSum <- op totalsSum s.[-offset]
                s.[0] <- totalsSum

            // Subtract total from totalsSum for an exclusive scan. 
            // TODO how to handle that
            totalsShared.[tid] <- totalsSum - total

        // Synchronize to make the block scan available to all warps.
        __syncthreads()

        // The total is the last element.
        totalRef := totalsShared.[2 * numWarps - 1]

        // Add the block scan to the inclusive sum for the block.
        sum + totalsShared.[warp]
        
    let inline upSweepKernel (plan:Plan) (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) (transfExpr:Expr<'T -> 'T>) =
        let numThreads = plan.numThreads
        let numWarps = plan.numWarps
        let logNumWarps = log2 numWarps
        <@ fun (dValues:DevicePtr<'T>) (dRanges:DevicePtr<int>) (dBlockTotals:DevicePtr<'T>) ->
            let init = %initExpr
            let op = %opExpr
            let transf = %transfExpr

            let block = blockIdx.x
            let tid = threadIdx.x
            let rangeX = dRanges.[block]
            let rangeY = dRanges.[block + 1]

            // Loop through all elements in the interval, adding up values.
            // There is no need to synchronize until we perform the multiscan.
            let mutable sum = init()
            let mutable index = rangeX + tid
            while index < rangeY do              
                sum <- op sum (transf dValues.[index]) 
                index <- index + numThreads

            // A full multiscan is unnecessary here - we really only need the total.
            let total:ref<'T> = ref (init())
            let blockSum = multiScan init op numWarps logNumWarps tid sum total

            if tid = 0 then dBlockTotals.[block] <- !total
        @>

    let inline reduceKernel (plan:Plan) (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) =
        let numThreads = plan.numThreadsReduction
        let numWarps = plan.numWarpsReduction
        let logNumWarps = log2 numWarps
        <@ fun numBlocks (dBlockTotals:DevicePtr<'T>) ->
            let init = %initExpr
            let op = %opExpr

            let tid = threadIdx.x
            let x = if tid < numBlocks then dBlockTotals.[tid] else 0G
            
            let total:ref<'T> = ref (init())
            let sum = multiScan init op numWarps logNumWarps tid x total

            // Subtract the value from the inclusive scan for the exclusive scan
            // ***** TODO how to handle that
            if tid < numBlocks then dBlockTotals.[tid] <- sum - x

            // Have the first thread in the block set the scan total.
            if tid = 0 then dBlockTotals.[numBlocks] <- !total
        @>

    let inline reduce (plan:Plan) (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (transf:Expr<'T -> 'T>) = cuda {
        let! upSweep = upSweepKernel plan init op transf |> defineKernelFuncWithName "upSweep"
        let! reduce = reduceKernel plan init op |> defineKernelFuncWithName "reduce"

        let launch (m:Module) (n:int) (values:DevicePtr<'T>) =
            let numSm = m.Worker.Device.Attribute(DeviceAttribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            let ranges = plan.blockRanges numSm n
            let numBlocks = ranges.Length - 1
            use dRanges = m.Worker.Malloc(ranges)
            use dBlockTotals = m.Worker.Malloc<'T>(Array.zeroCreate (numBlocks + 1))  
            
            // Launch block reduction kernel to calculate the totals per range
            let lp = LaunchParam(numBlocks, plan.numThreads)
            upSweep.Launch m lp values dRanges.Ptr dBlockTotals.Ptr

            if numBlocks = 1 then
                // We are finished 
                let blockTotals = dBlockTotals.ToHost()
                blockTotals.[0]
            else
                // Need to aggregate the block sums as well
                let lp = LaunchParam(1, plan.numThreadsReduction)
                reduce.Launch m lp numBlocks dBlockTotals.Ptr
                let blockTotals = dBlockTotals.ToHost()
                blockTotals.[blockTotals.Length - 1]

        return PFunc(fun (m:Module) ->
            let launch = launch m
            { new IReduce<'T> with
                member this.Reduce (n, values) = 
                    launch n values 
                member this.Reduce values =  
                    let dValues = m.Worker.Malloc(values)
                    launch values.Length dValues.Ptr
            } ) }