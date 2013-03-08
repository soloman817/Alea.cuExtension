﻿module Alea.CUDA.Extension.SegmentedScan

open Microsoft.FSharp.Quotations
open Alea.Interop.LLVM
open Alea.CUDA

open Util

module IRB = Alea.CUDA.IRBuilderUtil

[<IRB.LLVMFunctionBuilder>]
let bfi(x:int, y:int, bit:int, numBits:int):int = failwith "Device Only!"
let ``bfi [BUILDER]``(ctx:IRB.LLVMFunctionBuilderContext) =
    let args = ctx.LLVMValueArgs                // arguments LLVM values
    let i32t = ctx.LLVMHelper.i32_t             // int LLVM type
    let rett = i32t                             // return type
    let argst = [| i32t; i32t; i32t; i32t |]    // argument type list
    let funct = LLVMFunctionTypeEx(rett, argst, 0)
    let funcp = LLVMConstInlineAsm(funct, "bfi.b32 \t$0, $2, $1, $3, $4;", "=r,r,r,r,r", 0, 0)
    IRB.Value(LLVMBuildCallEx(ctx.Builder, funcp, args, ""))

// Scan : hint -> ranges -> rangeTotals -> headFlags -> marks(flags/keys) -> values -> results -> inclusive -> unit
type ISegmentedScan<'T when 'T : unmanaged> =
    abstract Ranges : int[]
    abstract NumRangeTotals : int
    abstract NumHeadFlags : int
    abstract Scan : ActionHint -> DevicePtr<int> -> DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<'T> -> DevicePtr<'T> -> bool -> unit

type Plan = Reduce.Plan
type Planner = Reduce.Planner

let boxMark = function
    | 4 ->            <@ fun (threadShared:SharedPtr<'T>) i mark -> let ptr = threadShared.Reinterpret<int>() in ptr.[i] <- mark @>
    | n when n > 4 -> <@ fun (threadShared:SharedPtr<'T>) i mark -> let ptr = (threadShared + i).Reinterpret<int>() in ptr.[0] <- mark @>
    | 1 ->            <@ fun (threadShared:SharedPtr<'T>) i mark -> let ptr = threadShared.Reinterpret<int8>() in ptr.[i] <- int8(mark) @>
    | 2 ->            <@ fun (threadShared:SharedPtr<'T>) i mark -> let ptr = threadShared.Reinterpret<int16>() in ptr.[i] <- int16(mark) @>
    | 3 ->            <@ fun (threadShared:SharedPtr<'T>) i mark -> let ptr = (threadShared + i).Reinterpret<int16>() in ptr.[0] <- int16(mark) @> // TODO: int8(flag) @>
    | _ ->            <@ fun (threadShared:SharedPtr<'T>) i mark -> let ptr = (threadShared + i).Reinterpret<int>() in ptr.[0] <- mark @>

let unboxMark = function
    | 4 ->            <@ fun (warpShared:SharedPtr<'T>) i -> let ptr = warpShared.Reinterpret<int>() in ptr.[i] @>
    | n when n > 4 -> <@ fun (warpShared:SharedPtr<'T>) i -> let ptr = (warpShared + i).Reinterpret<int>() in ptr.[0] @>
    | 1 ->            <@ fun (warpShared:SharedPtr<'T>) i -> let ptr = warpShared.Reinterpret<int8>() in int(ptr.[0]) @>
    | 2 ->            <@ fun (warpShared:SharedPtr<'T>) i -> let ptr = warpShared.Reinterpret<int16>() in int(ptr.[0]) @>
    | 3 ->            <@ fun (warpShared:SharedPtr<'T>) i -> let ptr = (warpShared + i).Reinterpret<int16>() in int(ptr.[0]) @>
    | _ ->            <@ fun (warpShared:SharedPtr<'T>) i -> let ptr = (warpShared + i).Reinterpret<int>() in ptr.[0] @>

module Generic =

    /// Reduction function for upsweep pass. 
    let [<ReflectedDefinition>] reduce (init:unit -> 'T) (op:'T -> 'T -> 'T) numWarps logNumWarps tid (x:'T) =
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)

        let scanStride = WARP_SIZE + WARP_SIZE / 2 + 1
        let scanSize = numWarps * scanStride
        
        let reductionShared = __shared__<'T>(scanSize).Ptr(0)
        let totalsShared = __shared__<'T>(2 * numWarps).Ptr(0)
    
        let s = (reductionShared + scanStride * warp + lane + WARP_SIZE / 2).Volatile()
        s.[-(WARP_SIZE / 2)] <- init()
        s.[0] <- x

        // Run inclusive scan on each warp's data.
        let mutable warpScan = x
        for i = 0 to LOG_WARP_SIZE - 1 do
            let offset = 1 <<< i
            warpScan <- op warpScan s.[-offset]   
            s.[0] <- warpScan

        __syncthreads()

        if tid < numWarps then
            let mutable totals = reductionShared.[scanStride * tid + WARP_SIZE / 2 + WARP_SIZE - 1]
            let s = (totalsShared + numWarps + tid).Volatile()
            s.[-numWarps] <- init()
            s.[0] <- totals

            for i = 0 to logNumWarps - 1 do
                let offset = 1 <<< i
                totals <- op totals s.[-offset]
                s.[0] <- totals

        // Synchronize to make the block scan available to all warps.
        __syncthreads()

        totalsShared.[2 * numWarps - 1]

    [<ReflectedDefinition>]
    let blockScan (init:unit -> 'T) (op:'T -> 'T -> 'T) numWarps logNumWarps tid warp lane last warpFlags mask (blockOffsetShared:SharedPtr<'T>) =
        let blockSharedValues = __shared__<'T>(2 * numWarps).Ptr(0).Volatile()
        let blockSharedFlags = __shared__<int>(numWarps).Ptr(0).Volatile()

        if lane = WARP_SIZE - 1 then
            blockSharedValues.[numWarps + warp] <- last
            blockSharedFlags.[warp] <- warpFlags

        __syncthreads()

        if tid < numWarps then
            // Pull out the sum and flags for each warp.
            let s = blockSharedValues + numWarps + tid

            let mutable warpLast = s.[0]
            let flag = blockSharedFlags.[tid]
            s.[-numWarps] <- init()

            let mutable blockFlags = __ballot flag

            // Mask out the bits at or above the current warp.
            blockFlags <- blockFlags &&& mask

            // Find the distance from the current warp to the warp at the start of this segment.
            let preceding = 31 - DeviceFunction.__clz blockFlags
            let distance = tid - preceding

            // Inter warp reduction.
            let mutable warpSum = init()
            let warpFirst = blockSharedValues.[numWarps + preceding]

            for i = 0 to logNumWarps - 1 do
                let offset = 1 <<< i
                if distance > offset then warpSum <- op warpSum s.[-offset]
                if i < logNumWarps - 1 then s.[0] <- op warpSum warpLast
            
            // Subtract warpLast to make exclusive and add first to grab the fragment sum of the preceding warp.
            warpSum <- op warpSum warpFirst

            // Store warpSum back into shared memory. This is added to all the lane sums and those are added into 
            // all the threads in the first  segment of each lane.
            blockSharedValues.[tid] <- warpSum

            // Set the block offset for the next brick of data.
            if tid = numWarps - 1 then
                if flag = 0 then warpLast <- op warpLast warpSum
                blockOffsetShared.[0] <- warpLast

        __syncthreads()

        blockSharedValues.[warp]

    [<ReflectedDefinition>]
    let segScanDownsweep (init:unit -> 'T) (op:'T -> 'T -> 'T) numWarps logNumWarps valuesPerThread tid lane warp (x:LocalArray<'T>) (flags:LocalArray<int>)
                         (warpShared:SharedPtr<'T>) (threadShared:SharedPtr<'T>) (blockOffsetShared:SharedPtr<'T>) inclusive =

        // Add sum to all the values in the continuing segment (that is, before the first start flag) in this thread.
        let blockOffset = if tid = 0 then blockOffsetShared.[0] else init()
        let mutable last = blockOffset

        // Compute the exclusive scan into scan. These values are then added to the
        // final thread offsets after the inter-warp multiscan pattern.
        let mutable hasHeadFlag = 0

        if inclusive <> 0 then
            for i = 0 to valuesPerThread - 1 do
                if flags.[i] <> 0 then last <- init()
                hasHeadFlag <- hasHeadFlag ||| flags.[i]
                x.[i] <- op last x.[i]
                last <- x.[i]
        else
            for i = 0 to valuesPerThread - 1 do              
                if flags.[i] <> 0 then last <- init()
                if flags.[i] <> 0 then hasHeadFlag <- hasHeadFlag ||| flags.[i]
                let incLast = last
                last <- op last x.[i]
                x.[i] <- incLast

        // Intra warp segment pass: run a ballot and clz to find the lane containing the start value for the segment that begins this thread.
        let warpFlags = __ballot hasHeadFlag

        // Mask out the bits at or above the current thread.
        let mask = bfi(0, 0xffffffff, 0, lane)
        let warpFlagsMask = warpFlags &&& mask

        // Find the distance from the current thread to the thread at the start of the segment.
        let preceding = 31 - DeviceFunction.__clz(warpFlagsMask)
        let distance = lane - preceding

        // Reduction pass: run a prefix sum scan over last to compute for each lane the sum of all
        // values in the segmented preceding the current lane, up to that point.
        // This is added back into the thread-local exclusive scan for the continued segment in each thread.
        let shifted = threadShared + 1
        shifted.[-1] <- init()
        shifted.[0] <- last 
        let first = warpShared.[1 + preceding]

        let mutable sum = init()
        for i = 0 to LOG_WARP_SIZE - 1 do 
            let offset = 1 <<< i
            if distance > offset then sum <- op sum shifted.[-offset]
            if i < LOG_WARP_SIZE - 1 then shifted.[0] <- op sum last

        // Subtract last to make exclusive and add first to grab the fragment sum of the preceding thread.
        sum <- op sum first

        // Call BlockScan for inter-warp scan on the reductions of the last segment in each warp.
        let lastSegLength = if hasHeadFlag = 0 then op last sum else last

        let blockScan = blockScan init op numWarps logNumWarps tid warp lane lastSegLength warpFlags mask blockOffsetShared
        if warpFlagsMask = 0 then sum <- op sum blockScan

        for i = 0 to valuesPerThread - 1 do
            if flags.[i] <> 0 then sum <- init()
            x.[i] <- op x.[i] sum

    let segScanReductionKernel (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (plan:Plan) =
        let NumThreads = plan.NumThreadsReduction
        let NumWarps = plan.NumWarpsReduction
        let LogNumWarps = log2 NumWarps

        <@ fun (headFlags:DevicePtr<int>) (blockLast:DevicePtr<'T>) (numBlocks:int) ->
            let init = %init
            let op = %op

            let tid = threadIdx.x
            let lane = (WARP_SIZE - 1) &&& tid
            let warp = tid / WARP_SIZE

            // load the head flag and last segment counts for each thread. These map
            // to blocks in the upsweep/downsweep passes.
            let mutable flag = 1
            let mutable x = init()
            if tid < numBlocks then
                flag <- headFlags.[tid]
                x <- blockLast.[tid]

            // get the start flags for each thread in the warp
            let flags = __ballot flag

            // mask out the bits at or above the current lane
            let mask = bfi(0, 0xffffffff, 0, lane)
            let flagsMasked = flags &&& mask

            // find the distance from the current thread to the thread at the start of 
            // the segment.
            let preceding = 31 - DeviceFunction.__clz(flagsMasked)
            let distance = lane - preceding

            let shared = __shared__<'T>(NumWarps * (WARP_SIZE + 1)).Ptr(0).Volatile()
            let blockShared = __shared__<'T>(2 * NumWarps).Ptr(0).Volatile()
            let flagShared = __shared__<int>(NumWarps).Ptr(0).Volatile()
            let warpShared = shared + warp * (WARP_SIZE + 1) + 1
            let threadShared = warpShared + lane

            // run an inclusive scan for each warp. this does not require any special
            // treatment of segment edges, as we have only one value per thread
            threadShared.[-1] <- init()
            threadShared.[0] <- x

            let mutable sum = init()
            let first = warpShared.[preceding]

            for i = 0 to LOG_WARP_SIZE - 1 do
                let offset = 1 <<< i
                if distance > offset then sum <- op sum threadShared.[-offset]
                threadShared.[0] <- op sum x

            sum <- op sum first

            let last = if flag <> 0 then x else op sum x

            // sum now holds the inclusive scan for the part of the segment within the
            // warp. run a multiscan by having each warp store its flags value to
            // shared memory.
            if lane = WARP_SIZE - 1 then
                flagShared.[warp] <- flags
                blockShared.[NumWarps + warp] <- last

            __syncthreads()

            if tid < NumWarps then
                // load the inclusive sums for the last value in each warp and the head
                // flags for each warp.
                let flag = flagShared.[tid]
                let x = blockShared.[NumWarps + tid]
                let flags = (__ballot flag) &&& mask

                let preceding = 31 - DeviceFunction.__clz(flags)
                let distance = tid - preceding

                // set the first value to zero to allow a -1 dereference
                blockShared.[tid] <- init()
                let s = blockShared + NumWarps + tid

                let mutable sum = init()
                let first = blockShared.[NumWarps + preceding]

                for i = 0 to LogNumWarps - 1 do
                    let offset = 1 <<< i
                    if distance > offset then sum <- op sum s.[-offset]
                    s.[0] <- op sum x

                // add preceding and subtract x to get an exclusive sum.
                sum <- op sum first

                blockShared.[tid] <- sum

            __syncthreads()

            let blockScan = blockShared.[warp]

            // add blockScan if the warp doesn't hasn't encountered a head flag yet.
            if flagsMasked = 0 then sum <- op sum blockScan

            if tid < numBlocks then blockLast.[tid] <- sum @>

    let segScanUpsweepFlagsKernel (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (transf:Expr<'T -> 'T>) (plan:Plan) =
        let NumThreads = plan.NumThreads
        let NumWarps = plan.NumWarps
        let LogNumWarps = log2 NumWarps
        let UpSweepValues = 4
        let NumValues = UpSweepValues * NumThreads

        <@ fun (values:DevicePtr<'T>) (flags:DevicePtr<int>) (blockLast:DevicePtr<'T>) (headFlags:DevicePtr<int>) (ranges:DevicePtr<int>) ->
            let init = %init
            let op = %op
            let transf = %transf

            let block = blockIdx.x
            let tid = threadIdx.x
            let rangeX = ranges.[block]
            let rangeY = ranges.[block + 1]

            // Start at the last tile (NumValues before the end iterator). Because 
            // upsweep isn't executed for the last block, we don't have to worry about
            // the ending edge case.
            let mutable current = rangeY - NumValues

            let mutable threadSum = init()
            let mutable segmentStart = -1

            while current + NumValues > rangeX do
               
                let x = __local__<'T>(UpSweepValues)
                let f = __local__<int>(UpSweepValues)

                for i = 0 to UpSweepValues - 1 do
                    let index = current + i * NumThreads + tid
                    if index >= rangeX then
                        x.[i] <- transf values.[index]
                        f.[i] <- flags.[index]
                    else
                        x.[i] <- init()
                        f.[i] <- 0

                // Find the index of the latest value loaded with a head flag set.
                let mutable lastHeadFlagPos = -1

                for i = 0 to UpSweepValues - 1 do
                    if f.[i] <> 0 then lastHeadFlagPos <- i

                if lastHeadFlagPos <> -1 then
                    lastHeadFlagPos <- tid + lastHeadFlagPos * NumThreads

                segmentStart <- reduce (fun () -> -1) max NumWarps LogNumWarps tid lastHeadFlagPos

                // Make a second pass and sum all the values that appear at or after segmentStart.
                // Add if tid + i * numThreads >= segmentStart. Subtract tid from both sides to simplify expression.
                let cmp = segmentStart - tid

                for i = 0 to UpSweepValues - 1 do
                    if i * NumThreads >= cmp then
                        threadSum <- op threadSum x.[i]

                match segmentStart = -1 with
                | true -> current <- current - NumValues; __syncthreads()
                | false -> current <- rangeX - NumValues - 1 // force break

            // We've either hit the head flag or run out of values. Do a horizontal sum
            // of the thread values and store to global memory.
            let total = reduce init op NumWarps LogNumWarps tid threadSum

            if tid = 0 then
                blockLast.[block] <- total
                headFlags.[block] <- if segmentStart = -1 then 0 else 1 @>

    let segScanDownsweepFlagsKernel (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (transf:Expr<'T -> 'T>) (plan:Plan) =
        let NumThreads = plan.NumThreads
        let ValuesPerThread = plan.ValuesPerThread
        let NumWarps = plan.NumWarps
        let LogNumWarps = Util.log2 NumWarps
        let ValuesPerWarp = plan.ValuesPerWarp
        let NumValues = NumThreads * ValuesPerThread

        let boxFlag = boxMark sizeof<'T>
        let unboxFlag = unboxMark sizeof<'T>
            
        <@ fun (dValues:DevicePtr<'T>) (dFlags:DevicePtr<int>) (dValuesOut:DevicePtr<'T>) (dStart:DevicePtr<'T>) (dRanges:DevicePtr<int>) inclusive ->
            let boxFlag = %boxFlag
            let unboxFlag = %unboxFlag
            let init = %init
            let op = %op
            let transf = %transf

            let tid = threadIdx.x
            let warp = tid / WARP_SIZE
            let lane = tid &&& (WARP_SIZE - 1)
            let block = blockIdx.x
            let index = ValuesPerWarp * warp + lane
            let mutable rangeX = dRanges.[block]
            let rangeY = dRanges.[block + 1]

            let blockOffsetShared = __shared__<'T>(1).Ptr(0).Volatile()
            let shared = __shared__<'T>(NumWarps * ValuesPerThread * (WARP_SIZE + 1)).Ptr(0).Volatile()

            // Use a stride of 33 slots per warp per value to allow conflict-free transposes from strided to thread order.
            let warpShared = shared + warp * ValuesPerThread * (WARP_SIZE + 1)
            let threadShared = warpShared + lane

            // Transpose values into thread order.
            let mutable offset = ValuesPerThread * lane
            offset <- offset + offset / WARP_SIZE

            if tid = 0 then blockOffsetShared.[0] <- dStart.[block]

            while rangeX < rangeY do

                // Load values into packed.
                let x = __local__<'T>(ValuesPerThread)
                let flags = __local__<int>(ValuesPerThread)

                // Load and transpose flags
                for i = 0 to ValuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let flag = if source < rangeY then dFlags.[source] else 0
                    boxFlag threadShared (i * (WARP_SIZE + 1)) flag

                // Transpose into thread order and separate values from head flags.
                for i = 0 to ValuesPerThread - 1 do
                    flags.[i] <- unboxFlag warpShared (offset + i)
        
                for i = 0 to ValuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let value = if source < rangeY then dValues.[source] else init()
                    threadShared.[i * (WARP_SIZE + 1)] <- transf value

                // Transpose into thread order and separate values from head flags.
                for i = 0 to ValuesPerThread - 1 do
                    x.[i] <- warpShared.[offset + i]
            
                // Run downsweep function on values and head flags.
                segScanDownsweep init op NumWarps LogNumWarps ValuesPerThread tid lane warp x flags warpShared threadShared blockOffsetShared inclusive

                // Transpose and store scanned values.
                for i = 0 to ValuesPerThread - 1 do
                    warpShared.[offset + i] <- x.[i]

                for i = 0 to ValuesPerThread - 1 do
                    let target = rangeX + index + i * WARP_SIZE
                    let value = threadShared.[i * (WARP_SIZE + 1)]
                    if target < rangeY then dValuesOut.[target] <- value

                rangeX <- rangeX + NumValues @>

    let segScanUpsweepKeysKernel (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (transf:Expr<'T -> 'T>) (plan:Plan) =
        let NumThreads = plan.NumThreads
        let NumWarps = plan.NumWarps
        let LogNumWarps = log2 NumWarps
        let UpSweepValues = 8
        let NumValues = UpSweepValues * NumThreads

        <@ fun (values:DevicePtr<'T>) (keys:DevicePtr<int>) (blockLast:DevicePtr<'T>) (headFlags:DevicePtr<int>) (ranges:DevicePtr<int>) ->
            let init = %init
            let op = %op
            let transf = %transf

            let block = blockIdx.x
            let tid = threadIdx.x
            let lane = (WARP_SIZE - 1) &&& tid
            let warp = tid / WARP_SIZE
            let rangeX = ranges.[block]
            let rangeY = ranges.[block + 1]

            // Start at the last tile (NumValues before the end iterator). Because 
            // upsweep isn't executed for the last block, we don't have to worry about
            // the ending edge case.
            let mutable current = rangeY - NumValues

            let mutable threadSum = init()
            let mutable blockFlags = 0

            // load the last key in the segment.
            let lastKey = keys.[rangeY - 1]
            let firstKey = keys.[rangeX]

            while current + NumValues > rangeX do
               
                let x = __local__<'T>(UpSweepValues)
                let k = __local__<int>(UpSweepValues)

                for i = 0 to UpSweepValues - 1 do
                    let index = current + i * NumThreads + tid
                    if index >= rangeX then
                        x.[i] <- transf values.[index]
                        k.[i] <- keys.[index]
                    else
                        x.[i] <- init()
                        k.[i] <- firstKey

                // add up all the values with a key that matches lastKey. If this thread
                // has any key that doesn't match lastKey, mark the prevSeg flag.
                let mutable prevSeg = 0

                for i = 0 to UpSweepValues - 1 do
                    if k.[i] = lastKey then threadSum <- op threadSum x.[i]
                    else prevSeg <- 1

                // use ballot to see if any threads in this warp encountered an earlier
                // segment.
                let mutable warpFlags = __ballot(prevSeg)

                let warpShared = __shared__<int>(NumWarps).Ptr(0).Volatile()
                if lane = 0 then warpShared.[warp] <- warpFlags
                __syncthreads()

                if tid < NumWarps then
                    warpFlags <- warpShared.[tid]
                    warpFlags <- __ballot warpFlags
                    warpShared.[tid] <- warpFlags
                __syncthreads()

                blockFlags <- warpShared.[0]

                match blockFlags <> 0 with
                | true -> current <- rangeX - NumValues - 1 // force break
                | false -> current <- current - NumValues

            // we've either hit the preceding segment or run out of values. do a
            // horizontal sum of the thread values and store to global memory
            let total = reduce init op NumWarps LogNumWarps tid threadSum

            if tid = 0 then
                blockLast.[block] <- total

                // prepare the head flag
                let mutable headFlag = blockFlags
                if headFlag = 0 && rangeX <> 0 then
                    // load the preceding key.
                    let precedingKey = keys.[rangeX - 1]
                    //headFlag <- if precedingKey <> lastKey then 1 else 0
                    headFlag <- precedingKey ^^^ lastKey
                headFlags.[block] <- headFlag @>

    let segScanDownsweepKeysKernel (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (transf:Expr<'T -> 'T>) (plan:Plan) =
        let NumThreads = plan.NumThreads
        let ValuesPerThread = plan.ValuesPerThread
        let NumWarps = plan.NumWarps
        let LogNumWarps = Util.log2 NumWarps
        let ValuesPerWarp = plan.ValuesPerWarp
        let NumValues = NumThreads * ValuesPerThread

        let boxKey = boxMark sizeof<'T>
        let unboxKey = unboxMark sizeof<'T>
            
        <@ fun (dValues:DevicePtr<'T>) (dKeys:DevicePtr<int>) (dValuesOut:DevicePtr<'T>) (dStart:DevicePtr<'T>) (dRanges:DevicePtr<int>) inclusive ->
            let init = %init
            let op = %op
            let transf = %transf
            let boxKey = %boxKey
            let unboxKey = %unboxKey

            let tid = threadIdx.x
            let warp = tid / WARP_SIZE
            let lane = tid &&& (WARP_SIZE - 1)
            let block = blockIdx.x
            let index = ValuesPerWarp * warp + lane
            let mutable rangeX = dRanges.[block]
            let rangeY = dRanges.[block + 1]

            let blockOffsetShared = __shared__<'T>(1).Ptr(0).Volatile()
            let shared = __shared__<'T>(NumWarps * ValuesPerThread * (WARP_SIZE + 1)).Ptr(0).Volatile()

            // Use a stride of 33 slots per warp per value to allow conflict-free transposes from strided to thread order.
            let warpShared = shared + warp * ValuesPerThread * (WARP_SIZE + 1)
            let threadShared = warpShared + lane

            // Transpose values into thread order.
            let mutable offset = ValuesPerThread * lane
            offset <- offset + offset / WARP_SIZE

            if tid = 0 then blockOffsetShared.[0] <- dStart.[block]

            let precedingKeyShared = __shared__<int>(1).Ptr(0).Volatile()
            if tid = 0 then precedingKeyShared.[0] <- if block <> 0 then dKeys.[rangeX - 1] else 0G

            while rangeX < rangeY do

                // Load values into packed.
                let x = __local__<'T>(ValuesPerThread)
                let keys = __local__<int>(ValuesPerThread)

                // Load and transpose flags
                for i = 0 to ValuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let key = if source < rangeY then dKeys.[source] else 0
                    boxKey threadShared (i * (WARP_SIZE + 1)) key

                // Transpose into thread order and separate values from head flags.
                for i = 0 to ValuesPerThread - 1 do
                    keys.[i] <- unboxKey warpShared (offset + i)
        
                for i = 0 to ValuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let value = if source < rangeY then dValues.[source] else init()
                    threadShared.[i * (WARP_SIZE + 1)] <- transf value

                // Transpose into thread order and separate values from head flags.
                for i = 0 to ValuesPerThread - 1 do
                    x.[i] <- warpShared.[offset + i]

                __syncthreads()

                // store the last key for each thread in shared memory
                boxKey shared (tid + 1) keys.[ValuesPerThread - 1]
                __syncthreads()

                // retrieve the last key for the preceding thread.
                let mutable precedingKey = unboxKey shared tid
                if tid = 0 then
                    precedingKey <- precedingKeyShared.[0]
                    precedingKeyShared.[0] <- unboxKey shared NumThreads

                // compare the adjacent keys in each thread to derive head flags
                let flags = __local__<int>(ValuesPerThread)

                for i = 0 to ValuesPerThread - 1 do
                    //if i <> 0 then flags.[i] <- if keys.[i - 1] <> keys.[i] then 1 else 0
                    //else flags.[0] <- if keys.[0] <> precedingKey then 1 else 0
                    if i <> 0 then flags.[i] <- keys.[i - 1] ^^^ keys.[i]
                    else flags.[0] <- keys.[0] ^^^ precedingKey
            
                // Run downsweep function on values and head flags.
                segScanDownsweep init op NumWarps LogNumWarps ValuesPerThread tid lane warp x flags warpShared threadShared blockOffsetShared inclusive

                // Transpose and store scanned values.
                for i = 0 to ValuesPerThread - 1 do
                    warpShared.[offset + i] <- x.[i]

                for i = 0 to ValuesPerThread - 1 do
                    let target = rangeX + index + i * WARP_SIZE
                    let value = threadShared.[i * (WARP_SIZE + 1)]
                    if target < rangeY then dValuesOut.[target] <- value

                rangeX <- rangeX + NumValues @>

module Sum =

    // Inter-warp reduction. Calculate the length of the last segment in the last lane in each warp. 
    // Also store the block offset to shared memory for the next pass.
    [<ReflectedDefinition>]
    let inline blockScan numWarps logNumWarps tid warp lane last warpFlags mask (blockOffsetShared:SharedPtr<'T>) =
        let blockSharedValues = __shared__<'T>(2 * numWarps).Ptr(0).Volatile()
        let blockSharedFlags = __shared__<int>(numWarps).Ptr(0).Volatile()

        if lane = WARP_SIZE - 1 then
            blockSharedValues.[numWarps + warp] <- last
            blockSharedFlags.[warp] <- warpFlags

        __syncthreads()

        if tid < numWarps then
            // Pull out the sum and flags for each warp.
            let s = blockSharedValues + numWarps + tid

            let mutable warpLast = s.[0]
            let flag = blockSharedFlags.[tid]
            s.[-numWarps] <- 0G

            let mutable blockFlags = __ballot flag

            // Mask out the bits at or above the current warp.
            blockFlags <- blockFlags &&& mask

            // Find the distance from the current warp to the warp at the start of this segment.
            let preceding = 31 - DeviceFunction.__clz blockFlags
            let distance = tid - preceding

            // Inter warp reduction.
            let mutable warpSum = warpLast
            let warpFirst = blockSharedValues.[numWarps + preceding]

            for i = 0 to logNumWarps - 1 do
                let offset = 1 <<< i
                if distance > offset then warpSum <- warpSum + s.[-offset]
                if i < logNumWarps - 1 then s.[0] <- warpSum
            
            // Subtract warpLast to make exclusive and add first to grab the fragment sum of the preceding warp.
            warpSum <- warpSum + warpFirst - warpLast

            // Store warpSum back into shared memory. This is added to all the lane sums and those are added into 
            // all the threads in the first  segment of each lane.
            blockSharedValues.[tid] <- warpSum

            // Set the block offset for the next brick of data.
            if tid = numWarps - 1 then
                if flag = 0 then warpLast <- warpLast + warpSum
                blockOffsetShared.[0] <- warpLast

        __syncthreads()

        blockSharedValues.[warp]

    // this one without (-) operator
    [<ReflectedDefinition>]
    let inline blockScan' numWarps logNumWarps tid warp lane last warpFlags mask (blockOffsetShared:SharedPtr<'T>) =
        let blockSharedValues = __shared__<'T>(2 * numWarps).Ptr(0).Volatile()
        let blockSharedFlags = __shared__<int>(numWarps).Ptr(0).Volatile()

        if lane = WARP_SIZE - 1 then
            blockSharedValues.[numWarps + warp] <- last
            blockSharedFlags.[warp] <- warpFlags

        __syncthreads()

        if tid < numWarps then
            // Pull out the sum and flags for each warp.
            let s = blockSharedValues + numWarps + tid

            let mutable warpLast = s.[0]
            let flag = blockSharedFlags.[tid]
            s.[-numWarps] <- 0G

            let mutable blockFlags = __ballot flag

            // Mask out the bits at or above the current warp.
            blockFlags <- blockFlags &&& mask

            // Find the distance from the current warp to the warp at the start of this segment.
            let preceding = 31 - DeviceFunction.__clz blockFlags
            let distance = tid - preceding

            // Inter warp reduction.
            let mutable warpSum = 0G
            let warpFirst = blockSharedValues.[numWarps + preceding]

            for i = 0 to logNumWarps - 1 do
                let offset = 1 <<< i
                if distance > offset then warpSum <- warpSum + s.[-offset]
                if i < logNumWarps - 1 then s.[0] <- warpSum + warpLast
            
            // Subtract warpLast to make exclusive and add first to grab the fragment sum of the preceding warp.
            warpSum <- warpSum + warpFirst

            // Store warpSum back into shared memory. This is added to all the lane sums and those are added into 
            // all the threads in the first  segment of each lane.
            blockSharedValues.[tid] <- warpSum

            // Set the block offset for the next brick of data.
            if tid = numWarps - 1 then
                if flag = 0 then warpLast <- warpLast + warpSum
                blockOffsetShared.[0] <- warpLast

        __syncthreads()

        blockSharedValues.[warp]

    // Segmented scan downsweep logic. Abstracts away loading of values and head flags.
    [<ReflectedDefinition>]
    let inline segScanDownsweep numWarps logNumWarps valuesPerThread tid lane warp (x:LocalArray<'T>) (flags:LocalArray<int>)
                                (warpShared:SharedPtr<'T>) (threadShared:SharedPtr<'T>) (blockOffsetShared:SharedPtr<'T>) inclusive =

        // Add sum to all the values in the continuing segment (that is, before the first start flag) in this thread.
        let blockOffset = if tid = 0 then blockOffsetShared.[0] else 0G
        let mutable last = blockOffset

        // Compute the exclusive scan into scan. These values are then added to the
        // final thread offsets after the inter-warp multiscan pattern.
        let mutable hasHeadFlag = 0

        if inclusive <> 0 then
            for i = 0 to valuesPerThread - 1 do
                if flags.[i] <> 0 then last <- 0G
                hasHeadFlag <- hasHeadFlag ||| flags.[i]
                x.[i] <- last + x.[i]
                last <- x.[i]
        else
            for i = 0 to valuesPerThread - 1 do              
                if flags.[i] <> 0 then last <- 0G
                if flags.[i] <> 0 then hasHeadFlag <- hasHeadFlag ||| flags.[i]
                let incLast = last
                last <- last + x.[i]
                x.[i] <- incLast

        // Intra warp segment pass: run a ballot and clz to find the lane containing the start value for the segment that begins this thread.
        let warpFlags = __ballot hasHeadFlag

        // Mask out the bits at or above the current thread.
        let mask = bfi(0, 0xffffffff, 0, lane)
        let warpFlagsMask = warpFlags &&& mask

        // Find the distance from the current thread to the thread at the start of the segment.
        let preceding = 31 - DeviceFunction.__clz(warpFlagsMask)
        let distance = lane - preceding

        // Reduction pass: run a prefix sum scan over last to compute for each lane the sum of all
        // values in the segmented preceding the current lane, up to that point.
        // This is added back into the thread-local exclusive scan for the continued segment in each thread.
        let shifted = threadShared + 1
        shifted.[-1] <- 0G
        shifted.[0] <- last
        let first = warpShared.[1 + preceding]

        let mutable sum = last
        for i = 0 to LOG_WARP_SIZE - 1 do 
            let offset = 1 <<< i
            if distance > offset then sum <- sum + shifted.[-offset]
            if i < LOG_WARP_SIZE - 1 then shifted.[0] <- sum

        // Subtract last to make exclusive and add first to grab the fragment sum of the preceding thread.
        sum <- sum + first - last

        // Call BlockScan for inter-warp scan on the reductions of the last segment in each warp.
        let lastSegLength = if hasHeadFlag = 0 then last + sum else last

        let blockScan = blockScan numWarps logNumWarps tid warp lane lastSegLength warpFlags mask blockOffsetShared
        if warpFlagsMask = 0 then sum <- sum + blockScan

        for i = 0 to valuesPerThread - 1 do
            if flags.[i] <> 0 then sum <- 0G
            x.[i] <- x.[i] + sum

    // this one without (-) operator
    [<ReflectedDefinition>]
    let inline segScanDownsweep' numWarps logNumWarps valuesPerThread tid lane warp (x:LocalArray<'T>) (flags:LocalArray<int>)
                                 (warpShared:SharedPtr<'T>) (threadShared:SharedPtr<'T>) (blockOffsetShared:SharedPtr<'T>) inclusive =

        // Add sum to all the values in the continuing segment (that is, before the first start flag) in this thread.
        let blockOffset = if tid = 0 then blockOffsetShared.[0] else 0G
        let mutable last = blockOffset

        // Compute the exclusive scan into scan. These values are then added to the
        // final thread offsets after the inter-warp multiscan pattern.
        let mutable hasHeadFlag = 0

        if inclusive <> 0 then
            for i = 0 to valuesPerThread - 1 do
                if flags.[i] <> 0 then last <- 0G
                hasHeadFlag <- hasHeadFlag ||| flags.[i]
                x.[i] <- last + x.[i]
                last <- x.[i]
        else
            for i = 0 to valuesPerThread - 1 do              
                if flags.[i] <> 0 then last <- 0G
                if flags.[i] <> 0 then hasHeadFlag <- hasHeadFlag ||| flags.[i]
                let incLast = last
                last <- last + x.[i]
                x.[i] <- incLast

        // Intra warp segment pass: run a ballot and clz to find the lane containing the start value for the segment that begins this thread.
        let warpFlags = __ballot hasHeadFlag

        // Mask out the bits at or above the current thread.
        let mask = bfi(0, 0xffffffff, 0, lane)
        let warpFlagsMask = warpFlags &&& mask

        // Find the distance from the current thread to the thread at the start of the segment.
        let preceding = 31 - DeviceFunction.__clz(warpFlagsMask)
        let distance = lane - preceding

        // Reduction pass: run a prefix sum scan over last to compute for each lane the sum of all
        // values in the segmented preceding the current lane, up to that point.
        // This is added back into the thread-local exclusive scan for the continued segment in each thread.
        let shifted = threadShared + 1
        shifted.[-1] <- 0G
        shifted.[0] <- last
        let first = warpShared.[1 + preceding]

        let mutable sum = 0G
        for i = 0 to LOG_WARP_SIZE - 1 do 
            let offset = 1 <<< i
            if distance > offset then sum <- sum + shifted.[-offset]
            if i < LOG_WARP_SIZE - 1 then shifted.[0] <- sum + last

        // Subtract last to make exclusive and add first to grab the fragment sum of the preceding thread.
        sum <- sum + first

        // Call BlockScan for inter-warp scan on the reductions of the last segment in each warp.
        let lastSegLength = if hasHeadFlag = 0 then last + sum else last

        let blockScan = blockScan' numWarps logNumWarps tid warp lane lastSegLength warpFlags mask blockOffsetShared
        if warpFlagsMask = 0 then sum <- sum + blockScan

        for i = 0 to valuesPerThread - 1 do
            if flags.[i] <> 0 then sum <- 0G
            x.[i] <- x.[i] + sum

    let inline segScanReductionKernel (plan:Plan) =
        let NumThreads = plan.NumThreadsReduction
        let NumWarps = plan.NumWarpsReduction
        let LogNumWarps = log2 NumWarps

        <@ fun (headFlags:DevicePtr<int>) (blockLast:DevicePtr<'T>) (numBlocks:int) ->
            let tid = threadIdx.x
            let lane = (WARP_SIZE - 1) &&& tid
            let warp = tid / WARP_SIZE

            // load the head flag and last segment counts for each thread. These map
            // to blocks in the upsweep/downsweep passes.
            let mutable flag = 1
            let mutable x = 0G
            if tid < numBlocks then
                flag <- headFlags.[tid]
                x <- blockLast.[tid]

            // get the start flags for each thread in the warp
            let flags = __ballot flag

            // mask out the bits at or above the current lane
            let mask = bfi(0, 0xffffffff, 0, lane)
            let flagsMasked = flags &&& mask

            // find the distance from the current thread to the thread at the start of 
            // the segment.
            let preceding = 31 - DeviceFunction.__clz(flagsMasked)
            let distance = lane - preceding

            let shared = __shared__<'T>(NumWarps * (WARP_SIZE + 1)).Ptr(0).Volatile()
            let blockShared = __shared__<'T>(2 * NumWarps).Ptr(0).Volatile()
            let flagShared = __shared__<int>(NumWarps).Ptr(0).Volatile()
            let warpShared = shared + warp * (WARP_SIZE + 1) + 1
            let threadShared = warpShared + lane

            // run an inclusive scan for each warp. this does not require any special
            // treatment of segment edges, as we have only one value per thread
            threadShared.[-1] <- 0G
            threadShared.[0] <- x

            let mutable sum = x + 0G
            let first = warpShared.[preceding]

            for i = 0 to LOG_WARP_SIZE - 1 do
                let offset = 1 <<< i
                if distance > offset then sum <- sum + threadShared.[-offset]
                threadShared.[0] <- sum

            sum <- sum + first

            let last = if flag <> 0 then x else sum

            // sum now holds the inclusive scan for the part of the segment within the
            // warp. run a multiscan by having each warp store its flags value to
            // shared memory.
            if lane = WARP_SIZE - 1 then
                flagShared.[warp] <- flags
                blockShared.[NumWarps + warp] <- last

            __syncthreads()

            if tid < NumWarps then
                // load the inclusive sums for the last value in each warp and the head
                // flags for each warp.
                let flag = flagShared.[tid]
                let x = blockShared.[NumWarps + tid]
                let flags = (__ballot flag) &&& mask

                let preceding = 31 - DeviceFunction.__clz(flags)
                let distance = tid - preceding

                // set the first value to zero to allow a -1 dereference
                blockShared.[tid] <- 0G
                let s = blockShared + NumWarps + tid

                let mutable sum = x + 0G
                let first = blockShared.[NumWarps + preceding]

                for i = 0 to LogNumWarps - 1 do
                    let offset = 1 <<< i
                    if distance > offset then sum <- sum + s.[-offset]
                    s.[0] <- sum

                // add preceding and subtract x to get an exclusive sum.
                sum <- sum + first - x

                blockShared.[tid] <- sum

            __syncthreads()

            let blockScan = blockShared.[warp]

            // add blockScan if the warp doesn't hasn't encountered a head flag yet.
            if flagsMasked = 0 then sum <- sum + blockScan
            sum <- sum - x

            if tid < numBlocks then blockLast.[tid] <- sum @>

    // this one without (-) operation
    let inline segScanReductionKernel' (plan:Plan) =
        let NumThreads = plan.NumThreadsReduction
        let NumWarps = plan.NumWarpsReduction
        let LogNumWarps = log2 NumWarps

        <@ fun (headFlags:DevicePtr<int>) (blockLast:DevicePtr<'T>) (numBlocks:int) ->
            let tid = threadIdx.x
            let lane = (WARP_SIZE - 1) &&& tid
            let warp = tid / WARP_SIZE

            // load the head flag and last segment counts for each thread. These map
            // to blocks in the upsweep/downsweep passes.
            let mutable flag = 1
            let mutable x = 0G
            if tid < numBlocks then
                flag <- headFlags.[tid]
                x <- blockLast.[tid]

            // get the start flags for each thread in the warp
            let flags = __ballot flag

            // mask out the bits at or above the current lane
            let mask = bfi(0, 0xffffffff, 0, lane)
            let flagsMasked = flags &&& mask

            // find the distance from the current thread to the thread at the start of 
            // the segment.
            let preceding = 31 - DeviceFunction.__clz(flagsMasked)
            let distance = lane - preceding

            let shared = __shared__<'T>(NumWarps * (WARP_SIZE + 1)).Ptr(0).Volatile()
            let blockShared = __shared__<'T>(2 * NumWarps).Ptr(0).Volatile()
            let flagShared = __shared__<int>(NumWarps).Ptr(0).Volatile()
            let warpShared = shared + warp * (WARP_SIZE + 1) + 1
            let threadShared = warpShared + lane

            // run an inclusive scan for each warp. this does not require any special
            // treatment of segment edges, as we have only one value per thread
            threadShared.[-1] <- 0G
            threadShared.[0] <- x

            let mutable sum = 0G
            let first = warpShared.[preceding]

            for i = 0 to LOG_WARP_SIZE - 1 do
                let offset = 1 <<< i
                if distance > offset then sum <- sum + threadShared.[-offset]
                threadShared.[0] <- sum + x

            sum <- sum + first

            let last = if flag <> 0 then x else sum + x

            // sum now holds the inclusive scan for the part of the segment within the
            // warp. run a multiscan by having each warp store its flags value to
            // shared memory.
            if lane = WARP_SIZE - 1 then
                flagShared.[warp] <- flags
                blockShared.[NumWarps + warp] <- last

            __syncthreads()

            if tid < NumWarps then
                // load the inclusive sums for the last value in each warp and the head
                // flags for each warp.
                let flag = flagShared.[tid]
                let x = blockShared.[NumWarps + tid]
                let flags = (__ballot flag) &&& mask

                let preceding = 31 - DeviceFunction.__clz(flags)
                let distance = tid - preceding

                // set the first value to zero to allow a -1 dereference
                blockShared.[tid] <- 0G
                let s = blockShared + NumWarps + tid

                let mutable sum = 0G
                let first = blockShared.[NumWarps + preceding]

                for i = 0 to LogNumWarps - 1 do
                    let offset = 1 <<< i
                    if distance > offset then sum <- sum + s.[-offset]
                    s.[0] <- sum + x

                // add preceding and subtract x to get an exclusive sum.
                sum <- sum + first

                blockShared.[tid] <- sum

            __syncthreads()

            let blockScan = blockShared.[warp]

            // add blockScan if the warp doesn't hasn't encountered a head flag yet.
            if flagsMasked = 0 then sum <- sum + blockScan

            if tid < numBlocks then blockLast.[tid] <- sum @>

    let inline segScanUpsweepFlagsKernel (plan:Plan) =
        let NumThreads = plan.NumThreads
        let NumWarps = plan.NumWarps
        let LogNumWarps = log2 NumWarps
        let UpSweepValues = 4
        let NumValues = UpSweepValues * NumThreads

        <@ fun (values:DevicePtr<'T>) (flags:DevicePtr<int>) (blockLast:DevicePtr<'T>) (headFlags:DevicePtr<int>) (ranges:DevicePtr<int>) ->
            let block = blockIdx.x
            let tid = threadIdx.x
            let rangeX = ranges.[block]
            let rangeY = ranges.[block + 1]

            // Start at the last tile (NumValues before the end iterator). Because 
            // upsweep isn't executed for the last block, we don't have to worry about
            // the ending edge case.
            let mutable current = rangeY - NumValues

            let mutable threadSum : 'T = 0G
            let mutable segmentStart = -1

            while current + NumValues > rangeX do
               
                let x = __local__<'T>(UpSweepValues)
                let f = __local__<int>(UpSweepValues)

                for i = 0 to UpSweepValues - 1 do
                    let index = current + i * NumThreads + tid
                    if index >= rangeX then
                        x.[i] <- values.[index]
                        f.[i] <- flags.[index]
                    else
                        x.[i] <- 0G
                        f.[i] <- 0

                // Find the index of the latest value loaded with a head flag set.
                let mutable lastHeadFlagPos = -1

                for i = 0 to UpSweepValues - 1 do
                    if f.[i] <> 0 then lastHeadFlagPos <- i

                if lastHeadFlagPos <> -1 then
                    lastHeadFlagPos <- tid + lastHeadFlagPos * NumThreads

                segmentStart <- Generic.reduce (fun () -> -1) max NumWarps LogNumWarps tid lastHeadFlagPos

                // Make a second pass and sum all the values that appear at or after segmentStart.
                // Add if tid + i * numThreads >= segmentStart. Subtract tid from both sides to simplify expression.
                let cmp = segmentStart - tid

                for i = 0 to UpSweepValues - 1 do
                    if i * NumThreads >= cmp then
                        threadSum <- threadSum + x.[i]

                match segmentStart = -1 with
                | true -> current <- current - NumValues; __syncthreads()
                | false -> current <- rangeX - NumValues - 1 // force break

            // We've either hit the head flag or run out of values. Do a horizontal sum
            // of the thread values and store to global memory.
            let total = Generic.reduce (fun () -> 0G) (+) NumWarps LogNumWarps tid threadSum

            if tid = 0 then
                blockLast.[block] <- total
                headFlags.[block] <- if segmentStart = -1 then 0 else 1 @>

    let inline segScanDownsweepFlagsKernel (plan:Plan) =
        let NumThreads = plan.NumThreads
        let ValuesPerThread = plan.ValuesPerThread
        let NumWarps = plan.NumWarps
        let LogNumWarps = Util.log2 NumWarps
        let ValuesPerWarp = plan.ValuesPerWarp
        let NumValues = NumThreads * ValuesPerThread

        let boxFlag = boxMark sizeof<'T>
        let unboxFlag = unboxMark sizeof<'T>
            
        <@ fun (dValues:DevicePtr<'T>) (dFlags:DevicePtr<int>) (dValuesOut:DevicePtr<'T>) (dStart:DevicePtr<'T>) (dRanges:DevicePtr<int>) inclusive ->
            let boxFlag = %boxFlag
            let unboxFlag = %unboxFlag

            let tid = threadIdx.x
            let warp = tid / WARP_SIZE
            let lane = tid &&& (WARP_SIZE - 1)
            let block = blockIdx.x
            let index = ValuesPerWarp * warp + lane
            let mutable rangeX = dRanges.[block]
            let rangeY = dRanges.[block + 1]

            let blockOffsetShared = __shared__<'T>(1).Ptr(0).Volatile()
            let shared = __shared__<'T>(NumWarps * ValuesPerThread * (WARP_SIZE + 1)).Ptr(0).Volatile()

            // Use a stride of 33 slots per warp per value to allow conflict-free transposes from strided to thread order.
            let warpShared = shared + warp * ValuesPerThread * (WARP_SIZE + 1)
            let threadShared = warpShared + lane

            // Transpose values into thread order.
            let mutable offset = ValuesPerThread * lane
            offset <- offset + offset / WARP_SIZE

            if tid = 0 then blockOffsetShared.[0] <- dStart.[block]

            while rangeX < rangeY do

                // Load values into packed.
                let x = __local__<'T>(ValuesPerThread)
                let flags = __local__<int>(ValuesPerThread)

                // Load and transpose flags
                for i = 0 to ValuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let flag = if source < rangeY then dFlags.[source] else 0
                    boxFlag threadShared (i * (WARP_SIZE + 1)) flag

                // Transpose into thread order and separate values from head flags.
                for i = 0 to ValuesPerThread - 1 do
                    flags.[i] <- unboxFlag warpShared (offset + i)
        
                for i = 0 to ValuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let value = if source < rangeY then dValues.[source] else 0G
                    threadShared.[i * (WARP_SIZE + 1)] <- value

                // Transpose into thread order and separate values from head flags.
                for i = 0 to ValuesPerThread - 1 do
                    x.[i] <- warpShared.[offset + i]
            
                // Run downsweep function on values and head flags.
                segScanDownsweep NumWarps LogNumWarps ValuesPerThread tid lane warp x flags warpShared threadShared blockOffsetShared inclusive

                // Transpose and store scanned values.
                for i = 0 to ValuesPerThread - 1 do
                    warpShared.[offset + i] <- x.[i]

                for i = 0 to ValuesPerThread - 1 do
                    let target = rangeX + index + i * WARP_SIZE
                    let value = threadShared.[i * (WARP_SIZE + 1)]
                    if target < rangeY then dValuesOut.[target] <- value

                rangeX <- rangeX + NumValues @>

    // this one without (-) operator
    let inline segScanDownsweepFlagsKernel' (plan:Plan) =
        let NumThreads = plan.NumThreads
        let ValuesPerThread = plan.ValuesPerThread
        let NumWarps = plan.NumWarps
        let LogNumWarps = Util.log2 NumWarps
        let ValuesPerWarp = plan.ValuesPerWarp
        let NumValues = NumThreads * ValuesPerThread

        let boxFlag = boxMark sizeof<'T>
        let unboxFlag = unboxMark sizeof<'T>
            
        <@ fun (dValues:DevicePtr<'T>) (dFlags:DevicePtr<int>) (dValuesOut:DevicePtr<'T>) (dStart:DevicePtr<'T>) (dRanges:DevicePtr<int>) inclusive ->
            let boxFlag = %boxFlag
            let unboxFlag = %unboxFlag

            let tid = threadIdx.x
            let warp = tid / WARP_SIZE
            let lane = tid &&& (WARP_SIZE - 1)
            let block = blockIdx.x
            let index = ValuesPerWarp * warp + lane
            let mutable rangeX = dRanges.[block]
            let rangeY = dRanges.[block + 1]

            let blockOffsetShared = __shared__<'T>(1).Ptr(0).Volatile()
            let shared = __shared__<'T>(NumWarps * ValuesPerThread * (WARP_SIZE + 1)).Ptr(0).Volatile()

            // Use a stride of 33 slots per warp per value to allow conflict-free transposes from strided to thread order.
            let warpShared = shared + warp * ValuesPerThread * (WARP_SIZE + 1)
            let threadShared = warpShared + lane

            // Transpose values into thread order.
            let mutable offset = ValuesPerThread * lane
            offset <- offset + offset / WARP_SIZE

            if tid = 0 then blockOffsetShared.[0] <- dStart.[block]

            while rangeX < rangeY do

                // Load values into packed.
                let x = __local__<'T>(ValuesPerThread)
                let flags = __local__<int>(ValuesPerThread)

                // Load and transpose flags
                for i = 0 to ValuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let flag = if source < rangeY then dFlags.[source] else 0
                    boxFlag threadShared (i * (WARP_SIZE + 1)) flag

                // Transpose into thread order and separate values from head flags.
                for i = 0 to ValuesPerThread - 1 do
                    flags.[i] <- unboxFlag warpShared (offset + i)
        
                for i = 0 to ValuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let value = if source < rangeY then dValues.[source] else 0G
                    threadShared.[i * (WARP_SIZE + 1)] <- value

                // Transpose into thread order and separate values from head flags.
                for i = 0 to ValuesPerThread - 1 do
                    x.[i] <- warpShared.[offset + i]
            
                // Run downsweep function on values and head flags.
                segScanDownsweep' NumWarps LogNumWarps ValuesPerThread tid lane warp x flags warpShared threadShared blockOffsetShared inclusive

                // Transpose and store scanned values.
                for i = 0 to ValuesPerThread - 1 do
                    warpShared.[offset + i] <- x.[i]

                for i = 0 to ValuesPerThread - 1 do
                    let target = rangeX + index + i * WARP_SIZE
                    let value = threadShared.[i * (WARP_SIZE + 1)]
                    if target < rangeY then dValuesOut.[target] <- value

                rangeX <- rangeX + NumValues @>

    let inline segScanUpsweepKeysKernel (plan:Plan) =
        let NumThreads = plan.NumThreads
        let NumWarps = plan.NumWarps
        let LogNumWarps = log2 NumWarps
        let UpSweepValues = 8
        let NumValues = UpSweepValues * NumThreads

        <@ fun (values:DevicePtr<'T>) (keys:DevicePtr<int>) (blockLast:DevicePtr<'T>) (headFlags:DevicePtr<int>) (ranges:DevicePtr<int>) ->
            let block = blockIdx.x
            let tid = threadIdx.x
            let lane = (WARP_SIZE - 1) &&& tid
            let warp = tid / WARP_SIZE
            let rangeX = ranges.[block]
            let rangeY = ranges.[block + 1]

            // Start at the last tile (NumValues before the end iterator). Because 
            // upsweep isn't executed for the last block, we don't have to worry about
            // the ending edge case.
            let mutable current = rangeY - NumValues

            let mutable threadSum : 'T = 0G
            let mutable blockFlags = 0

            // load the last key in the segment.
            let lastKey = keys.[rangeY - 1]
            let firstKey = keys.[rangeX]

            while current + NumValues > rangeX do
               
                let x = __local__<'T>(UpSweepValues)
                let k = __local__<int>(UpSweepValues)

                for i = 0 to UpSweepValues - 1 do
                    let index = current + i * NumThreads + tid
                    if index >= rangeX then
                        x.[i] <- values.[index]
                        k.[i] <- keys.[index]
                    else
                        x.[i] <- 0G
                        k.[i] <- firstKey

                // add up all the values with a key that matches lastKey. If this thread
                // has any key that doesn't match lastKey, mark the prevSeg flag.
                let mutable prevSeg = 0

                for i = 0 to UpSweepValues - 1 do
                    if k.[i] = lastKey then threadSum <- threadSum + x.[i]
                    else prevSeg <- 1

                // use ballot to see if any threads in this warp encountered an earlier
                // segment.
                let mutable warpFlags = __ballot(prevSeg)

                let warpShared = __shared__<int>(NumWarps).Ptr(0).Volatile()
                if lane = 0 then warpShared.[warp] <- warpFlags
                __syncthreads()

                if tid < NumWarps then
                    warpFlags <- warpShared.[tid]
                    warpFlags <- __ballot warpFlags
                    warpShared.[tid] <- warpFlags
                __syncthreads()

                blockFlags <- warpShared.[0]

                match blockFlags <> 0 with
                | true -> current <- rangeX - NumValues - 1 // force break
                | false -> current <- current - NumValues

            // we've either hit the preceding segment or run out of values. do a
            // horizontal sum of the thread values and store to global memory
            let total = Generic.reduce (fun () -> 0G) (+) NumWarps LogNumWarps tid threadSum

            if tid = 0 then
                blockLast.[block] <- total

                // prepare the head flag
                let mutable headFlag = blockFlags
                if headFlag = 0 && rangeX <> 0 then
                    // load the preceding key.
                    let precedingKey = keys.[rangeX - 1]
                    //headFlag <- if precedingKey <> lastKey then 1 else 0
                    headFlag <- precedingKey ^^^ lastKey
                headFlags.[block] <- headFlag @>

    let inline segScanDownsweepKeysKernel (plan:Plan) =
        let NumThreads = plan.NumThreads
        let ValuesPerThread = plan.ValuesPerThread
        let NumWarps = plan.NumWarps
        let LogNumWarps = Util.log2 NumWarps
        let ValuesPerWarp = plan.ValuesPerWarp
        let NumValues = NumThreads * ValuesPerThread

        let boxKey = boxMark sizeof<'T>
        let unboxKey = unboxMark sizeof<'T>
            
        <@ fun (dValues:DevicePtr<'T>) (dKeys:DevicePtr<int>) (dValuesOut:DevicePtr<'T>) (dStart:DevicePtr<'T>) (dRanges:DevicePtr<int>) inclusive ->
            let boxKey = %boxKey
            let unboxKey = %unboxKey

            let tid = threadIdx.x
            let warp = tid / WARP_SIZE
            let lane = tid &&& (WARP_SIZE - 1)
            let block = blockIdx.x
            let index = ValuesPerWarp * warp + lane
            let mutable rangeX = dRanges.[block]
            let rangeY = dRanges.[block + 1]

            let blockOffsetShared = __shared__<'T>(1).Ptr(0).Volatile()
            let shared = __shared__<'T>(NumWarps * ValuesPerThread * (WARP_SIZE + 1)).Ptr(0).Volatile()

            // Use a stride of 33 slots per warp per value to allow conflict-free transposes from strided to thread order.
            let warpShared = shared + warp * ValuesPerThread * (WARP_SIZE + 1)
            let threadShared = warpShared + lane

            // Transpose values into thread order.
            let mutable offset = ValuesPerThread * lane
            offset <- offset + offset / WARP_SIZE

            if tid = 0 then blockOffsetShared.[0] <- dStart.[block]

            let precedingKeyShared = __shared__<int>(1).Ptr(0).Volatile()
            if tid = 0 then precedingKeyShared.[0] <- if block <> 0 then dKeys.[rangeX - 1] else 0G

            while rangeX < rangeY do

                // Load values into packed.
                let x = __local__<'T>(ValuesPerThread)
                let keys = __local__<int>(ValuesPerThread)

                // Load and transpose flags
                for i = 0 to ValuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let key = if source < rangeY then dKeys.[source] else 0
                    boxKey threadShared (i * (WARP_SIZE + 1)) key

                // Transpose into thread order and separate values from head flags.
                for i = 0 to ValuesPerThread - 1 do
                    keys.[i] <- unboxKey warpShared (offset + i)
        
                for i = 0 to ValuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let value = if source < rangeY then dValues.[source] else 0G
                    threadShared.[i * (WARP_SIZE + 1)] <- value

                // Transpose into thread order and separate values from head flags.
                for i = 0 to ValuesPerThread - 1 do
                    x.[i] <- warpShared.[offset + i]

                __syncthreads()

                // store the last key for each thread in shared memory
                boxKey shared (tid + 1) keys.[ValuesPerThread - 1]
                __syncthreads()

                // retrieve the last key for the preceding thread.
                let mutable precedingKey = unboxKey shared tid
                if tid = 0 then
                    precedingKey <- precedingKeyShared.[0]
                    precedingKeyShared.[0] <- unboxKey shared NumThreads

                // compare the adjacent keys in each thread to derive head flags
                let flags = __local__<int>(ValuesPerThread)

                for i = 0 to ValuesPerThread - 1 do
                    //if i <> 0 then flags.[i] <- if keys.[i - 1] <> keys.[i] then 1 else 0
                    //else flags.[0] <- if keys.[0] <> precedingKey then 1 else 0
                    if i <> 0 then flags.[i] <- keys.[i - 1] ^^^ keys.[i]
                    else flags.[0] <- keys.[0] ^^^ precedingKey
            
                // Run downsweep function on values and head flags.
                segScanDownsweep NumWarps LogNumWarps ValuesPerThread tid lane warp x flags warpShared threadShared blockOffsetShared inclusive

                // Transpose and store scanned values.
                for i = 0 to ValuesPerThread - 1 do
                    warpShared.[offset + i] <- x.[i]

                for i = 0 to ValuesPerThread - 1 do
                    let target = rangeX + index + i * WARP_SIZE
                    let value = threadShared.[i * (WARP_SIZE + 1)]
                    if target < rangeY then dValuesOut.[target] <- value

                rangeX <- rangeX + NumValues @>

// UpsweepKernel values flags rangeTotals headFlags headFlags ranges
type UpsweepKernel<'T> = DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<int> -> unit
// ReduceKernel headFlags rangeTotals numRanges
type ReduceKernel<'T> = DevicePtr<int> -> DevicePtr<'T> -> int -> unit
// DownsweepKernel values -> marks(flags/keys) -> results -> rangeTotals -> ranges -> inclusive
type DownsweepKernel<'T> = DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<'T> -> DevicePtr<'T> -> DevicePtr<int> -> int -> unit

let plan32 : Plan = { NumThreads = 256; ValuesPerThread = 16; NumThreadsReduction = 256; BlockPerSm = 2 }
let plan64 : Plan = { NumThreads = 256; ValuesPerThread = 4; NumThreadsReduction = 256; BlockPerSm = 6 }

let build (planner:Planner) (upsweep:Plan -> Expr<UpsweepKernel<'T>>) (reduce:Plan -> Expr<ReduceKernel<'T>>) (downsweep:Plan -> Expr<DownsweepKernel<'T>>) = cuda {
    let plan =
        match planner with
        | Planner.Default -> if sizeof<'T> > 4 then plan64 else plan32
        | Planner.Specific(plan) -> plan
        | Planner.ByWorker(worker) -> failwith "TODO"

    let! upsweep = upsweep plan |> defineKernelFuncWithName "segscan_upsweep"
    let! reduce = reduce plan |> defineKernelFuncWithName "segscan_reduce"
    let! downsweep = downsweep plan |> defineKernelFuncWithName "segscan_downsweep"

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let upsweep = upsweep.Apply m
        let reduce = reduce.Apply m
        let downsweep = downsweep.Apply m
        let numSm = m.Worker.Device.NumSm

        fun (n:int) ->
            let ranges = plan.BlockRanges numSm n
            let numRanges = ranges.Length - 1

            let lpUpsweep = LaunchParam(numRanges, plan.NumThreads)
            let lpReduce = LaunchParam(1, plan.NumThreadsReduction)
            let lpDownsweep = LaunchParam(numRanges, plan.NumThreads)

            let launch (hint:ActionHint) (ranges:DevicePtr<int>) (rangeTotals:DevicePtr<'T>) (headFlags:DevicePtr<int>) (marks:DevicePtr<int>) (values:DevicePtr<'T>) (results:DevicePtr<'T>) (inclusive:bool) =
                let inclusive = if inclusive then 1 else 0
                let lpUpsweep = lpUpsweep |> hint.ModifyLaunchParam
                let lpReduce = lpReduce |> hint.ModifyLaunchParam
                let lpDownsweep = lpDownsweep |> hint.ModifyLaunchParam

                fun () ->
                    upsweep.Launch lpUpsweep values marks rangeTotals headFlags ranges
                    reduce.Launch lpReduce headFlags rangeTotals numRanges
                    downsweep.Launch lpDownsweep values marks results rangeTotals ranges inclusive
                |> worker.Eval // the three kernels should be launched together without interrupt.

            { new ISegmentedScan<'T> with
                member this.Ranges = ranges
                member this.NumRangeTotals = numRanges
                member this.NumHeadFlags = numRanges
                member this.Scan lphint ranges rangeTotals headFlags flags values results inclusive = launch lphint ranges rangeTotals headFlags flags values results inclusive
            } ) }

let genericf planner init op transf =
    let upsweep = Generic.segScanUpsweepFlagsKernel init op transf
    let reduce = Generic.segScanReductionKernel init op
    let downsweep = Generic.segScanDownsweepFlagsKernel init op transf
    build planner upsweep reduce downsweep

let inline sumf planner = 
    let upsweep = Sum.segScanUpsweepFlagsKernel
    let reduce = Sum.segScanReductionKernel
    let downsweep = Sum.segScanDownsweepFlagsKernel
    build planner upsweep reduce downsweep

let generick planner init op transf =
    let upsweep = Generic.segScanUpsweepKeysKernel init op transf
    let reduce = Generic.segScanReductionKernel init op
    let downsweep = Generic.segScanDownsweepKeysKernel init op transf
    build planner upsweep reduce downsweep

let inline sumk planner =
    let upsweep = Sum.segScanUpsweepKeysKernel
    let reduce = Sum.segScanReductionKernel
    let downsweep = Sum.segScanDownsweepKeysKernel
    build planner upsweep reduce downsweep

