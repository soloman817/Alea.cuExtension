module Alea.CUDA.Extension.SegmentedScan

open Microsoft.FSharp.Quotations
open Alea.Interop.LLVM
open Alea.CUDA

open Util
open Reduce

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

// Scan : hint -> ranges -> rangeTotals -> headFlagPos -> values -> falgs -> results -> inclusive -> unit
type ISegmentedScan<'T when 'T : unmanaged> =
    abstract Ranges : int[]
    abstract NumRangeTotals : int
    abstract NumHeadFlags : int
    abstract Scan : ActionHint -> DevicePtr<int> -> DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<'T> -> bool -> unit

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

    let segScanUpsweepFlagsKernel (plan:Plan) (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) =
        let NumThreads = plan.NumThreads
        let NumWarps = plan.NumWarps
        let LogNumWarps = log2 NumWarps
        let UpSweepValues = 4
        let NumValues = UpSweepValues * NumThreads

        <@ fun (values:DevicePtr<'T>) (flags:DevicePtr<int>) (blockLast:DevicePtr<'T>) (headFlagPos:DevicePtr<int>) (ranges:DevicePtr<int>) ->
            let init = %init
            let op = %op

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

            while current + NumValues >= rangeX do
               
                let x = __local__<'T>(UpSweepValues)
                let f = __local__<int>(UpSweepValues)

                for i = 0 to UpSweepValues - 1 do
                    let index = current + i * NumThreads + tid
                    if index >= rangeX then
                        x.[i] <- values.[index]
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
            let total = reduce (fun () -> init()) op NumWarps LogNumWarps tid threadSum

            if tid = 0 then
                blockLast.[block] <- total
                headFlagPos.[block] <- if segmentStart = -1 then 0 else 1 @>

    let segScanReductionKernel (plan:Plan) (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) =
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

            let mutable sum = op x (init())
            let first = warpShared.[preceding]

            for i = 0 to LOG_WARP_SIZE - 1 do
                let offset = 1 <<< i
                if distance > offset then sum <- op sum threadShared.[-offset]
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
                blockShared.[tid] <- init()
                let s = blockShared + NumWarps + tid

                let mutable sum = op x (init())
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
            if flagsMasked = 0 then
                sum <- sum + blockScan
            sum <- sum - x

            if tid < numBlocks then blockLast.[tid] <- sum @>

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
            let sv = blockSharedValues + numWarps + tid
            let sf = blockSharedFlags + tid

            let mutable warpLast = sv.[0]
            let flag = sf.[0]
            sv.[-numWarps] <- 0G

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
                if distance > offset then warpSum <- warpSum + sv.[-offset]
                if i < logNumWarps - 1 then sv.[0] <- warpSum;
            
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
                //if flags.[i] <> 0 then hasHeadFlag <- hasHeadFlag ||| flags.[i]
                if flags.[i] <> 0 then hasHeadFlag <- 1
                let incLast = last + 0G // the bug, must add 0G
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
        let shifted = (threadShared + 1).Volatile()
        shifted.[-1] <- 0G
        shifted.[0] <- last
        let first = warpShared.[1 + preceding]

        let mutable sum = last
        for i = 0 to LOG_WARP_SIZE - 1 do 
            let offset = 1 <<< i
            if distance > offset then sum <- sum + shifted.[-offset]
            if i < LOG_WARP_SIZE - 1 then shifted.[0] <- sum

        // Subtract last to make exclusive and add first to grab the fragment sum of the preceding thread.
        sum <- sum + first - last;

        // Call BlockScan for inter-warp scan on the reductions of the last segment in each warp.
        let mutable lastSegLength = last
        if hasHeadFlag = 0 then lastSegLength <- lastSegLength + sum

        let blockScan = blockScan numWarps logNumWarps tid warp lane lastSegLength warpFlags mask blockOffsetShared
        if warpFlagsMask = 0 then sum <- sum + blockScan

        for i = 0 to valuesPerThread - 1 do
            if flags.[i] <> 0 then sum <- 0G
            x.[i] <- x.[i] + sum

    let inline segScanDownsweepFlagsKernel (plan:Plan) =
        let numThreads = plan.NumThreads
        let valuesPerWarp = plan.ValuesPerWarp
        let valuesPerThread = plan.ValuesPerThread
        let numWarps = plan.NumWarps
        let logNumWarps = log2 numWarps
        let upSweepValues = 4
        let numValues = upSweepValues * numThreads

        let size = numWarps * valuesPerThread * (WARP_SIZE + 1)
        let size = size * (max sizeof<int> sizeof<'T>)

        <@ fun (dValues:DevicePtr<'T>) (dFlags:DevicePtr<int>) (dValuesOut:DevicePtr<'T>) (dStart:DevicePtr<'T>) (dRanges:DevicePtr<int>) count inclusive ->
            let tid = threadIdx.x
            let warp = tid / WARP_SIZE
            let lane = tid &&& (WARP_SIZE - 1)
            let block = blockIdx.x
            let index = valuesPerWarp * warp + lane
            let mutable rangeX = dRanges.[block]
            let rangeY = dRanges.[block + 1]

            let _shared = __shared__<byte>(size).Ptr(0).Volatile()
            let blockOffsetShared = __shared__<'T>(1).Ptr(0).Volatile()
            let sharedValues = _shared.Reinterpret<'T>()
            let sharedFlags = _shared.Reinterpret<int>()

            // Use a stride of 33 slots per warp per value to allow conflict-free transposes from strided to thread order.
            let warpSharedValues = sharedValues + warp * valuesPerThread * (WARP_SIZE + 1)
            let threadSharedValues = warpSharedValues + lane
            let warpSharedFlags = sharedFlags + warp * valuesPerThread * (WARP_SIZE + 1)
            let threadSharedFlags = warpSharedFlags + lane

            // Transpose values into thread order.
            let mutable offset = valuesPerThread * lane
            offset <- offset + offset / WARP_SIZE

            //let lastOffset = ~~~(numValues - 1) &&& count

            if tid = 0 then blockOffsetShared.[0] <- dStart.[block]
            __syncthreads()

            while rangeX < rangeY do

                // Load values into packed.
                let x = __local__<'T>(valuesPerThread)
                let flags = __local__<int>(valuesPerThread)

                for i = 0 to valuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let value = if source < rangeY then dValues.[source] else 0G
                    threadSharedValues.[i * (WARP_SIZE + 1)] <- value

                // Transpose into thread order and separate values from head flags.
                for i = 0 to valuesPerThread - 1 do
                    x.[i] <- warpSharedValues.[offset + i]
            
                // Load and transpose flags
                for i = 0 to valuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let flag = dFlags.[source]
                    threadSharedFlags.[i * (WARP_SIZE + 1)] <- flag

                // Transpose into thread order and separate values from head flags.
                for i = 0 to valuesPerThread - 1 do
                    flags.[i] <- warpSharedFlags.[offset + i]
        
                // Run downsweep function on values and head flags.
                segScanDownsweep numWarps logNumWarps valuesPerThread tid lane warp x flags warpSharedValues threadSharedValues blockOffsetShared inclusive

                // Transpose and store scanned values.
                for i = 0 to valuesPerThread - 1 do
                    warpSharedValues.[offset + i] <- x.[i]

                for i = 0 to valuesPerThread - 1 do
                    let target = rangeX + index + i * WARP_SIZE
                    let value = threadSharedValues.[i * (WARP_SIZE + 1)]
                    if target < rangeY then dValuesOut.[target] <- value

                rangeX <- rangeX + numValues @>

module Sum =

    let inline segScanUpsweepFlagsKernel (plan:Plan) =
        let NumThreads = plan.NumThreads
        let NumWarps = plan.NumWarps
        let LogNumWarps = log2 NumWarps
        let UpSweepValues = 4
        let NumValues = UpSweepValues * NumThreads

        <@ fun (values:DevicePtr<'T>) (flags:DevicePtr<int>) (blockLast:DevicePtr<'T>) (headFlagPos:DevicePtr<int>) (ranges:DevicePtr<int>) ->
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

            while current + NumValues >= rangeX do
               
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
                headFlagPos.[block] <- if segmentStart = -1 then 0 else 1 @>

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
            let sv = blockSharedValues + numWarps + tid
            let sf = blockSharedFlags + tid

            let mutable warpLast = sv.[0]
            let flag = sf.[0]
            sv.[-numWarps] <- 0G

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
                if distance > offset then warpSum <- warpSum + sv.[-offset]
                if i < logNumWarps - 1 then sv.[0] <- warpSum;
            
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
                //if flags.[i] <> 0 then hasHeadFlag <- hasHeadFlag ||| flags.[i]
                if flags.[i] <> 0 then hasHeadFlag <- 1
                let incLast = last + 0G // the bug, must add 0G
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
        let shifted = (threadShared + 1).Volatile()
        shifted.[-1] <- 0G
        shifted.[0] <- last
        let first = warpShared.[1 + preceding]

        let mutable sum = last
        for i = 0 to LOG_WARP_SIZE - 1 do 
            let offset = 1 <<< i
            if distance > offset then sum <- sum + shifted.[-offset]
            if i < LOG_WARP_SIZE - 1 then shifted.[0] <- sum

        // Subtract last to make exclusive and add first to grab the fragment sum of the preceding thread.
        sum <- sum + first - last;

        // Call BlockScan for inter-warp scan on the reductions of the last segment in each warp.
        let mutable lastSegLength = last
        if hasHeadFlag = 0 then lastSegLength <- lastSegLength + sum

        let blockScan = blockScan numWarps logNumWarps tid warp lane lastSegLength warpFlags mask blockOffsetShared
        if warpFlagsMask = 0 then sum <- sum + blockScan

        for i = 0 to valuesPerThread - 1 do
            if flags.[i] <> 0 then sum <- 0G
            x.[i] <- x.[i] + sum

    let inline segScanDownsweepFlagsKernel (plan:Plan) =
        let numThreads = plan.NumThreads
        let valuesPerWarp = plan.ValuesPerWarp
        let valuesPerThread = plan.ValuesPerThread
        let numWarps = plan.NumWarps
        let logNumWarps = log2 numWarps
        let upSweepValues = 4
        let numValues = upSweepValues * numThreads

        let size = numWarps * valuesPerThread * (WARP_SIZE + 1)
        let size = size * (max sizeof<int> sizeof<'T>)

        <@ fun (dValues:DevicePtr<'T>) (dFlags:DevicePtr<int>) (dValuesOut:DevicePtr<'T>) (dStart:DevicePtr<'T>) (dRanges:DevicePtr<int>) count inclusive ->
            let tid = threadIdx.x
            let warp = tid / WARP_SIZE
            let lane = tid &&& (WARP_SIZE - 1)
            let block = blockIdx.x
            let index = valuesPerWarp * warp + lane
            let mutable rangeX = dRanges.[block]
            let rangeY = dRanges.[block + 1]

            let _shared = __shared__<byte>(size).Ptr(0).Volatile()
            let blockOffsetShared = __shared__<'T>(1).Ptr(0).Volatile()
            let sharedValues = _shared.Reinterpret<'T>()
            let sharedFlags = _shared.Reinterpret<int>()

            // Use a stride of 33 slots per warp per value to allow conflict-free transposes from strided to thread order.
            let warpSharedValues = sharedValues + warp * valuesPerThread * (WARP_SIZE + 1)
            let threadSharedValues = warpSharedValues + lane
            let warpSharedFlags = sharedFlags + warp * valuesPerThread * (WARP_SIZE + 1)
            let threadSharedFlags = warpSharedFlags + lane

            // Transpose values into thread order.
            let mutable offset = valuesPerThread * lane
            offset <- offset + offset / WARP_SIZE

            //let lastOffset = ~~~(numValues - 1) &&& count

            if tid = 0 then blockOffsetShared.[0] <- dStart.[block]
            __syncthreads()

            while rangeX < rangeY do

                // Load values into packed.
                let x = __local__<'T>(valuesPerThread)
                let flags = __local__<int>(valuesPerThread)

                for i = 0 to valuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let value = if source < rangeY then dValues.[source] else 0G
                    threadSharedValues.[i * (WARP_SIZE + 1)] <- value

                // Transpose into thread order and separate values from head flags.
                for i = 0 to valuesPerThread - 1 do
                    x.[i] <- warpSharedValues.[offset + i]
            
                // Load and transpose flags
                for i = 0 to valuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let flag = dFlags.[source]
                    threadSharedFlags.[i * (WARP_SIZE + 1)] <- flag

                // Transpose into thread order and separate values from head flags.
                for i = 0 to valuesPerThread - 1 do
                    flags.[i] <- warpSharedFlags.[offset + i]
        
                // Run downsweep function on values and head flags.
                segScanDownsweep numWarps logNumWarps valuesPerThread tid lane warp x flags warpSharedValues threadSharedValues blockOffsetShared inclusive

                // Transpose and store scanned values.
                for i = 0 to valuesPerThread - 1 do
                    warpSharedValues.[offset + i] <- x.[i]

                for i = 0 to valuesPerThread - 1 do
                    let target = rangeX + index + i * WARP_SIZE
                    let value = threadSharedValues.[i * (WARP_SIZE + 1)]
                    if target < rangeY then dValuesOut.[target] <- value

                rangeX <- rangeX + numValues @>

// UpsweepKernel values flags rangeTotals headFlags headFlagPos ranges
type UpsweepKernel<'T> = DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<int> -> unit
// ReduceKernel headFlags rangeTotals numRanges
type ReduceKernel<'T> = DevicePtr<int> -> DevicePtr<'T> -> int -> unit
// DownsweepKernel values -> flags -> results -> rangeTotals -> ranges -> count -> inclusive
type DownsweepKernel<'T> = DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<'T> -> DevicePtr<'T> -> DevicePtr<int> -> int -> int -> unit

/// Scan builder to unify scan cuda monad with a function taking the kernel1, kernel2, kernel3 as args.
let build (upsweep:Plan -> Expr<UpsweepKernel<'T>>) (reduce:Plan -> Expr<ReduceKernel<'T>>) (downsweep:Plan -> Expr<DownsweepKernel<'T>>) = cuda {
    let plan = if sizeof<'T> > 4 then plan64 else plan32
    let plan = { plan with NumThreads = plan.NumThreads / 2 }
    let! upsweep = upsweep plan |> defineKernelFuncWithName "segscan_upsweep"
    let! reduce = reduce plan |> defineKernelFuncWithName "segscan_reduce"
    let! downsweep = downsweep plan |> defineKernelFuncWithName "segscan_downsweep"

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

            let launch (hint:ActionHint) (ranges:DevicePtr<int>) (rangeTotals:DevicePtr<'T>) (headFlags:DevicePtr<int>) (values:DevicePtr<'T>) (flags:DevicePtr<int>) (results:DevicePtr<'T>) (inclusive:bool) =
                let inclusive = if inclusive then 1 else 0
                let lpUpsweep = lpUpsweep |> hint.ModifyLaunchParam
                let lpReduce = lpReduce |> hint.ModifyLaunchParam
                let lpDownsweep = lpDownsweep |> hint.ModifyLaunchParam

                fun () ->
                    upsweep.Launch lpUpsweep values flags rangeTotals headFlags ranges
                    reduce.Launch lpReduce headFlags rangeTotals numRanges
                    downsweep.Launch lpDownsweep values flags results rangeTotals ranges n inclusive
                |> worker.Eval // the three kernels should be launched together without interrupt.

            { new ISegmentedScan<'T> with
                member this.Ranges = ranges
                member this.NumRangeTotals = numRanges + 1
                member this.NumHeadFlags = numRanges
                member this.Scan lphint ranges rangeTotals headFlags values flags results inclusive = launch lphint ranges rangeTotals headFlags values flags results inclusive
            } ) }

/// <summary>
/// Global scan algorithm template. 
/// </summary>
//let generic (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (transf:Expr<'T -> 'T>) =
//    let upsweep = Generic.reduceUpSweepKernel init op transf
//    let reduce = Generic.scanReduceKernel init op transf
//    let downsweep = Generic.scanDownSweepKernel init op transf
//    build upsweep reduce downsweep

/// <summary>
/// Global scan algorithm template. 
/// </summary>
let inline sum () = 
    let upsweep = Sum.segScanUpsweepFlagsKernel
    let reduce = Sum.segScanReductionKernel
    let downsweep = Sum.segScanDownsweepFlagsKernel
    build upsweep reduce downsweep

