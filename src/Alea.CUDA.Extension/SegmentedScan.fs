module Alea.CUDA.Extension.SegmentedScan

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.Timing
open Alea.CUDA.Extension.Reduce

module Sum =

    /// Reduction function for upsweep pass. 
    let [<ReflectedDefinition>] inline reduce (init: unit -> 'T) (op:'T -> 'T -> 'T) numWarps logNumWarps tid (x:'T) =
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)

        let scanStride = WARP_SIZE + WARP_SIZE / 2 + 1
        let scanSize = numWarps * scanStride
        
        let reductionShared = __shared__<'T>(scanSize).Ptr(0)
        let totalsShared = __shared__<'T>(2 * WARP_SIZE).Ptr(0)
    
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
            let s = (totalsShared + numWarps / 2 + tid).Volatile()
            s.[-(numWarps / 2)] <- init()
            s.[0] <- totals

            for i = 0 to logNumWarps - 1 do
                let offset = 1 <<< i
                totals <- op totals s.[-offset]
                if i < logNumWarps then s.[0] <- totals

            totalsShared.[tid] <- totals
        
        // Synchronize to make the block scan available to all warps.
        __syncthreads()

        totalsShared.[numWarps - 1]

    /// Test caller for reduce.
    let inline reduceTest (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) = cuda {
        let plan = if sizeof<'T> >= 8 then plan64 else plan32
        let numWarps = plan.numWarps
        let logNumWarps = log2 numWarps
        let! kernel = 
            <@ fun (dValues:DevicePtr<'T>) (dValuesOut:DevicePtr<'T>) ->
                let init = %initExpr
                let op = %opExpr
                let block = blockIdx.x
                let tid = threadIdx.x
                let x = dValues.[tid]
                dValuesOut.[0] <- reduce init op numWarps logNumWarps tid x
            @> |> defineKernelFunc

        return PFunc(fun (m:Module) (values:'T[]) ->
            if values.Length > plan.numThreads then failwith "array size too large"
            use dValues = m.Worker.Malloc(values)
            use dValuesOut = m.Worker.Malloc(1)
            let lp = LaunchParam (1, plan.numThreads)
            kernel.Launch m lp dValues.Ptr dValuesOut.Ptr
            dValuesOut.ToHost().[0]
         ) }

(*
////////////////////////////////////////////////////////////////////////////////
// INTER-WARP REDUCTION 
// Calculate the length of the last segment in the last lane in each warp. Also
// store the block offset to shared memory for the next pass.

template<int NumWarps>
DEVICE2 uint BlockScan(uint tid, uint warp, uint lane, uint last,
    uint warpFlags, uint mask, volatile uint* blockOffsetShared) {

    const int LogNumWarps = LOG_BASE_2(NumWarps);

    __shared__ volatile uint blockShared[3 * NumWarps];
    if(WARP_SIZE - 1 == lane) {
        blockShared[NumWarps + warp] = last;
        blockShared[2 * NumWarps + warp] = warpFlags;
    }
    __syncthreads();

    if(tid < NumWarps) {
        // Pull out the sum and flags for each warp.
        volatile uint* s = blockShared + NumWarps + tid;
        uint warpLast = s[0];
        uint flag = s[NumWarps];
        s[-NumWarps] = 0;

        uint blockFlags = __ballot(flag);

        // Mask out the bits at or above the current warp.
        blockFlags &= mask;

        // Find the distance from the current warp to the warp at the start of 
        // this segment.
        int preceding = 31 - __clz(blockFlags);
        uint distance = tid - preceding;


        // INTER-WARP reduction
        uint warpSum = warpLast;
        uint warpFirst = blockShared[NumWarps + preceding];

        #pragma unroll
        for(int i = 0; i < LogNumWarps; ++i) {
            uint offset = 1<< i;
            if(distance > offset) warpSum += s[-offset];
            if(i < LogNumWarps - 1) s[0] = warpSum;
        }
        // Subtract warpLast to make exclusive and add first to grab the
        // fragment sum of the preceding warp.
        warpSum += warpFirst - warpLast;

        // Store warpSum back into shared memory. This is added to all the
        // lane sums and those are added into all the threads in the first 
        // segment of each lane.
        blockShared[tid] = warpSum;

        // Set the block offset for the next brick of data.
        if(NumWarps - 1 == tid) {
            if(!flag) warpLast += warpSum;
            *blockOffsetShared = warpLast;
        }
    }
    __syncthreads();

    return blockShared[warp];
}
*)

    // Inter-warp reduction. Calculate the length of the last segment in the last lane in each warp. 
    // Also store the block offset to shared memory for the next pass.
    let [<ReflectedDefinition>] inline blockScan numWarps logNumWarps tid warp lane last warpFlags mask (blockOffsetShared:SharedPtr<int>) =

        let blockShared = __shared__<'T>(3 * numWarps).Ptr(0)

        if lane = WARP_SIZE - 1 then
            blockShared.[numWarps + warp] <- last
            blockShared.[2 * numWarps + warp] <- warpFlags

        __syncthreads()

        if tid < numWarps then
            // Pull out the sum and flags for each warp.
            let s = (blockShared + numWarps + tid).Volatile()

            let mutable warpLast = s.[0]
            let flag = s.[numWarps]
            s.[-numWarps] <- 0G

            let mutable blockFlags = __ballot flag

            // Mask out the bits at or above the current warp.
            blockFlags <- blockFlags &&& mask

            // Find the distance from the current warp to the warp at the start of this segment.
            let preceding = 31 - DeviceFunction.__clz blockFlags
            let distance = tid - preceding

            // Inter warp reduction.
            let mutable warpSum = warpLast;
            let warpFirst = blockShared.[numWarps + preceding];

            for i = 0 to logNumWarps - 1 do
                let offset = 1 <<< i
                if distance > offset then warpSum <- warpSum + s.[-offset]
                if i < logNumWarps - 1 then s.[0] <- warpSum;
            
            // Subtract warpLast to make exclusive and add first to grab the fragment sum of the preceding warp.
            warpSum <- warpSum + warpFirst - warpLast;

            // Store warpSum back into shared memory. This is added to all the lane sums and those are added into 
            // all the threads in the first  segment of each lane.
            blockShared.[tid] <- warpSum

            // Set the block offset for the next brick of data.
            if tid = numWarps - 1 then
                if flag = 0 then warpLast <- warpLast + warpSum
                blockOffsetShared.[0] <- warpLast

        __syncthreads()

        blockShared.[warp]

(*
// Segmented scan downsweep logic. Abstracts away loading of values and head 
// flags.

template<int NumWarps, int ValuesPerThread>
DEVICE2 void SegScanDownsweep(uint tid, uint lane, uint warp, 
    uint x[ValuesPerThread], const uint flags[ValuesPerThread],
    volatile uint* warpShared, volatile uint* threadShared, int inclusive, 
    volatile uint* blockOffsetShared) {

    ////////////////////////////////////////////////////////////////////////////
    // INTRA-WARP PASS
    // Add sum to all the values in the continuing segment (that is, before the
    // first start flag) in this thread.

    uint blockOffset = 0;
    if(!tid) blockOffset = *blockOffsetShared;
    uint last = blockOffset;

    // Compute the exclusive scan into scan. These values are then added to the
    // final thread offsets after the inter-warp multiscan pattern.
    uint hasHeadFlag = 0;

    
    if(inclusive) {
        #pragma unroll
        for(int i = 0; i < ValuesPerThread; ++i) {
            if(flags[i]) last = 0;
            hasHeadFlag |= flags[i];
            x[i] += last;
            last = x[i];
        }
    } else {
        #pragma unroll
        for(int i = 0; i < ValuesPerThread; ++i) {
            if(flags[i]) last = 0;
            if(flags[i]) hasHeadFlag |= flags[i];
            uint incLast = last;
            last += x[i];
            x[i] = incLast;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // INTRA-WARP SEGMENT PASS
    // Run a ballot and clz to find the lane containing the start value for the
    // segment that begins this thread.

    uint warpFlags = __ballot(hasHeadFlag);

    // Mask out the bits at or above the current thread.
    uint mask = bfi(0, 0xffffffff, 0, lane);
    uint warpFlagsMask = warpFlags & mask;

    // Find the distance from the current thread to the thread at the start of
    // the segment.
    int preceding = 31 - __clz(warpFlagsMask);
    uint distance = lane - preceding;


    ////////////////////////////////////////////////////////////////////////////
    // REDUCTION PASS
    // Run a prefix sum scan over last to compute for each lane the sum of all
    // values in the segmented preceding the current lane, up to that point.
    // This is added back into the thread-local exclusive scan for the continued
    // segment in each thread.

    volatile uint* shifted = threadShared + 1;
    shifted[-1] = 0;
    shifted[0] = last;
    uint sum = last;
    uint first = warpShared[1 + preceding];

    #pragma unroll
    for(int i = 0; i < LOG_WARP_SIZE; ++i) {
        uint offset = 1<< i;
        if(distance > offset) sum += shifted[-offset];
        if(i < LOG_WARP_SIZE - 1) shifted[0] = sum;
    }
    // Subtract last to make exclusive and add first to grab the fragment
    // sum of the preceding thread.
    sum += first - last;

    // Call BlockScan for inter-warp scan on the reductions of the last
    // segment in each warp.
    uint lastSegLength = last;
    if(!hasHeadFlag) lastSegLength += sum;

    uint blockScan = BlockScan<NumWarps>(tid, warp, lane, lastSegLength,
        warpFlags, mask, blockOffsetShared);
    if(!warpFlagsMask) sum += blockScan;

    #pragma unroll
    for(int i = 0; i < ValuesPerThread; ++i) {
        if(flags[i]) sum = 0;
        x[i] += sum;
    }
}
*)

    // Segmented scan downsweep logic. Abstracts away loading of values and head flags.
    let [<ReflectedDefinition>] inline segScanDownsweep numWarps logNumWarps valuesPerThread tid lane warp (x:LocalArray<'T>) (flags:LocalArray<int>)
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
        let mask = int(DeviceFunction.bfi(0u, 0xffffffffu, 0u, uint32(lane)))
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
            if flags.[i] <> 0 then sum <- 0
            x.[i] <- x.[i] + sum


(*
extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanUpsweepFlags(const uint* dValues, 
    const uint* dFlags, uint* dBlockLast, int* dHeadFlagPos, 
    const int2* rangePairs_global) {

    uint tid = threadIdx.x;
    uint block = blockIdx.x;

    int2 range = rangePairs_global[block];

    const int UpsweepValues = 4;
    const int NumValues = UpsweepValues * NUM_THREADS;

    // Start at the last tile (NUM_VALUES before the end iterator). Because
    // upsweep isn't executed for the last block, we don't have to worry about
    // the ending edge case.
    int current = range.y - NumValues;

    uint threadSum = 0;
    int segmentStart = -1;

    while(current >= range.x) {

        uint x[UpsweepValues], flags[UpsweepValues];
    
        #pragma unroll
        for(int i = 0; i < UpsweepValues; ++i) {
            x[i] = dValues[current + tid + i * NUM_THREADS];
            flags[i] = dFlags[current + tid + i * NUM_THREADS];
        }

        // Find the index of the latest value loaded with a head flag set.
        int lastHeadFlagPos = -1;

        #pragma unroll
        for(int i = 0; i < UpsweepValues; ++i)
            if(flags[i]) lastHeadFlagPos = i;
        
        if(-1 != lastHeadFlagPos)
            lastHeadFlagPos = tid + lastHeadFlagPos * NUM_THREADS;

        segmentStart = Reduce<NUM_WARPS>(tid, lastHeadFlagPos, 1);

        // Make a second pass and sum all the values that appear at or after
        // segmentStart.

        // Add if tid + i * NUM_THREADS >= segmentStart.
        // Subtract tid from both sides to simplify expression.
        int cmp = segmentStart - tid;
        #pragma unroll
        for(int i = 0; i < UpsweepValues; ++i)
            if(i * NUM_THREADS >= cmp)
                threadSum += x[i];
        
        if(-1 != segmentStart) break;

        __syncthreads();

        current -= NumValues;
    }

    // We've either hit the head flag or run out of values. Do a horizontal sum
    // of the thread values and store to global memory.
    uint total = (uint)Reduce<NUM_WARPS>(tid, (int)threadSum, 0);

    if(0 == tid) {
        dBlockLast[block] = total;
        dHeadFlagPos[block] = -1 != segmentStart;
    }
}
*)

    let inline segScanUpsweepFlagsKernel (plan:Plan) =
        let numThreads = plan.numThreads
        let numWarps = plan.numWarps
        let logNumWarps = log2 numWarps
        let upSweepValues = 4
        let numValues = upSweepValues * numThreads
        <@ fun (dValues:DevicePtr<'T>) (dFlags:DevicePtr<int>) (dBlockLast:DevicePtr<'T>) (dHeadFlagPos:DevicePtr<int>) (dRanges:DevicePtr<int>) ->
            let block = blockIdx.x
            let tid = threadIdx.x
            let rangeX = dRanges.[block]
            let rangeY = dRanges.[block + 1]

            // Start at the last tile numValues before the end. 
            // Because upsweep isn't executed for the last block, we don't have to worry about the ending edge case.
            let mutable current = rangeY - numValues
            let mutable threadSum = 0G
            let mutable segmentStart = -1

            let x = __local__<'T>(upSweepValues)
            let flags = __local__<int>(upSweepValues)

            while current >= rangeX do
               
                for i = 0 to upSweepValues - 1 do
                    x.[i] <- dValues.[current + tid + i * numThreads] 
                    flags.[i] <- dFlags.[current + tid + i * numThreads]

                // Find the index of the latest value loaded with a head flag set.
                let mutable lastHeadFlagPos = -1

                for i = 0 to upSweepValues - 1 do
                    if flags.[i] <> 0 then lastHeadFlagPos <- i

                if lastHeadFlagPos <> -1 then
                    lastHeadFlagPos <- tid + lastHeadFlagPos * numThreads

                let segmentStart = reduce (fun () -> -1) (max) numWarps logNumWarps tid lastHeadFlagPos

                // Make a second pass and sum all the values that appear at or after segmentStart.
                // Add if tid + i * numThreads >= segmentStart. Subtract tid from both sides to simplify expression.
                let cmp = segmentStart - tid

                for i = 0 to upSweepValues do
                    if i * numThreads >= cmp then
                        threadSum <- threadSum + x.[i]
        
                if segmentStart = -1 then  
                    __syncthreads()
                    current <- current - numValues
                else
                    current <- -1  // force break

            // We've either hit the head flag or run out of values. Do a horizontal sum
            // of the thread values and store to global memory.
            let total = reduce (fun () -> 0G) (+) numWarps logNumWarps  tid threadSum 

            if tid = 0 then
                dBlockLast.[block] <- total
                dHeadFlagPos.[block] <- if segmentStart <> -1 then 1 else 0
        @>
    

(*
extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanDownsweepFlags(const uint* dValues, 
    const uint* dFlags, uint* dValuesOut, 
    const uint* dStart, const int2* rangePairs_global, int count,
    int inclusive) {

    uint tid = threadIdx.x;
    uint lane = (WARP_SIZE - 1) & tid;
    uint warp = tid / WARP_SIZE;
    uint block = blockIdx.x;
    uint index = VALUES_PER_WARP * warp + lane;

    int2 range = rangePairs_global[block];

    const int Size = NUM_WARPS * valuesPerThread * (WARP_SIZE + 1);
    __shared__ volatile uint shared[Size];
    __shared__ volatile uint blockOffsetShared;

    // Use a stride of 33 slots per warp per value to allow conflict-free
    // transposes from strided to thread order.
    volatile uint* warpShared = shared + 
        warp * valuesPerThread * (WARP_SIZE + 1);
    volatile uint* threadShared = warpShared + lane;

    // Transpose values into thread order.
    uint offset = valuesPerThread * lane;
    offset += offset / WARP_SIZE;

    int lastOffset = ~(NUM_VALUES - 1) & count;

    if(!tid) blockOffsetShared = dStart[block];

    while(range.x < range.y) {
        // Load values into packed.
        uint x[valuesPerThread];
        uint flags[valuesPerThread];


        ////////////////////////////////////////////////////////////////////////
        // Load and transpose values.

            #pragma unroll
            for(int i = 0; i < valuesPerThread; ++i) {
                uint source = range.x + index + i * WARP_SIZE;
                uint value = dValues[source];
                threadShared[i * (WARP_SIZE + 1)] = value;
            }

        // Transpose into thread order and separate values from head flags.
        #pragma unroll
        for(int i = 0; i < valuesPerThread; ++i)
            x[i] = warpShared[offset + i];
            

        ////////////////////////////////////////////////////////////////////////
        // Load and transpose flags

            #pragma unroll
            for(int i = 0; i < valuesPerThread; ++i) {
                uint source = range.x + index + i * WARP_SIZE;
                uint flag = dFlags[source];
                threadShared[i * (WARP_SIZE + 1)] = flag;
            }

        // Transpose into thread order and separate values from head flags.
        #pragma unroll
        for(int i = 0; i < valuesPerThread; ++i)
            flags[i] = warpShared[offset + i];
        

        ////////////////////////////////////////////////////////////////////////
        // Run downsweep function on values and head flags.

        SegScanDownsweep<NUM_WARPS, valuesPerThread>(tid, lane, warp, x,
            flags, warpShared, threadShared, inclusive, &blockOffsetShared);

        ////////////////////////////////////////////////////////////////////////
        // Transpose and store scanned values.

        #pragma unroll
        for(int i = 0; i < valuesPerThread; ++i)
            warpShared[offset + i] = x[i];

            #pragma unroll
            for(int i = 0; i < valuesPerThread; ++i) {
                uint target = range.x + index + i * WARP_SIZE;
                uint value = threadShared[i * (WARP_SIZE + 1)];
                dValuesOut[target] = value;
            }

        range.x += NUM_VALUES;
    }
}
*)

    let inline segScanDownsweepFlags (plan:Plan) =
        let numThreads = plan.numThreads
        let valuesPerWarp = plan.valuesPerWarp
        let valuesPerThread = plan.valuesPerThread
        let numWarps = plan.numWarps
        let logNumWarps = log2 numWarps
        let upSweepValues = 4
        let numValues = upSweepValues * numThreads
        <@ fun (dValues:DevicePtr<'T>) (dFlags:DevicePtr<int>) (dValuesOut:DevicePtr<'T>) (dStart:DevicePtr<'T>) (dRanges:DevicePtr<int>) count inclusive ->
            let tid = threadIdx.x
            let warp = tid / WARP_SIZE
            let lane = tid &&& (WARP_SIZE - 1)
            let block = blockIdx.x
            let index = valuesPerWarp * warp + lane
            let mutable rangeX = dRanges.[block]
            let rangeY = dRanges.[block + 1]

            let size = numWarps * valuesPerThread * (WARP_SIZE + 1)
            let shared = __shared__<'T>(size).Ptr(0).Volatile()
            let blockOffsetShared = __shared__<'T>(1).Ptr(0).Volatile()

            // Use a stride of 33 slots per warp per value to allow conflict-free transposes from strided to thread order.
            let warpShared = (shared + warp * valuesPerThread * (WARP_SIZE + 1)).Volatile()
            let threadShared = (warpShared + lane).Volatile()

            // Transpose values into thread order.
            let mutable offset = valuesPerThread * lane
            offset <- offset + offset / WARP_SIZE

            let lastOffset = ~~~(numValues - 1) &&& count

            if tid = 0 then blockOffsetShared.[0] <- dStart.[block]

            while rangeX < rangeY do

                // Load values into packed.
                let x = __local__<'T>(valuesPerThread)
                let flags = __local__<'T>(valuesPerThread)

                for i = 0 to valuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let value = dValues.[source];
                    threadShared.[i * (WARP_SIZE + 1)] <- value

                // Transpose into thread order and separate values from head flags.
                for i = 0 to valuesPerThread - 1 do
                    x.[i] <- warpShared.[offset + i]
            
                // Load and transpose flags
                for i = 0 to valuesPerThread - 1 do
                    let source = rangeX + index + i * WARP_SIZE
                    let flag = dFlags.[source]
                    threadShared.[i * (WARP_SIZE + 1)] <- flag

                // Transpose into thread order and separate values from head flags.
                for i = 0 to valuesPerThread - 1 do
                    flags.[i] <- warpShared.[offset + i]
        
                // Run downsweep function on values and head flags.
                segScanDownsweep numWarps logNumWarps valuesPerThread tid lane warp x flags warpShared threadShared blockOffsetShared inclusive

                // Transpose and store scanned values.
                for i = 0 to valuesPerThread - 1 do
                    warpShared.[offset + i] <- x.[i]

                for i = 0 to valuesPerThread - 1 do
                    let target = rangeX + index + i * WARP_SIZE
                    let value = threadShared.[i * (WARP_SIZE + 1)]
                    dValuesOut.[target] <- value

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
 

//        <@ fun (dValues:DevicePtr<'T>) (dFlags:DevicePtr<int>) (dBlockLast:DevicePtr<'T>) (dHeadFlagPos:DevicePtr<int>) (dRanges:DevicePtr<int>) 
            
/// Scan builder to unify scan cuda monad with a function taking the kernel1, kernel2, kernel3 as args.
let inline segScan () = cuda {
    let plan = if sizeof<'T> >= 8 then plan64 else plan32

    let! kernel1 = Sum.segScanUpsweepFlagsKernel plan |> defineKernelFunc
    let! kernel2 = Scan.Sum.scanReduceKernel plan |> defineKernelFunc
    let! kernel3 = Sum.segScanDownsweepFlags plan |> defineKernelFunc

    let launch (m:Module) numValues (dValuesIn:DevicePtr<'T>) (dFlags:DevicePtr<int>) (dValuesOut:DevicePtr<'T>) (inclusive:bool) =
        let numSm = m.Worker.Device.Attribute(DeviceAttribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        let ranges = plan.blockRanges numSm numValues
        let numRanges = ranges.Length - 1
        let inclusive = if inclusive then 1 else 0
        use dRanges = m.Worker.Malloc(ranges)
        use dRangeTotals = m.Worker.Malloc<'T>(Array.zeroCreate (numRanges + 1))  
        use dHeadFlags = m.Worker.Malloc<'T>(numRanges)   

        printfn "====> ranges = %A" ranges
        printfn "0) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())

        //let lp = LaunchParam (numRanges-1, plan.numThreads) |> setDiagnoser (diagnose "upSweep")
        let lp = LaunchParam (numRanges, plan.numThreads)
        kernel1.Launch m lp dValuesIn dFlags dRangeTotals.Ptr dHeadFlags.Ptr dRanges.Ptr  

        printfn "1) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())
        printfn "1) dHeadFlags = %A" (dHeadFlags.ToHost())  

        let lp = LaunchParam(1, plan.numThreadsReduction)
        kernel2.Launch m lp numRanges dRangeTotals.Ptr

        printfn "2) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())

        let lp = LaunchParam(numRanges, plan.numThreads)
        kernel3.Launch m lp dValuesIn dFlags dValuesOut dRangeTotals.Ptr dRanges.Ptr numValues inclusive

        printfn "3) dRangeTotals = %A dRanges = %A" (dRangeTotals.ToHost()) (dRanges.ToHost())

    return PFunc(fun (m:Module) (values:'T[]) (flags:int[]) inclusive ->
        let launch = launch m
        let padding = scanPadding plan values.Length
        let size = values.Length + padding
        use dValuesIn = m.Worker.Malloc<'T>(size)
        use dFlags = m.Worker.Malloc<int>(size)
        use dValuesOut = m.Worker.Malloc<'T>(size)                 
        scatter m values padding dValuesIn
        scatter m flags padding dFlags
        launch values.Length dValuesIn.Ptr dFlags.Ptr dValuesOut.Ptr inclusive
        gather m values.Length dValuesOut.Ptr
     ) }

