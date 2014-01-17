[<AutoOpen>]
module Alea.cuExtension.MGPU.CTASegsort

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension
//open Alea.cuExtension.Util
open Alea.cuExtension.MGPU
open Alea.cuExtension.MGPU.Static
//open Alea.cuExtension.MGPU.QuotationUtil
open Alea.cuExtension.MGPU.DeviceUtil
open Alea.cuExtension.MGPU.Intrinsics
open Alea.cuExtension.MGPU.LoadStore
open Alea.cuExtension.MGPU.SortedNetwork
open Alea.cuExtension.MGPU.CTASearch
open Alea.cuExtension.MGPU.CTAMerge


let extractThreadHeadFlags =
    <@ fun (flags:deviceptr<uint32>) (index:int) (numBits:int) ->
        let index2 = index >>> 5
        let shift = 31u &&& (uint32 index)
        let mutable headFlags = flags.[index2] >>> (int32 shift)
        let shifted = 32u - shift

        if (int shifted) < numBits then
            headFlags <- bfi flags.[index2 + 1] headFlags shifted shift
        headFlags <- headFlags &&& (1u <<< numBits) - 1u
        
        headFlags
    @>

let segmentedSerialMerge (VT:int) (compOp:IComp<'TV>) =
    let comp = compOp.Device
    <@ fun  (keys_shared    :deviceptr<'TV>)
            (aBegin         :int)
            (aEnd           :int)
            (bBegin         :int)
            (bEnd           :int)
            (results        :deviceptr<'TV>)
            (indices        :deviceptr<int>)
            (leftEnd        :int)
            (rightStart     :int)
            (sync           :bool)
            ->
        let comp = %comp

        let mutable aBegin = aBegin
        let mutable bBegin = bBegin

        let bEnd = min rightStart bEnd
        let mutable aKey = keys_shared.[aBegin]
        let mutable bKey = keys_shared.[bBegin]

        for i = 0 to VT - 1 do
            let mutable p = false
            if aBegin >= aEnd then
                p <- false
            elif (bBegin >= bEnd) || (aBegin < leftEnd) then
                p <- true
            else
                p <- not (comp bKey aKey)

            results.[i] <- if p then aKey else bKey
            indices.[i] <- if p then aBegin else bBegin
            if p then
                aBegin <- aBegin + 1
                aKey <- keys_shared.[aBegin]
            else
                bBegin <- bBegin + 1
                bKey <- keys_shared.[bBegin]
        
        if sync then __syncthreads()
        @>


//////////////////////////////////////////////////////////////////////////////////
//// CTASegsortPass
//
//template<int NT, int VT, typename T, typename Comp>
//MGPU_DEVICE void CTASegsortPass(T* keys_shared, int* ranges_shared, int tid,
//	int pass, T results[VT], int indices[VT], int2& activeRange, Comp comp) {
let ctaSegsortPass (NT:int) (VT:int) (compOp:IComp<'TV>) =
    let findMergesortFrame = findMergesortFrame.Device
    let segmentedMergePath = (segmentedMergePath compOp).DSegmentedMergePath
    let segmentedSerialMerge = segmentedSerialMerge VT compOp

    <@ fun  (keys_shared:deviceptr<'TV>)
            (ranges_shared:deviceptr<int>)
            (tid:int)
            (pass:int)
            (results:deviceptr<'TV>)
            (indices:deviceptr<int>)
            (activeRange:int2)
            ->
        let findMergesortFrame = %findMergesortFrame
        let segmentedMergePath = %segmentedMergePath
        let segmentedSerialMerge = %segmentedSerialMerge

        let mutable activeRange = activeRange
        let frame = findMergesortFrame (2 <<< pass) tid VT
        let a0 = frame.x
        let b0 = frame.y
        let listLen = frame.z
        let list = tid >>> pass
        let listParity = 1 &&& pass
        let diag = VT * tid - frame.x

        let siblingRange = ranges_shared.[1 ^^^ list]
        let siblingStart = 0x0000ffff &&& siblingRange
        let siblingEnd = siblingRange >>> 16

        let leftEnd = if listParity <> 0 then siblingEnd else activeRange.y
        let rightStart = if listParity <> 0 then activeRange.x else siblingStart
        activeRange.x <- min activeRange.x siblingStart
        activeRange.y <- max activeRange.y siblingEnd

        let p = segmentedMergePath keys_shared a0 listLen b0 listLen leftEnd rightStart diag
        
        let a0tid = a0 + p
        let b0tid = b0 + diag - p

        segmentedSerialMerge keys_shared a0tid b0 b0tid (b0 + listLen) results indices leftEnd rightStart true

        if diag = 0 then ranges_shared.[list >>> 1] <- int(bfi (uint32 activeRange.y) (uint32 activeRange.x) 16u 16u)
    @>


//////////////////////////////////////////////////////////////////////////////////
//// CTASegsortLoop
//
let ctaSegsortLoop (NT:int) (VT:int) (hasValues:bool) (compOp:IComp<'TV>) =
    let deviceThreadToShared = deviceThreadToShared VT
    let deviceGather = deviceGather NT VT
    let ctaSegsortPass = ctaSegsortPass NT VT compOp

    let _, numPasses = sLogPow2 NT 1
    <@ fun  (threadKeys:deviceptr<'TV>)
            (threadValues:deviceptr<'TV>)
            (keys_shared:deviceptr<'TV>)
            (values_shared:deviceptr<'TV>)
            (ranges_shared:deviceptr<int>)
            (tid:int)
            (activeRange:int2)
            ->
        let deviceThreadToShared = %deviceThreadToShared
        let deviceGather = %deviceGather
        let ctaSegsortPass = %ctaSegsortPass

        for pass = 0 to numPasses do
            let indices = __local__.Array<int>(VT) |> __array_to_ptr
            ctaSegsortPass keys_shared ranges_shared tid pass threadKeys indices activeRange

            if hasValues then
                deviceThreadToShared threadValues tid values_shared true
                deviceGather (NT * VT) values_shared indices tid threadValues true

            deviceThreadToShared threadKeys tid keys_shared true

        activeRange
    @>


//////////////////////////////////////////////////////////////////////////////////
//// CTASegsort
//// Pass keys and values in register. On return, values are returned in register
//// and keys returned in shared memory.
//
let ctaSegsort (NT:int) (VT:int) (stable:bool) (hasValues:bool) (compOp:IComp<'TV>) =
    let oddEvenTransposeSortFlags = oddEvenTransposeSortFlags VT compOp
    let oddEvenMergesortFlags = oddEvenMergesortFlags VT compOp
    let deviceThreadToShared = deviceThreadToShared VT
    let ctaSegsortLoop = ctaSegsortLoop NT VT hasValues compOp
    
    <@ fun  (threadKeys:deviceptr<'TV>)
            (threadValues:deviceptr<'TV>)
            (tid:int)
            (headFlags:int)
            (keys_shared:deviceptr<'TV>)
            (values_shared:deviceptr<'TV>)
            (ranges_shared:deviceptr<int>)
            ->
        let oddEvenTransposeSortFlags = %oddEvenTransposeSortFlags
        let oddEvenMergesortFlags = %oddEvenMergesortFlags
        let deviceThreadToShared = %deviceThreadToShared
        let ctaSegsortLoop = %ctaSegsortLoop

        if stable then
            oddEvenTransposeSortFlags threadKeys threadValues headFlags
        else
            oddEvenMergesortFlags threadKeys threadValues headFlags

        let mutable blockEnd = 31 - clz headFlags
        if (-1 <> blockEnd) then blockEnd <- blockEnd + VT * tid

        let mutable blockStart = ffs headFlags
        blockStart <- if blockStart <> 0 then (VT * tid - 1 + blockStart) else (NT * VT)

        ranges_shared.[tid] <- int( bfi (uint32 blockEnd) (uint32 blockStart) 16u 16u)

        deviceThreadToShared threadKeys tid keys_shared true

        let activeRange = ctaSegsortLoop threadKeys threadValues keys_shared values_shared ranges_shared tid (int2(blockStart,blockEnd))

        activeRange @>


//////////////////////////////////////////////////////////////////////////////////
//// DeviceSegBlocksort
//// Load keys and values from global memory, sort in shared memory, and store
//// back to global memory. Store the left-most and right-most encountered 
//// headflag locations to ranges_global to prepare for the next pass.
//// This function is factored out of the blocksort kernel to allow easier
//// customization of that kernel - we have two implementations currently:
//// sort over indices and sort over bitfield.
let deviceSegBlocksort (NT:int) (VT:int) (stable:bool) (hasValues:bool) (compOp:IComp<'TV>) =
    let deviceGlobalToShared = deviceGlobalToShared NT VT
    let deviceSharedToThread = deviceSharedToThread VT
    let ctaSegsort = ctaSegsort NT VT stable hasValues compOp
    let deviceSharedToGlobal = deviceSharedToGlobal NT VT
    let deviceThreadToShared = deviceThreadToShared VT

    <@ fun  (keys_global    :deviceptr<'TV>)
            (values_global  :deviceptr<'TV>)
            (count2         :int)
            (keys_shared    :deviceptr<'TV>)
            (values_shared  :deviceptr<'TV>)
            (ranges_shared  :deviceptr<int>)
            (headFlags      :int)
            (tid            :int)
            (block          :int)
            (keysDest_global:deviceptr<'TV>)
            (valsDest_global:deviceptr<'TV>)
            (ranges_global  :deviceptr<int>)
            ->
        let deviceGlobalToShared = %deviceGlobalToShared
        let deviceSharedToThread = %deviceSharedToThread
        let ctaSegsort = %ctaSegsort
        let deviceSharedToGlobal = %deviceSharedToGlobal
        let deviceThreadToShared = %deviceThreadToShared

        let gid = NT * VT * block
        let threadKeys = __local__.Array<'TV>(VT) |> __array_to_ptr
        deviceGlobalToShared count2 (keys_global + gid) tid keys_shared true
        deviceSharedToThread keys_shared tid threadKeys true

        let threadValues = __local__.Array<'TV>(VT) |> __array_to_ptr
        if hasValues then
            deviceGlobalToShared count2 (values_global + gid) tid values_shared true
            deviceSharedToThread values_shared tid threadValues true

        let activeRange = ctaSegsort threadKeys threadValues tid headFlags keys_shared values_shared ranges_shared

        deviceSharedToGlobal count2 keys_shared tid (keysDest_global + gid) true

        if hasValues then
            deviceThreadToShared threadValues tid values_shared true
            deviceSharedToGlobal count2 values_shared tid (valsDest_global + gid) false

        if tid = 0 then ranges_global.[block] <- int(bfi (uint32 activeRange.y) (uint32 activeRange.x) 16u 16u)
        @>


//////////////////////////////////////////////////////////////////////////////////
//// DeviceIndicesToHeadFlags
//// Load indices from an array and cooperatively turn into a head flag bitfield
//// for each thread.
let deviceIndicesToHeadFlags (NT:int) (VT:int) =
    <@ fun  (indices_global     :deviceptr<int>) 
            (partitions_global  :deviceptr<int>) 
            (tid:               int) 
            (block              :int) 
            (count2             :int) 
            (words_shared       :deviceptr<int>)
            (flags_shared       :deviceptr<byte>)
            ->
        let flagWordsPerThread = divup VT 4
        let gid = NT * VT * block
        let p0 = partitions_global.[block]
        let p1 = partitions_global.[block + 1]

        let mutable headFlags = 0u
        if (p1 > p0) || (count2 < NT * VT) then
            for i = 0 to flagWordsPerThread - 1 do
                words_shared.[NT * i + tid] <- 0
            __syncthreads()
            
            let mutable index = p0
            while index < p1 do
                let headFlag = indices_global.[index]
                flags_shared.[headFlag - gid] <- 1uy
                index <- index + NT
            __syncthreads()

            let first = VT * tid
            let offset = first / 4
            let mutable prev = uint32 words_shared.[offset]
            let mask = 0x3210u + 0x1111u * (3u &&& uint32 first)
            for i = 0 to flagWordsPerThread - 1 do
                let next = uint32 words_shared.[offset + 1 + i]
                let x = prmt prev next mask
                prev <- next
                if (0x00000001u &&& x) <> 0u then headFlags <- headFlags ||| 1u <<< (4 * i)
                if (0x00000100u &&& x) <> 0u then headFlags <- headFlags ||| 1u <<< (4 * i + 1)
                if (0x00010000u &&& x) <> 0u then headFlags <- headFlags ||| 1u <<< (4 * i + 2)
                if (0x01000000u &&& x) <> 0u then headFlags <- headFlags ||| 1u <<< (4 * i + 3)
            __syncthreads()
            
            let outOfRange = min VT (first + VT - count2)
            if outOfRange > 0 then
                headFlags <- bfi 0xffffffffu headFlags (uint32(VT - outOfRange)) (uint32 outOfRange)

            headFlags <- headFlags &&& (1u <<< VT) - 1u
        
        headFlags @>

//////////////////////////////////////////////////////////////////////////////////
//// SegSortSupport
//
//struct SegSortSupport {
//	int* ranges_global;
//	int2* ranges2_global;
//	
//	int4* mergeList_global;
//	int* copyList_global;
//	int2* queueCounters_global;
//	int2* nextCounters_global;
//
//	byte* copyStatus_global;
//};
//
//////////////////////////////////////////////////////////////////////////////////
//// DeviceSegSortMerge
//
//template<int NT, int VT, bool HasValues, typename KeyType, typename ValueType,
//	typename Comp>
//MGPU_DEVICE void DeviceSegSortMerge(const KeyType* keys_global,
//	const ValueType* values_global, SegSortSupport support, int tid, 
//	int block, int4 range, int pass, KeyType* keys_shared, 
//	int* indices_shared, KeyType* keysDest_global, ValueType* valsDest_global, 
//	Comp comp) {
//
//	const int NV = NT * VT;
//	int gid = NV * block;
//
//	// Load the local compressed segment indices.
//	int compressedRange = support.ranges_global[block];
//	int a0 = range.x;
//	int aCount = range.y - range.x;
//	int b0 = range.z;
//	int bCount = range.w - range.z;
//
//	DeviceLoad2ToShared<NT, VT, VT>(keys_global + a0, aCount, keys_global + b0, 
//		bCount, tid, keys_shared);
//
//	////////////////////////////////////////////////////////////////////////////
//	// Run a merge path to find the starting point for each thread to merge.
//	// If the entire warp fits into the already-sorted segments, we can skip
//	// sorting it and leave its keys in shared memory. Doing this on the warp
//	// level rather than thread level (also legal) gives slightly better 
//	// performance.
//	
//	int segStart = 0x0000ffff & compressedRange;
//	int segEnd = compressedRange>> 16;
//	int listParity = 1 & (block>> pass);
//
//	int warpOffset = VT * (~31 & tid);
//	bool sortWarp = listParity ? 
//		// The spliced segment is to the left (segStart).
//		(warpOffset < segStart) :
//		// The spliced segment is to the right (segEnd).
//		(warpOffset + 32 * VT > segEnd);
//
//	KeyType threadKeys[VT];
//	int indices[VT];
//	if(sortWarp) {
//		int diag = VT * tid;
//		int mp = SegmentedMergePath(keys_shared, 0, aCount, aCount, bCount,
//			listParity ? 0 : segEnd, listParity ? segStart : NV, diag, comp);
//		int a0tid = mp;
//		int a1tid = aCount;
//		int b0tid = aCount + diag - mp;
//		int b1tid = aCount + bCount;
//
//		// Serial merge into register. All threads in the CTA so we hoist the
//		// check for list parity outside the function call to simplify the 
//		// logic. Unlike in the blocksort, this does not cause warp divergence.
//		SegmentedSerialMerge<VT>(keys_shared, a0tid, a1tid, b0tid, b1tid,
//			threadKeys, indices, listParity ? 0 : segEnd,
//			listParity ? segStart : NV, comp, false);
//	} 
//	__syncthreads();
//
//	// Store sorted data in register back to shared memory. Then copy to global.
//	if(sortWarp)
//		DeviceThreadToShared<VT>(threadKeys, tid, keys_shared, false);
//	__syncthreads();
//
//	DeviceSharedToGlobal<NT, VT>(aCount + bCount, keys_shared, tid, 
//		keysDest_global + gid);
//
//	////////////////////////////////////////////////////////////////////////////
//	// Use the merge indices to gather values from global memory. Store directly
//	// to valsDest_global.
//
//	if(HasValues) {
//		// Transpose the gather indices to help coalesce loads.
//		if(sortWarp) 
//			DeviceThreadToShared<VT>(indices, tid, indices_shared, false);
//		else {
//			#pragma unroll
//			for(int i = 0; i < VT; ++i)
//				indices_shared[VT * tid + i] = VT * tid + i;
//		}
//		__syncthreads();
//
//		DeviceTransferMergeValues<NT, VT>(aCount + bCount, values_global + a0,  
//			values_global + b0, aCount, indices_shared, tid, 
//			valsDest_global + NV * block);
//	}
//}
//
//////////////////////////////////////////////////////////////////////////////////
//// DeviceSegSortCopy
//
//template<int NT, int VT, bool HasValues, typename KeyType, typename ValueType>
//MGPU_DEVICE void DeviceSegSortCopy(const KeyType* keys_global,
//	const ValueType* values_global, int tid, int block, int count,
//	KeyType* keysDest_global, ValueType* valsDest_global) {
//
//	int gid = NT * VT * block;
//	int count2 = min(NT * VT, count - gid);
//
//	DeviceGlobalToGlobal<NT, VT>(count2, keys_global + gid, tid, 
//		keysDest_global + gid);
//	if(HasValues)
//		DeviceGlobalToGlobal<NT, VT>(count2, values_global + gid, tid,
//			valsDest_global + gid);
//}
//
//} // namespace mgpu