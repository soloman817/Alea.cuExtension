module Alea.cuExtension.MGPU.Join

// NOT IMPLEMENTED YET

//open System.Runtime.InteropServices
//open Microsoft.FSharp.Collections
//open Alea.CUDA
//open Alea.cuExtension
//open Alea.cuExtension.Util
//open Alea.cuExtension.MGPU
//open Alea.cuExtension.MGPU.Intrinsics
//open Alea.cuExtension.MGPU.QuotationUtil
//open Alea.cuExtension.MGPU.DeviceUtil
//open Alea.cuExtension.MGPU.LoadStore
//open Alea.cuExtension.MGPU.CTAScan
//open Alea.cuExtension.MGPU.CTALoadBalance
//

//type Plan =
//    {
//        NT : int
//        VT : int
//    }
//
//type MgpuJoinKind =
//    | MgpuJoinKindInner
//    | MgpuJoinKindLeft
//    | MgpuJoinKindRight
//    | MgpuJoinKindOuter
//
//
//
//let kernelLeftJoin (plan:Plan) (leftJoin:int) =
//    let NT = plan.NT
//    let VT = plan.VT
//    let NV = NT * VT
//
//    let sharedSize = NT * (VT + 1)
//    
//    let ulonglong_as_uint2 = ulonglong_as_uint2
//
//    let deviceMemToMemLoop = deviceMemToMemLoop NT
//    let ctaLoadBalance = ctaLoadBalance NT VT
//     
//    <@ fun (total:int) (aLowerBound_global:DevicePtr<int>) (aCountsScan_global:DevicePtr<int>) (aCount:int) (mp_global:DevicePtr<int>) (aIndices_global:DevicePtr<int>) (bIndices_global:DevicePtr<int>) ->
//        let ulonglong_as_uint2 = %ulonglong_as_uint2
//        let ctaLoadBalance = %ctaLoadBalance
//        let deviceMemToMemLoop = %deviceMemToMemLoop
//
//        let shared = __shared__<int>(sharedSize).Ptr(0)
//        let indices_shared = shared
//
//        let tid = threadIdx.x
//        let block = blockIdx.x
//
//        let range = ctaLoadBalance total aCountsScan_global aCount block tid mp_global indices_shared 1
//        let outputCount = range.y - range.x
//        let inputCount = range.w - range.z
//        let output_shared = indices_shared
//        let input_shared = indices_shared + outputCount
//
//        let aIndex = __local__<int>(VT).Ptr(0)
//        let rank = __local__<int>(VT).Ptr(0)
//        for i = 0 to VT - 1 do
//            let index = NT * i + tid
//            if index < outputCount then
//                let gid = range.x + index
//                aIndex.[i] <- output_shared.[index]
//                rank.[i] <- gid - input_shared.[aIndex.[i] - range.z]
//                aIndices_global.[gid] <- aIndex.[i]
//        __syncthreads()
//
//        deviceMemToMemLoop inputCount (aLowerBound_global + range.z) tid input_shared true
//
//        for i = 0 to VT - 1 do
//            let index = NT * i + tid
//            if index < outputCount then
//                let gid = range.x + index
//                let lb = input_shared.[aIndex.[i] - range.z]
//                let mutable bIndex = 0
//                if leftJoin = 1 then
//                    bIndex <- if (0x80000000 &&& lb) > 0 then ((0x7fffffff &&& lb) + rank.[i]) else -1
//                else
//                    bIndex <- lb + rank.[i]
//                bIndices_global.[gid] <- bIndex
//        @>
//
//
//let kernelRightJoinUpsweep (NT:int) =
//    let capacity, reduce = ctaReduce NT (scanOp ScanOpTypeAdd 0)
//    let sharedSize = capacity
//    let ulonglong_as_uint2 = ulonglong_as_uint2
//
//    <@ fun (matches_global:DevicePtr<uint64>) (count:int) (totals_global:DevicePtr<int>) ->
//        let reduce = %reduce
//        let ulonglong_as_uint2 = %ulonglong_as_uint2
//
//        let shared = __shared__<int>(sharedSize).Ptr(0)
//        let sharedReduce = shared
//        
//        let tid = threadIdx.x
//        let block = blockIdx.x
//        let gid = 8 * NT * block
//        let mutable count2 = min (8 * NT) (count - gid)
//
//        let mutable x = 0u
//        if (gid + 8 * tid) < count then
//            let mutable packed = matches_global.[NT * block + tid]
//            count2 <- count2 - (8 * tid)
//            if (count2 < 8) then
//                let mask = (1UL <<< (8 * count2)) - 1UL
//                packed <- (packed &&& mask) ||| (0101010101010101UL &&& ~~~mask)
//            let pair = ulonglong_as_uint2(packed)
//            x <- 8u - popc(pair.x) - popc(pair.y)
//
//        let total = reduce tid (int x) sharedReduce
//        if tid = 0 then totals_global.[block] <- total
//        @>
//
//
//let kernelRightJoinDownsweep (NT:int) =
//    
//    let capacity, scan = ctaScan NT (scanOp ScanOpTypeAdd 0)
//    let sharedSize = 2 * NT + 1
//
//    let ulonglong_as_uint2 = ulonglong_as_uint2
//    let deviceMemToMemLoop = deviceMemToMemLoop NT
//    
//    <@ fun (matches_global:DevicePtr<uint64>) (count:int) (scan_global:DevicePtr<int>) (rightJoin_global:DevicePtr<int>) ->
//        let scan = %scan
//        let ulonglong_as_uint2 = %ulonglong_as_uint2
//        let deviceMemToMemLoop = %deviceMemToMemLoop
//
//        let shared = __shared__<int>(sharedSize).Ptr(0)
//        let sharedIndices = shared
//        let sharedScan = shared
//
//        let tid = threadIdx.x
//        let block = blockIdx.x
//        let gid = 8 * NT * block
//        let mutable count2 = uint64(min (8 * NT) (count - gid))
//
//        let start = scan_global.[block]
//
//        let mutable packed = 0UL
//        let mutable x = 0
//        if (gid + 8 * tid) < count then
//            packed <- matches_global.[NT * block + tid]
//            count2 <- count2 - uint64(8 * tid)
//            if count2 < 8UL then
//                let mask = uint64((1 <<< (8 * (int count2))) - 1)
//                packed <- (packed &&& mask) ||| (0101010101010101UL &&& ~~~mask)
//            let pair = ulonglong_as_uint2(packed)
//            x <- int(8u - popc(pair.x) - popc(pair.y))
//
//            packed <- packed ^^^ 0101010101010101UL
//
//        let total = __local__<int>(1).Ptr(0)
//        let mutable scan = scan tid (int x) sharedScan total ExclusiveScan
//
//        if x <> 0 then
//            for i = 0 to 8 - 1 do
//                if ((1UL <<< (8 * i)) &&& packed) <> 0UL then
//                    sharedIndices.[scan] <- gid + 8 * tid + i
//                    scan <- scan + 1
//        __syncthreads()
//
//        deviceMemToMemLoop total.[0] sharedIndices tid (rightJoin_global + start) false
//    @>
//
//type IJoin<'T> =
//    {
//        //Action : ActionHint -> Device
//    }
//
//let relationalJoin (kind:MgpuJoinKind) (compOp:IComp<'TI>) =
//
//    let supportLeft = (kind = MgpuJoinKindLeft) || (kind = MgpuJoinKindOuter)
//    let supportRight = (kind = MgpuJoinKindRight) || (kind = MgpuJoinKindOuter)
//
//    let leftType = if supportLeft then MgpuSearchTypeIndexMatch else MgpuSearchTypeIndex
//
////    MGPU_MEM(int) aLowerBound = context.Malloc<int>(aCount);
////	MGPU_MEM(byte) bMatches;
//    let mutable rightJoinTotal = 0
//    if supportRight then
//        
        

      
//////////////////////////////////////////////////////////////////////////////////
//// KernelLeftJoin
//
//template<typename Tuning, bool LeftJoin>
//MGPU_LAUNCH_BOUNDS void KernelLeftJoin(int total, const int* aLowerBound_global,
//	const int* aCountsScan_global, int aCount, const int* mp_global,
//	int* aIndices_global, int* bIndices_global) { 
//
//	typedef MGPU_LAUNCH_PARAMS Params;
//	const int NT = Params::NT;
//	const int VT = Params::VT;
//
//	__shared__ int indices_shared[NT * (VT + 1)];
//	int tid = threadIdx.x;
//	int block = blockIdx.x;
//
//	int4 range = CTALoadBalance<NT, VT>(total, aCountsScan_global, aCount,
//		block, tid, mp_global, indices_shared, true);
//	int outputCount = range.y - range.x;
//	int inputCount = range.w - range.z;
//	int* output_shared = indices_shared;
//	int* input_shared = indices_shared + outputCount;
//
//	int aIndex[VT], rank[VT];
//	#pragma unroll
//	for(int i = 0; i < VT; ++i) {
//		int index = NT * i + tid;
//		if(index < outputCount) {
//			int gid = range.x + index;
//			aIndex[i] = output_shared[index];
//			rank[i] = gid - input_shared[aIndex[i] - range.z];
//			aIndices_global[gid] = aIndex[i];
//		}
//	}
//	__syncthreads();
//
//	// Load the lower bound of A into B for each element of A.
//	DeviceMemToMemLoop<NT>(inputCount, aLowerBound_global + range.z, tid,
//		input_shared);
//
//	// Store the lower bound of A into B back for every output.
//	#pragma unroll
//	for(int i = 0; i < VT; ++i) {
//		int index = NT * i + tid;
//		if(index < outputCount) {
//			int gid = range.x + index;
//			int lb = input_shared[aIndex[i] - range.z];
//			int bIndex;
//			if(LeftJoin)
//				bIndex = (0x80000000 & lb) ? 
//					((0x7fffffff & lb) + rank[i]) :
//					-1;
//			else
//				bIndex = lb + rank[i];
//			bIndices_global[gid] = bIndex;
//		}
//	}
//}
//
//struct LeftJoinEqualityOp {
//	MGPU_HOST_DEVICE int operator()(int lb, int ub) const {
//		lb &= 0x7fffffff;
//		return max(1, ub - lb);
//	}
//};
//
//////////////////////////////////////////////////////////////////////////////////
//// Right-join compaction kernels.
//
//template<int NT>
//__global__ void KernelRightJoinUpsweep(const uint64* matches_global, int count,
//	int* totals_global) {
//
//	typedef CTAReduce<NT> R;
//	__shared__ typename R::Storage reduce;
//
//	int tid = threadIdx.x;
//	int block = blockIdx.x;
//	int gid = 8 * NT * block;
//	int count2 = min(8 * NT, count - gid);
//
//	int x = 0;
//	if(gid + 8 * tid < count) {
//		uint64 packed = matches_global[NT * block + tid];
//		count2 -= 8 * tid;
//		if(count2 < 8) {
//			// Clear the bits above count2
//			uint64 mask = (1ll<< (8 * count2)) - 1;
//			packed = (packed & mask) | (0x0101010101010101ll & ~mask);
//		}
//		uint2 pair = ulonglong_as_uint2(packed);
//		x = 8 - popc(pair.x) - popc(pair.y);
//	}
//
//	int total = R::Reduce(tid, x, reduce);
//	if(!tid) totals_global[block] = total;
//}
//
//template<int NT>
//__global__ void KernelRightJoinDownsweep(const uint64* matches_global, 
//	int count, const int* scan_global, int* rightJoin_global) {
//
//	typedef CTAScan<NT> S;
//	union Shared {
//		int indices[8 * NT];
//		typename S::Storage scan;
//	};
//	__shared__ Shared shared;
//	
//	int tid = threadIdx.x;
//	int block = blockIdx.x;
//	int gid = 8 * NT * block;
//	int count2 = min(8 * NT, count - gid);
//
//	int start = scan_global[block];
//
//	uint64 packed;
//	int x = 0;
//	if(gid + 8 * tid < count) {
//		packed = matches_global[NT * block + tid];
//		count2 -= 8 * tid;
//		if(count2 < 8) {
//			// Clear the bits above count2
//			uint64 mask = (1ll<< (8 * count2)) - 1;
//			packed = (packed & mask) | (0x0101010101010101ll & ~mask);
//		}
//		uint2 pair = ulonglong_as_uint2(packed);
//		x = 8 - popc(pair.x) - popc(pair.y);
//
//		packed ^= 0x0101010101010101ll;
//	}
//
//	int total;
//	int scan = S::Scan(tid, x, shared.scan, &total);
//
//	if(x) {
//		#pragma unroll
//		for(int i = 0; i < 8; ++i)
//			if((1ll<< (8 * i)) & packed) 
//				shared.indices[scan++] = gid + 8 * tid + i;
//	}
//	__syncthreads();
//
//	DeviceMemToMemLoop<NT>(total, shared.indices, tid, 
//		rightJoin_global + start, false);
//}
//
//////////////////////////////////////////////////////////////////////////////////
//// RelationalJoin
//
//template<MgpuJoinKind Kind, typename InputIt1, typename InputIt2,
//	typename Comp>
//MGPU_HOST int RelationalJoin(InputIt1 a_global, int aCount, InputIt2 b_global,
//	int bCount, MGPU_MEM(int)* ppAJoinIndices, MGPU_MEM(int)* ppBJoinIndices, 
//	Comp comp, CudaContext& context) {
//
//	typedef typename std::iterator_traits<InputIt1>::value_type T;
//	const bool SupportLeft = MgpuJoinKindLeft == Kind || 
//		MgpuJoinKindOuter == Kind;
//	const bool SupportRight = MgpuJoinKindRight == Kind ||
//		MgpuJoinKindOuter == Kind;
//
//	const MgpuSearchType LeftType = SupportLeft ? 
//		MgpuSearchTypeIndexMatch : MgpuSearchTypeIndex;
//
//	MGPU_MEM(int) aLowerBound = context.Malloc<int>(aCount);
//	MGPU_MEM(byte) bMatches;
//
//	// Find the lower bound of A into B. If we are right joining also return the
//	// set of matches of B into A.
//	int rightJoinTotal = 0;
//	if(SupportRight) {
//		// Support a right or outer join. Count the number of B elements that
//		// have matches in A. These matched values are included in the inner
//		// join part. The other values (bCount - bMatchCount) are copied to the
//		// end for the right join part.
//		bMatches = context.Malloc<byte>(bCount);
//		int bMatchCount;
//		SortedSearch<MgpuBoundsLower, LeftType, MgpuSearchTypeMatch>(a_global,
//			aCount, b_global, bCount, aLowerBound->get(), bMatches->get(), comp,
//			context, 0, &bMatchCount);
//		rightJoinTotal = bCount - bMatchCount;
//	} else 
//		SortedSearch<MgpuBoundsLower, LeftType, MgpuSearchTypeNone>(a_global,
//			aCount, b_global, bCount, aLowerBound->get(), (int*)0, comp,
//			context, 0, 0);
//
//	// Use the lower bounds to compute the counts for each element.
//	MGPU_MEM(int) aCounts = context.Malloc<int>(aCount);
//	if(SupportLeft) 
//		SortedEqualityCount(a_global, aCount, b_global, bCount,
//			aLowerBound->get(), aCounts->get(), comp, LeftJoinEqualityOp(), 
//			context);
//	else
//		SortedEqualityCount(a_global, aCount, b_global, bCount,
//			aLowerBound->get(), aCounts->get(), comp, SortedEqualityOp(), 
//			context);
//
//	// Scan the product counts. This is part of the load-balancing search.
//	int leftJoinTotal = Scan(aCounts->get(), aCount, context);
//
//	// Allocate space for the join indices from the sum of left and right join
//	// sizes.
//	int joinTotal = leftJoinTotal + rightJoinTotal;
//	MGPU_MEM(int) aIndicesDevice = context.Malloc<int>(joinTotal);
//	MGPU_MEM(int) bIndicesDevice = context.Malloc<int>(joinTotal);
//
//	// Launch the inner/left join kernel. Run an upper-bounds partitioning 
//	// to load-balance the data.
//	const int NT = 128;
//	const int VT = 7;
//	typedef LaunchBoxVT<NT, VT> Tuning;
//	int2 launch = Tuning::GetLaunchParams(context);
//	int NV = launch.x * launch.y;
//	
//	MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsUpper>(
//		mgpu::counting_iterator<int>(0), leftJoinTotal, aCounts->get(),
//		aCount, NV, 0, mgpu::less<int>(), context);
//
//	int numBlocks = MGPU_DIV_UP(leftJoinTotal + aCount, NV);
//	KernelLeftJoin<Tuning, SupportLeft>
//		<<<numBlocks, launch.x, 0, context.Stream()>>>(leftJoinTotal, 
//		aLowerBound->get(), aCounts->get(), aCount, partitionsDevice->get(),
//		aIndicesDevice->get(), bIndicesDevice->get());
//
//	// Launch the right join kernel. Compact the non-matches from B into A.
//	if(SupportRight) {
//		const int NT = 128;
//		int numBlocks = MGPU_DIV_UP(bCount, 8 * NT);
//
//		MGPU_MEM(int) totals = context.Malloc<int>(numBlocks);
//		KernelRightJoinUpsweep<NT><<<numBlocks, NT>>>(
//			(const uint64*)bMatches->get(), bCount, totals->get());
//		
//		Scan<MgpuScanTypeExc>(totals->get(), numBlocks, totals->get(),
//			ScanOpAdd(), (int*)0, false, context);
//
//		KernelRightJoinDownsweep<NT><<<numBlocks, NT>>>(
//			(const uint64*)bMatches->get(), bCount, totals->get(), 
//			bIndicesDevice->get() + leftJoinTotal);
//
//		cudaMemset(aIndicesDevice->get() + leftJoinTotal, -1, 
//			sizeof(int) * rightJoinTotal);
//	}
//
//	*ppAJoinIndices = aIndicesDevice;
//	*ppBJoinIndices = bIndicesDevice;
//	return joinTotal;
//}
//
//template<MgpuJoinKind Kind, typename InputIt1, typename InputIt2>
//MGPU_HOST int RelationalJoin(InputIt1 a_global, int aCount, InputIt2 b_global,
//	int bCount, MGPU_MEM(int)* ppAJoinIndices, MGPU_MEM(int)* ppBJoinIndices, 
//	CudaContext& context) {
//
//	typedef typename std::iterator_traits<InputIt1>::value_type T;
//	return RelationalJoin<Kind>(a_global, aCount, b_global, bCount,
//		ppAJoinIndices, ppBJoinIndices, mgpu::less<T>(), context);
//}
//
//} // namespace mgpu
