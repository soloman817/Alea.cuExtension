module Alea.CUDA.Extension.MGPU.BulkInsert

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTAScan


//type Plan =
//    {
//        NT : int
//        VT: int
//    }
//
//let kernelBulkInsert (plan:Plan) =
//    let NT = plan.NT
//    let VT = plan.VT
//    let NV = NT * VT
//
//    
//    let capacity, bulkInsert = ctaScan NT (scanOp ScanOpTypeAdd 0)
//    let alignOfTI, sizeOfTI = TypeUtil.cudaAlignOf typeof<'TI>, sizeof<'TI>
//    let alignOfTV, sizeOfTV = TypeUtil.cudaAlignOf typeof<'TV>, sizeof<'TV>
//    let sharedAlign = max alignOfTI alignOfTV
//    let sharedSize = max (sizeOfTI * NV) (sizeOfTV * capacity)
//    let createSharedExpr = createSharedExpr sharedAlign sharedSize
//
//    
//    
//    let deviceGlobalToReg = deviceGlobalToReg NT VT
//
//    <@ fun (a_global:DevicePtr<'TI>) (indices_global:DevicePtr<'TI>) (aCount:int) (b_global:DevicePtr<'TI>) (bCount:int) (mp_global:RWPtr<int>) (dest_global:DevicePtr<'TV>) ->
//        let deviceGlobalToReg = %deviceGlobalToReg
//        let bulkInsert = %bulkInsert
//
//        let shared = %(createSharedExpr)
//        let sharedbulkInsert = shared.Reinterpret<'TV>()
//        let sharedInputs = shared.Reinterpret<'TI>()
//
//        let tid = threadIdx.x
//        let block = blockIdx.x
//        
//        let mutable range = computeMergeRange aCount bCount block 0 NV mp_global
//        let a0 = range.x
//        let a1 = range.y
//        let b0 = range.z
//        let b1 = range.w
//
//
//
//        let mutable total = extract identity -1
//        let mutable totalDefined = false
//
//        while range.x < range.y do
//            let count2 = min NV (count - range.x)
//                        
//            let inputs = __local__<'TI>(VT).Ptr(0)
//            deviceGlobalToReg count2 (data_global + range.x) tid inputs 0
//
//            if commutative <> 0 then
//                for i = 0 to VT - 1 do
//                    let index = NT * i + tid
//                    if index < count2 then
//                        let x = extract inputs.[i] (range.x + index)
//                        total <- if i > 0 || totalDefined then plus total x else x
//            else 
//                // TODO
//                ()
//
//            range.x <- range.x + NV
//            totalDefined <- true
//
//        if commutative <> 0 then
//            total <- bulkInsert tid total sharedbulkInsert
//
//        if tid = 0 then reduction_global.[block] <- total @>
//
//type IbulkInsert<'TI, 'TV> =
//    {
//        NumBlocks : int
//        Action : ActionHint -> DevicePtr<'TI> ->DevicePtr<'TV> -> unit
//        Result : 'TV[] -> 'TV
//    }
//
//let bulkInsert (op:IScanOp<'TI, 'TV, 'TR>) = cuda {
//    let cutOff = 20000
//    let plan1 = { NT = 512; VT = 5 }
//    let plan2 = { NT = 128; VT = 9 }
//    let! kernelbulkInsert1 = kernelbulkInsert plan1 op |> defineKernelFunc
//    let! kernelbulkInsert2 = kernelbulkInsert plan2 op |> defineKernelFunc
//    let hplus = op.HPlus
//
//    return PFunc(fun (m:Module) ->
//        let worker = m.Worker
//        let kernelbulkInsert1 = kernelbulkInsert1.Apply m
//        let kernelbulkInsert2 = kernelbulkInsert2.Apply m
//
//        fun (count:int) ->
//            let numBlocks, task, lp, kernelbulkInsert =
//                if count < cutOff then
//                    let plan = plan1
//                    let kernelbulkInsert = kernelbulkInsert1
//                    let NV = plan.NT * plan.VT
//                    let numTiles = divup count NV
//                    let numBlocks = 1
//                    let task = int2(numTiles, 1)
//                    let lp = LaunchParam(1, plan.NT)
//                    numBlocks, task, lp, kernelbulkInsert
//                else
//                    let plan = plan2
//                    let kernelbulkInsert = kernelbulkInsert2
//                    let NV = plan.NT * plan.VT
//                    let numTiles = divup count NV
//                    let numBlocks = min (worker.Device.NumSm * 25) numTiles
//                    let task = divideTaskRange numTiles numBlocks
//                    let lp = LaunchParam(numBlocks, plan.NT)
//                    numBlocks, task, lp, kernelbulkInsert
//
//            let action (hint:ActionHint) (data:DevicePtr<'TI>) (reduction:DevicePtr<'TV>) =
//                let lp = lp |> hint.ModifyLaunchParam
//                kernelbulkInsert.Launch lp data count task reduction
//
//            let result (reduction:'TV[]) =
//                reduction |> Array.bulkInsert hplus
//
//            { NumBlocks = numBlocks; Action = action; Result = result } ) }



//////////////////////////////////////////////////////////////////////////////////
//// KernelBulkInsert
//
//// Insert the values from a_global into the positions marked by indices_global.
//template<typename Tuning, typename InputIt1, typename IndicesIt, 
//	typename InputIt2, typename OutputIt>
//MGPU_LAUNCH_BOUNDS void KernelBulkInsert(InputIt1 a_global, 
//	IndicesIt indices_global, int aCount, InputIt2 b_global, int bCount, 
//	const int* mp_global, OutputIt dest_global) {
//
//	typedef MGPU_LAUNCH_PARAMS Params;
//	typedef typename std::iterator_traits<InputIt1>::value_type T;
//	const int NT = Params::NT;
//	const int VT = Params::VT;
//	const int NV = NT * VT;
//
//	typedef CTAScan<NT, ScanOpAdd> S;
//	union Shared {
//		int indices[NV];
//		typename S::Storage scan;
//	};
//	__shared__ Shared shared;
//
//	int tid = threadIdx.x;
//	int block = blockIdx.x;
//
//	int4 range = ComputeMergeRange(aCount, bCount, block, 0, NV, mp_global);
//	int a0 = range.x;		// A is array of values to insert.
//	int a1 = range.y;
//	int b0 = range.z;		// B is source array.
//	int b1 = range.w;
//	aCount = a1 - a0;
//	bCount = b1 - b0;
//
//	// Initialize the indices to 0.
//	#pragma unroll
//	for(int i = 0; i < VT; ++i)
//		shared.indices[NT * i + tid] = 0;
//	__syncthreads();
//
//	// Load the indices.
//	int indices[VT];
//	DeviceGlobalToReg<NT, VT>(aCount, indices_global + a0, tid, indices);
//
//	// Set the counters for all the loaded indices. This has the effect of 
//	// pushing the scanned values to the right, causing the B data to be 
//	// inserted to the right of each insertion point.
//	#pragma unroll
//	for(int i = 0; i < VT; ++i) {
//		int index = NT * i + tid;
//		if(index < aCount) shared.indices[index + indices[i] - b0] = 1;
//	}
//	__syncthreads();
//
//	// Run a raking scan over the indices.
//	int x = 0;
//	#pragma unroll
//	for(int i = 0; i < VT; ++i)
//		x += indices[i] = shared.indices[VT * tid + i];
//	__syncthreads();
//
//	// Run a CTA scan over the thread totals.
//	int scan = S::Scan(tid, x, shared.scan);
//
//	// Complete the scan to compute merge-style gather indices. Indices between
//	// in the interval (0, aCount) are from array A (the new values). Indices in
//	// (aCount, aCount + bCount) are from array B (the sources). This style of
//	// indexing lets us use DeviceTransferMergeValues to do global memory 
//	// transfers. 
//	#pragma unroll
//	for(int i = 0; i < VT; ++i) {
//		int index = VT * tid + i;
//		int gather = indices[i] ? scan++ : aCount + index - scan;
//		shared.indices[index] = gather;
//	}
//	__syncthreads();
//
//	DeviceTransferMergeValues<NT, VT>(aCount + bCount, a_global + a0, 
//		b_global + b0, aCount, shared.indices, tid, dest_global + a0 + b0,
//		false);
//}
//
//
//////////////////////////////////////////////////////////////////////////////////
//// BulkInsert
//// Insert elements from A into elements from B before indices.
//
//template<typename InputIt1, typename IndicesIt, typename InputIt2,
//	typename OutputIt>
//MGPU_HOST void BulkInsert(InputIt1 a_global, IndicesIt indices_global, 
//	int aCount, InputIt2 b_global, int bCount, OutputIt dest_global,
//	CudaContext& context) {
//
//	const int NT = 128;
//	const int VT = 7;
//	typedef LaunchBoxVT<NT, VT> Tuning;
//	int2 launch = Tuning::GetLaunchParams(context);
//	const int NV = launch.x * launch.y;
//
//	MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsLower>(
//		indices_global, aCount, mgpu::counting_iterator<int>(0), bCount, NV, 0,
//		mgpu::less<int>(), context);
//
//	int numBlocks = MGPU_DIV_UP(aCount + bCount, NV);
//	KernelBulkInsert<Tuning><<<numBlocks, launch.x, 0, context.Stream()>>>(
//		a_global, indices_global, aCount, b_global, bCount, 
//		partitionsDevice->get(), dest_global);
//}
//
//} // namespace mgpu
