module Alea.cuExtension.MGPU.Merge
//
//open System.Runtime.InteropServices
//open Microsoft.FSharp.Collections
//open Alea.CUDA
//open Alea.CUDA.Utilities
//open Alea.cuExtension
////open Alea.cuExtension.Util
//open Alea.cuExtension.MGPU
////open Alea.cuExtension.MGPU.QuotationUtil
//open Alea.cuExtension.MGPU.DeviceUtil
//open Alea.cuExtension.MGPU.LoadStore
//open Alea.cuExtension.MGPU.CTASearch
//open Alea.cuExtension.MGPU.CTAMerge
//
//
//
//let kernelMerge (plan:Plan) (hasValues:int) (mergeSort:int) (compOp:IComp<'TV>) =
//    let NT = plan.NT
//    let VT = plan.VT
//    let NV = NT * VT
//
//    let hasValues = if hasValues = 1 then true else false
//
//    let comp = compOp.Device
//    let computeMergeRange = computeMergeRange.Device
//    let deviceMerge = deviceMerge NT VT hasValues compOp
//    let sharedSize = max NV (NT * (VT + 1))
//
//    <@ fun  (aKeys_global:deviceptr<'TV>) (aVals_global:deviceptr<'TV>) 
//            (aCount:int) 
//            (bKeys_global:deviceptr<'TV>) (bVals_global:deviceptr<'TV>) 
//            (bCount:int) 
//            (mp_global:deviceptr<int>) 
//            (coop:int) 
//            (keys_global:deviceptr<'TV>) (vals_global:deviceptr<'TV>) 
//            ->
//
//        let comp = %comp
//        let computeMergeRange = %computeMergeRange
//        let deviceMerge = %deviceMerge
//
//        let shared = __shared__.Array<'TV>(sharedSize) |> __array_to_ptr
//        let sharedKeys = shared
//        let sharedIndices = __shared__.Array<int>()
//
//        let tid = threadIdx.x
//        let block = blockIdx.x
//
//        let range = computeMergeRange aCount bCount block coop (NT * VT) mp_global
//        
//        deviceMerge aKeys_global aVals_global bKeys_global bVals_global tid block range sharedKeys sharedIndices keys_global vals_global
//        @>
//
////type IMergeKeys<'TV> =
////    {
////        Action : ActionHint -> deviceptr<'TV> -> deviceptr<'TV> -> deviceptr<int> -> deviceptr<'TV> -> unit
////        NumPartitions : int
////    }
//
//let mergeKeys (compOp:IComp<'TV>) = cuda {
//    let plan = { NT = 128; VT = 11 }
//    let NT = plan.NT
//    let VT = plan.VT
//    let NV = NT * VT
//
//    let! kernelMerge = (kernelMerge plan 0 0 compOp) |> Compiler.DefineKernel
//    let! mpp = Search.mergePathPartitions MgpuBoundsLower compOp
//
//    return Entry(fun program ->
//        let worker = program.Worker
//        let kernelMerge = program.Apply kernelMerge
//        //let mpp = mpp.Apply m
//
//        fun (aCount:int) (bCount:int) ->
//            let numBlocks = divup (aCount + bCount) NV
//            let lp = LaunchParam(numBlocks, NT)
//
//            let run (aKeys_global:deviceptr<'TV>) (bKeys_global:deviceptr<'TV>) (parts:deviceptr<int>) (keys_global:deviceptr<'TV>) =
//                fun () ->
//                    
//                    let mpp = mpp aCount bCount NV 0
//                    let partitions = mpp aKeys_global bKeys_global parts
//                    kernelMerge.Launch lp aKeys_global (deviceptr<'TV>(0n)) aCount bKeys_global (deviceptr<'TV>(0n)) bCount parts 0 keys_global (deviceptr<'TV>(0n))
//                |> worker.Eval
//            { NumPartitions = numBlocks + 1 } ) }
//
////type IMergePairs<'TV> =
////    {
////        Action : ActionHint -> deviceptr<'TV> -> deviceptr<'TV> -> deviceptr<'TV> -> deviceptr<'TV> -> deviceptr<int> -> deviceptr<'TV> -> deviceptr<'TV> -> unit
////        NumPartitions : int
////    }
//
//
//let mergePairs (compOp:IComp<'TV>) = cuda {
//    let plan = { NT = 128; VT = 7 }
//    let NT = plan.NT
//    let VT = plan.VT
//    let NV = NT * VT
//
//    let! kernelMerge = kernelMerge plan 1 0 compOp |> Compiler.DefineKernel //"mp"
//    let! mpp = Search.mergePathPartitions MgpuBoundsLower compOp
//
//    return Entry(fun program ->
//        let worker = program.Worker
//        let kernelMerge = program.Apply kernelMerge
//        //let mpp = mpp.Apply m
//
//        fun (aCount:int) (bCount:int) ->
//            let numBlocks = divup (aCount + bCount) NV
//            let lp = LaunchParam(numBlocks, NT)
//
//            let run (aKeys_global:deviceptr<'TV>) (aVals_global:deviceptr<'TV>) (bKeys_global:deviceptr<'TV>) (bVals_global:deviceptr<'TV>) (parts:deviceptr<int>) (keys_global:deviceptr<'TV>) (vals_global:deviceptr<'TV>) =
//                fun () ->
//                    
//                    let mpp = mpp aCount bCount NV 0
//                    let partitions = mpp aKeys_global bKeys_global parts
//                    kernelMerge.Launch lp aKeys_global aVals_global aCount bKeys_global bVals_global bCount parts 0 keys_global vals_global
//                |> worker.Eval
//            { NumPartitions = numBlocks + 1 } ) }
//
//
////type IPKernelMergesort<'TV> =
////    {
////        Action : ActionHint -> deviceptr<'TV> -> deviceptr<int> -> deviceptr<'TV> -> unit
////        NumPartitions : int
////    }
//
//let pKernelMergesort (plan:Plan) (compOp:IComp<'TV>) = cuda {
//    let NV = plan.NT * plan.VT
//    
//    let hasValues = 0
//    let mergeSort = 1
//    
//    let! kernelMerge = (kernelMerge plan hasValues mergeSort compOp) |> Compiler.DefineKernel
//
//    return Entry(fun program ->
//        let worker = program.Worker
//        let kernelMerge = program.Apply kernelMerge
//
//        fun (count:int) (coop:int) ->
//            let numBlocks = divup count NV
//            let lp = LaunchParam(numBlocks, plan.NT)
//
//            let run (source:deviceptr<'TV>) (parts:deviceptr<int>) (dest:deviceptr<'TV>) =
//                fun () ->
//                    
//                    kernelMerge.Launch lp source (deviceptr<'TV>(0n)) count source (deviceptr<'TV>(0n)) 0 parts coop dest (deviceptr<'TV>(0n))
//                |> worker.Eval
//            { NumPartitions = numBlocks + 1 } ) }
//
////////////////////////////////////////////////////////////////////////////////////
////// KernelMerge
////
////template<typename Tuning, bool HasValues, bool MergeSort, typename KeysIt1, 
////	typename KeysIt2, typename KeysIt3, typename ValsIt1, typename ValsIt2,
////	typename ValsIt3, typename Comp>
////MGPU_LAUNCH_BOUNDS void KernelMerge(KeysIt1 aKeys_global, ValsIt1 aVals_global,
////	int aCount, KeysIt2 bKeys_global, ValsIt2 bVals_global, int bCount,
////	const int* mp_global, int coop, KeysIt3 keys_global, ValsIt3 vals_global,
////	Comp comp) {
////
////	typedef MGPU_LAUNCH_PARAMS Params;
////	typedef typename std::iterator_traits<KeysIt1>::value_type KeyType;
////	typedef typename std::iterator_traits<ValsIt1>::value_type ValType;
////
////	const int NT = Params::NT;
////	const int VT = Params::VT;
////	const int NV = NT * VT;
////	union Shared {
////		KeyType keys[NT * (VT + 1)];
////		int indices[NV];
////	};
////	__shared__ Shared shared;
////
////	int tid = threadIdx.x;
////	int block = blockIdx.x;
////
////	int4 range = ComputeMergeRange(aCount, bCount, block, coop, NT * VT, 
////		mp_global);
////
////	DeviceMerge<NT, VT, HasValues>(aKeys_global, aVals_global, bKeys_global,
////		bVals_global, tid, block, range, shared.keys, shared.indices, 
////		keys_global, vals_global, comp);
////}
////
////////////////////////////////////////////////////////////////////////////////////
////// MergeKeys
////
////template<typename KeysIt1, typename KeysIt2, typename KeysIt3, typename Comp>
////MGPU_HOST void MergeKeys(KeysIt1 aKeys_global, int aCount, KeysIt2 bKeys_global,
////	int bCount, KeysIt3 keys_global, Comp comp, CudaContext& context) {
////
////	const int NT = 128;
////	const int VT = 11;
////	typedef LaunchBoxVT<NT, VT> Tuning;
////	int2 launch = Tuning::GetLaunchParams(context);
////
////	const int NV = launch.x * launch.y;
////	MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsLower>(
////		aKeys_global, aCount, bKeys_global, bCount, NV, 0, comp, context);
////
////	int numBlocks = MGPU_DIV_UP(aCount + bCount, NV);
////	KernelMerge<Tuning, false, false>
////		<<<numBlocks, launch.x, 0, context.Stream()>>>(aKeys_global, 
////		(const int*)0, aCount, bKeys_global, (const int*)0, bCount, 
////		partitionsDevice->get(), 0, keys_global, (int*)0, comp);
////}
////template<typename KeysIt1, typename KeysIt2, typename KeysIt3>
////MGPU_HOST void MergeKeys(KeysIt1 aKeys_global, int aCount, KeysIt2 bKeys_global,
////	int bCount, KeysIt3 keys_global, CudaContext& context) {
////
////	typedef mgpu::less<typename std::iterator_traits<KeysIt1>::value_type> Comp;
////	return MergeKeys(aKeys_global, aCount, bKeys_global, bCount, keys_global,
////		Comp(), context);
////}
////
////////////////////////////////////////////////////////////////////////////////////
////// MergePairs
////
////template<typename KeysIt1, typename KeysIt2, typename KeysIt3, typename ValsIt1,
////	typename ValsIt2, typename ValsIt3, typename Comp>
////MGPU_HOST void MergePairs(KeysIt1 aKeys_global, ValsIt1 aVals_global, 
////	int aCount, KeysIt2 bKeys_global, ValsIt2 bVals_global, int bCount,
////	KeysIt3 keys_global, ValsIt3 vals_global, Comp comp, CudaContext& context) {
////
////	const int NT = 128;
////	const int VT = 7;
////	typedef LaunchBoxVT<NT, VT> Tuning;
////	int2 launch = Tuning::GetLaunchParams(context);
////
////	const int NV = launch.x * launch.y;
////	MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsLower>(
////		aKeys_global, aCount, bKeys_global, bCount, NV, 0, comp, context);
////
////	int numBlocks = MGPU_DIV_UP(aCount + bCount, NV);
////	KernelMerge<Tuning, true, false>
////		<<<numBlocks, launch.x, 0, context.Stream()>>>(aKeys_global,
////		aVals_global, aCount, bKeys_global, bVals_global, bCount, 
////		partitionsDevice->get(), 0, keys_global, vals_global, comp);
////}
////template<typename KeysIt1, typename KeysIt2, typename KeysIt3, typename ValsIt1,
////	typename ValsIt2, typename ValsIt3>
////MGPU_HOST void MergePairs(KeysIt1 aKeys_global, ValsIt1 aVals_global, 
////	int aCount, KeysIt2 bKeys_global, ValsIt2 bVals_global, int bCount,
////	KeysIt3 keys_global, ValsIt3 vals_global, CudaContext& context) {
////
////	typedef mgpu::less<typename std::iterator_traits<KeysIt1>::value_type> Comp;
////	return MergePairs(aKeys_global, aVals_global, aCount, bKeys_global, 
////		bVals_global, bCount, keys_global, vals_global, Comp(), context);
////}
////
////} // namespace mgpu