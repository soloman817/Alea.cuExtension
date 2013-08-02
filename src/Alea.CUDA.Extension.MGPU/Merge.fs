module Alea.CUDA.Extension.MGPU.Merge

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.CTAMerge



let kernelMerge (plan:Plan) (hasValues:int) (mergeSort:int) (compOp:IComp<'TV>) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    let hasValues = if hasValues = 1 then true else false

    let comp = compOp.Device
    let computeMergeRange = computeMergeRange.Device
    let deviceMerge = deviceMerge NT VT hasValues compOp
    let sharedSize = max NV (NT * (VT + 1))

    <@ fun  (aKeys_global:DevicePtr<'TV>) (aVals_global:DevicePtr<'TV>) 
            (aCount:int) 
            (bKeys_global:DevicePtr<'TV>) (bVals_global:DevicePtr<'TV>) 
            (bCount:int) 
            (mp_global:DevicePtr<int>) 
            (coop:int) 
            (keys_global:DevicePtr<'TV>) (vals_global:DevicePtr<'TV>) 
            ->

        let comp = %comp
        let computeMergeRange = %computeMergeRange
        let deviceMerge = %deviceMerge

        let shared = __shared__<'TV>(sharedSize).Ptr(0)
        let sharedKeys = shared
        let sharedIndices = shared.Reinterpret<int>()

        let tid = threadIdx.x
        let block = blockIdx.x

        let range = computeMergeRange aCount bCount block coop (NT * VT) mp_global
        
        deviceMerge aKeys_global aVals_global bKeys_global bVals_global tid block range sharedKeys sharedIndices keys_global vals_global
        @>

type IMergeKeys<'TV> =
    {
        Action : ActionHint -> DevicePtr<'TV> -> DevicePtr<'TV> -> DevicePtr<int> -> DevicePtr<'TV> -> unit
        NumPartitions : int
    }

let mergeKeys (compOp:IComp<'TV>) = cuda {
    let plan = { NT = 128; VT = 11 }
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    let! kernelMerge = (kernelMerge plan 0 0 compOp) |> defineKernelFunc
    let! mpp = Search.mergePathPartitions MgpuBoundsLower compOp

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelMerge = kernelMerge.Apply m
        let mpp = mpp.Apply m

        fun (aCount:int) (bCount:int) ->
            let numBlocks = divup (aCount + bCount) NV
            let lp = LaunchParam(numBlocks, NT)

            let action (hint:ActionHint) (aKeys_global:DevicePtr<'TV>) (bKeys_global:DevicePtr<'TV>) (parts:DevicePtr<int>) (keys_global:DevicePtr<'TV>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
                    let mpp = mpp aCount bCount NV 0
                    let partitions = mpp.Action hint aKeys_global bKeys_global parts
                    kernelMerge.Launch lp aKeys_global (DevicePtr<'TV>(0n)) aCount bKeys_global (DevicePtr<'TV>(0n)) bCount parts 0 keys_global (DevicePtr<'TV>(0n))
                |> worker.Eval
            { Action = action; NumPartitions = numBlocks + 1 } ) }

type IMergePairs<'TV> =
    {
        Action : ActionHint -> DevicePtr<'TV> -> DevicePtr<'TV> -> DevicePtr<'TV> -> DevicePtr<'TV> -> DevicePtr<int> -> DevicePtr<'TV> -> DevicePtr<'TV> -> unit
        NumPartitions : int
    }


let mergePairs (compOp:IComp<'TV>) = cuda {
    let plan = { NT = 128; VT = 7 }
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    let! kernelMerge = kernelMerge plan 1 0 compOp |> defineKernelFuncWithName "mp"
    let! mpp = Search.mergePathPartitions MgpuBoundsLower compOp

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelMerge = kernelMerge.Apply m
        let mpp = mpp.Apply m

        fun (aCount:int) (bCount:int) ->
            let numBlocks = divup (aCount + bCount) NV
            let lp = LaunchParam(numBlocks, NT)

            let action (hint:ActionHint) (aKeys_global:DevicePtr<'TV>) (aVals_global:DevicePtr<'TV>) (bKeys_global:DevicePtr<'TV>) (bVals_global:DevicePtr<'TV>) (parts:DevicePtr<int>) (keys_global:DevicePtr<'TV>) (vals_global:DevicePtr<'TV>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
                    let mpp = mpp aCount bCount NV 0
                    let partitions = mpp.Action hint aKeys_global bKeys_global parts
                    kernelMerge.Launch lp aKeys_global aVals_global aCount bKeys_global bVals_global bCount parts 0 keys_global vals_global
                |> worker.Eval
            { Action = action; NumPartitions = numBlocks + 1 } ) }


type IPKernelMergesort<'TV> =
    {
        Action : ActionHint -> DevicePtr<'TV> -> DevicePtr<int> -> DevicePtr<'TV> -> unit
        NumPartitions : int
    }

let pKernelMergesort (plan:Plan) (compOp:IComp<'TV>) = cuda {
    let NV = plan.NT * plan.VT
    
    let hasValues = 0
    let mergeSort = 1
    
    let! kernelMerge = (kernelMerge plan hasValues mergeSort compOp) |> defineKernelFunc

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelMerge = kernelMerge.Apply m

        fun (count:int) (coop:int) ->
            let numBlocks = divup count NV
            let lp = LaunchParam(numBlocks, plan.NT)

            let action (hint:ActionHint) (source:DevicePtr<'TV>) (parts:DevicePtr<int>) (dest:DevicePtr<'TV>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
                    kernelMerge.Launch lp source (DevicePtr<'TV>(0n)) count source (DevicePtr<'TV>(0n)) 0 parts coop dest (DevicePtr<'TV>(0n))
                |> worker.Eval
            { Action = action; NumPartitions = numBlocks + 1 } ) }

//////////////////////////////////////////////////////////////////////////////////
//// KernelMerge
//
//template<typename Tuning, bool HasValues, bool MergeSort, typename KeysIt1, 
//	typename KeysIt2, typename KeysIt3, typename ValsIt1, typename ValsIt2,
//	typename ValsIt3, typename Comp>
//MGPU_LAUNCH_BOUNDS void KernelMerge(KeysIt1 aKeys_global, ValsIt1 aVals_global,
//	int aCount, KeysIt2 bKeys_global, ValsIt2 bVals_global, int bCount,
//	const int* mp_global, int coop, KeysIt3 keys_global, ValsIt3 vals_global,
//	Comp comp) {
//
//	typedef MGPU_LAUNCH_PARAMS Params;
//	typedef typename std::iterator_traits<KeysIt1>::value_type KeyType;
//	typedef typename std::iterator_traits<ValsIt1>::value_type ValType;
//
//	const int NT = Params::NT;
//	const int VT = Params::VT;
//	const int NV = NT * VT;
//	union Shared {
//		KeyType keys[NT * (VT + 1)];
//		int indices[NV];
//	};
//	__shared__ Shared shared;
//
//	int tid = threadIdx.x;
//	int block = blockIdx.x;
//
//	int4 range = ComputeMergeRange(aCount, bCount, block, coop, NT * VT, 
//		mp_global);
//
//	DeviceMerge<NT, VT, HasValues>(aKeys_global, aVals_global, bKeys_global,
//		bVals_global, tid, block, range, shared.keys, shared.indices, 
//		keys_global, vals_global, comp);
//}
//
//////////////////////////////////////////////////////////////////////////////////
//// MergeKeys
//
//template<typename KeysIt1, typename KeysIt2, typename KeysIt3, typename Comp>
//MGPU_HOST void MergeKeys(KeysIt1 aKeys_global, int aCount, KeysIt2 bKeys_global,
//	int bCount, KeysIt3 keys_global, Comp comp, CudaContext& context) {
//
//	const int NT = 128;
//	const int VT = 11;
//	typedef LaunchBoxVT<NT, VT> Tuning;
//	int2 launch = Tuning::GetLaunchParams(context);
//
//	const int NV = launch.x * launch.y;
//	MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsLower>(
//		aKeys_global, aCount, bKeys_global, bCount, NV, 0, comp, context);
//
//	int numBlocks = MGPU_DIV_UP(aCount + bCount, NV);
//	KernelMerge<Tuning, false, false>
//		<<<numBlocks, launch.x, 0, context.Stream()>>>(aKeys_global, 
//		(const int*)0, aCount, bKeys_global, (const int*)0, bCount, 
//		partitionsDevice->get(), 0, keys_global, (int*)0, comp);
//}
//template<typename KeysIt1, typename KeysIt2, typename KeysIt3>
//MGPU_HOST void MergeKeys(KeysIt1 aKeys_global, int aCount, KeysIt2 bKeys_global,
//	int bCount, KeysIt3 keys_global, CudaContext& context) {
//
//	typedef mgpu::less<typename std::iterator_traits<KeysIt1>::value_type> Comp;
//	return MergeKeys(aKeys_global, aCount, bKeys_global, bCount, keys_global,
//		Comp(), context);
//}
//
//////////////////////////////////////////////////////////////////////////////////
//// MergePairs
//
//template<typename KeysIt1, typename KeysIt2, typename KeysIt3, typename ValsIt1,
//	typename ValsIt2, typename ValsIt3, typename Comp>
//MGPU_HOST void MergePairs(KeysIt1 aKeys_global, ValsIt1 aVals_global, 
//	int aCount, KeysIt2 bKeys_global, ValsIt2 bVals_global, int bCount,
//	KeysIt3 keys_global, ValsIt3 vals_global, Comp comp, CudaContext& context) {
//
//	const int NT = 128;
//	const int VT = 7;
//	typedef LaunchBoxVT<NT, VT> Tuning;
//	int2 launch = Tuning::GetLaunchParams(context);
//
//	const int NV = launch.x * launch.y;
//	MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsLower>(
//		aKeys_global, aCount, bKeys_global, bCount, NV, 0, comp, context);
//
//	int numBlocks = MGPU_DIV_UP(aCount + bCount, NV);
//	KernelMerge<Tuning, true, false>
//		<<<numBlocks, launch.x, 0, context.Stream()>>>(aKeys_global,
//		aVals_global, aCount, bKeys_global, bVals_global, bCount, 
//		partitionsDevice->get(), 0, keys_global, vals_global, comp);
//}
//template<typename KeysIt1, typename KeysIt2, typename KeysIt3, typename ValsIt1,
//	typename ValsIt2, typename ValsIt3>
//MGPU_HOST void MergePairs(KeysIt1 aKeys_global, ValsIt1 aVals_global, 
//	int aCount, KeysIt2 bKeys_global, ValsIt2 bVals_global, int bCount,
//	KeysIt3 keys_global, ValsIt3 vals_global, CudaContext& context) {
//
//	typedef mgpu::less<typename std::iterator_traits<KeysIt1>::value_type> Comp;
//	return MergePairs(aKeys_global, aVals_global, aCount, bKeys_global, 
//		bVals_global, bCount, keys_global, vals_global, Comp(), context);
//}
//
//} // namespace mgpu