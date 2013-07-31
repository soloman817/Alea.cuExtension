module Alea.CUDA.Extension.MGPU.Mergesort

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Intrinsics
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTAScan
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.CTAMerge
open Alea.CUDA.Extension.MGPU.Merge



type Plan =
    {
        NT : int
        VT : int
    }

let kernelBlocksort (plan:Plan) (hasValues:int) (compOp:IComp<'TV>) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    let hasValues = if hasValues = 1 then true else false

    let sharedSize = max NV (NT * (VT + 1))
    let comp = compOp.Device
    
    let deviceGlobalToShared = deviceGlobalToShared NT VT
    let deviceSharedToThread = deviceSharedToThread VT
    let deviceSharedToGlobal = deviceSharedToGlobal NT VT
    let deviceThreadToShared = deviceThreadToShared VT
    let ctaMergesort = ctaMergesort NT VT hasValues compOp
                                                    
    <@ fun  (keysSource_global:DevicePtr<'TV>) (valsSource_global:DevicePtr<'TV>) 
            (count:int) 
            (keysDest_global:DevicePtr<'TV>) (valsDest_global:DevicePtr<'TV>) ->

        let comp = %comp
        let deviceGlobalToShared = %deviceGlobalToShared
        let deviceSharedToThread = %deviceSharedToThread
        let deviceSharedToGlobal = %deviceSharedToGlobal
        let deviceThreadToShared = %deviceThreadToShared
        let ctaMergesort = %ctaMergesort

        let shared = __shared__<'TV>(sharedSize).Ptr(0)
        let sharedKeys = shared
        let sharedValues = shared

        let tid = threadIdx.x
        let block = blockIdx.x
        let gid = NV * block
        let count2 = min NV (count - gid)            

        let threadValues = __local__<'TV>(VT).Ptr(0) 
        if hasValues then
            deviceGlobalToShared count2 (valsSource_global + gid) tid sharedValues true
            deviceSharedToThread sharedValues tid threadValues true
        
        let threadKeys = __local__<'TV>(VT).Ptr(0)
        deviceGlobalToShared count2 (keysSource_global + gid) tid sharedKeys true
        deviceSharedToThread sharedKeys tid threadKeys true

        let first = VT * tid
        let mutable maxKey = threadKeys.[0]
        if ((first + VT) > count2) && (first < count2) then
            
            for i = 1 to VT - 1 do
                if (first + i) < count2 then
                    maxKey <- if comp maxKey threadKeys.[i] then threadKeys.[i] else maxKey
            for i = 0 to VT - 1 do
                if (first + i) >= count2 then threadKeys.[i] <- maxKey

        ctaMergesort threadKeys threadValues sharedKeys sharedValues count2 tid
        deviceSharedToGlobal count2 sharedKeys tid (keysDest_global + gid) true

        if hasValues then
            deviceThreadToShared threadValues tid sharedValues true
            deviceSharedToGlobal count2 sharedValues tid (valsDest_global + gid) true
        @>

type IBlocksort<'TV> =
    {
        Action : ActionHint -> DevicePtr<'TV> -> unit //DevicePtr<'TV> -> unit
        NumPartitions : int
        //NumPasses : int
    }


let mergesortKeys (compOp:IComp<'TV>) = cuda {
    let plan = { NT = 257; VT = 7 }
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT
    
    let! kernelBlocksort = kernelBlocksort plan 0 compOp |> defineKernelFunc //WithName "kbs"
    //let! mpp = Search.mergePathPartitions MgpuBoundsLower compOp    
    //let! kernelMerge = Merge.pKernelMergesort {NT = 256; VT = 7} compOp
    //let swap = (swap (DevicePtr<'TV>(0n))).Host
    //let swap = (swap compOp.Identity).Host

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelBlocksort = kernelBlocksort.Apply m
        //let mpp = mpp.Apply m
        //let kernelMerge = kernelMerge.Apply m

        fun (count:int) ->
            let numBlocks = divup count NV
            let numPasses = findLog2 numBlocks true
            let lp = LaunchParam(numBlocks, NT)

            let action (hint:ActionHint) (source:DevicePtr<'TV>) = //(dest:DevicePtr<'TV>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
//                    kernelBlocksort.Launch lp source (DevicePtr<'TV>(0n)) count (if (1 &&& numPasses) <> 0 then dest else source) (DevicePtr<'TV>(0n))
                    kernelBlocksort.Launch lp source (DevicePtr<'TV>(0n)) count source (DevicePtr<'TV>(0n))

//                    if (1 &&& numPasses) <> 0 then
//                        swap source dest
//                    for pass = 0 to numPasses - 1 do
//                        let coop = 2 <<< pass
//                        let mpp = mpp count 0 NV coop
//                        let partitions = mpp.Action hint source source parts
//                        let kernelMerge = kernelMerge count coop                        
//                        let merged = kernelMerge.Action hint source parts dest
//                        //swap dest source
//                        printfn "pass %d" pass
//                        swap dest source
                |> worker.Eval
            { Action = action; NumPartitions = numBlocks + 1 (*;  NumPasses = numPasses*) } ) }


//type IBlocksort2<'TV> =
//    {
//        Action : ActionHint -> DevicePtr<'TV> -> DevicePtr<'TV> -> unit
//        NumPartitions : int
//        NumPasses : int
//    }
//
//let mergesortKeys2 (compOp:IComp<'TV>) = cuda {
//    let plan = { NT = 256; VT = 7 }
//    let NT = plan.NT
//    let VT = plan.VT
//    let NV = NT * VT
//    
//    let! kernelBlocksort = kernelBlocksort plan 0 compOp |> defineKernelFunc //WithName "kbs"
//    //let! mpp = Search.mergePathPartitions MgpuBoundsLower compOp    
//    //let! kernelMerge = Merge.pKernelMergesort {NT = 256; VT = 7} compOp
//    //let swap = (swap (DevicePtr<'TV>(0n))).Host
//    //let swap = (swap compOp.Identity).Host
//
//    return PFunc(fun (m:Module) ->
//        let worker = m.Worker
//        let kernelBlocksort = kernelBlocksort.Apply m
//        //let mpp = mpp.Apply m
//        //let kernelMerge = kernelMerge.Apply m
//
//        fun (count:int) ->
//            let numBlocks = divup count NV
//            printfn "numBlocks: %d" numBlocks
//            let numPasses = findLog2 numBlocks true
//            printfn "numPasses: %d" numPasses
//            let lp = LaunchParam(numBlocks, NT)
//
//            let action (hint:ActionHint) (source:DevicePtr<'TV>) (dest:DevicePtr<'TV>) =
//                fun () ->
//                    let lp = lp |> hint.ModifyLaunchParam
//                    kernelBlocksort.Launch lp source (DevicePtr<'TV>(0n)) count (if (1 &&& numPasses) <> 0 then dest else source) (DevicePtr<'TV>(0n))
//                |> worker.Eval
//            { Action = action; NumPartitions = numBlocks + 1;  NumPasses = numPasses } ) }
//






//////////////////////////////////////////////////////////////////////////////////
//// MergesortPairs
//
//template<typename KeyType, typename ValType, typename Comp>
//MGPU_HOST void MergesortPairs(KeyType* keys_global, ValType* values_global,
//	int count, Comp comp, CudaContext& context) {
//
//	const int NT = 256;
//	const int VT = 11;
//	typedef LaunchBoxVT<NT, VT> Tuning;
//	int2 launch = Tuning::GetLaunchParams(context);
//
//	const int NV = launch.x * launch.y;
//	int numBlocks = MGPU_DIV_UP(count, NV);
//	int numPasses = FindLog2(numBlocks, true);
//
//	MGPU_MEM(KeyType) keysDestDevice = context.Malloc<KeyType>(count);
//	MGPU_MEM(ValType) valsDestDevice = context.Malloc<ValType>(count);
//	KeyType* keysSource = keys_global;
//	KeyType* keysDest = keysDestDevice->get();
//	ValType* valsSource = values_global;
//	ValType* valsDest = valsDestDevice->get();
//
//	KernelBlocksort<Tuning, true><<<numBlocks, launch.x, 0, context.Stream()>>>(
//		keysSource, valsSource, count, (1 & numPasses) ? keysDest : keysSource, 
//		(1 & numPasses) ? valsDest : valsSource, comp);
//	if(1 & numPasses) {
//		std::swap(keysSource, keysDest);
//		std::swap(valsSource, valsDest);
//	}
//
//	for(int pass = 0; pass < numPasses; ++pass) {
//		int coop = 2<< pass;
//		MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsLower>(
//			keysSource, count, keysSource, 0, NV, coop, comp, context);
//
//		KernelMerge<Tuning, true, true>
//			<<<numBlocks, launch.x, 0, context.Stream()>>>(keysSource, 
//			valsSource, count, keysSource, valsSource, 0, 
//			partitionsDevice->get(), coop, keysDest, valsDest, comp);
//		std::swap(keysDest, keysSource);
//		std::swap(valsDest, valsSource);
//	}
//}
//template<typename KeyType, typename ValType>
//MGPU_HOST void MergesortPairs(KeyType* keys_global, ValType* values_global,
//	int count, CudaContext& context) {
//	MergesortPairs(keys_global, values_global, count, mgpu::less<KeyType>(),
//		context);
//}
//
//template<typename KeyType, typename Comp>
//MGPU_HOST void MergesortIndices(KeyType* keys_global, int* values_global,
//	int count, Comp comp, CudaContext& context) {
//
//	const int NT = 256;
//	const int VT = 11;
//	typedef LaunchBoxVT<NT, VT> Tuning;
//	int2 launch = Tuning::GetLaunchParams(context);
//
//	const int NV = launch.x * launch.y;
//	int numBlocks = MGPU_DIV_UP(count, NV);
//	int numPasses = FindLog2(numBlocks, true);
//
//	MGPU_MEM(KeyType) keysDestDevice = context.Malloc<KeyType>(count);
//	MGPU_MEM(int) valsDestDevice = context.Malloc<int>(count);
//	KeyType* keysSource = keys_global;
//	KeyType* keysDest = keysDestDevice->get();
//	int* valsSource = values_global;
//	int* valsDest = valsDestDevice->get();
//
//	KernelBlocksort<Tuning, true><<<numBlocks, launch.x, 0, context.Stream()>>>(
//		keysSource, mgpu::counting_iterator<int>(0), count, 
//		(1 & numPasses) ? keysDest : keysSource, 
//		(1 & numPasses) ? valsDest : valsSource, comp);
//	if(1 & numPasses) {
//		std::swap(keysSource, keysDest);
//		std::swap(valsSource, valsDest);
//	}
//
//	for(int pass = 0; pass < numPasses; ++pass) {
//		int coop = 2<< pass;
//		MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsLower>(
//			keysSource, count, keysSource, 0, NV, coop, comp, context);
//
//		KernelMerge<Tuning, true, true>
//			<<<numBlocks, launch.x, 0, context.Stream()>>>(keysSource, 
//			valsSource, count, keysSource, valsSource, 0, 
//			partitionsDevice->get(), coop, keysDest, valsDest, comp);
//		std::swap(keysDest, keysSource);
//		std::swap(valsDest, valsSource);
//	}
//}
//template<typename KeyType>
//MGPU_HOST void MergesortIndices(KeyType* keys_global, int* values_global,
//	int count, CudaContext& context) {
//	MergesortIndices(keys_global, values_global, count, mgpu::less<KeyType>(),
//		context);
//}
//
//} // namespace mgpu