module Alea.CUDA.Extension.MGPU.Search

// this file maps to search.cuh

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


type Plan =
    {
        NT : int
        Bounds : int
    }

// @COMMENTS@ : uhmm, here , because the (min (nv * gid) count is always int, so 
// it cannot be generic, so I made it to be int, cause I found it is just for
// index calculation. But if it need to be generic, it is also doable, just
// make the (min (nv * gid) count) to be a input generic argument. 
let kernelBinarySearch (plan:Plan) (binarySearch:IBinarySearch<int>) =
    let NT = plan.NT
    let bounds = plan.Bounds
    let binarySearch = binarySearch.DBinarySearch
    
    <@ fun (count:int) (data_global:DevicePtr<int>) (numItems:int) (nv:int) (partitions_global:DevicePtr<int>) (numSearches:int) ->
        let binarySearch = %binarySearch
        let gid = NT * blockIdx.x + threadIdx.x
        if (gid < numSearches) then
            let p = binarySearch data_global numItems (min (nv * gid) count)
            partitions_global.[gid] <- p @>

// @COMMENTS@ : also here, cause I think it is just for index calculation. Tell me if
// I was wrong that there is cases that will use it for non-int usage.
type IBinarySearchPartitions =
    {
        Action : ActionHint -> DevicePtr<int> -> DevicePtr<int> -> unit        
    }

let binarySearchPartitions (bounds:int) (compOp:IComp<int>) = cuda { 
    let plan = { NT = 64; Bounds = bounds }
    let binarySearch = binarySearchFun bounds compOp
    let! kernelBinarySearch = (kernelBinarySearch plan binarySearch) |> defineKernelFuncWithName "bsp"

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelBinarySearch = kernelBinarySearch.Apply m
        
        fun (count:int) (numItems:int) (nv:int) ->
            let numBlocks = divup count nv
            let numPartitionBlocks = divup (numBlocks + 1) plan.NT
            let lp = LaunchParam(numPartitionBlocks, plan.NT)
            
            let action (hint:ActionHint) (data_global:DevicePtr<int>) (partitionsDevice:DevicePtr<int>) =
                let lp = lp |> hint.ModifyLaunchParam
                kernelBinarySearch.Launch lp count data_global numItems nv partitionsDevice (numBlocks + 1)
                            
            { Action = action } ) }



// MergePathPartitions
let kernelMergePartition (NT:int) (bounds:int) (comp:IComp<'TC>) = 
    let mergePath = (mergeSearch bounds comp).DMergePath
    let findMergesortFrame = findMergesortFrame.Device

    <@ fun (a_global:DevicePtr<'TI1>) (aCount:int) (b_global:DevicePtr<'TI2>) (bCount:int) (nv:int) (coop:int) (mp_global:DevicePtr<int>) (numSearches:int) ->
        let mergePath = %mergePath
        let findMergesortFrame = %findMergesortFrame

        let partition = NT * blockIdx.x * threadIdx.x
        let mutable aCount, bCount = aCount, bCount
        if partition < numSearches then
            let mutable a0, b0 = 0, 0
            let mutable gid = nv * partition
            if coop <> 0 then
                let frame = findMergesortFrame coop partition nv
                a0 <- frame.x
                b0 <- min aCount frame.y
                bCount <- (min aCount (frame.y + frame.z)) - b0
                aCount <- (min aCount (frame.x + frame.z)) - a0
                
                gid <- gid - a0
            let mp = mergePath (a_global + a0) aCount (b_global + b0) bCount (min gid (aCount + bCount))
            mp_global.[partition] <- mp @>


type IMergePathPartitions<'TI1, 'TI2> =
    {
        Action : ActionHint -> DevicePtr<'TI1> -> DevicePtr<'TI2> -> int -> int -> unit        
    }


let mergePathPartitions (bounds:int) (comp:IComp<'TC>) = cuda {
    let NT = 64
    let! kernelMergePartition = (kernelMergePartition 64 bounds comp) |> defineKernelFunc

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelMergePartition = kernelMergePartition.Apply m

        fun (aCount:int) (bCount:int) (nv:int) (partitionsDevice:DevicePtr<int>) ->
            let numPartitions = divup (aCount + bCount) nv
            let numPartitionBlocks = divup (numPartitions + 1) NT
            let lp = LaunchParam(numPartitionBlocks, NT)
            
            let action (hint:ActionHint) (a_global:DevicePtr<'TI1>) (b_global:DevicePtr<'TI2>) (coop:int) (numSearches:int) =
                let lp = lp |> hint.ModifyLaunchParam
                kernelMergePartition.Launch lp a_global aCount b_global bCount nv coop partitionsDevice numSearches

            { Action = action } ) }





// MODERN GPU C++ CODE
//
//
//////////////////////////////////////////////////////////////////////////////////
//// MergePathPartitions
//
//template<int NT, MgpuBounds Bounds, typename It1, typename It2, typename Comp>
//__global__ void KernelMergePartition(It1 a_global, int aCount, It2 b_global, 
//	int bCount, int nv, int coop, int* mp_global, int numSearches, Comp comp) {
//
//	int partition = NT * blockIdx.x + threadIdx.x;
//	if(partition < numSearches) {
//		int a0 = 0, b0 = 0;
//		int gid = nv * partition;
//		if(coop) {
//			int3 frame = FindMergesortFrame(coop, partition, nv);
//			a0 = frame.x;
//			b0 = min(aCount, frame.y);
//			bCount = min(aCount, frame.y + frame.z) - b0;
//			aCount = min(aCount, frame.x + frame.z) - a0;
//
//			// Put the cross-diagonal into the coordinate system of the input
//			// lists.
//			gid -= a0;
//		}
//		int mp = MergePath<Bounds>(a_global + a0, aCount, b_global + b0, bCount,
//			min(gid, aCount + bCount), comp);
//		mp_global[partition] = mp;
//	}
//}
//
//template<MgpuBounds Bounds, typename It1, typename It2, typename Comp>
//MGPU_MEM(int) MergePathPartitions(It1 a_global, int aCount, It2 b_global,
//	int bCount, int nv, int coop, Comp comp, CudaContext& context) {
//
//	const int NT = 64;
//	int numPartitions = MGPU_DIV_UP(aCount + bCount, nv);
//	int numPartitionBlocks = MGPU_DIV_UP(numPartitions + 1, NT);
//	MGPU_MEM(int) partitionsDevice = context.Malloc<int>(numPartitions + 1);
//
//	KernelMergePartition<NT, Bounds>
//		<<<numPartitionBlocks, NT, 0, context.Stream()>>>(a_global, aCount,
//		b_global, bCount, nv, coop, partitionsDevice->get(), numPartitions + 1, 
//		comp);
//	return partitionsDevice;
//}
//
//
//////////////////////////////////////////////////////////////////////////////////
//// FindSetPartitions
//
//template<int NT, bool Duplicates, typename InputIt1, typename InputIt2,
//	typename Comp>
//__global__ void KernelSetPartition(InputIt1 a_global, int aCount, 
//	InputIt2 b_global, int bCount, int nv, int* bp_global, int numSearches,
//	Comp comp) {
//
//	int gid = NT * blockIdx.x + threadIdx.x;
//	if(gid < numSearches) {
//		int diag = min(aCount + bCount, gid * nv);
//
//		// Search with level 4 bias. This helps the binary search converge 
//		// quickly for small runs of duplicates (the common case).
//		int2 bp = BalancedPath<Duplicates, int64>(a_global, aCount, b_global,
//			bCount, diag, 4, comp);
//
//		if(bp.y) bp.x |= 0x80000000;
//		bp_global[gid] = bp.x;
//	}
//}
//
//template<bool Duplicates, typename It1, typename It2, typename Comp>
//MGPU_MEM(int) FindSetPartitions(It1 a_global, int aCount, It2 b_global,
//	int bCount, int nv, Comp comp, CudaContext& context) {
//
//	const int NT = 64;
//	int numPartitions = MGPU_DIV_UP(aCount + bCount, nv);
//	int numPartitionBlocks = MGPU_DIV_UP(numPartitions + 1, NT);
//	MGPU_MEM(int) partitionsDevice = context.Malloc<int>(numPartitions + 1);
//
//	KernelSetPartition<NT, Duplicates>
//		<<<numPartitionBlocks, NT, 0, context.Stream()>>>(a_global, aCount,
//		b_global, bCount, nv, partitionsDevice->get(), numPartitions + 1, comp);
//	return partitionsDevice;
//}