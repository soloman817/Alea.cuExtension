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


type Plan =
    {
        NT : int
        Bounds : int
    }


let kernelBinarySearch (plan:Plan) (compOp:CompType) =
    let NT = plan.NT
    let bounds = plan.Bounds
    let binarySearch = (BinarySearch bounds compOp).DBinarySearch
    
    <@ fun (count:int) (data_global:DevicePtr<'TI>) (numItems:int) (nv:int) (partitions_global:DevicePtr<int>) (numSearches:int) ->
        let binarySearch = %binarySearch

        let gid = NT * blockIdx.x + threadIdx.x
        if (gid < numSearches) then
            let p = binarySearch data_global numItems (min (nv * gid) count)
            partitions_global.[gid] <- p
        @>

type IBinarySearchPartitions<'TI> =
    {
        NumBlocks : int
        Action : ActionHint -> DevicePtr<'TI> -> unit
        Partitions : DevicePtr<'TI>
    }

let binarySearchPartitions (bounds:int) (compOp:CompType) = cuda { 
    let plan = { NT = 64; Bounds = bounds }
    let! kernelBinarySearch = (kernelBinarySearch plan compOp) |> defineKernelFunc

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelBinarySearch = kernelBinarySearch.Apply m
        
        fun (count:int) (numItems:int) (nv:int) ->
            let numBlocks = divup count nv
            let numPartitionBlocks = divup (numBlocks + 1) plan.NT
            let partitionsDevice = worker.Malloc<int>(numBlocks + 1)
            let lp = LaunchParam(numPartitionBlocks, plan.NT)

            let action (hint:ActionHint) (data_global:DevicePtr<'TI>) =
                let lp = lp |> hint.ModifyLaunchParam
                kernelBinarySearch.Launch lp count data_global numItems nv partitionsDevice.Ptr (numBlocks + 1)
                
            { NumBlocks = numBlocks; Action = action; Partitions = partitionsDevice.Ptr} ) }


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