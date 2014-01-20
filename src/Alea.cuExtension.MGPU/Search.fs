module Alea.cuExtension.MGPU.Search

// this file maps to search.cuh
open System.Runtime.InteropServices
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension
//open Alea.cuExtension.Util
open Alea.cuExtension.MGPU
//open Alea.cuExtension.MGPU.QuotationUtil
open Alea.cuExtension.MGPU.DeviceUtil
open Alea.cuExtension.MGPU.LoadStore
open Alea.cuExtension.MGPU.CTASearch
open Alea.cuExtension.MGPU.CTAMerge

// @COMMENTS@ : uhmm, here , because the (min (nv * gid) count is always int, so 
// it cannot be generic, so I made it to be int, cause I found it is just for
// index calculation. But if it need to be generic, it is also doable, just
// make the (min (nv * gid) count) to be a input generic argument. 
let kernelBinarySearch (NT:int) (bounds:int) (compOp:IComp<int>) =    
    let bounds = if bounds = 0 then MgpuBoundsLower else MgpuBoundsUpper
    let binarySearch = (binarySearch bounds compOp).DBinarySearch
    
    <@ fun (count:int) (data_global:deviceptr<int>) (numItems:int) (nv:int) (partitions_global:deviceptr<int>) (numSearches:int) ->
        let binarySearch = %binarySearch
        let gid = NT * blockIdx.x + threadIdx.x
        if (gid < numSearches) then
            let p = binarySearch data_global numItems (min (nv * gid) count)
            partitions_global.[gid] <- p @>

//// @COMMENTS@ : also here, cause I think it is just for index calculation. Tell me if
//// I was wrong that there is cases that will use it for non-int usage.
//type IBinarySearchPartitions =
//    {
//        Action : ActionHint -> deviceptr<int> -> deviceptr<int> -> unit        
//    }

let binarySearchPartitions (bounds:int) (compOp:IComp<int>) = cuda { 
    let NT = 64
    let bounds = if bounds = MgpuBoundsLower then 0 else 1
    let! kernelBinarySearch = (kernelBinarySearch NT bounds compOp) |> Compiler.DefineKernel //"bsp"

    return Entry(fun program ->
        let worker = program.Worker
        let kernelBinarySearch = program.Apply kernelBinarySearch
        
        fun (count:int) (numItems:int) (nv:int) ->
            let numBlocks = divup count nv
            let numPartitionBlocks = divup (numBlocks + 1) NT
            let lp = LaunchParam(numPartitionBlocks, NT)
            
            let run (data_global:deviceptr<int>) (partitionsDevice:deviceptr<int>) =
                kernelBinarySearch.Launch lp count data_global numItems nv partitionsDevice (numBlocks + 1)
            run            
             ) }


// Leaving as int for now
// MergePathPartitions
let kernelMergePartition (NT:int) (bounds:int) (compOp:IComp<'T>) =     
    let bounds = if bounds = 0 then MgpuBoundsLower else MgpuBoundsUpper
    let mergePath = (mergePath bounds compOp).DMergePath
    let findMergesortFrame = findMergesortFrame.Device

    <@ fun (a_global:deviceptr<'T>) (aCount:int) (b_global:deviceptr<'T>) (bCount:int) (nv:int) (coop:int) (mp_global:deviceptr<int>) (numSearches:int) ->
        let mergePath = %mergePath
        let findMergesortFrame = %findMergesortFrame

        let partition = NT * blockIdx.x + threadIdx.x
        let mutable aCount = aCount
        let mutable bCount = bCount
        
        if partition < numSearches then
            let mutable a0 = 0
            let mutable b0 = 0
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

type API<'T> =
    {
        MergePartition : Resources.Kernel<deviceptr<'T> -> deviceptr<'T> -> deviceptr<int> -> unit>
    }
//type IMergePathPartitions<'T> =
//    {
//        Action : ActionHint -> deviceptr<'T> -> deviceptr<'T> -> deviceptr<int> -> unit                
//    }


let mergePathPartitions (bounds:int) (compOp:IComp<'T>) = cuda {
    let NT = 64
    let bounds = if bounds = MgpuBoundsLower then 0 else 1
    let! kernelMergePartition = (kernelMergePartition NT bounds compOp) |> Compiler.DefineKernel //"kmp"

    return Entry(fun program ->
        let worker = program.Worker
        let kernelMergePartition = program.Apply kernelMergePartition

        fun (aCount:int) (bCount:int) (nv:int) (coop:int) ->
            let numPartitions = divup (aCount + bCount) nv
            let numPartitionBlocks = divup (numPartitions + 1) NT
            let lp = LaunchParam(numPartitionBlocks, NT)
            
            let run (a_global:deviceptr<'T>) (b_global:deviceptr<'T>) (partitionsDevice:deviceptr<int>) =
                kernelMergePartition.Launch lp a_global aCount b_global bCount nv coop partitionsDevice (numPartitions + 1)
            run
             ) }


let kernelSetPartition (NT:int) (duplicates:int) (compOp:IComp<'T>) =
    let duplicates = if duplicates = 1 then true else false

    let balancedPathSearch = (balancedPathSearch duplicates 0L compOp).DBalancedPath
    
    <@ fun (a_global:deviceptr<'T>) (aCount:int) (b_global:deviceptr<'T>) (bCount:int) (nv:int) (bp_global:deviceptr<int>) (numSearches:int) ->
        let balancedPathSearch = %balancedPathSearch
        let gid = NT * blockIdx.x + threadIdx.x
        if gid < numSearches then
            let diag = min (aCount + bCount) (gid * nv)
            let mutable bp = balancedPathSearch a_global aCount b_global bCount diag 4
            if bp.y <> 0 then bp.x <- bp.x ||| 0x80000000
            bp_global.[gid] <- bp.x
    @>


//type IFindSetPartitions<'T> =
//    {
//        Action : ActionHint -> deviceptr<'T> -> deviceptr<'T> -> deviceptr<int> -> unit
//    }


let findSetPartitions (duplicates:bool) (compOp:IComp<'T>) = cuda {
    let NT = 64
    let duplicates = if duplicates then 1 else 0
    let! kernelSetPartition = kernelSetPartition NT duplicates compOp |> Compiler.DefineKernel //"fsp"

    return Entry(fun program ->
        let worker = program.Worker
        let kernelSetPartition = program.Apply kernelSetPartition

        fun (aCount:int) (bCount:int) (nv:int) ->
            let numPartitions = divup (aCount + bCount) nv
            let numPartitionBlocks = divup (numPartitions + 1) NT
            let lp = LaunchParam(numPartitionBlocks, NT)

            let run (a_global:deviceptr<'T>) (b_global:deviceptr<'T>) (parts:deviceptr<int>) =
                kernelSetPartition.Launch lp a_global aCount b_global bCount nv parts (numPartitions + 1)
            run
             ) }


// MODERN GPU C++ CODE
//
//
//////////////////////////////////////////////////////////////////////////////////
//// MergePathPartitions
//
//template<int NT, int Bounds, typename It1, typename It2, typename Comp>
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
//template<int Bounds, typename It1, typename It2, typename Comp>
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