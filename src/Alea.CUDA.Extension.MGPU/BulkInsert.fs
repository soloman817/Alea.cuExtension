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
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.CTAMerge


type Plan =
    {
        NT : int
        VT: int
    }

let kernelBulkInsert (plan:Plan) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    
    let capacity, S = ctaScan2 NT (scanOp ScanOpTypeAdd 0)
    let sharedSize = max NV capacity
 
    let deviceGlobalToReg = deviceGlobalToReg NT VT
    let computeMergeRange = computeMergeRange().Device
    let deviceTransferMergeValues = deviceTransferMergeValuesA NT VT

    <@ fun (a_global:DevicePtr<'T>) (indices_global:DevicePtr<int>) (aCount:int) (b_global:DevicePtr<'T>) (bCount:int) (mp_global:DevicePtr<int>) (dest_global:DevicePtr<'T>) ->
        let deviceGlobalToReg = %deviceGlobalToReg
        let computeMergeRange = %computeMergeRange
        let deviceTransferMergeValues = %deviceTransferMergeValues
        let S = %S
        

        let shared = __shared__<int>(sharedSize).Ptr(0)
        let sharedScan = shared
        let sharedIndices = shared

        let tid = threadIdx.x
        let block = blockIdx.x
        
        let mutable range = computeMergeRange aCount bCount block 0 NV mp_global
        let mutable aCount = aCount
        let mutable bCount = bCount
        let a0 = range.x
        let a1 = range.y
        let b0 = range.z
        let b1 = range.w
        aCount <- a1 - a0
        bCount <- b1 - b0

        for i = 0 to VT - 1 do
            sharedIndices.[NT * i + tid] <- 0
        __syncthreads()

        let indices = __local__<int>(VT).Ptr(0)
        deviceGlobalToReg aCount (indices_global + a0) tid indices true
        
        for i = 0 to VT - 1 do
            let index = NT * i + tid
            if index < aCount then
                sharedIndices.[index + indices.[i] - b0] <- 1
        __syncthreads()

        let mutable x = 0
        for i = 0 to VT - 1 do
            indices.[i] <- sharedIndices.[VT * tid + i]
            x <- x + indices.[i]
        __syncthreads()

        let mutable scan = S tid x sharedScan

        for i = 0 to VT - 1 do
            let index = VT * tid + i
            let gather = if indices.[i] <> 0 then scan else aCount + index - scan
            scan <- scan + 1
            sharedIndices.[index] <- gather
        __syncthreads()

        deviceTransferMergeValues (aCount + bCount) (a_global + a0) (b_global + b0) aCount sharedIndices tid (dest_global + a0 + b0) false 
        @>



type IBulkInsert<'T> =
    {
        Action : ActionHint -> DevicePtr<'T> -> DevicePtr<int> -> int -> DevicePtr<'T> -> int -> DevicePtr<int> -> DevicePtr<'T> -> unit
        NumPartitions : int
    }



let bulkInsert()  = cuda {
    let plan = { NT = 128; VT = 7 }
    let NV = plan.NT * plan.VT

    let! kernelBulkInsert = kernelBulkInsert plan |> defineKernelFuncWithName "bi"
    let! mpp = Search.mergePathPartitions MgpuBoundsLower (comp CompTypeLess 0)

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelBulkInsert = kernelBulkInsert.Apply m
        let mpp = mpp.Apply m
        
        fun (aCount:int) (bCount:int) ->
            let numBlocks = divup (aCount + bCount) NV
            let numPartitions = numBlocks
            let lp = LaunchParam(numBlocks, plan.NT)
                        
            let action (hint:ActionHint) (a_global:DevicePtr<'T>) (indices_global:DevicePtr<int>) (aCount:int) (b_global:DevicePtr<'T>) (bCount:int) (partitions:DevicePtr<int>) (dest_global:DevicePtr<'T>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
                    let mpp = mpp aCount bCount NV
                    let partitionsDevice = mpp.Action hint indices_global (DevicePtr<int>(0L)) 0 partitions
                                
                    kernelBulkInsert.Launch lp a_global indices_global aCount b_global bCount partitions dest_global
                |> worker.Eval
            
            { Action = action; NumPartitions = numBlocks + 1 } ) }



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
