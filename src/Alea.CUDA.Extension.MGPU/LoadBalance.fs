module Alea.CUDA.Extension.MGPU.LoadBalance
// NOT IMPLEMENTED YET
open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTALoadBalance

type Plan =
    {
        NT : int
        VT : int
    }

let kernelLoadBalance (plan:Plan) =
    let NT = plan.NT
    let VT = plan.VT

    let ctaLoadBalance = ctaLoadBalance NT VT
    let deviceSharedToGlobal = deviceSharedToGlobal NT VT
    <@ fun (aCount:int) (b_global:DevicePtr<int>) (bCount:int) (mp_global:DevicePtr<int>) (indices_global:DevicePtr<int>) (mpCountingItr:DevicePtr<int>) ->
        let ctaLoadBalance = %ctaLoadBalance
        let deviceSharedToGlobal = %deviceSharedToGlobal

        let indices_shared = __shared__<int>(NT * (VT + 1)).Ptr(0)

        let mutable aCount = aCount

        let tid = threadIdx.x
        let block = blockIdx.x
        
        let range = ctaLoadBalance aCount b_global bCount block tid mp_global indices_shared false mpCountingItr
        aCount <- range.y - range.x

        deviceSharedToGlobal aCount indices_shared tid (indices_global + range.x) false
    @>


type ILoadBalanceSearch =
    {
        Action : ActionHint -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> unit
        NumPartitions : int
    }


let loadBalanceSearch() = cuda {
    let plan = { NT = 128; VT = 7 }
    let NV = plan.NT * plan.VT

    let! kernelLoadBalance = kernelLoadBalance plan |> defineKernelFuncWithName "lbs"
    let! mpp = Search.mergePathPartitions MgpuBoundsUpper (comp CompTypeLess 0)

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelLoadBalance = kernelLoadBalance.Apply m
        let mpp = mpp.Apply m

        fun (aCount:int) (bCount:int) ->
            let numBlocks = divup (aCount + bCount) NV
            let lp = LaunchParam(numBlocks, plan.NT)

            let action (hint:ActionHint) (b_global:DevicePtr<int>) (parts:DevicePtr<int>) (indices_global:DevicePtr<int>) (zeroItr:DevicePtr<int>) (mpCountingItr:DevicePtr<int>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
                    let mpp = mpp aCount bCount NV 0
                    let partitions = mpp.Action hint zeroItr b_global parts
                    kernelLoadBalance.Launch lp aCount b_global bCount parts indices_global mpCountingItr
                |> worker.Eval
            
            { Action = action; NumPartitions = numBlocks + 1 } ) }




//////////////////////////////////////////////////////////////////////////////////
//// KernelLoadBalance
//
//template<typename Tuning>
//MGPU_LAUNCH_BOUNDS void KernelLoadBalance(int aCount, const int* b_global,
//	int bCount, const int* mp_global, int* indices_global) {
//
//	typedef MGPU_LAUNCH_PARAMS Params;
//	const int NT = Params::NT;
//	const int VT = Params::VT;
//	__shared__ int indices_shared[NT * (VT + 1)];
//	
//	int tid = threadIdx.x;
//	int block = blockIdx.x;
//	int4 range = CTALoadBalance<NT, VT>(aCount, b_global, bCount, block, tid,
//		mp_global, indices_shared, false);
//	aCount = range.y - range.x;
//
//	DeviceSharedToGlobal<NT, VT>(aCount, indices_shared, tid, 
//		indices_global + range.x, false);
//}
//
//////////////////////////////////////////////////////////////////////////////////
//// LoadBalanceSearch
//
//MGPU_HOST void LoadBalanceSearch(int aCount, const int* b_global, int bCount,
//	int* indices_global, CudaContext& context) {
//
//	const int NT = 128;
//	const int VT = 7;
//	typedef LaunchBoxVT<NT, VT> Tuning;
//	int2 launch = Tuning::GetLaunchParams(context);
//	const int NV = launch.x * launch.y;
//	  
//	MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsUpper>(
//		mgpu::counting_iterator<int>(0), aCount, b_global, bCount, NV, 0,
//		mgpu::less<int>(), context);
//
//	int numBlocks = MGPU_DIV_UP(aCount + bCount, NV);
//	KernelLoadBalance<Tuning><<<numBlocks, launch.x, 0, context.Stream()>>>(
//		aCount, b_global, bCount, partitionsDevice->get(), indices_global);
//}
//
//} // namespace mgpu