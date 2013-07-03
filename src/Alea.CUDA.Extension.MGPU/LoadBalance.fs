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
open Alea.CUDA.Extension.MGPU.CTAScan





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