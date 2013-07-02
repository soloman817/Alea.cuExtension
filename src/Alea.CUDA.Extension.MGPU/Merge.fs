module Alea.CUDA.Extension.MGPU.Merge
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