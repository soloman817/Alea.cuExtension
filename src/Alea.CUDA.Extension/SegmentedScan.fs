module Alea.CUDA.Extension.SegmentedScan

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.Timing
open Alea.CUDA.Extension.Reduce

type ISegmentedScanFlags<'T> =
    abstract Scan : int * DevicePtr<int> * DevicePtr<'T> -> DevicePtr<'T>
    abstract Scan : int[] * 'T[] -> 'T[] 

//module Sum =
//
///// Reduction function for upsweep pass. This performs addition for code 0 and max for code 1.
//let [<ReflectedDefinition>] inline reduce (init: unit -> 'T) (op:'T -> 'T -> 'T) numWarps logNumWarps tid (x:'T) =
//    let warp = tid / WARP_SIZE
//    let lane = tid &&& (WARP_SIZE - 1)
//    let warpStride = WARP_SIZE + WARP_SIZE / 2
//    let sharedSize = numWarps * warpStride
//    
//    let shared = __shared__<'T>(sharedSize).Ptr(0)
//    let shared = __shared__<'T>(sharedSize).Ptr(0)
//    
//    let warpShared = (shared + warp * warpStride).Volatile()      
//    let s = warpShared + (lane + WARP_SIZE / 2)
//
//    warpShared.[lane] <- init()  
//    s.[0] <- x
//
//    // Run inclusive scan on each warp's data.
//    let mutable warpScan = x
//    for i = 0 to LOG_WARP_SIZE - 1 do
//        let offset = 1 <<< i
//        warpScan <- op warpScan s.[-offset]   
//        if i < LOG_WARP_SIZE - 1 then s.[0] <- warpScan
        

(*


template<int NumWarps>
DEVICE2 int Reduce(uint tid, int x, int code) {

	const int LogNumWarps = LOG_BASE_2(NumWarps);

	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	int init = code ? -1 : 0;

	const int ScanStride = WARP_SIZE + WARP_SIZE / 2 + 1;
	const int ScanSize = NumWarps * ScanStride;
	__shared__ volatile int reduction_shared[ScanSize];
	__shared__ volatile int totals_shared[2 * WARP_SIZE];

	volatile int* s = reduction_shared + ScanStride * warp + lane +
		WARP_SIZE / 2;
	s[-(WARP_SIZE / 2)] = init;
	s[0] = x;

	// Run intra-warp max reduction.
	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		uint offset = 1<< i;
		if(0 == code) x += s[-offset];
		else if(1 == code) x = max(x, s[-offset]);
		s[0] = x;
	}

	// Synchronize to make all the totals available to the reduction code.
	__syncthreads();
	if(tid < NumWarps) {
		// Grab the block total for the tid'th block. This is the last element
		// in the block's scanned sequence. This operation avoids bank 
		// conflicts.
		x = reduction_shared[ScanStride * tid + WARP_SIZE / 2 + WARP_SIZE - 1];

		volatile int* s = totals_shared + NumWarps / 2 + tid;
		s[-(NumWarps / 2)] = init;
		s[0] = x;

		#pragma unroll
		for(int i = 0; i < LogNumWarps; ++i) {
			int offset = 1<< i;
			if(0 == code) x += s[-offset];
			else if(1 == code) x = max(x, s[-offset]);
			if(i < LogNumWarps - 1) s[0] = x;
		}
		totals_shared[tid] = x;
	}

	// Synchronize to make the block scan available to all warps.
	__syncthreads();

	return totals_shared[NumWarps - 1];
}


*)
