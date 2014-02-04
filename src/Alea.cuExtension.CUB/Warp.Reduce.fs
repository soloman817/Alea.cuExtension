module Alea.cuExtension.CUB.Warp.Reduce

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Common

let privateStorage() = __null()

let POW_OF_TWO =
    fun logical_warp_threads ->
        ((logical_warp_threads &&& (logical_warp_threads - 1)) = 0)


[<Record>]
type ThreadFields<'T> =
    {
        temp_storage : deviceptr<'T>
        warp_id : int
        lane_id : int
    }

    static member Init(temp_storage, warp_id, lane_id) =
        { 
            temp_storage = temp_storage
            warp_id = warp_id
            lane_id = lane_id
        }


//[<Record>]
//type WarpReduce<'T> =
//    {
//        ThreadFields : ThreadFields<'T>
//        LOGICAL_WARPS : int
//        LOGICAL_WARP_THREADS : int
//    }
//
//    static member Init(threadFields, logical_warps, logical_warp_threads) =
//        {
//            ThreadFields = threadFields
//            LOGICAL_WARPS = logical_warps
//            LOGICAL_WARP_THREADS = logical_warp_threads
//        }
//
//
//    static member Init(logical_warps, logical_warp_threads) =
//        let threadFields = ThreadFields.Init(
//            privateStorage(),
//            (if logical_warps = 1 then 0 else threadIdx.x / logical_warp_threads),
//            (if (logical_warps = 1) || (logical_warp_threads = CUB_PTX_WARP_THREADS then LaneId() else threadIdx.x % LOGICAL_WARP_THREADS
//                )))
//        {
//            ThreadFields = ThreadFields.Init(
//                privateStorage(),
//                )
//        }