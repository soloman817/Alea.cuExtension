module Alea.cuExtension.CUB.Warp.Scan

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Common
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

open Specializations.ScanShfl




let POW_OF_TWO =
    fun logical_warp_threads ->
        ((logical_warp_threads &&& (logical_warp_threads - 1)) = 0)

let InternalWarpScan =
    fun logical_warps logical_warp_threads ->
        if (CUB_PTX_VERSION >= 300) && ((logical_warps = 1) || (logical_warps |> POW_OF_TWO)) then
            (logical_warps, logical_warp_threads) |> WarpScanShfl.Create
        else
            (logical_warps, logical_warp_threads) |> WarpScanShfl.Create //WarpScanSmem

let privateStorage() = __shared__.Extern<'T>() 

let warp_id =
    fun logical_warps logical_warp_threads ->
        if logical_warps = 1 then 0 else threadIdx.x / logical_warp_threads

let lane_id =
    fun logical_warps logical_warp_threads ->
        if ((logical_warps = 1) || (logical_warp_threads = CUB_PTX_WARP_THREADS)) then __ptx__.LaneId() else threadIdx.x % logical_warp_threads



[<Record>]
type TempStorage<'T> =
    {
        temp_storage : deviceptr<'T>
    }

    static member Create() = 
        {
            temp_storage = privateStorage()
        }


[<Record>]
type ThreadFields<'T> =
    {
        temp_storage : TempStorage<'T>
        warp_id : int
        lane_id : int
    }

    static member Create(temp_storage, warp_id, lane_id) =
        {
            temp_storage = temp_storage
            warp_id = warp_id
            lane_id = lane_id
        }

    static member Create(logical_warps, logical_warp_threads) =
        {
            temp_storage = TempStorage<'T>.Create()
            warp_id = (logical_warps, logical_warp_threads) ||> warp_id
            lane_id = (logical_warps, logical_warp_threads) ||> lane_id
        }


//let warpScan logical_warps logical_warp_threads =
//    fun 


[<Record>]
type WarpScan<'T> =
    {
        LOGICAL_WARPS : int
        LOGICAL_WARP_THREADS : int
        ThreadFields : ThreadFields<'T>
    }

    static member Create(logical_warps, logical_warp_threads, threadFields) =
        {
            LOGICAL_WARPS = logical_warps
            LOGICAL_WARP_THREADS = logical_warp_threads
            ThreadFields = threadFields
        }

//    static member Create() =
//        {
//            LOGICAL_WARPS = 1
//            LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS
//            ThreadFields = ThreadFields.Create( privateStorage(),
//                                                
//        }