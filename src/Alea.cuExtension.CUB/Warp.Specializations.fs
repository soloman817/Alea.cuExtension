module Alea.cuExtension.CUB.Warp.Specializations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Common

open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread
    
module ReduceShfl =
    let f() = "reduce shfl"

module ReduceSmem =
    let f() = "reduce smem"

module ScanShfl =
    
    let STEPS = 
        fun logical_warp_threads ->
            logical_warp_threads |> log2

    let SHFL_C =
        fun logical_warp_threads ->
            let STEPS = logical_warp_threads |> STEPS
            ((-1 <<< STEPS) &&& 31) <<< 8

    let broadCast =
        fun logical_warp_threads ->
            fun input src_lane ->
                ()//(input, src_lane, logical_warp_threads) |||> ShuffleBroadcast

    [<Record>]
    type ThreadFields<'T> =
        {
            warp_id : int
            lane_id : int
        }

        static member Create(warp_id, lane_id) =
            {
                warp_id = warp_id
                lane_id = lane_id
            }


    [<Record>]
    type WarpScanShfl =
        {
            LOGICAL_WARPS : int
            LOGICAL_WARP_THREADS : int
        }

        static member Create(logical_warps, logical_warp_threads) =
            {
                LOGICAL_WARPS = logical_warps
                LOGICAL_WARP_THREADS = logical_warp_threads
            }

module ScanSmem =
    
    /// The number of warp scan steps
    let STEPS =
        fun logical_warp_threads ->
            logical_warp_threads |> log2

    /// The number of threads in half a warp
    let HALF_WARP_THREADS =
        fun logical_warp_threads ->
            let STEPS = logical_warp_threads |> STEPS
            1 <<< (STEPS - 1)

    /// The number of shared memory elements per warp
    let WARP_SMEM_ELEMENTS =
        fun logical_warp_threads ->
            logical_warp_threads + (logical_warp_threads |> HALF_WARP_THREADS)

    let inline _TempStorage<'T>() =
        fun logical_warps warp_smem_elements ->
            __shared__.Array2D(logical_warps, warp_smem_elements)

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

    let initIdentity (has_identity:bool) =
        fun warp_id lane_id ->
            match has_identity with
            | true ->
                let identity = ZeroInitialize<'T>()
                (warp_id, lane_id) ||> _TempStorage<'T>() |> __array_to_ptr
                    |> ((warp_id * lane_id) |> ThreadStore<'T>.DefaultStore)