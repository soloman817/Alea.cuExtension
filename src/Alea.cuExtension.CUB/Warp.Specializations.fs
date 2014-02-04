module Alea.cuExtension.CUB.Warp.Specializations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Common

    
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
                (input, src_lane, logical_warp_threads) |||> ShuffleBroadcast

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

module ScanSmem =
    let f() = "scan smem"