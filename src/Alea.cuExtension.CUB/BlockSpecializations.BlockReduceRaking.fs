[<AutoOpen>]
module Alea.cuExtension.CUB.Block.BlockSpecializations.BlockReduceRaking

open Alea.CUDA
open Alea.CUDA.Utilities




//    let RAKING_THREADS =
//        fun block_threads ->
//            (block_threads, 1) ||> RakingLayout.RAKING_THREADS
//    
//    let SEGMENT_LENGTH =
//        fun block_threads ->
//            (block_threads, 1) ||> RakingLayout.SEGMENT_LENGTH
//
//    let WARP_SYNCHRONOUS =
//        fun block_threads ->
//            let RAKING_THREADS = block_threads |> RAKING_THREADS
//            (RAKING_THREADS = block_threads)
//
//    let WARP_SYNCHRONOUS_UNGUARDED =
//        fun block_threads ->
//            let RAKING_THREADS = block_threads |> RAKING_THREADS
//            ((RAKING_THREADS &&& (RAKING_THREADS - 1)) = 0)
//
//    let RAKING_UNGUARDED =
//        fun block_threads ->
//        (block_threads, 1) ||> RakingLayout.UNGUARDED


[<Record>]
type TempStorage<'T> =
    {
        warp_scan : deviceptr<'T>
        raking_grid : deviceptr<'T>
    }


[<Record>]
type ThreadFields<'T> =
    {
        temp_storage : deviceptr<'T>
        linear_tid : int
    }

    static member Init(temp_storage, linear_tid) =
        {
            temp_storage = temp_storage
            linear_tid = linear_tid
        }

//    
//    let rakingReduction (block_threads:int) = 
//        let RAKING_UNGUARDED = block_threads |> RAKING_UNGUARDED
//
//        fun (full_tile:bool) (iteration:int) ->
//            fun (reductionOp:'T -> 'T -> 'R) (raking_segment:deviceptr<'T>) (partial:'T) (num_valid:int) ->
//                
//                if (full_tile && RAKING_UNGUARDED) || ((linear_tid * SEGMENT_LENGTH) + iteration < num_valid) then
//                    let addend = raking_segment.[iteration]
//                    partial <- (partial, addend) ||> reduction_op


[<Record>]
type BlockReduceRaking<'T> =
    {
        BLOCK_THREADS : int
    }