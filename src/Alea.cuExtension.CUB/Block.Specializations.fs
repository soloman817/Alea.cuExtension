module Alea.cuExtension.CUB.Block.Specializations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

module HistogramAtomic =
    let f() = "histogram atomic"

module HistogramSort =
    let f() = "histogram sort"

module ReduceRaking =

    let RAKING_THREADS =
        fun block_threads ->
            (block_threads, 1) ||> RakingLayout.RAKING_THREADS
    
    let SEGMENT_LENGTH =
        fun block_threads ->
            (block_threads, 1) ||> RakingLayout.SEGMENT_LENGTH

    let WARP_SYNCHRONOUS =
        fun block_threads ->
            let RAKING_THREADS = block_threads |> RAKING_THREADS
            (RAKING_THREADS = block_threads)

    let WARP_SYNCHRONOUS_UNGUARDED =
        fun block_threads ->
            let RAKING_THREADS = block_threads |> RAKING_THREADS
            ((RAKING_THREADS &&& (RAKING_THREADS - 1)) = 0)

    let RAKING_UNGUARDED =
        fun block_threads ->
        (block_threads, 1) ||> RakingLayout.UNGUARDED


    [<Record>]
    type TempStorage<'T> =
        {
            warp_storage : deviceptr<'T>
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

module ReduceWarpReduction =
    let f() = "reduce warp reduction"

module ScanRaking =
    open RakingLayout

    let BlockRakingLayout = 
        fun block_threads ->
            block_threads |> RakingLayout.Constants.Init

    let WARPS =
        fun block_threads ->
            (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS

    let RAKING_THREADS = 
        RakingLayout.RAKING_THREADS

    let SEGMENT_LENGTH = 
        RakingLayout.SEGMENT_LENGTH

    let WARP_SYNCHRONOUS = 
        fun block_threads raking_threads ->
            block_threads = raking_threads

    let WarpScan = 


    [<Record>]
    type TempStorage<'T> =
        {
            warp_scan : deviceptr<'T>
            raking_grid : deviceptr<'T>
            block_aggregate : 'T
        }

        static member Create(warp_scan, raking_grid, block_aggregate) =
            {
                warp_scan = warp_scan
                raking_grid = raking_grid
                block_aggregate = block_aggregate
            }      
   
    
    let guardedReduce =
        fun block_threads ->
            fun unguarded segment_length ->
                fun linear_tid ->
                    fun warps raking_threads segment_length warp_synchronous ->
                        fun iteration ->
                            fun (raking_ptr:deviceptr<'T>) (scan_op:IScanOp<'T>) (raking_partial:'T) ->
                                let mutable raking_partial = raking_partial
                                if unguarded || (((linear_tid * segment_length) + iteration) < block_threads) then
                                    let addend = raking_ptr.[iteration]
                                    raking_partial <- (raking_partial, addend) ||> scan_op.Sum

    let upsweep =
        fun (scan_op:IScanOp<'T>) ->
            let smem_raking_ptr = 
        

    let blockScanRaking block_threads memoize =
        let BlockRakingLayout = block_threads |> BlockRakingLayout.Init
        ()
    



    
    //let guardedReduce (iteration:int) (scanOp:)


    [<Record>]
    type ThreadFields<'T> =
        {
            temp_storage    : TempStorage<'T>
            linear_tid      : int
            cached_segment  : deviceptr<'T>
        }

        static member Create(temp_storage, linear_tid) =
            {
                temp_storage = temp_storage
                linear_tid = linear_tid
                cached_segment = __null()
            }


    [<Record>]
    type BlockScanRaking =
        {
            BLOCK_THREADS : int
            MEMOIZE : bool
        }





module ScanWarpScans =

    let WARPS =
        fun block_threads ->
            (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS    

    [<Record>]
    type TempStorage<'T> =
        {
            warp_scan : deviceptr<'T>
            warp_aggregates : deviceptr<'T>
            block_prefix : 'T
        }

        static member Create(warp_scan, warp_aggregates, block_prefix) =
            {
                warp_scan = warp_scan
                warp_aggregates = warp_aggregates
                block_prefix = block_prefix
            }



    [<Record>]
    type ThreadFields<'T> =
        {
            temp_storage : TempStorage<'T>
            linear_tid : int
            warp_id : int
            lane_id : int
        }

        static member Create(temp_storage, linear_tid, warp_id, lane_id) =
            {
                temp_storage = temp_storage
                linear_tid = linear_tid
                warp_id = warp_id
                lane_id = lane_id
            }


    let applyWarpAggregates block_threads = 
        let WARPS = block_threads |> WARPS
        fun (partial:Ref<'T>) (scan_op:IScanOp<'T>) (warp_aggregate:'T) (block_aggregate:Ref<'T>) (lane_valid:bool option) ->
            let lane_valid = if lane_valid.IsSome then lane_valid.Value else true
            fun temp_storage warp_id ->
                temp_storage.warp_aggregates.[warp_id] <- warp_aggregate

                __syncthreads()

                block_aggregate := temp_storage.warp_aggregates.[0]

                for WARP = 1 to WARPS - 1 do
                    if warp_id = WARP then
                        partial := if lane_valid then (!block_aggregate, !partial) ||> scan_op.Sum else !block_aggregate
                    block_aggregate := (!block_aggregate, temp_storage.warp_aggregates.[WARP]) ||> scan_op.Sum

     
    [<Record>]
    type BlockScanWarpScans<'T> =
        {
            BLOCK_THREADS : int
            ThreadFields : ThreadFields<'T>
        }

//        static member Create(temp_storage, linear_tid) =
//            {
//                ThreadFields = ThreadFields.Create(
//                                                    temp_storage,
//                                                    linear_tid,
//                                                    )
//            }