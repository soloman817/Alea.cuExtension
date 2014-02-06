[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Scan

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

open ScanOperators



type ReductionOpKind =
    | ReduceByKey
    | SegmentedOp

//type ReduceByKeyOp<'K,'V> =
//    abstract op : Expr<KeyValuePair<'K,'V> -> KeyValuePair<'K,'V> -> KeyValuePair<'K,'V>>

let reductionOp (kind:ReductionOpKind) (op:('V -> 'V -> 'V)) =
    match kind with
    | ReduceByKey ->
        fun (first:KeyValuePair<'K,'V>, second:KeyValuePair<'K,'V>) ->
            KeyValuePair<'K,'V>(second.Key,
                if second.Key <> first.Key then second.Value else (first.Value, second.Value) ||> op )
    | SegmentedOp ->
        fun (first:KeyValuePair<'K,'V>, second:KeyValuePair<'K,'V>) ->
            if second.Key > 0G then KeyValuePair<'K,'V>(first.Key + second.Key, second.Value)
            else KeyValuePair<'K,'V>(first.Key + second.Key, (first.Value, second.Value) ||> op)


type BlockScanAlgorithm =
    | BLOCK_SCAN_RAKING
    | BLOCK_SCAN_RAKING_MEMOIZE
    | BLOCK_SCAN_WARP_SCANS


let SAFE_ALGORITHM = 
    fun block_threads algorithm ->
        if (algorithm = BLOCK_SCAN_WARP_SCANS) && ((block_threads % CUB_PTX_WARP_THREADS) <> 0) then
            BLOCK_SCAN_RAKING
        else
            algorithm

let InternalBlockScan =
    fun block_threads algorithm ->
        match (block_threads, algorithm) ||> SAFE_ALGORITHM with
        | BLOCK_SCAN_WARP_SCANS -> () //BlockScanWarpScans()
        | _ -> () //BlockScanRaking()
