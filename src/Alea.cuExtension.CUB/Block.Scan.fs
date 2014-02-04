[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Scan

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common


type BlockScanAlgorithm =
    | BLOCK_SCAN_RAKING
    | BLOCK_SCAN_RAKING_MEMOIZE
    | BLOCK_SCAN_WARP_SCANS

type IScanOp<'T> =
    abstract Sum : ('T -> 'T -> 'T)

type ReductionOp = Expr

[<Record>]
type ReduceByKeyOp =
    {
        op : ReductionOp
    }

