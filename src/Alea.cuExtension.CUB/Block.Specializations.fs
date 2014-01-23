module Alea.cuExtension.CUB.Block.Specializations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common

module HistogramAtomic =
    let f() = "histogram atomic"

module HistogramSort =
    let f() = "histogram sort"

module ReduceRanking =
    let f() = "reduce ranking"

module ReduceWarpReduction =
    let f() = "reduce warp reduction"

module ScanRanking =
    let f() = "scan ranking"

module ScanWarpScans =
    let f() = "scan warp scans"