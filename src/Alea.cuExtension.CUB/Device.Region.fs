module Alea.cuExtension.CUB.Device.Region

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common

module HistoRegion =
    let f() = "histo region"

module RadixSortDowsweepRegion =
    let f() = "radix sort downsweep region"

module RadixSortUpsweepRegion =
    let f() = "radix sort upsweep region"

module ReduceRegion =
    let f() = "reduce region"

module ScanRegion =
    let f() = "scan region"

module SelectRegion =
    let f() = "select region"
    
module ScanTypes =
    let f() = "scan types"