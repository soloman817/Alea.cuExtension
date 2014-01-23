module Test.Alea.cuExtension.CUB.Device.Region

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

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

//module Specializations =
//    module HistoRegionGAtomic =
//        let f() = "histo region gatomic"
//
//    module HistoRegionSAtomic =
//     let f() = "histo region satomic"
//
//    module HistoRegionSort =
//        let f() = "histo region sort"