module Alea.cuExtension.CUB.Device.Region.Specializations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common

module HistoRegionGAtomic =
    let f() = "histo region gatomic"

module HistoRegionSAtomic =
    let f() = "histo region satomic"

module HistoRegionSort =
    let f() = "histo region sort"