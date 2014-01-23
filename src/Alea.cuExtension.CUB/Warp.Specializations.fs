module Alea.cuExtension.CUB.Warp.Specializations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Common

    
module ReduceShfl =
    let f() = "reduce shfl"

module ReduceSmem =
    let f() = "reduce smem"

module ScanShfl =
    let f() = "scan shfl"

module ScanSmem =
    let f() = "scan smem"