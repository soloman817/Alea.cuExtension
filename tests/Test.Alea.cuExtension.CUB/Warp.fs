module Test.Alea.cuExtension.CUB.Warp

module Reduce =
    let f() = "reduce"

module Scan =
    let f() = "scan"


module Specializations =
    
    module ReduceShfl =
        let f() = "reduce shfl"

    module ReduceSmem =
        let f() = "reduce smem"

    module ScanShfl =
        let f() = "scan shfl"

    module ScanSmem =
        let f() = "scan smem"