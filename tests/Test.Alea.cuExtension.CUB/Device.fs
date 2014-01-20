module Test.Alea.cuExtension.CUB.Device

module Histogram =
    let f() = "histogram"
module Partition =
    let f() = "partition"
module RadixSort =
    let f() = "radix sort"
module Reduce =
    let f() = "reduce"
module ReduceByKey =
    let f() = "reduce by key"
module Scan =
    let f() = "scan"


module Region =
    
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

    
    
    module Specializations =

        module HistoRegionGAtomic =
            let f() = "histo region gatomic"

        module HistoRegionSAtomic =
            let f() = "histo region satomic"

        module HistoRegionSort =
            let f() = "histo region sort"