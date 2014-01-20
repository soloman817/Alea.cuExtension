module Test.Alea.cuExtension.CUB.Block

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB
open NUnit.Framework

module Discontinuity =
    let f() = "discontinuity"
module Exchange =
    let f() = "exchange"
module Histogram =
    let f() = "histogram"

module Load =
    open Block.Load

    [<Test>]
    let ``CUB.Block.LoadDirectBlocked`` () = 
        //let items_per_thread = 100

        let inline test (loadDirectBlocked:LoadDirectBlocked<'T>) = cuda {
            let! load = loadDirectBlocked.Invoke

            let! kernel =
                <@ fun (ipt:int) ->
                    let tid = threadIdx.x
                    let temp_storage = __shared__.Array(128)
                    load.Invoke ipt @>
                |> Compiler.DefineKernel

            return Entry(fun program ->
                let worker = program.Worker
                let kernel = program.Apply kernel
        
                let run (a:'T[]) (b:'T[]) =
                    use a = worker.Malloc(a)
                    use b = worker.Malloc(a.Length)
                    let lp = LaunchParam(256, 1)
                    kernel.Launch lp a.Ptr b.Ptr
                run )}
        ()

module RadixRank =
    let f() = "radix rank"
module RadixSort =
    let f() = "radix sort"
module RakingLayout =
    let f() = "raking layout"
module Reduce =
    let f() = "reduce"
module Scan =
    let f() = "scan"
module Store =
    let f() = "store"


module Specializations =

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