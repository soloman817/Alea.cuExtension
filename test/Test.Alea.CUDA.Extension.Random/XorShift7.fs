module Test.Alea.CUDA.Extension.Random.XorShift7

open System.IO
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Random
open Test.Alea.CUDA.Extension

let worker = getDefaultWorker()

[<Test>]
let simple() =
    let streams = 114688
    let steps = 5
    let runs = 1
    let rank = 0
    let seed = 42u

    let xorshift7 = worker.LoadPModule(PRandom.xorshift7 <@ Util.identity @>).Invoke

    let calc = pcalc {
        let! rn = xorshift7 streams steps seed runs rank
        return! rn.Numbers.Gather() }

    let rn = calc |> PCalc.run
    for i = 0 to 10 do
        printf "%d " rn.[i]
    printfn ""
