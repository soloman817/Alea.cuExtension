module Test.Alea.cuExtension.CUB.Block.Exchange

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

open Alea.cuExtension.CUB.Block.Exchange


[<Test>]
let ``block exchange test`` () =
    let x = 0
    ()