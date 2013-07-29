module Test.Alea.CUDA.Extension.Finance.LiborMarketModel

open System
open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Finance.LiborMarketModel
open Test.Alea.CUDA.Extension

[<Test>]
let ``LiborMarketModel`` () =

    let N = 80
    let nMat = 40
    let nOpt = 15
    let nPath = 96000

    let worker = getDefaultWorker()
    let liborMarketModelPricer = worker.LoadPModule(liborMarketModel N nMat nOpt).Invoke

    let delta = 0.25
    let maturities = [| 4; 4; 4; 8; 8; 8; 20; 20; 20; 28; 28; 28; 40; 40; 40 |]
    let swaprates = [| 0.045; 0.05; 0.055; 0.045; 0.05; 0.055; 0.045; 0.05; 0.055; 0.045; 0.05; 0.055; 0.045; 0.05; 0.055 |]

    let v1 = [| 1; 2; 3|]
    let v2 = [| 1; 2; 3|]


    ()

//    let pricer = pcalc {
//        let! s;  v;  u = eulerSolver heston optionType strike timeToMaturity param
//        return! u.Gather()
//    }

//    let result = pricer |> PCalc.runWithKernelTiming(10)
//    let result;  loggers = pricer |> PCalc.runWithTimingLogger
//    loggers.["default"].DumpLogs()
//    printfn "%A" result

