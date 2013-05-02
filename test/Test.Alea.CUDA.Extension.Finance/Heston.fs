module Test.Alea.CUDA.Extension.Finance.Heston

open System
open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Finance.Heston
open Alea.CUDA.Extension.Finance.Grid
open Test.Alea.CUDA.Extension

    
[<Test>]
let ``finite difference weights`` () =
      
    let worker = getDefaultWorker()

    let s = concentratedGrid 0.0 250.0 50.0 100 10.0
    let finiteDifferenceWeights = worker.LoadPModule(finiteDifferenceWeights).Invoke

    //let fdWeights = worker.LoadPModule(fdWeights).Invoke
    let fd = pcalc {
        let! s = DArray.scatterInBlob worker s
        let! sDiff = finiteDifferenceWeights s.Length s.Ptr  
        let a = sDiff.Alpha0
        return sDiff
    } 

    let fd = fd |> PCalc.run


    Assert.IsTrue(true)

