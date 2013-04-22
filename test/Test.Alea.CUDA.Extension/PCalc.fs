module Test.Alea.CUDA.Extension.PCalc

open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

let worker = getDefaultWorker()

[<Test>]
let ``DArray.subView``() =
    let calc = pcalc {
        let input = [| 1; 2; 3; 4; 5; 6 |]
        let! input = DArray.scatterInBlob worker input
        let output1 = DArray.subView input 0 3
        let output2 = DArray.subView input 3 3
        let! output1 = output1.Gather()
        let! output2 = output2.Gather()
        return output1, output2 }
    let output1, output2 = calc |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    printfn "%A" output1
    printfn "%A" output2
    Assert.That(output1.Length = 3)
    Assert.That(output2.Length = 3)
    Assert.AreEqual(output1.[0], 1)
    Assert.AreEqual(output1.[1], 2)
    Assert.AreEqual(output1.[2], 3)
    Assert.AreEqual(output2.[0], 4)
    Assert.AreEqual(output2.[1], 5)
    Assert.AreEqual(output2.[2], 6)
