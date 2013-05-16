module Test.Alea.CUDA.Extension.TransformPArray

open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

let worker = getDefaultWorker()

/// All tests for PArray ??
/// init,initp,fill,fillp,filli,fillip,transform,transformp,transformi,transformip,transform2
//  transformp2,transformi2,transformip2,map,mapp,mapi,mapip,map2,mapp2,mapi2,mapip2,


[<Test>]
let ``transform: (x:float) -> log x``() =
    let transform = worker.LoadPModule(PArray.transform <@ log @>).Invoke
    let test n eps = pcalc {
        let hInput = Array.init n (fun _ -> rng.NextDouble())
        let hOutput = hInput |> Array.map log
        let! dInput = DArray.scatterInBlob worker hInput
        let! dOutput = DArray.createInBlob worker n
        do! transform dInput dOutput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>]
let ``transform: (x:float32) -> log x``() =
    let transform = worker.LoadPModule(PArray.transform <@ log @>).Invoke
    let test n eps = pcalc {
        let hInput = Array.init n (fun _ -> rng.NextDouble() |> float32)
        let hOutput = hInput |> Array.map log
        let! dInput = DArray.scatterInBlob worker hInput
        let! dOutput = DArray.createInBlob worker n
        do! transform dInput dOutput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-5 |> PCalc.run
    test (1<<<22) 1e-5 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-5 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>]
let ``transform: (x:float) -> float32(log x)``() =
    let transform = worker.LoadPModule(PArray.transform <@ fun x -> float32(log x) @>).Invoke
    let test n eps = pcalc {
        let hInput = Array.init n (fun _ -> rng.NextDouble())
        let hOutput = hInput |> Array.map log |> Array.map float32
        let! dInput = DArray.scatterInBlob worker hInput
        let! dOutput = DArray.createInBlob worker n
        do! transform dInput dOutput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-5 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>]
let ``transformi: int sequence``() =
    let transformi = worker.LoadPModule(PArray.transformi <@ fun i _ -> i @>).Invoke
    let test n = pcalc {
        let hOutput = Array.init n (fun i -> i)
        let! dOutput = DArray.createInBlob worker n
        do! transformi dOutput dOutput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d)) }

    test (1<<<22) |> PCalc.run
    test (1<<<22) |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>]
let ``transform2: (x:float) (y:float32) -> x + float(y)``() =
    let transform2 = worker.LoadPModule(PArray.transform2 <@ fun x y -> x + float(y) @>).Invoke
    let test n eps = pcalc {
        let hInput1 = Array.init n (fun _ -> rng.NextDouble())
        let hInput2 = Array.init n (fun _ -> rng.NextDouble() |> float32)
        let hOutput = (hInput1, hInput2) ||> Array.map2 (fun x y -> x + float(y))
        let! dInput1 = DArray.scatterInBlob worker hInput1
        let! dInput2 = DArray.scatterInBlob worker hInput2
        let! dOutput = DArray.createInBlob worker n
        do! transform2 dInput1 dInput2 dOutput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>]
let ``map: (x:float) -> log x``() =
    let map = worker.LoadPModule(PArray.map <@ log @>).Invoke
    let test n eps = pcalc {
        let hInput = Array.init n (fun _ -> rng.NextDouble())
        let hOutput = hInput |> Array.map log
        let! dInput = DArray.scatterInBlob worker hInput
        let! dOutput = dInput |> map
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
        

