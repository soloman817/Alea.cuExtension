module Test.Alea.CUDA.Extension.Transformation

open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

let worker = getDefaultWorker()

[<Struct;Align(16)>]
    type Param =
        val rows : int
        val cols : int
        val time : float

        new (rows, cols, time) = { rows = rows; cols = cols; time = time }

/// All tests for PArray ??
/// init,initp,fill,fillp,filli,fillip,transform,transformp,transformi,transformip,transform2
//  transformp2,transformi2,transformip2,map,mapp,mapi,mapip,map2,mapp2,mapi2,mapip2,

//[<Test>] //init

//[<Test>] //initp

//[<Test>] //fill

//[<Test>] //fillp

//[<Test>] //filli

//[<Test>] //fillip

[<Test>]//transform
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

[<Test>]//transform
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

[<Test>]//transform
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

//[<Test>] //transformp

[<Test>] //transformi
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

//[<Test>] //transformip



[<Test>] //transform2
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

//[<Test>] //transformp2

//[<Test>] //transformi2

//[<Test>] //transformip2

[<Test>] //map
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

[<Test>] //mapp
let ``mapp: (param:'P) (x:float) -> log(x) + p``() =
    let mapp = worker.LoadPModule(PArray.mapp  <@ (fun (x:float) (p:float) -> log(x) + p ) @>).Invoke
    let test n eps = pcalc {
        let hInput = Array.init n (fun _ -> rng.NextDouble() )
        let hOutput p = hInput |> Array.map log |> Array.map (fun x -> x + p)
        let hOutput = hOutput 2.0
        let! dInput = DArray.scatterInBlob worker hInput
        let dOutput p = mapp p dInput
        let! dOutput = dOutput 2.0
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    
//[<Test>] //mapi

//[<Test>] //mapip

//[<Test>] //map2

//[<Test>] //mapp2

//[<Test>] //mapi2

//[<Test>] //mapip2
