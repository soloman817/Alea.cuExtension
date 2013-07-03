module Test.Alea.CUDA.Extension.Transformation

open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

let worker = getDefaultWorker()

[<Struct>]
    type Param =
        val x : float
        val y : float
        val time : float

        new (x, y, time) = { x = x; y = y; time = time }

/// All tests for PArray ??
/// init,initp,fill,fillp,filli,fillip,transform,transformp,transformi,transformip,transform2
//  transformp2,transformi2,transformip2,map,mapp,mapi,mapip,map2,mapp2,mapi2,mapip2,

[<Test>] //init
let ``init: (i:int seq) (n:int) ``() =
    let init = worker.LoadPModule(PArray.init <@ fun i -> i @>).Invoke
    let test n eps = pcalc {
        let hInput = Array.init n (fun i -> i)
        let hOutput = hInput
        let! dOutput = init n
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>] //initp
let ``initp: (i:int seq) (p:Param) (n:int) ``() =
    let initp = worker.LoadPModule(PArray.initp <@ fun i (p:Param) -> p.x @>).Invoke
    let test n eps = pcalc {
        let par = new Param(2.0,2.0,2.0)
        let hInput (p:Param) = Array.init n (fun i -> p.x)
        let hOutput = hInput par
        let! dOutput = initp par n
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()


[<Test>] //fill
let ``fill: (x:float) -> x``() =
    let fill = worker.LoadPModule(PArray.fill <@ 0.0 @>).Invoke
    let test n eps = pcalc {
        let hInput = Array.init n (fun _ -> rng.NextDouble())
        Array.fill hInput 0 n 0.0        
        let hOutput = hInput
        let! dInput = DArray.scatterInBlob worker hInput
        let! dOutput = fill dInput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>] //fillp
let ``fillp: (p:Param) -> p.x*p.y``() =
    let fillp = worker.LoadPModule(PArray.fillp <@ fun (p:Param) -> p.x*p.y @>).Invoke
    let test n eps = pcalc {
        let par = new Param(3.0, 7.0, 1.0)
        let hInput = Array.init n (fun _ -> rng.NextDouble())
        let hOutput (p:Param) (hIn:float[]) = 
            Array.fill hIn 0 n (p.x*p.y)
            hIn
        let hOutput = hOutput par hInput
        let! dInput = DArray.scatterInBlob worker hInput
        let! dOutput = fillp par dInput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>] //filli
let ``filli: (i:seq int) -> i``() =
    let filli = worker.LoadPModule(PArray.filli <@ fun i -> i @>).Invoke
    let test n eps = pcalc {
        let hInput = Array.init n (fun i -> i)
        let hOutput = hInput
        let! dInput = DArray.scatterInBlob worker hInput
        let! dOutput = filli dInput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>] //fillip
let ``fillip: (i:seq int) (p:Param) -> float(i)*p.x``() =
    let fillip = worker.LoadPModule(PArray.fillip <@ fun i (p:Param) -> float(i)*p.x @>).Invoke
    let test n eps = pcalc {
        let par = new Param(2.0,1.0,1.0)
        let hInput = Array.init n (fun i -> i)
        let hOutput (p:Param) = hInput |> Array.map (fun i -> float(i)*p.x)
        let hOutput = hOutput par
        let! dInput = DArray.scatterInBlob worker hOutput
        let! dOutput = fillip par dInput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

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

[<Test>] //transformp
let ``transformp: (p:Param) (x:float) -> p.x * log(x)``() =
    let transformp = worker.LoadPModule(PArray.transformp <@ fun (p:Param) x -> p.x*log(x) @>).Invoke
    let test n eps = pcalc {
        let par = new Param(2.0, 1.0, 1.0)
        let hInput = Array.init n (fun _ -> rng.NextDouble())
        let hOutput (p:Param) = hInput |> Array.map (fun x -> p.x*log(x))
        let hOutput = hOutput par
        let! dInput = DArray.scatterInBlob worker hInput
        let! dOutput = DArray.createInBlob worker n
        do! transformp par dInput dOutput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

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

[<Test>] //transformip
let ``transformip: (i:seq int) (p:Param) (x:float) -> p.x*(log(x) + float(i))``() =
    let transformip = worker.LoadPModule(PArray.transformip <@ fun i (p:Param) x -> p.x*(log(x) + float(i)) @>).Invoke
    let test n eps = pcalc {
        let par = new Param(2.0, 1.0, 1.0)
        let hInput = Array.init n (fun _ -> rng.NextDouble())
        let hOutput (p:Param) = hInput |> Array.mapi (fun i x -> p.x*(log(x) + float(i)))
        let hOutput = hOutput par
        let! dInput = DArray.scatterInBlob worker hInput
        let! dOutput = DArray.createInBlob worker n
        do! transformip par dInput dOutput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

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

[<Test>] //transformp2
let ``transformp2: (p:Param) (x:float) (y:float) -> (x*p.x + y*p.y)*p.time``() =
    let transformp2 = worker.LoadPModule(PArray.transformp2 <@ fun (p:Param) x y -> (x*p.x + y*p.y)*p.time @>).Invoke
    let test n eps = pcalc {
        let par = new Param(2.0, 2.0, 3.0)
        let hInput1 = Array.init n (fun _ -> rng.NextDouble())
        let hInput2 = Array.init n (fun _ -> rng.NextDouble())
        let hOutput (p:Param) = (hInput1, hInput2) ||> Array.map2 (fun x y -> (x*p.x + y*p.y)*p.time)
        let hOutput = hOutput par
        let! dInput1 = DArray.scatterInBlob worker hInput1
        let! dInput2 = DArray.scatterInBlob worker hInput2
        let! dOutput = DArray.createInBlob worker n
        do! transformp2 par dInput1 dInput2 dOutput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>] //transformi2
let ``transformi2: (i:seq int) (x:float) (y:float) -> (x + y)*float(i)``() =
    let transformi2 = worker.LoadPModule(PArray.transformi2 <@ fun i x y -> (x + y)*float(i) @>).Invoke
    let test n eps = pcalc {
        let hInput1 = Array.init n (fun _ -> rng.NextDouble())
        let hInput2 = Array.init n (fun _ -> rng.NextDouble())
        let hOutput = (hInput1, hInput2) ||> Array.mapi2 (fun i x y -> (x + y)*float(i))
        let! dInput1 = DArray.scatterInBlob worker hInput1
        let! dInput2 = DArray.scatterInBlob worker hInput2
        let! dOutput = DArray.createInBlob worker n
        do! transformi2 dInput1 dInput2 dOutput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>] //transformip2
let ``transformip2: (i:seq int) (p:Param) (x:float) (y:float) -> (x*p.x + y*p.y)*float(i)``() =
    let transformip2 = worker.LoadPModule(PArray.transformip2 <@ fun i (p:Param) x y -> (x*p.x + y*p.y)*float(i) @>).Invoke
    let test n eps = pcalc {
        let par = new Param(2.0, 3.0, 1.0)
        let hInput1 = Array.init n (fun _ -> rng.NextDouble())
        let hInput2 = Array.init n (fun _ -> rng.NextDouble())
        let hOutput (p:Param) = (hInput1, hInput2) ||> Array.mapi2 (fun i x y -> (x*p.x + y*p.y)*float(i))
        let hOutput = hOutput par
        let! dInput1 = DArray.scatterInBlob worker hInput1
        let! dInput2 = DArray.scatterInBlob worker hInput2
        let! dOutput = DArray.createInBlob worker n
        do! transformip2 par dInput1 dInput2 dOutput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

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
let ``mapp: (param:float) (x:float) -> log(x) + param``() =
    let mapp = worker.LoadPModule(PArray.mapp  <@ (fun (p:float) (x:float) -> log(x) + p ) @>).Invoke
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
    
[<Test>] //mapp (axpy using struct)
let ``mapp: (param:Param) (a:float) -> a*param.x + param.y``() =
    let mapp = worker.LoadPModule(PArray.mapp  <@ (fun (p:Param) (a:float) -> a*p.x + p.y ) @>).Invoke
    let test n eps = pcalc {
        let par = new Param(2.0, 3.0, 0.0) 
        let hInput = Array.init n (fun _ -> rng.NextDouble() )
        let hOutput (p:Param) = hInput |> Array.map (fun a -> a*p.x + p.y) 
        let hOutput = hOutput par
        let! dInput = DArray.scatterInBlob worker hInput
        let! dOutput = mapp par dInput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>] //mapi
let ``mapi: (i:seq int) (x:float) -> x*float(i)``() =
    let mapi = worker.LoadPModule(PArray.mapi  <@ (fun i x -> x*float(i) ) @>).Invoke
    let test n eps = pcalc {
        let hInput = Array.init n (fun _ -> rng.NextDouble())
        let hOutput = hInput |> Array.mapi (fun i x -> x*float(i))
        let! dInput = DArray.scatterInBlob worker hInput
        let! dOutput = mapi dInput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>] //mapip
let ``mapip: (i:seq int) (p:Param) (x:float) -> (x + p.x)*float(i)``() =
    let mapip = worker.LoadPModule(PArray.mapip  <@ (fun i (p:Param) x -> (x + p.x) * float(i) ) @>).Invoke
    let test n eps = pcalc {
        let par = new Param(1.0,2.0,3.0)
        let hInput = Array.init n (fun _ -> rng.NextDouble())
        let hOutput (p:Param) = hInput |> Array.mapi (fun i x -> (x + p.x)*float(i))
        let hOutput = hOutput par
        let! dInput = DArray.scatterInBlob worker hInput
        let! dOutput = mapip par dInput
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>] //map2
let ``map2: (x:float) (y:float) -> x + y``() =
    let map2 = worker.LoadPModule(PArray.map2 <@ fun x y -> x + y @>).Invoke
    let test n eps = pcalc {
        let hInput1 = Array.init n (fun _ -> rng.NextDouble())
        let hInput2 = Array.init n (fun _ -> rng.NextDouble())
        let hOutput = (hInput1, hInput2) ||> Array.map2 (fun x y -> x + y)
        let! dInput1 = DArray.scatterInBlob worker hInput1
        let! dInput2 = DArray.scatterInBlob worker hInput2
        let! dOutput = map2 dInput1 dInput2
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>] //mapp2
let ``mapp2: (p:Param) (x:float) (y:float) -> x + y*p.time``() =
    let mapp2 = worker.LoadPModule(PArray.mapp2 <@ fun (p:Param) (x:float) (y:float) -> x + y*p.time @>).Invoke
    let test n eps = pcalc {
        let par = new Param(1.0, 1.0, 2.0)
        let hInput1 = Array.init n (fun _ -> rng.NextDouble())
        let hInput2 = Array.init n (fun _ -> rng.NextDouble())
        let hOutput (p:Param) = (hInput1, hInput2) ||> Array.map2 (fun x y -> x + y*p.time)
        let hOutput = hOutput par
        let! dInput1 = DArray.scatterInBlob worker hInput1
        let! dInput2 = DArray.scatterInBlob worker hInput2
        let! dOutput = mapp2 par dInput1 dInput2
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>] //mapi2
let ``mapi2: (i:seq int) (x:float) (y:float) -> (x + y)*float(i)``() =
    let mapi2 = worker.LoadPModule(PArray.mapi2 <@ fun i x y -> (x + y)*float(i) @>).Invoke
    let test n eps = pcalc {
        let hInput1 = Array.init n (fun _ -> rng.NextDouble())
        let hInput2 = Array.init n (fun _ -> rng.NextDouble())
        let hOutput = (hInput1, hInput2) ||> Array.mapi2 (fun i x y -> (x + y) *float(i))
        let! dInput1 = DArray.scatterInBlob worker hInput1
        let! dInput2 = DArray.scatterInBlob worker hInput2
        let! dOutput = mapi2 dInput1 dInput2
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>] //mapip2
let ``mapip2: (i:seq int) (p:Param) (x:float) (y:float) -> (x*p.x + y*p.y)*float(i)``() =
    let mapip2 = worker.LoadPModule(PArray.mapip2 <@ fun i (p:Param) x y -> (x*p.x + y*p.y)*float(i) @>).Invoke
    let test n eps = pcalc {
        let par = new Param(2.0, 3.0, 1.0)
        let hInput1 = Array.init n (fun _ -> rng.NextDouble())
        let hInput2 = Array.init n (fun _ -> rng.NextDouble())
        let hOutput (p:Param) = (hInput1, hInput2) ||> Array.mapi2 (fun i x y -> (x*p.x + y*p.y) *float(i))
        let hOutput = hOutput par
        let! dInput1 = DArray.scatterInBlob worker hInput1
        let! dInput2 = DArray.scatterInBlob worker hInput2
        let! dOutput = mapip2 par dInput1 dInput2
        let! dOutput = dOutput.Gather()
        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }

    test (1<<<22) 1e-10 |> PCalc.run
    test (1<<<22) 1e-10 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) 1e-10 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()