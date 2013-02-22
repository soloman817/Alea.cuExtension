module Test.Alea.CUDA.Extension.Reduce

open System
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

let worker = getDefaultWorker()
let rng = System.Random()
let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608; 33554432]

[<Test>]
let ``sum: int``() =
    let sum = worker.LoadPModule(PArray.sum()).Invoke
    let test n init = pcalc {
        let hValues = Array.init<int> n init
        let hResult = Array.sum hValues
        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult = sum dValues
        let! dResult = dResult.Gather()
        printfn "[Size %d] h(%d) d(%d)" n hResult dResult
        Assert.AreEqual(hResult, dResult) }

    sizes |> Seq.iter (fun n -> test n (fun _ -> 1) |> PCalc.run)
    sizes |> Seq.iter (fun n -> test n (fun _ -> rng.Next(-100, 100)) |> PCalc.run)

    test (1<<<22) (fun _ -> rng.Next(-100, 100)) |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) (fun _ -> rng.Next(-100, 100)) |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>]
let ``sum: float``() =
    let sum = worker.LoadPModule(PArray.sum()).Invoke
    let test n init eps = pcalc {
        let hValues = Array.init<float> n init
        let hResult = Array.sum hValues
        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult = sum dValues
        let! dResult = dResult.Gather()
        let err = abs (dResult - hResult) / abs hResult
        printfn "[Size %d] h(%f) d(%f) e(%f)" n hResult dResult err
        Assert.That(err < eps) }

    let eps = 1e-10
    sizes |> Seq.iter (fun n -> test n (fun _ -> 1.0) eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> test n (fun _ -> rng.NextDouble() - 0.5) eps |> PCalc.run)

    test (1<<<22) (fun _ -> rng.NextDouble()) eps |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) (fun _ -> rng.NextDouble()) eps |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>]
let ``reduce: sum<float>``() =
    let reduce = worker.LoadPModule(PArray.reduce <@ fun () -> 0.0 @> <@ (+) @> <@ Util.identity @>).Invoke
    let test n init eps = pcalc {
        let hValues = Array.init n init
        let hResult = Array.sum hValues
        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult = reduce dValues
        let! dResult = dResult.Gather()
        let err = abs (dResult - hResult) / abs hResult
        printfn "[Size %d] h(%f) d(%f) e(%f)" n hResult dResult err
        Assert.That(err < eps) }

    let eps = 1e-10
    sizes |> Seq.iter (fun n -> test n (fun _ -> 1.0) eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> test n (fun _ -> rng.NextDouble() - 0.5) eps |> PCalc.run)

    test (1<<<22) (fun _ -> rng.NextDouble()) eps |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) (fun _ -> rng.NextDouble()) eps |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>]
let ``reduce: max<float>``() =
    let reduce = worker.LoadPModule(PArray.reduce <@ fun () -> Double.NegativeInfinity @> <@ max @> <@ Util.identity @>).Invoke
    let test n init eps = pcalc {
        let hValues = Array.init n init
        let hResult = Array.max hValues
        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult = reduce dValues
        let! dResult = dResult.Gather()
        let err = abs (dResult - hResult) / abs hResult
        printfn "[Size %d] h(%f) d(%f) e(%f)" n hResult dResult err
        Assert.That(err < eps) }

    let eps = 1e-10
    sizes |> Seq.iter (fun n -> test n (fun _ -> -1.0) eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> test n (fun _ -> rng.NextDouble() - 0.5) eps |> PCalc.run)

    test (1<<<22) (fun _ -> rng.NextDouble()) eps |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) (fun _ -> rng.NextDouble()) eps |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>]
let ``reduce: min<float>``() =
    let reduce = worker.LoadPModule(PArray.reduce <@ fun () -> Double.PositiveInfinity @> <@ min @> <@ Util.identity @>).Invoke
    let test n init eps = pcalc {
        let hValues = Array.init n init
        let hResult = Array.min hValues
        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult = reduce dValues
        let! dResult = dResult.Gather()
        let err = abs (dResult - hResult) / abs hResult
        printfn "[Size %d] h(%f) d(%f) e(%f)" n hResult dResult err
        Assert.That(err < eps) }

    let eps = 1e-10
    sizes |> Seq.iter (fun n -> test n (fun _ -> 1.0) eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> test n (fun _ -> rng.NextDouble() - 0.5) eps |> PCalc.run)

    test (1<<<22) (fun _ -> rng.NextDouble()) eps |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) (fun _ -> rng.NextDouble()) eps |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>]
let ``reduce: sum square<float>``() =
    let reduce = worker.LoadPModule(PArray.reduce <@ fun () -> 0.0 @> <@ (+) @> <@ fun x -> x * x @>).Invoke
    let test n init eps = pcalc {
        let hValues = Array.init n init
        let hResult = hValues |> Array.map (fun x -> x * x) |> Array.sum
        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult = reduce dValues
        let! dResult = dResult.Gather()
        let err = abs (dResult - hResult) / abs hResult
        printfn "[Size %d] h(%f) d(%f) e(%f)" n hResult dResult err
        Assert.That(err < eps) }

    let eps = 1e-10
    sizes |> Seq.iter (fun n -> test n (fun _ -> 1.0) eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> test n (fun _ -> rng.NextDouble()) eps |> PCalc.run)

    test (1<<<22) (fun _ -> rng.NextDouble()) eps |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) (fun _ -> rng.NextDouble()) eps |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

