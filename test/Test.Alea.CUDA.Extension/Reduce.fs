﻿module Test.Alea.CUDA.Extension.Reduce

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
let ``sum: int x 2``() =
    let sum = worker.LoadPModule(PArray.sum()).Invoke
    let test n init1 init2 = pcalc {
        let hValues1 = Array.init<int> n init1
        let hValues2 = Array.init<int> n init2

        let hResult1 = Array.sum hValues1
        let hResult2 = Array.sum hValues2

        let! dValues1 = DArray.scatterInBlob worker hValues1
        let! dValues2 = DArray.scatterInBlob worker hValues2

        let! dResult1 = sum dValues1
        let! dResult2 = sum dValues2

        let! dResult1 = dResult1.Gather()
        let! dResult2 = dResult2.Gather()

        printfn "[Size %d] h(%d) d(%d)" n hResult1 dResult1
        printfn "[Size %d] h(%d) d(%d)" n hResult2 dResult2
        Assert.AreEqual(hResult1, dResult1)
        Assert.AreEqual(hResult2, dResult2) }

    let init1 i = 1
    let init2 i = rng.Next(-100, 100)

    sizes |> Seq.iter (fun n -> test n init1 init2 |> PCalc.run)
    test (1<<<22) init1 init2 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) init1 init2 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    
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
let ``sumer: int x 2``() =
    let sumer = worker.LoadPModule(PArray.sumer()).Invoke
    let test n init1 init2 = pcalc {
        let! sum = sumer n

        let hValues1 = Array.init<int> n init1
        let hValues2 = Array.init<int> n init2

        let hResult1 = Array.sum hValues1
        let hResult2 = Array.sum hValues2

        let! dValues1 = DArray.scatterInBlob worker hValues1
        let! dValues2 = DArray.scatterInBlob worker hValues2

        let! dResult1 = sum dValues1
        let! dResult1 = dResult1.Gather()

        let! dResult2 = sum dValues2
        let! dResult2 = dResult2.Gather()

        printfn "[Size %d] h(%d) d(%d)" n hResult1 dResult1
        printfn "[Size %d] h(%d) d(%d)" n hResult2 dResult2
        Assert.AreEqual(hResult1, dResult1)
        Assert.AreEqual(hResult2, dResult2) }

    let init1 i = 1
    let init2 i = rng.Next(-100, 100)

    sizes |> Seq.iter (fun n -> test n init1 init2 |> PCalc.run)
    test (1<<<22) init1 init2 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) init1 init2 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

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

[<Test>]
let ``reduce: sum square<float> x 2``() =
    let reduce = worker.LoadPModule(PArray.reduce <@ fun () -> 0.0 @> <@ (+) @> <@ fun x -> x * x @>).Invoke
    let test n init1 init2 eps = pcalc {
        let hValues1 = Array.init<float> n init1
        let hValues2 = Array.init<float> n init2

        let hResult1 = hValues1 |> Array.map (fun x -> x * x) |> Array.sum
        let hResult2 = hValues2 |> Array.map (fun x -> x * x) |> Array.sum

        let! dValues1 = DArray.scatterInBlob worker hValues1
        let! dValues2 = DArray.scatterInBlob worker hValues2

        let! dResult1 = reduce dValues1
        let! dResult2 = reduce dValues2

        let! dResult1 = dResult1.Gather()
        let! dResult2 = dResult2.Gather()

        let err1 = abs (dResult1 - hResult1) / abs hResult1
        let err2 = abs (dResult2 - hResult2) / abs hResult2
        printfn "[Size %d] h(%f) d(%f) e(%f)" n hResult1 dResult1 err1
        printfn "[Size %d] h(%f) d(%f) e(%f)" n hResult2 dResult2 err2
        Assert.That(err1 < eps)
        Assert.That(err2 < eps) }

    let init1 i = 1.0
    let init2 i = rng.NextDouble() - 0.5
    let eps = 1e-10

    sizes |> Seq.iter (fun n -> test n init1 init2 eps |> PCalc.run)
    test (1<<<22) init1 init2 eps |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) init1 init2 eps |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>]
let ``reducer: sum square<float> x 2``() =
    let reducer = worker.LoadPModule(PArray.reducer <@ fun () -> 0.0 @> <@ (+) @> <@ fun x -> x * x @>).Invoke
    let test n init1 init2 eps = pcalc {
        let! reduce = reducer n

        let hValues1 = Array.init<float> n init1
        let hValues2 = Array.init<float> n init2

        let hResult1 = hValues1 |> Array.map (fun x -> x * x) |> Array.sum
        let hResult2 = hValues2 |> Array.map (fun x -> x * x) |> Array.sum

        let! dValues1 = DArray.scatterInBlob worker hValues1
        let! dValues2 = DArray.scatterInBlob worker hValues2

        let! dResult1 = reduce dValues1
        let! dResult1 = dResult1.Gather()

        let! dResult2 = reduce dValues2
        let! dResult2 = dResult2.Gather()

        let err1 = abs (dResult1 - hResult1) / abs hResult1
        let err2 = abs (dResult2 - hResult2) / abs hResult2
        printfn "[Size %d] h(%f) d(%f) e(%f)" n hResult1 dResult1 err1
        printfn "[Size %d] h(%f) d(%f) e(%f)" n hResult2 dResult2 err2
        Assert.That(err1 < eps)
        Assert.That(err2 < eps) }

    let init1 i = 1.0
    let init2 i = rng.NextDouble() - 0.5
    let eps = 1e-10

    sizes |> Seq.iter (fun n -> test n init1 init2 eps |> PCalc.run)
    test (1<<<22) init1 init2 eps |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = test (1<<<22) init1 init2 eps |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

