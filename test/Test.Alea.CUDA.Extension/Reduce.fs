module Test.Alea.CUDA.Extension.Reduce

open System
open System.IO
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

let worker = getDefaultWorker()
let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608; 33554432; 33554431; 33554433]

//[<Test>]
let ``[DEBUG] save data``() =
    sizes |> Seq.iteri (fun i n ->
        let rng = Random(2)
        let values = Array.init n (fun _ -> rng.Next(-100, 100))
        use file = File.OpenWrite(sprintf "TestData_%d.dat" i)
        use writer = new BinaryWriter(file)
        writer.Write(values.Length)
        values |> Array.iter (fun value -> writer.Write(value))
        printfn "Size = %d" n
        let ranges = Reduce.plan32.BlockRanges worker.Device.NumSm n
        printfn "%A" ranges
        //printfn "%A" values
        )

//[<Test>]
let ``[DEBUG]`` () =
    let calc = pcalc {
        let hArray = [| 1; 2; 3 |]
        let! dArray = DArray.scatterInBlob worker hArray
        return! dArray.Gather() }
    let result = calc |> PCalc.run
    printfn "%A" result

[<Test>]
let ``sum: int``() =
    let sum = worker.LoadPModule(PArray.sum()).Invoke
    let test (hValues:int[]) = pcalc {
        let n = hValues.Length
        let hResult = Array.sum hValues
        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult = sum dValues
        let! dResult = dResult.Gather()
        printfn "[Size %d] h(%d) d(%d)" n hResult dResult
        Assert.AreEqual(hResult, dResult) }

    let values1 n = Array.init n (fun _ -> 1)
    let values2 n = Array.init n (fun _ -> -1)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.Next(-100, 100))

    sizes |> Seq.iter (fun n -> values1 n |> test |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test |> PCalc.run)

    let n = 8388608
    let test = values3 n |> test

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``sum: int (2)``() =
    let sum = worker.LoadPModule(PArray.sum()).Invoke
    let test (hValues1:int[]) (hValues2:int[]) = pcalc {
        let n1 = hValues1.Length
        let n2 = hValues2.Length

        let hResult1 = Array.sum hValues1
        let hResult2 = Array.sum hValues2

        let! dValues1 = DArray.scatterInBlob worker hValues1
        let! dValues2 = DArray.scatterInBlob worker hValues2

        let! dResult1 = sum dValues1
        let! dResult2 = sum dValues2

        let! dResult1 = dResult1.Gather()
        let! dResult2 = dResult2.Gather()

        printfn "[Size %d %d] h1(%d) d1(%d) h2(%d) d2(%d)" n1 n2 hResult1 dResult1 hResult2 dResult2
        Assert.AreEqual(hResult1, dResult1)
        Assert.AreEqual(hResult2, dResult2) }

    let values1 n = Array.init n (fun _ -> 1)
    let values2 n = let rng = Random(2) in Array.init n (fun _ -> rng.Next(-100, 100))

    sizes |> Seq.iter (fun n -> (values1 n, values2 n) ||> test |> PCalc.run)

    let n = 8388608
    let test = (values1 n, values2 n) ||> test

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    
[<Test>]
let ``sum: float``() =
    let sum = worker.LoadPModule(PArray.sum()).Invoke
    let test eps (hValues:float[]) = pcalc {
        let n = hValues.Length
        let hResult = Array.sum hValues
        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult = sum dValues
        let! dResult = dResult.Gather()
        let err = abs (dResult - hResult) / abs hResult
        printfn "[Size %d] h(%f) d(%f) e(%f)" n hResult dResult err
        Assert.That(err < eps) }

    let eps = 1e-10
    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = Array.init n (fun _ -> -1.0)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)

    sizes |> Seq.iter (fun n -> values1 n |> test eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test eps |> PCalc.run)

    let n = 8388608
    let test = values3 n |> test eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``sumer: int (2)``() =
    let sumer = worker.LoadPModule(PArray.sumer()).Invoke
    let test (hValues1:int[]) (hValues2:int[]) = pcalc {
        if hValues1.Length <> hValues2.Length then failwith "input arrays should be equal length"
        let n = hValues1.Length
        let! sum = sumer n

        let hResult1 = Array.sum hValues1
        let hResult2 = Array.sum hValues2

        let! dValues1 = DArray.scatterInBlob worker hValues1
        let! dValues2 = DArray.scatterInBlob worker hValues2
        let! dResult1 = DScalar.createInBlob worker
        let! dResult2 = DScalar.createInBlob worker

        do! sum dValues1 dResult1
        do! sum dValues2 dResult2

        let! dResult1 = dResult1.Gather()
        let! dResult2 = dResult2.Gather()

        printfn "[Size %d] h1(%d) d1(%d) h2(%d) d2(%d)" n hResult1 dResult1 hResult2 dResult2
        Assert.AreEqual(hResult1, dResult1)
        Assert.AreEqual(hResult2, dResult2) }

    let values1 n = Array.init n (fun _ -> 1)
    let values2 n = let rng = Random(2) in Array.init n (fun _ -> rng.Next(-100, 100))

    sizes |> Seq.iter (fun n -> (values1 n, values2 n) ||> test |> PCalc.run)

    let n = 8388608
    let test = (values1 n, values2 n) ||> test

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``reduce: sum<int>``() =
    let reduce = worker.LoadPModule(PArray.reduce <@ fun () -> 0 @> <@ (+) @> <@ Util.identity @>).Invoke
    let test (hValues:int[]) = pcalc {
        let n = hValues.Length
        let hResult = Array.sum hValues
        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult = reduce dValues
        let! dResult = dResult.Gather()
        printfn "[Size %d] h(%d) d(%d)" n hResult dResult
        Assert.AreEqual(hResult, dResult) }

    let values1 n = Array.init n (fun _ -> 1)
    let values2 n = Array.init n (fun _ -> -1)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.Next(-100, 100))

    sizes |> Seq.iter (fun n -> values1 n |> test |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test |> PCalc.run)

    let n = 8388608
    let test = values3 n |> test

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``reduce: sum<float>``() =
    let reduce = worker.LoadPModule(PArray.reduce <@ fun () -> 0.0 @> <@ (+) @> <@ Util.identity @>).Invoke
    let test eps (hValues:float[]) = pcalc {
        let n = hValues.Length
        let hResult = Array.sum hValues
        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult = reduce dValues
        let! dResult = dResult.Gather()
        let err = abs (dResult - hResult) / abs hResult
        printfn "[Size %d] h(%f) d(%f) e(%f)" n hResult dResult err
        Assert.That(err < eps) }

    let eps = 1e-10
    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = Array.init n (fun _ -> -1.0)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)

    sizes |> Seq.iter (fun n -> values1 n |> test eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test eps |> PCalc.run)

    let n = 8388608
    let test = values3 n |> test eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``reduce: max<float>``() =
    let reduce = worker.LoadPModule(PArray.reduce <@ fun () -> Double.NegativeInfinity @> <@ max @> <@ Util.identity @>).Invoke
    let test eps (hValues:float[]) = pcalc {
        let n = hValues.Length
        let hResult = Array.max hValues
        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult = reduce dValues
        let! dResult = dResult.Gather()
        let err = abs (dResult - hResult) / abs hResult
        printfn "[Size %d] h(%f) d(%f) e(%f)" n hResult dResult err
        Assert.That(err < eps) }

    let eps = 1e-10
    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = Array.init n (fun _ -> -1.0)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)

    sizes |> Seq.iter (fun n -> values1 n |> test eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test eps |> PCalc.run)

    let n = 8388608
    let test = values3 n |> test eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``reduce: min<float>``() =
    let reduce = worker.LoadPModule(PArray.reduce <@ fun () -> Double.PositiveInfinity @> <@ min @> <@ Util.identity @>).Invoke
    let test eps (hValues:float[]) = pcalc {
        let n = hValues.Length
        let hResult = Array.min hValues
        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult = reduce dValues
        let! dResult = dResult.Gather()
        let err = abs (dResult - hResult) / abs hResult
        printfn "[Size %d] h(%f) d(%f) e(%f)" n hResult dResult err
        Assert.That(err < eps) }

    let eps = 1e-10
    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = Array.init n (fun _ -> -1.0)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)

    sizes |> Seq.iter (fun n -> values1 n |> test eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test eps |> PCalc.run)

    let n = 8388608
    let test = values3 n |> test eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``reduce: sum square<float>``() =
    let reduce = worker.LoadPModule(PArray.reduce <@ fun () -> 0.0 @> <@ (+) @> <@ fun x -> x * x @>).Invoke
    let test eps (hValues:float[]) = pcalc {
        let n = hValues.Length
        let hResult = hValues |> Array.map (fun x -> x * x) |> Array.sum
        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult = reduce dValues
        let! dResult = dResult.Gather()
        let err = abs (dResult - hResult) / abs hResult
        printfn "[Size %d] h(%f) d(%f) e(%f)" n hResult dResult err
        Assert.That(err < eps) }

    let eps = 1e-10
    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = Array.init n (fun _ -> -1.0)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)

    sizes |> Seq.iter (fun n -> values1 n |> test eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test eps |> PCalc.run)

    let n = 8388608
    let test = values3 n |> test eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``reduce: sum square<float> (2)``() =
    let reduce = worker.LoadPModule(PArray.reduce <@ fun () -> 0.0 @> <@ (+) @> <@ fun x -> x * x @>).Invoke
    let test eps (hValues1:float[]) (hValues2:float[]) = pcalc {
        let n1 = hValues1.Length
        let n2 = hValues2.Length

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
        printfn "[Size %d %d] h1(%f) d1(%f) e1(%f) h2(%f) d2(%f) e2(%f)" n1 n2 hResult1 dResult1 err1 hResult2 dResult2 err2
        Assert.That(err1 < eps)
        Assert.That(err2 < eps) }

    let eps = 1e-10
    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)
    let sizes = sizes |> List.filter (fun n -> n <5000000) // to avoid out of memory exception when x86

    sizes |> Seq.iter (fun n -> (values1 n, values2 n) ||> test eps |> PCalc.run)

    let n = 8388608
    let test = (values1 n, values2 n) ||> test eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``reducer: sum square<float> (2)``() =
    let reducer = worker.LoadPModule(PArray.reducer <@ fun () -> 0.0 @> <@ (+) @> <@ fun x -> x * x @>).Invoke
    let test eps (hValues1:float[]) (hValues2:float[]) = pcalc {
        if hValues1.Length <> hValues2.Length then failwith "input arrays should be equal length"
        let n = hValues1.Length
        let! reduce = reducer n

        let hResult1 = hValues1 |> Array.map (fun x -> x * x) |> Array.sum
        let hResult2 = hValues2 |> Array.map (fun x -> x * x) |> Array.sum

        let! dValues1 = DArray.scatterInBlob worker hValues1
        let! dValues2 = DArray.scatterInBlob worker hValues2
        let! dResult1 = DScalar.createInBlob worker
        let! dResult2 = DScalar.createInBlob worker

        do! reduce dValues1 dResult1
        do! reduce dValues2 dResult2

        let! dResult1 = dResult1.Gather()
        let! dResult2 = dResult2.Gather()

        let err1 = abs (dResult1 - hResult1) / abs hResult1
        let err2 = abs (dResult2 - hResult2) / abs hResult2
        printfn "[Size %d] h1(%f) d1(%f) e1(%f) h2(%f) d2(%f) e2(%f)" n hResult1 dResult1 err1 hResult2 dResult2 err2
        Assert.That(err1 < eps)
        Assert.That(err2 < eps) }

    let eps = 1e-10
    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)

    sizes |> Seq.iter (fun n -> (values1 n, values2 n) ||> test eps |> PCalc.run)

    let n = 8388608
    let test = (values1 n, values2 n) ||> test eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Struct;Align(16)>]
type IterativeMean =
    val mean : float
    val count : float

    [<ReflectedDefinition>]
    new (mean, count) = { mean = mean; count = count }

    [<ReflectedDefinition>]
    new (mean) = { mean = mean; count = 1.0 }

    [<ReflectedDefinition>]
    static member ( + ) (lhs:IterativeMean, rhs:IterativeMean) =
        if lhs.count = 0.0 then rhs
        elif rhs.count = 0.0 then lhs
        else
            let count = lhs.count + rhs.count
            let mean = (lhs.count * lhs.mean + rhs.count * rhs.mean) / count
            IterativeMean(mean, count)

    [<ReflectedDefinition>]
    static member get_Zero() = IterativeMean()

[<Test>]
let ``reduce: iterative mean <float>``() =
    let reduce = worker.LoadPModule(PArray.reduce <@ IterativeMean.get_Zero @> <@ (+) @> <@ Util.identity @>).Invoke
    let test eps1 eps2 (hValues:IterativeMean[]) = pcalc {
        let n = hValues.Length
        let hResult = Array.sum hValues
        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult = reduce dValues
        let! dResult = dResult.Gather()
        let average = hValues |> Array.averageBy (fun v -> v.mean)
        let err1 = abs (dResult.mean - average)
        let err2 = abs (dResult.mean - hResult.mean) / abs hResult.mean
        let err3 = abs (dResult.count - hResult.count) / abs hResult.count
        printfn "[Size %d] a(%f) h(%f,%f) d(%f,%f) e(%f,%f,%f)" n average hResult.mean hResult.count dResult.mean dResult.count err1 err2 err3
        Assert.That(err1 < eps1)
        Assert.That(err2 < eps2)
        Assert.AreEqual(err3, 0.0)
        Assert.AreEqual(dResult.count |> int, n) }

    let eps1 = 1e-16
    let eps2 = 1e-12
    let values1 n = Array.init n (fun _ -> IterativeMean(1.0))
    let values2 n = Array.init n (fun _ -> IterativeMean(-1.0))
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> IterativeMean(rng.NextDouble() - 0.5))

    sizes |> Seq.iter (fun n -> values1 n |> test eps1 eps2 |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test eps1 eps2 |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test eps1 eps2 |> PCalc.run)

    let n = 8388608
    let test = values3 n |> test eps1 eps2

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

