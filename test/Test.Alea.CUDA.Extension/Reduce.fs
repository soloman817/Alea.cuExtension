module Test.Alea.CUDA.Extension.Reduce

open System
open System.IO
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

let worker = getDefaultWorker()
let rng = System.Random()
let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608; 33554432; 33554431; 33554433]

[<Test>]
let ``[DEBUG] sum<int> with reduce int + int``() =
    let init = <@ fun () -> 0 @>
    let op = <@ (+) @>
    let transf = <@ Util.identity @>

    let reducert = Reduce.reduceBuilder (Reduce.Generic.reduceUpSweepKernel init op transf)
                                        (Reduce.Generic.reduceRangeTotalsKernel init op)
    use reducerm = worker.LoadPModule(reducert)

    let test (hValues:int[]) = pcalc {
        let n = hValues.Length
        let hResult = Array.sum hValues

        let reducer = reducerm.Invoke n
        let! dValues = DArray.scatterInBlob worker hValues
        let! dRanges = DArray.scatterInBlob worker reducer.Ranges
        let! dRangeTotals = DArray.createInBlob worker reducer.NumRangeTotals
        let dResult = DScalar.ofArray dRangeTotals 0

        do! PCalc.action (fun lphint -> reducer.Reduce lphint dRanges.Ptr dRangeTotals.Ptr dValues.Ptr)

        let! dResult = dResult.Gather()
        let! hValues' = dValues.Gather()
        let hResult' = Array.sum hValues'

        printfn "[Size %d] h(%d) h'(%d) d(%d)" n hResult hResult' dResult
        if hResult <> dResult then
            let! hRangeTotals = dRangeTotals.Gather()
            printfn "%A" hRangeTotals

        Assert.AreEqual(hResult, dResult) }

    sizes |> Seq.iter (fun n ->
        let rng = Random(2)
        let values = Array.init n (fun _ -> rng.Next(-100, 100))
        test values |> PCalc.run)

[<Test>]
let ``[DEBUG] sum<int> with reduce float + float``() =
    let init = <@ fun () -> 0 @>
    let op = <@ fun x y -> int(float(x) + float(y)) @>
    let transf = <@ Util.identity @>

    let reducert = Reduce.reduceBuilder (Reduce.Generic.reduceUpSweepKernel init op transf)
                                        (Reduce.Generic.reduceRangeTotalsKernel init op)
    use reducerm = worker.LoadPModule(reducert)

    let test (hValues:int[]) = pcalc {
        let n = hValues.Length
        let hResult = Array.sum hValues

        let reducer = reducerm.Invoke n
        let! dValues = DArray.scatterInBlob worker hValues
        let! dRanges = DArray.scatterInBlob worker reducer.Ranges
        let! dRangeTotals = DArray.createInBlob worker reducer.NumRangeTotals
        let dResult = DScalar.ofArray dRangeTotals 0

        do! PCalc.action (fun lphint -> reducer.Reduce lphint dRanges.Ptr dRangeTotals.Ptr dValues.Ptr)

        let! dResult = dResult.Gather()
        let! hValues' = dValues.Gather()
        let hResult' = Array.sum hValues'

        printfn "[Size %d] h(%d) h'(%d) d(%d)" n hResult hResult' dResult
        if hResult <> dResult then
            let! hRangeTotals = dRangeTotals.Gather()
            printfn "%A" hRangeTotals

        Assert.AreEqual(hResult, dResult) }

    sizes |> Seq.iter (fun n ->
        let rng = Random(2)
        let values = Array.init n (fun _ -> rng.Next(-100, 100))
        test values |> PCalc.run)

[<Test>]
let ``[DEBUG] sum<int> with sum``() =
    let reducert = Reduce.reduceBuilder Reduce.Sum.reduceUpSweepKernel Reduce.Sum.reduceRangeTotalsKernel
    use reducerm = worker.LoadPModule(reducert)

    let test (hValues:int[]) = pcalc {
        let n = hValues.Length
        let hResult = Array.sum hValues

        let reducer = reducerm.Invoke n
        let! dValues = DArray.scatterInBlob worker hValues
        let! dRanges = DArray.scatterInBlob worker reducer.Ranges
        let! dRangeTotals = DArray.createInBlob worker reducer.NumRangeTotals
        let dResult = DScalar.ofArray dRangeTotals 0

        do! PCalc.action (fun lphint -> reducer.Reduce lphint dRanges.Ptr dRangeTotals.Ptr dValues.Ptr)

        let! dResult = dResult.Gather()
        let! hValues' = dValues.Gather()
        let hResult' = Array.sum hValues'

        printfn "[Size %d] h(%d) h'(%d) d(%d)" n hResult hResult' dResult
        if hResult <> dResult then
            let! hRangeTotals = dRangeTotals.Gather()
            printfn "%A" hRangeTotals

        Assert.AreEqual(hResult, dResult) }

    sizes |> Seq.iter (fun n ->
        let rng = Random(2)
        let values = Array.init n (fun _ -> rng.Next(-100, 100))
        test values |> PCalc.run)

[<Test>]
let ``sum: debug``() =
    // use raw impl
    let reducert = Reduce.reduceBuilder Reduce.Sum.reduceUpSweepKernel Reduce.Sum.reduceRangeTotalsKernel
    use reducerm = worker.LoadPModule(reducert)

    let debug n = pcalc {
        let reduce = reducerm.Invoke n
        let hRanges = reduce.Ranges
        let nRangeTotals = reduce.NumRangeTotals

        printfn "%A" hRanges
        printfn "%A" nRangeTotals

        let! dRanges = DArray.scatterInBlob worker hRanges
        let! dRangeTotals = DArray.createInBlob worker nRangeTotals

        let hValues = Array.init n (fun i -> if i % 2 = 0 then -1 else 1)
        let! dValues = DArray.scatterInBlob worker hValues

        do! PCalc.action (fun lphint -> reduce.Reduce lphint dRanges.Ptr dRangeTotals.Ptr dValues.Ptr)

        let! hRangeTotals = dRangeTotals.Gather()
        printfn "%A" hRangeTotals
        }

    let n = 33554431
    let debug = debug n

    debug |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

//[<Test>]
//let ``sum: int``() =
//    let sum = worker.LoadPModule(PArray.sum()).Invoke
//    let test (hValues:int[]) = pcalc {
//        let n = hValues.Length
//        let hResult = Array.sum hValues
//        let! dValues = DArray.scatterInBlob worker hValues
//        let! dResult = sum dValues
//        let! dResult = dResult.Gather()
//        let! hValues' = dValues.Gather()
//        let hResult' = hValues' |> Array.sum
//        printfn "[Size %d] h(%d) h'(%d) d(%d)" n hResult hResult' dResult
//        Assert.AreEqual(hResult, dResult) }
//
//    
////    sizes |> Seq.iter (fun n ->
////        let values = Array.init n (fun _ -> 1)
////        test values |> PCalc.run)
////
////    sizes |> Seq.iter (fun n ->
////        let values = Array.init n (fun _ -> -1)
////        test values |> PCalc.run)
////
////    sizes |> Seq.iter (fun n ->
////        let values = Array.init n (fun _ -> 10)
////        test values |> PCalc.run)
////
////    sizes |> Seq.iter (fun n ->
////        let values = Array.init n (fun _ -> -10)
////        test values |> PCalc.run)
//
//    sizes |> Seq.iter (fun n ->
//        let rng = Random(2)
//        let values = Array.init n (fun _ -> rng.Next(-100, 100))
//        test values |> PCalc.run)
//
////    let values = 
////        let rng = Random(0)
////        Array.init 33554432 (fun _ -> rng.Next(-100, 100))
////    test values |> PCalc.run
//
//
//
////    let rng = System.Random(0)
////    let init1 i = 1
////    //let init2 i = rng.Next(-20, 20)
////    let init2 i = rng.Next(-50, 50)
////
////    sizes |> Seq.iter (fun n -> test n init1 |> PCalc.run)
////    sizes |> Seq.iter (fun n -> test n init2 |> PCalc.run)
////
////    test (1<<<22) init2 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
////    let _, loggers = test (1<<<22) init2 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
////
////    for i = 1 to 10 do [ 2097152; 8388608; 33554432; 33554431; 33554433 ] |> Seq.iter (fun n -> test n init2 |> PCalc.run)
////
////    let _, tc = test (1<<<26) init2 |> PCalc.runWithKernelTiming 10 in tc.Dump()
//
//[<Test>]
//let ``sum: misdata!``() =
//    let sum = worker.LoadPModule(PArray.sum()).Invoke
//    let test (hValues:int[]) = pcalc {
//        let n = hValues.Length
//        let hResult = Array.sum hValues
//        let! dValues = DArray.scatterInBlob worker hValues
//        let! dResult = sum dValues
//        let! dResult = dResult.Gather()
//        printfn "[Size %d] h(%d) d(%d)" n hResult dResult
//        Assert.AreEqual(hResult, dResult) }
//
//    let s = File.OpenRead("_MismatchData1.dat")
//    let ss = new BinaryReader(s)
//    let n = ss.ReadInt32()
//    printfn "%A" n
//    let hValues = Array.init n (fun _ -> ss.ReadInt32())
//    ss.Dispose()
//    s.Close()
//    
//    let test = test hValues
//
//    test |> PCalc.run
//    
//
//[<Test>]
//let ``sum: int x 2``() =
//    let sum = worker.LoadPModule(PArray.sum()).Invoke
//    let test n init1 init2 = pcalc {
//        let hValues1 = Array.init<int> n init1
//        let hValues2 = Array.init<int> n init2
//
//        let hResult1 = Array.sum hValues1
//        let hResult2 = Array.sum hValues2
//
//        let! dValues1 = DArray.scatterInBlob worker hValues1
//        let! dValues2 = DArray.scatterInBlob worker hValues2
//
//        let! dResult1 = sum dValues1
//        let! dResult2 = sum dValues2
//
//        let! dResult1 = dResult1.Gather()
//        let! dResult2 = dResult2.Gather()
//
//        printfn "[Size %d] h(%d) d(%d)" n hResult1 dResult1
//        printfn "[Size %d] h(%d) d(%d)" n hResult2 dResult2
//        if hResult1 <> dResult1 then
//            let s = new StreamWriter("MismatchData.dat")
//            s.Write(hValues1.Length)
//            for i = 0 to hValues1.Length - 1 do
//                s.Write(hValues1.[i])
//            s.Dispose()
//        if hResult2 <> dResult2 then
//            let s = new StreamWriter("MismatchData.dat")
//            s.Write(hValues2.Length)
//            for i = 0 to hValues2.Length - 1 do
//                s.Write(hValues2.[i])
//            s.Dispose()
//        Assert.AreEqual(hResult1, dResult1)
//        Assert.AreEqual(hResult2, dResult2) }
//
//    let init1 i = 1
//    let init2 i = rng.Next(-50, 50)
//
//    sizes |> Seq.iter (fun n -> test n init1 init2 |> PCalc.run)
//    test (1<<<22) init1 init2 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
//    let _, loggers = test (1<<<22) init1 init2 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    
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

//[<Test>]
//let ``sumer: int x 2``() =
//    let sumer = worker.LoadPModule(PArray.sumer()).Invoke
//    let test n init1 init2 = pcalc {
//        let! sum = sumer n
//
//        let hValues1 = Array.init<int> n init1
//        let hValues2 = Array.init<int> n init2
//
//        let hResult1 = Array.sum hValues1
//        let hResult2 = Array.sum hValues2
//
//        let! dValues1 = DArray.scatterInBlob worker hValues1
//        let! dValues2 = DArray.scatterInBlob worker hValues2
//
//        let! dResult1 = sum dValues1
//        let! dResult1 = dResult1.Gather()
//
//        let! dResult2 = sum dValues2
//        let! dResult2 = dResult2.Gather()
//
//        printfn "[Size %d] h(%d) d(%d)" n hResult1 dResult1
//        printfn "[Size %d] h(%d) d(%d)" n hResult2 dResult2
//        Assert.AreEqual(hResult1, dResult1)
//        Assert.AreEqual(hResult2, dResult2) }
//
//    let init1 i = 1
//    let init2 i = rng.Next(-50, 50)
//
//    sizes |> Seq.iter (fun n -> test n init1 init2 |> PCalc.run)
//    test (1<<<22) init1 init2 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
//    let _, loggers = test (1<<<22) init1 init2 |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

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
let ``reduce: sum<int>``() =
    //let sum = worker.LoadPModule(PArray.reduce <@ fun () -> 0 @> <@ (+) @> <@ Util.identity @>).Invoke
    //let sum = worker.LoadPModule(PArray.reduce <@ fun () -> 0 @> <@ fun (x:int) (y:int) -> int(float32(x) + float32(y)) @> <@ Util.identity @>).Invoke
    //let sum = worker.LoadPModule(PArray.reduce <@ fun () -> 0 @> <@ (+) @> <@ Util.identity @>).Invoke
    let sum = worker.LoadPModule(PArray.reduce <@ fun () -> 0 @> <@ fun x y -> int(float(x) + float(y)) @> <@ Util.identity @>).Invoke
    let test (hValues:int[]) = pcalc {
        let n = hValues.Length
        let hResult = Array.sum hValues
        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult = sum dValues
        let! dResult = dResult.Gather()
        let! hValues' = dValues.Gather()
        let hResult' = hValues' |> Array.sum
        printfn "[Size %d] h(%d) h'(%d) d(%d)" n hResult hResult' dResult
        Assert.AreEqual(hResult, dResult) }

    sizes |> Seq.iter (fun n ->
        let rng = Random(2)
        let values = Array.init n (fun _ -> rng.Next(-100, 100))
        test values |> PCalc.run)

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

