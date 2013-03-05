module Test.Alea.CUDA.Extension.Scan

open System
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

let worker = getDefaultWorker()
let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608]

//[<Test>]
let ``debug sum`` () =
    let scanner = worker.LoadPModule(Scan.sum None).Invoke
    let test verify eps (hValues:int[]) = pcalc {
        let n = hValues.Length
        printfn "Testing size %d..." n
        let scanner = scanner n

        let hResults = if verify then hValues |> Array.scan (+) 0 else Array.empty

        let! dValues = DArray.scatterInBlob worker hValues
        let! dResults = DArray.createInBlob worker n
        let! dRanges = DArray.scatterInBlob worker scanner.Ranges
        let! dRangeTotals = DArray.createInBlob worker scanner.NumRangeTotals

        do! PCalc.action (fun lphint -> scanner.Scan lphint dRanges.Ptr dRangeTotals.Ptr dValues.Ptr dResults.Ptr true)
        let! dResultsIncl = dResults.Gather()
        if verify then
            let hResultsIncl = Array.sub hResults 1 n
            (hResultsIncl, dResultsIncl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

        do! PCalc.action (fun lphint -> scanner.Scan lphint dRanges.Ptr dRangeTotals.Ptr dValues.Ptr dResults.Ptr false)
        let! dResultsExcl = dResults.Gather()
        if verify then
            let hResultsExcl = Array.sub hResults 0 n
            (hResultsExcl, dResultsExcl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps))) }


    let eps = 1e-10
    let values1 n = Array.init n (fun _ -> 1)
    let values2 n = Array.init n (fun _ -> -1)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.Next(-100, 100))

    sizes |> Seq.iter (fun n -> values1 n |> test true eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test true eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test true eps |> PCalc.run)

    let n = 2097152
    let test = values1 n |> test false eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``sumscan: int`` () =
    let scan = worker.LoadPModule(PArray.sumscan()).Invoke
    let test verify (hValues:int[]) = pcalc {
        let n = hValues.Length
        printfn "Testing size %d..." n

        let! dValues = DArray.scatterInBlob worker hValues
        let! dResultsIncl = scan true dValues
        let! dResultsExcl = scan false dValues

        if verify then
            let hResults = hValues |> Array.scan (+) 0
            
            let hResultsIncl = Array.sub hResults 1 n
            let! dResultsIncl = dResultsIncl.Gather()
            (hResultsIncl, dResultsIncl) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))

            let hResultsExcl = Array.sub hResults 0 n
            let! dResultsExcl = dResultsExcl.Gather()
            (hResultsExcl, dResultsExcl) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))
        else do! PCalc.force() }

    let values1 n = Array.init n (fun _ -> 1)
    let values2 n = Array.init n (fun _ -> -1)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.Next(-100, 100))

    sizes |> Seq.iter (fun n -> values1 n |> test true |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test true |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test true |> PCalc.run)

    let n = 2097152
    let test = values1 n |> test false

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``sumscan: float`` () =
    let scan = worker.LoadPModule(PArray.sumscan()).Invoke
    let test verify eps (hValues:float[]) = pcalc {
        let n = hValues.Length
        printfn "Testing size %d..." n

        let! dValues = DArray.scatterInBlob worker hValues
        let! dResultsIncl = scan true dValues
        let! dResultsExcl = scan false dValues

        if verify then
            let hResults = hValues |> Array.scan (+) 0.0
            
            let hResultsIncl = Array.sub hResults 1 n
            let! dResultsIncl = dResultsIncl.Gather()
            (hResultsIncl, dResultsIncl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            let hResultsExcl = Array.sub hResults 0 n
            let! dResultsExcl = dResultsExcl.Gather()
            (hResultsExcl, dResultsExcl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
        else do! PCalc.force() }

    let eps = 1e-10
    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = Array.init n (fun _ -> -1.0)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)

    sizes |> Seq.iter (fun n -> values1 n |> test true eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test true eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test true eps |> PCalc.run)

    let n = 2097152
    let test = values1 n |> test false eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``sumscanner: int (2)`` () =
    let scanner = worker.LoadPModule(PArray.sumscanner()).Invoke
    let test verify (hValues1:int[]) (hValues2:int[]) = pcalc {
        if hValues1.Length <> hValues2.Length then failwith "inputs should be equal length"
        let n = hValues1.Length
        printfn "Testing size %d..." n
        let! scan = scanner n

        let! dValues1 = DArray.scatterInBlob worker hValues1
        let! dResults1Incl = DArray.createInBlob worker n
        let! dResults1Excl = DArray.createInBlob worker n
        do! scan true dValues1 dResults1Incl
        do! scan false dValues1 dResults1Excl

        let! dValues2 = DArray.scatterInBlob worker hValues2
        let! dResults2Incl = DArray.createInBlob worker n
        let! dResults2Excl = DArray.createInBlob worker n
        do! scan true dValues2 dResults2Incl
        do! scan false dValues2 dResults2Excl
        
        if verify then
            let hResults1 = hValues1 |> Array.scan (+) 0
            
            let hResults1Incl = Array.sub hResults1 1 n
            let! dResults1Incl = dResults1Incl.Gather()
            (hResults1Incl, dResults1Incl) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))

            let hResults1Excl = Array.sub hResults1 0 n
            let! dResults1Excl = dResults1Excl.Gather()
            (hResults1Excl, dResults1Excl) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))

            let hResults2 = hValues2 |> Array.scan (+) 0
            
            let hResults2Incl = Array.sub hResults2 1 n
            let! dResults2Incl = dResults2Incl.Gather()
            (hResults2Incl, dResults2Incl) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))

            let hResults2Excl = Array.sub hResults2 0 n
            let! dResults2Excl = dResults2Excl.Gather()
            (hResults2Excl, dResults2Excl) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))
        else do! PCalc.force() }

    let values1 n = Array.init n (fun _ -> 1)
    let values2 n = let rng = Random(2) in Array.init n (fun _ -> rng.Next(-100, 100))

    sizes |> Seq.iter (fun n -> (values1 n, values2 n) ||> test true |> PCalc.run)

    let n = 2097152
    let test = (values1 n, values2 n) ||> test false

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``sumscanner: float (2)`` () =
    let scanner = worker.LoadPModule(PArray.sumscanner()).Invoke
    let test verify eps (hValues1:float[]) (hValues2:float[]) = pcalc {
        if hValues1.Length <> hValues2.Length then failwith "inputs should be equal length"
        let n = hValues1.Length
        printfn "Testing size %d..." n
        let! scan = scanner n

        let! dValues1 = DArray.scatterInBlob worker hValues1
        let! dResults1Incl = DArray.createInBlob worker n
        let! dResults1Excl = DArray.createInBlob worker n
        do! scan true dValues1 dResults1Incl
        do! scan false dValues1 dResults1Excl

        let! dValues2 = DArray.scatterInBlob worker hValues2
        let! dResults2Incl = DArray.createInBlob worker n
        let! dResults2Excl = DArray.createInBlob worker n
        do! scan true dValues2 dResults2Incl
        do! scan false dValues2 dResults2Excl
        
        if verify then
            let hResults1 = hValues1 |> Array.scan (+) 0.0
            
            let hResults1Incl = Array.sub hResults1 1 n
            let! dResults1Incl = dResults1Incl.Gather()
            (hResults1Incl, dResults1Incl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            let hResults1Excl = Array.sub hResults1 0 n
            let! dResults1Excl = dResults1Excl.Gather()
            (hResults1Excl, dResults1Excl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            let hResults2 = hValues2 |> Array.scan (+) 0.0
            
            let hResults2Incl = Array.sub hResults2 1 n
            let! dResults2Incl = dResults2Incl.Gather()
            (hResults2Incl, dResults2Incl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            let hResults2Excl = Array.sub hResults2 0 n
            let! dResults2Excl = dResults2Excl.Gather()
            (hResults2Excl, dResults2Excl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
        else do! PCalc.force() }

    let eps = 1e-10
    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)

    sizes |> Seq.iter (fun n -> (values1 n, values2 n) ||> test true eps |> PCalc.run)

    let n = 2097152
    let test = (values1 n, values2 n) ||> test false eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``scan: sum<int>`` () =
    let scan = worker.LoadPModule(PArray.scan <@ fun () -> 0 @> <@ (+) @> <@ Util.identity @>).Invoke
    let test verify (hValues:int[]) = pcalc {
        let n = hValues.Length
        printfn "Testing size %d..." n

        let! dValues = DArray.scatterInBlob worker hValues
        let! dResultsIncl = scan true dValues
        let! dResultsExcl = scan false dValues

        if verify then
            let hResults = hValues |> Array.scan (+) 0
            
            let hResultsIncl = Array.sub hResults 1 n
            let! dResultsIncl = dResultsIncl.Gather()
            (hResultsIncl, dResultsIncl) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))

            let hResultsExcl = Array.sub hResults 0 n
            let! dResultsExcl = dResultsExcl.Gather()
            (hResultsExcl, dResultsExcl) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))
        else do! PCalc.force() }

    let values1 n = Array.init n (fun _ -> 1)
    let values2 n = Array.init n (fun _ -> -1)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.Next(-100, 100))

    sizes |> Seq.iter (fun n -> values1 n |> test true |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test true |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test true |> PCalc.run)

    let n = 2097152
    let test = values1 n |> test false

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``scan: sum<float>`` () =
    let scan = worker.LoadPModule(PArray.scan <@ fun () -> 0.0 @> <@ (+) @> <@ Util.identity @>).Invoke
    let test verify eps (hValues:float[]) = pcalc {
        let n = hValues.Length
        printfn "Testing size %d..." n

        let! dValues = DArray.scatterInBlob worker hValues
        let! dResultsIncl = scan true dValues
        let! dResultsExcl = scan false dValues

        if verify then
            let hResults = hValues |> Array.scan (+) 0.0
            
            let hResultsIncl = Array.sub hResults 1 n
            let! dResultsIncl = dResultsIncl.Gather()
            (hResultsIncl, dResultsIncl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            let hResultsExcl = Array.sub hResults 0 n
            let! dResultsExcl = dResultsExcl.Gather()
            (hResultsExcl, dResultsExcl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
        else do! PCalc.force() }

    let eps = 1e-10
    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = Array.init n (fun _ -> -1.0)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)

    sizes |> Seq.iter (fun n -> values1 n |> test true eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test true eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test true eps |> PCalc.run)

    let n = 2097152
    let test = values1 n |> test false eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``scanner: sum<int> (2)`` () =
    let scanner = worker.LoadPModule(PArray.scanner <@ fun () -> 0 @> <@ (+) @> <@ Util.identity @>).Invoke
    let test verify (hValues1:int[]) (hValues2:int[]) = pcalc {
        if hValues1.Length <> hValues2.Length then failwith "inputs should be equal length"
        let n = hValues1.Length
        printfn "Testing size %d..." n
        let! scan = scanner n

        let! dValues1 = DArray.scatterInBlob worker hValues1
        let! dResults1Incl = DArray.createInBlob worker n
        let! dResults1Excl = DArray.createInBlob worker n
        do! scan true dValues1 dResults1Incl
        do! scan false dValues1 dResults1Excl

        let! dValues2 = DArray.scatterInBlob worker hValues2
        let! dResults2Incl = DArray.createInBlob worker n
        let! dResults2Excl = DArray.createInBlob worker n
        do! scan true dValues2 dResults2Incl
        do! scan false dValues2 dResults2Excl
        
        if verify then
            let hResults1 = hValues1 |> Array.scan (+) 0
            
            let hResults1Incl = Array.sub hResults1 1 n
            let! dResults1Incl = dResults1Incl.Gather()
            (hResults1Incl, dResults1Incl) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))

            let hResults1Excl = Array.sub hResults1 0 n
            let! dResults1Excl = dResults1Excl.Gather()
            (hResults1Excl, dResults1Excl) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))

            let hResults2 = hValues2 |> Array.scan (+) 0
            
            let hResults2Incl = Array.sub hResults2 1 n
            let! dResults2Incl = dResults2Incl.Gather()
            (hResults2Incl, dResults2Incl) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))

            let hResults2Excl = Array.sub hResults2 0 n
            let! dResults2Excl = dResults2Excl.Gather()
            (hResults2Excl, dResults2Excl) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))
        else do! PCalc.force() }

    let values1 n = Array.init n (fun _ -> 1)
    let values2 n = let rng = Random(2) in Array.init n (fun _ -> rng.Next(-100, 100))

    sizes |> Seq.iter (fun n -> (values1 n, values2 n) ||> test true |> PCalc.run)

    let n = 2097152
    let test = (values1 n, values2 n) ||> test false

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``scanner: sum<float> (2)`` () =
    let scanner = worker.LoadPModule(PArray.scanner <@ fun () -> 0.0 @> <@ (+) @> <@ Util.identity @>).Invoke
    let test verify eps (hValues1:float[]) (hValues2:float[]) = pcalc {
        if hValues1.Length <> hValues2.Length then failwith "inputs should be equal length"
        let n = hValues1.Length
        printfn "Testing size %d..." n
        let! scan = scanner n

        let! dValues1 = DArray.scatterInBlob worker hValues1
        let! dResults1Incl = DArray.createInBlob worker n
        let! dResults1Excl = DArray.createInBlob worker n
        do! scan true dValues1 dResults1Incl
        do! scan false dValues1 dResults1Excl

        let! dValues2 = DArray.scatterInBlob worker hValues2
        let! dResults2Incl = DArray.createInBlob worker n
        let! dResults2Excl = DArray.createInBlob worker n
        do! scan true dValues2 dResults2Incl
        do! scan false dValues2 dResults2Excl
        
        if verify then
            let hResults1 = hValues1 |> Array.scan (+) 0.0
            
            let hResults1Incl = Array.sub hResults1 1 n
            let! dResults1Incl = dResults1Incl.Gather()
            (hResults1Incl, dResults1Incl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            let hResults1Excl = Array.sub hResults1 0 n
            let! dResults1Excl = dResults1Excl.Gather()
            (hResults1Excl, dResults1Excl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            let hResults2 = hValues2 |> Array.scan (+) 0.0
            
            let hResults2Incl = Array.sub hResults2 1 n
            let! dResults2Incl = dResults2Incl.Gather()
            (hResults2Incl, dResults2Incl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            let hResults2Excl = Array.sub hResults2 0 n
            let! dResults2Excl = dResults2Excl.Gather()
            (hResults2Excl, dResults2Excl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
        else do! PCalc.force() }

    let eps = 1e-10
    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)

    sizes |> Seq.iter (fun n -> (values1 n, values2 n) ||> test true eps |> PCalc.run)

    let n = 2097152
    let test = (values1 n, values2 n) ||> test false eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``scan: sum squared<float>`` () =
    let scan = worker.LoadPModule(PArray.scan <@ fun () -> 0.0 @> <@ (+) @> <@ fun x -> x * x @>).Invoke
    let test verify eps (hValues:float[]) = pcalc {
        let n = hValues.Length
        printfn "Testing size %d..." n

        let! dValues = DArray.scatterInBlob worker hValues
        let! dResultsIncl = scan true dValues
        let! dResultsExcl = scan false dValues

        if verify then
            let hResults = hValues |> Array.map (fun x -> x * x) |> Array.scan (+) 0.0
            
            let hResultsIncl = Array.sub hResults 1 n
            let! dResultsIncl = dResultsIncl.Gather()
            (hResultsIncl, dResultsIncl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            let hResultsExcl = Array.sub hResults 0 n
            let! dResultsExcl = dResultsExcl.Gather()
            (hResultsExcl, dResultsExcl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
        else do! PCalc.force() }

    let eps = 1e-7
    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = Array.init n (fun _ -> -1.0)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)

    sizes |> Seq.iter (fun n -> values1 n |> test true eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test true eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test true eps |> PCalc.run)

    let n = 2097152
    let test = values1 n |> test false eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``scan: max<float>`` () =
    let scan = worker.LoadPModule(PArray.scan <@ fun () -> Double.NegativeInfinity @> <@ max @> <@ Util.identity @>).Invoke
    let test verify eps (hValues:float[]) = pcalc {
        let n = hValues.Length
        printfn "Testing size %d..." n

        let! dValues = DArray.scatterInBlob worker hValues
        let! dResultsIncl = scan true dValues
        let! dResultsExcl = scan false dValues

        if verify then
            let hResults = hValues |> Array.scan max Double.NegativeInfinity
            
            let hResultsIncl = Array.sub hResults 1 n
            let! dResultsIncl = dResultsIncl.Gather()
            (hResultsIncl, dResultsIncl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            let hResultsExcl = Array.sub hResults 0 n
            let! dResultsExcl = dResultsExcl.Gather()
            (hResultsExcl, dResultsExcl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
        else do! PCalc.force() }

    let eps = 1e-10
    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = Array.init n (fun _ -> -1.0)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)

    sizes |> Seq.iter (fun n -> values1 n |> test true eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test true eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test true eps |> PCalc.run)

    let n = 2097152
    let test = values1 n |> test false eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``scan: min<float>`` () =
    let scan = worker.LoadPModule(PArray.scan <@ fun () -> Double.PositiveInfinity @> <@ min @> <@ Util.identity @>).Invoke
    let test verify eps (hValues:float[]) = pcalc {
        let n = hValues.Length
        printfn "Testing size %d..." n

        let! dValues = DArray.scatterInBlob worker hValues
        let! dResultsIncl = scan true dValues
        let! dResultsExcl = scan false dValues

        if verify then
            let hResults = hValues |> Array.scan min Double.PositiveInfinity
            
            let hResultsIncl = Array.sub hResults 1 n
            let! dResultsIncl = dResultsIncl.Gather()
            (hResultsIncl, dResultsIncl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            let hResultsExcl = Array.sub hResults 0 n
            let! dResultsExcl = dResultsExcl.Gather()
            (hResultsExcl, dResultsExcl) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
        else do! PCalc.force() }

    let eps = 1e-10
    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = Array.init n (fun _ -> -1.0)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)

    sizes |> Seq.iter (fun n -> values1 n |> test true eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values2 n |> test true eps |> PCalc.run)
    sizes |> Seq.iter (fun n -> values3 n |> test true eps |> PCalc.run)

    let n = 2097152
    let test = values1 n |> test false eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
