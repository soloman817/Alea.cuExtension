module Test.Alea.CUDA.Extension.SegmentedScan

open System
open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.Interop.CUDA
open Alea.CUDA
open Alea.CUDA.Extension

let worker = getDefaultWorker()

let sizes = [| 12; 2597152; 128; 4002931; 511; 1024; 8191; 1200; 4096; 5000; 8192; 12; 8193; 9000; 10000; 2097152 |]

let zeroCreate : int -> PCalc<DArray<int>> = worker.LoadPModule(PArray.zeroCreate()).Invoke

let createFlags (valuess:'T[][]) = pcalc {
    let ns = valuess |> Array.map (fun x -> x.Length)
    let n = ns |> Array.sum
    let headIndices = ns |> Array.scan (+) 0

    let! flags = zeroCreate n
    let setIndices (hint:ActionHint) =
        fun () ->
            for i = 0 to headIndices.Length - 2 do // bypass the last one becuase it is the cpu scan algorithm
                let ptr = flags.Ptr + headIndices.[i]
                cuSafeCall(cuMemsetD32Async(ptr.Handle, 1u, 1n, hint.Stream.Handle))
        |> worker.Eval
    do! PCalc.action setIndices

    return flags }

let testSegScanInt32 (scan:bool -> DArray<int> -> DArray<int> -> PCalc<DArray<int>>) (init:int) (op:int -> int -> int) (transf:int -> int) =
    let test verify (hValuess:int[][]) = pcalc {
        let! dValues = DArray.scatterInBlob worker (hValuess |> Array.concat)
        let! dFlags = createFlags hValuess
        let! dResultsIncl = scan true dValues dFlags
        let! dResultsExcl = scan false dValues dFlags

        match verify with
        | true ->
            let hResultss = hValuess |> Array.map (fun hValues -> hValues |> Array.map transf |> Array.scan op init)

            // check inclusive
            let hResults = hResultss |> Array.map (fun x -> Array.sub x 1 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsIncl.Gather()
            (hResults, dResults) ||> Array.iteri2 (fun i h d -> Assert.AreEqual(d, h))

            // check exclusive
            let hResults = hResultss |> Array.map (fun x -> Array.sub x 0 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsExcl.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))
        | false -> do! PCalc.force() }

    let valuess1() = sizes |> Array.map (fun n -> Array.init n (fun _ -> 1))
    let valuess2() = let rng = Random(2) in sizes |> Array.map (fun n -> Array.init n (fun _ -> rng.Next(-100, 100)))

    test true (valuess1()) |> PCalc.run
    test true (valuess2()) |> PCalc.run

    let test = test false (valuess2())
    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

let testSegScanFloat32 (scan:bool -> DArray<float32> -> DArray<int> -> PCalc<DArray<float32>>) (eps:float) (init:float32) (op:float32 -> float32 -> float32) (transf:float32 -> float32) =
    let test (verify:float option) (hValuess:float32[][]) = pcalc {
        let! dValues = DArray.scatterInBlob worker (hValuess |> Array.concat)
        let! dFlags = createFlags hValuess
        let! dResultsIncl = scan true dValues dFlags
        let! dResultsExcl = scan false dValues dFlags
        
        match verify with
        | Some(eps) ->
            let hResultss = hValuess |> Array.map (fun hValues -> hValues |> Array.map transf |> Array.scan op init)

            // check inclusive
            let hResults = hResultss |> Array.map (fun x -> Array.sub x 1 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsIncl.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            // check exclusive
            let hResults = hResultss |> Array.map (fun x -> Array.sub x 0 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsExcl.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
        | None -> do! PCalc.force() }

    let eps = Some eps
    let valuess1() = sizes |> Array.map (fun n -> Array.init n (fun _ -> 1.0f))
    let valuess2() = let rng = Random(2) in sizes |> Array.map (fun n -> Array.init n (fun _ -> (rng.NextDouble() - 0.5) |> float32))

    test eps (valuess1()) |> PCalc.run
    test eps (valuess2()) |> PCalc.run

    let test = test None (valuess2())
    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

let testSegScanFloat64 (scan:bool -> DArray<float> -> DArray<int> -> PCalc<DArray<float>>) (eps:float) (init:float) (op:float -> float -> float) (transf:float -> float) =
    let test (verify:float option) (hValuess:float[][]) = pcalc {
        let! dValues = DArray.scatterInBlob worker (hValuess |> Array.concat)
        let! dFlags = createFlags hValuess
        let! dResultsIncl = scan true dValues dFlags
        let! dResultsExcl = scan false dValues dFlags
        
        match verify with
        | Some(eps) ->
            let hResultss = hValuess |> Array.map (fun hValues -> hValues |> Array.map transf |> Array.scan op init)

            // check inclusive
            let hResults = hResultss |> Array.map (fun x -> Array.sub x 1 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsIncl.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            // check exclusive
            let hResults = hResultss |> Array.map (fun x -> Array.sub x 0 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsExcl.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
        | None -> do! PCalc.force() }

    let eps = Some eps
    let valuess1() = sizes |> Array.map (fun n -> Array.init n (fun _ -> 1.0))
    let valuess2() = let rng = Random(2) in sizes |> Array.map (fun n -> Array.init n (fun _ -> rng.NextDouble() - 0.5))

    test eps (valuess1()) |> PCalc.run
    test eps (valuess2()) |> PCalc.run

    let test = test None (valuess2())
    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

let testSegScannerFloat64 (scanner:int -> PCalc<bool -> DArray<float> -> DArray<int> -> DArray<float> -> PCalc<unit>>) (eps:float) (init:float) (op:float -> float -> float) (transf:float -> float) =
    let test (verify:float option) (hValuess1:float[][]) (hValuess2:float[][]) = pcalc {
        let n1 = hValuess1 |> Array.map (fun hValues -> hValues.Length) |> Array.sum
        let n2 = hValuess2 |> Array.map (fun hValues -> hValues.Length) |> Array.sum
        if n1 <> n2 then failwith "must equal length"
        let n = n1
        let! scan = scanner n

        let! dValues1 = DArray.scatterInBlob worker (hValuess1 |> Array.concat)
        let! dFlags1 = createFlags hValuess1
        let! dResultsIncl1 = DArray.createInBlob worker n
        let! dResultsExcl1 = DArray.createInBlob worker n
        do! scan true dValues1 dFlags1 dResultsIncl1
        do! scan false dValues1 dFlags1 dResultsExcl1
        
        let! dValues2 = DArray.scatterInBlob worker (hValuess2 |> Array.concat)
        let! dFlags2 = createFlags hValuess2
        let! dResultsIncl2 = DArray.createInBlob worker n
        let! dResultsExcl2 = DArray.createInBlob worker n
        do! scan true dValues2 dFlags2 dResultsIncl2
        do! scan false dValues2 dFlags2 dResultsExcl2

        match verify with
        | Some(eps) ->
            let hResultss1 = hValuess1 |> Array.map (fun hValues -> hValues |> Array.map transf |> Array.scan op init)

            // check inclusive
            let hResults = hResultss1 |> Array.map (fun x -> Array.sub x 1 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsIncl1.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            // check exclusive
            let hResults = hResultss1 |> Array.map (fun x -> Array.sub x 0 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsExcl1.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            let hResultss2 = hValuess2 |> Array.map (fun hValues -> hValues |> Array.map transf |> Array.scan op init)

            // check inclusive
            let hResults = hResultss2 |> Array.map (fun x -> Array.sub x 1 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsIncl2.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            // check exclusive
            let hResults = hResultss2 |> Array.map (fun x -> Array.sub x 0 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsExcl2.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
        | None -> do! PCalc.force() }

    let eps = Some eps
    let valuess1() = sizes |> Array.map (fun n -> Array.init n (fun _ -> 1.0))
    let valuess2() = let rng = Random(2) in sizes |> Array.map (fun n -> Array.init n (fun _ -> rng.NextDouble() - 0.5))

    test eps (valuess1()) (valuess2()) |> PCalc.run

    let test = test None (valuess1()) (valuess2())
    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

let [<Test>] ``sumsegscan: int``() = let scan = worker.LoadPModule(PArray.sumsegscan()).Invoke in testSegScanInt32 scan 0 (+) Util.identity
let [<Test>] ``segscan: sum<int>``() = let scan = worker.LoadPModule(PArray.segscan <@ fun () -> 0 @> <@ (+) @> <@ Util.identity @>).Invoke in testSegScanInt32 scan 0 (+) Util.identity
let [<Test>] ``segscan: min<int>``() = let scan = worker.LoadPModule(PArray.segscan <@ fun () -> Int32.MaxValue @> <@ min @> <@ Util.identity @>).Invoke in testSegScanInt32 scan Int32.MaxValue min Util.identity
let [<Test>] ``segscan: max<int>``() = let scan = worker.LoadPModule(PArray.segscan <@ fun () -> Int32.MinValue @> <@ max @> <@ Util.identity @>).Invoke in testSegScanInt32 scan Int32.MinValue max Util.identity
let [<Test>] ``segscan: sum<square<int>>``() = let scan = worker.LoadPModule(PArray.segscan <@ fun () -> 0 @> <@ (+) @> <@ fun x -> x * x @>).Invoke in testSegScanInt32 scan 0 (+) (fun x -> x * x)

let [<Test>] ``sumsegscan: float32``() = let scan = worker.LoadPModule(PArray.sumsegscan()).Invoke in testSegScanFloat32 scan 1e-1 0.0f (+) Util.identity
let [<Test>] ``segscan: sum<float32>``() = let scan = worker.LoadPModule(PArray.segscan <@ fun () -> 0.0f @> <@ (+) @> <@ Util.identity @>).Invoke in testSegScanFloat32 scan 1e-1 0.0f (+) Util.identity
let [<Test>] ``segscan: min<float32>``() = let scan = worker.LoadPModule(PArray.segscan <@ fun () -> Single.PositiveInfinity @> <@ min @> <@ Util.identity @>).Invoke in testSegScanFloat32 scan 1e-14 Single.PositiveInfinity min Util.identity
let [<Test>] ``segscan: max<float32>``() = let scan = worker.LoadPModule(PArray.segscan <@ fun () -> Single.NegativeInfinity @> <@ max @> <@ Util.identity @>).Invoke in testSegScanFloat32 scan 1e-14 Single.NegativeInfinity max Util.identity
let [<Test>] ``segscan: sum<square<float32>>``() = let scan = worker.LoadPModule(PArray.segscan <@ fun () -> 0.0f @> <@ (+) @> <@ fun x -> x * x @>).Invoke in testSegScanFloat32 scan 1e3 0.0f (+) (fun x -> x * x)

let [<Test>] ``sumsegscan: float``() = let scan = worker.LoadPModule(PArray.sumsegscan()).Invoke in testSegScanFloat64 scan 1e-10 0.0 (+) Util.identity
let [<Test>] ``segscan: sum<float>``() = let scan = worker.LoadPModule(PArray.segscan <@ fun () -> 0.0 @> <@ (+) @> <@ Util.identity @>).Invoke in testSegScanFloat64 scan 1e-10 0.0 (+) Util.identity
let [<Test>] ``segscan: min<float>``() = let scan = worker.LoadPModule(PArray.segscan <@ fun () -> Double.PositiveInfinity @> <@ min @> <@ Util.identity @>).Invoke in testSegScanFloat64 scan 1e-14 Double.PositiveInfinity min Util.identity
let [<Test>] ``segscan: max<float>``() = let scan = worker.LoadPModule(PArray.segscan <@ fun () -> Double.NegativeInfinity @> <@ max @> <@ Util.identity @>).Invoke in testSegScanFloat64 scan 1e-14 Double.NegativeInfinity max Util.identity
let [<Test>] ``segscan: sum<square<float>>``() = let scan = worker.LoadPModule(PArray.segscan <@ fun () -> 0.0 @> <@ (+) @> <@ fun x -> x * x @>).Invoke in testSegScanFloat64 scan 1e-7 0.0 (+) (fun x -> x * x)

let [<Test>] ``sumsegscanner: float``() = let scan = worker.LoadPModule(PArray.sumsegscanner()).Invoke in testSegScannerFloat64 scan 1e-10 0.0 (+) Util.identity
let [<Test>] ``segscanner: sum<float>``() = let scan = worker.LoadPModule(PArray.segscanner <@ fun () -> 0.0 @> <@ (+) @> <@ Util.identity @>).Invoke in testSegScannerFloat64 scan 1e-10 0.0 (+) Util.identity
let [<Test>] ``segscanner: min<float>``() = let scan = worker.LoadPModule(PArray.segscanner <@ fun () -> Double.PositiveInfinity @> <@ min @> <@ Util.identity @>).Invoke in testSegScannerFloat64 scan 1e-14 Double.PositiveInfinity min Util.identity
let [<Test>] ``segscanner: max<float>``() = let scan = worker.LoadPModule(PArray.segscanner <@ fun () -> Double.NegativeInfinity @> <@ max @> <@ Util.identity @>).Invoke in testSegScannerFloat64 scan 1e-14 Double.NegativeInfinity max Util.identity
let [<Test>] ``segscanner: sum<square<float>>``() = let scan = worker.LoadPModule(PArray.segscanner <@ fun () -> 0.0 @> <@ (+) @> <@ fun x -> x * x @>).Invoke in testSegScannerFloat64 scan 1e-7 0.0 (+) (fun x -> x * x)

