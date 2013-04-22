module Test.Alea.CUDA.Extension.Random.Sobol

open System.IO
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Random
open Test.Alea.CUDA.Extension

let worker = getDefaultWorker()

[<Test>]
let stream () =
    let sobolRng = worker.LoadPModule(PRandom.sobolRng <@ Sobol.toFloat64 @>).Invoke
    let reducer = worker.LoadPModule(PArray.reducer <@ fun () -> 0.0 @> <@ (+) @> <@ Util.identity @>).Invoke

    let calc dimensions vectors (streams:Stream[]) = pcalc {
        let n = dimensions * vectors
        let ns = streams.Length
        let! sobol = sobolRng dimensions
        let sobolIter = sobol vectors
        let! reduce = reducer n

        //let streams = Array.zeroCreate<Stream> ns
        let dmems = Array.zeroCreate<DArray<float>> ns
        let dresults = Array.zeroCreate<DScalar<float>> ns
        let hresults = Array.zeroCreate<float> ns

        // create streams (if you use this way, the stream will be disposed after the pcalc finished)
        //for i = 0 to ns - 1 do
        //    let! stream = DStream.create worker
        //    streams.[i] <- stream

        // create working memory
        for i = 0 to ns - 1 do
            let! dmem = DArray.createInBlob worker n
            let! dresult = DScalar.createInBlob worker
            dmems.[i] <- dmem
            dresults.[i] <- dresult

        // sobol
        for i = 0 to ns - 1 do
            let offset = i * vectors
            do! PCalc.stream streams.[i]
            do! sobolIter offset dmems.[i]

        // reduce
        for i = 0 to ns - 1 do
            do! PCalc.stream streams.[i]
            do! reduce dmems.[i] dresults.[i]

        // gather
        for i = 0 to ns - 1 do
            let! hresult = dresults.[i].Gather()
            let hresult = hresult / float(n)
            hresults.[i] <- hresult

        return hresults |> Array.average }

    let calc' dimensions vectors ns = pcalc {
        let n = dimensions * vectors
        let! sobol = sobolRng dimensions
        let sobolIter = sobol (vectors * ns)
        let! reduce = reducer (n * ns)

        let! dmem = DArray.createInBlob worker (n * ns)
        let! dresult = DScalar.createInBlob worker
        do! sobolIter 0 dmem
        do! reduce dmem dresult
        let! hresult = dresult.Gather()
        let hresult = hresult / float(n * ns)

        return hresult }

    let ns = 4
    let streams = Array.init ns (fun _ -> worker.CreateStream())

    let calc = calc 2048 4096 streams
    let calc' = calc' 2048 4096 ns

    printfn "calc  = %f" (calc |> PCalc.run)
    printfn "calc' = %f" (calc' |> PCalc.run)

    let _, timings = calc |> PCalc.runWithTiming 10 in printfn "t(calc)  = %.6f ms" (timings |> Array.average)
    let _, timings = calc' |> PCalc.runWithTiming 10 in printfn "t(calc') = %.6f ms" (timings |> Array.average)

    let _, loggers = calc |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = calc |> PCalc.runWithKernelTiming 10 in ktc.Dump()

    let _, loggers = calc' |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = calc' |> PCalc.runWithKernelTiming 10 in ktc.Dump()

    calc |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1)) |> ignore
    calc' |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1)) |> ignore

    streams |> Array.iter (fun s -> s.Dispose())

[<Test>]
let simple() =
    let dimensions = 4096
    let vectors = 4096
    let offset = 234

    let dSobol = worker.LoadPModule(PRandom.sobol <@ Sobol.toUInt32 @>).Invoke
    let hSobol = SobolGold.Sobol(dimensions, offset)

    let calc = pcalc {
        let! dOutput = dSobol dimensions vectors offset
        return! dOutput.Gather() }
    let dOutput = calc |> PCalc.run
    
    // verify
    let hOutput = Array.init vectors (fun _ -> hSobol.NextPoint) |> Array.concat
    let dOutput = dOutput |> SobolGold.reorderPoints dimensions vectors
    (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))

    calc |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1)) |> ignore
    let _, loggers = calc |> PCalc.runWithTimingLogger
    loggers.["default"].DumpLogs()

let rngUInt32 = worker.LoadPModule(PRandom.sobolRng <@ Sobol.toUInt32 @>).Invoke
let testUInt32 verify dimensions vectors iters = pcalc {
    let! logger = PCalc.tlogger("verify")
    let! sobol = rngUInt32 dimensions
    let sobolIter = sobol vectors

    let! dOutput = DArray.createInBlob worker (dimensions * vectors)

    for i = 0 to iters - 1 do
        let offset = i * vectors
        do! sobolIter offset dOutput

        if verify then
            logger.Log(sprintf "generate houtput on offset %d" offset)
            let hSobol = SobolGold.Sobol(dimensions, offset)
            let hOutput = Array.init vectors (fun _ -> hSobol.NextPoint) |> Array.concat
            logger.Log("gather doutput")
            let! dOutput = dOutput.Gather()
            logger.Log("reorder doutput")
            let dOutput = dOutput |> SobolGold.reorderPoints dimensions vectors
            logger.Log("verify doutput vs houtput")
            (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))
            logger.Touch() }

let [<Test>] ``UInt32: [V] 32 x 256 5`` () = testUInt32 true 32 256 5 |> PCalc.run
let [<Test>] ``UInt32: [V] 32 x 4096 5`` () = testUInt32 true 32 4096 5 |> PCalc.run
let [<Test>] ``UInt32: [V] 32 x 65536 5`` () = testUInt32 true 32 65536 5 |> PCalc.run
let [<Test>] ``UInt32: [_] 32 x 1048576 5`` () = testUInt32 false 32 1048576 5 |> PCalc.run
let [<Test>] ``UInt32: [V] 1024 x 256 5`` () = testUInt32 true 1024 256 5 |> PCalc.run
let [<Test>] ``UInt32: [V] 1024 x 4096 5`` () = testUInt32 true 1024 4096 5 |> PCalc.run
let [<Test>] ``UInt32: [_] 1024 x 65536 5`` () = testUInt32 false 1024 65536 5 |> PCalc.run
let [<Test>] ``UInt32: [V] 4096 x 256 5`` () = testUInt32 true 4096 256 5 |> PCalc.run
let [<Test>] ``UInt32: [V] 4096 x 4096 5`` () = testUInt32 true 4096 4096 5 |> PCalc.run
let [<Test>] ``UInt32: [_] 4096 x 8192 5`` () = testUInt32 false 4096 8192 5 |> PCalc.run
let [<Test>] ``UInt32: [_] 4096 x 16384 5`` () = testUInt32 false 4096 16384 5 |> PCalc.run

let [<Test>] ``UInt32: [D] 1024 x 4096 3`` () =
    testUInt32 true 1024 4096 3 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = testUInt32 true 1024 4096 3 |> PCalc.runWithTimingLogger
    loggers.["default"].DumpLogs()
    loggers.["verify"].DumpLogs()

let rngFloat32 = worker.LoadPModule(PRandom.sobolRng <@ Sobol.toFloat32 @>).Invoke
let testFloat32 verify dimensions vectors iters = pcalc {
    let! logger = PCalc.tlogger("verify")
    let! sobol = rngFloat32 dimensions
    let sobolIter = sobol vectors

    let! dOutput = DArray.createInBlob worker (dimensions * vectors)

    for i = 0 to iters - 1 do
        let offset = i * vectors
        do! sobolIter offset dOutput

        if verify then
            logger.Log(sprintf "generate houtput on offset %d" offset)
            let hSobol = SobolGold.Sobol(dimensions, offset)
            let hOutput = Array.init vectors (fun _ -> hSobol.NextPoint |> Array.map Sobol.toFloat32) |> Array.concat
            logger.Log("gather doutput")
            let! dOutput = dOutput.Gather()
            logger.Log("reorder doutput")
            let dOutput = dOutput |> SobolGold.reorderPoints dimensions vectors
            logger.Log("verify doutput vs houtput")
            (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))
            logger.Touch() }

let [<Test>] ``Float32: [V] 1024 x 256 5`` () = testFloat32 true 1024 256 5 |> PCalc.run
let [<Test>] ``Float32: [V] 1024 x 4096 5`` () = testFloat32 true 1024 4096 5 |> PCalc.run

let [<Test>] ``Float32: [D] 1024 x 4096 3`` () =
    testFloat32 true 1024 4096 3 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = testFloat32 true 1024 4096 3 |> PCalc.runWithTimingLogger
    loggers.["default"].DumpLogs()
    loggers.["verify"].DumpLogs()

let rngFloat64 = worker.LoadPModule(PRandom.sobolRng <@ Sobol.toFloat64 @>).Invoke
let testFloat64 verify dimensions vectors iters = pcalc {
    let! logger = PCalc.tlogger("verify")
    let! sobol = rngFloat64 dimensions
    let sobolIter = sobol vectors

    let! dOutput = DArray.createInBlob worker (dimensions * vectors)

    for i = 0 to iters - 1 do
        let offset = i * vectors
        do! sobolIter offset dOutput

        if verify then
            logger.Log(sprintf "generate houtput on offset %d" offset)
            let hSobol = SobolGold.Sobol(dimensions, offset)
            let hOutput = Array.init vectors (fun _ -> hSobol.NextPoint |> Array.map Sobol.toFloat64) |> Array.concat
            logger.Log("gather doutput")
            let! dOutput = dOutput.Gather()
            logger.Log("reorder doutput")
            let dOutput = dOutput |> SobolGold.reorderPoints dimensions vectors
            logger.Log("verify doutput vs houtput")
            (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))
            logger.Touch() }

let [<Test>] ``Float64: [V] 1024 x 256 5`` () = testFloat64 true 1024 256 5 |> PCalc.run
let [<Test>] ``Float64: [V] 1024 x 4096 5`` () = testFloat64 true 1024 4096 5 |> PCalc.run

let [<Test>] ``Float64: [D] 1024 x 4096 3`` () =
    testFloat64 true 1024 4096 3 |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = testFloat64 true 1024 4096 3 |> PCalc.runWithTimingLogger
    loggers.["default"].DumpLogs()
    loggers.["verify"].DumpLogs()
