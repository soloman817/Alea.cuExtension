module Test.Alea.CUDA.Extension.Sobol

open System.IO
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

let worker = getDefaultWorker()

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

    calc |> PCalc.runWithDiagnoser(Diagnoser.All(1)) |> ignore
    let _, loggers = calc |> PCalc.runWithTimingLogger
    loggers.["default"].DumpLogs()

let sobolIterUInt32 = worker.LoadPModule(PRandom.sobolIter <@ Sobol.toUInt32 @>).Invoke
let sobolTestUInt32 verify dimensions vectors iters = pcalc {
    let! logger = PCalc.tlogger("verify")
    let! sobolIter = sobolIterUInt32 dimensions
    let sobolIter = sobolIter vectors

    let! dOutput = DArray.CreateInBlob(worker, dimensions * vectors)

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

let [<Test>] ``UInt32: [V] 32 x 256 5`` () = sobolTestUInt32 true 32 256 5 |> PCalc.run
let [<Test>] ``Uint32: [V] 32 x 256 5`` () = sobolTestUInt32 true 32 256 5 |> PCalc.run
let [<Test>] ``Uint32: [V] 32 x 4096 5`` () = sobolTestUInt32 true 32 4096 5 |> PCalc.run
let [<Test>] ``Uint32: [V] 32 x 65536 5`` () = sobolTestUInt32 true 32 65536 5 |> PCalc.run
let [<Test>] ``Uint32: [_] 32 x 1048576 5`` () = sobolTestUInt32 false 32 1048576 5 |> PCalc.run
let [<Test>] ``Uint32: [V] 1024 x 256 5`` () = sobolTestUInt32 true 1024 256 5 |> PCalc.run
let [<Test>] ``Uint32: [V] 1024 x 4096 5`` () = sobolTestUInt32 true 1024 4096 5 |> PCalc.run
let [<Test>] ``Uint32: [_] 1024 x 65536 5`` () = sobolTestUInt32 false 1024 65536 5 |> PCalc.run
let [<Test>] ``Uint32: [V] 4096 x 256 5`` () = sobolTestUInt32 true 4096 256 5 |> PCalc.run
let [<Test>] ``Uint32: [V] 4096 x 4096 5`` () = sobolTestUInt32 true 4096 4096 5 |> PCalc.run
let [<Test>] ``Uint32: [_] 4096 x 8192 5`` () = sobolTestUInt32 false 4096 8192 5 |> PCalc.run
let [<Test>] ``Uint32: [_] 4096 x 16384 5`` () = sobolTestUInt32 false 4096 16384 5 |> PCalc.run

let [<Test>] ``Uint32: [D] 1024 x 4096 3`` () =
    sobolTestUInt32 true 1024 4096 3 |> PCalc.runWithDiagnoser(Diagnoser.All(1))
    let _, loggers = sobolTestUInt32 true 1024 4096 3 |> PCalc.runWithTimingLogger
    loggers.["default"].DumpLogs()
    loggers.["verify"].DumpLogs()

let sobolIterFloat32 = worker.LoadPModule(PRandom.sobolIter <@ Sobol.toFloat32 @>).Invoke
let sobolTestFloat32 verify dimensions vectors iters = pcalc {
    let! logger = PCalc.tlogger("verify")
    let! sobolIter = sobolIterFloat32 dimensions
    let sobolIter = sobolIter vectors

    let! dOutput = DArray.CreateInBlob(worker, dimensions * vectors)

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

let [<Test>] ``Float32: [V] 1024 x 256 5`` () = sobolTestFloat32 true 1024 256 5 |> PCalc.run
let [<Test>] ``Float32: [V] 1024 x 4096 5`` () = sobolTestFloat32 true 1024 4096 5 |> PCalc.run

let [<Test>] ``Float32: [D] 1024 x 4096 3`` () =
    sobolTestFloat32 true 1024 4096 3 |> PCalc.runWithDiagnoser(Diagnoser.All(1))
    let _, loggers = sobolTestFloat32 true 1024 4096 3 |> PCalc.runWithTimingLogger
    loggers.["default"].DumpLogs()
    loggers.["verify"].DumpLogs()

let sobolIterFloat64 = worker.LoadPModule(PRandom.sobolIter <@ Sobol.toFloat64 @>).Invoke
let sobolTestFloat64 verify dimensions vectors iters = pcalc {
    let! logger = PCalc.tlogger("verify")
    let! sobolIter = sobolIterFloat64 dimensions
    let sobolIter = sobolIter vectors

    let! dOutput = DArray.CreateInBlob(worker, dimensions * vectors)

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

let [<Test>] ``Float64: [V] 1024 x 256 5`` () = sobolTestFloat64 true 1024 256 5 |> PCalc.run
let [<Test>] ``Float64: [V] 1024 x 4096 5`` () = sobolTestFloat64 true 1024 4096 5 |> PCalc.run

let [<Test>] ``Float64: [D] 1024 x 4096 3`` () =
    sobolTestFloat64 true 1024 4096 3 |> PCalc.runWithDiagnoser(Diagnoser.All(1))
    let _, loggers = sobolTestFloat64 true 1024 4096 3 |> PCalc.runWithTimingLogger
    loggers.["default"].DumpLogs()
    loggers.["verify"].DumpLogs()
