module Test.Alea.CUDA.Extension.PCalc

open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

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
let ``DArray.subView``() =
    let calc = pcalc {
        let input = [| 1; 2; 3; 4; 5; 6 |]
        let! input = DArray.scatterInBlob worker input
        let output1 = DArray.subView input 0 3
        let output2 = DArray.subView input 3 3
        let! output1 = output1.Gather()
        let! output2 = output2.Gather()
        return output1, output2 }
    let output1, output2 = calc |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    printfn "%A" output1
    printfn "%A" output2
    Assert.That(output1.Length = 3)
    Assert.That(output2.Length = 3)
    Assert.AreEqual(output1.[0], 1)
    Assert.AreEqual(output1.[1], 2)
    Assert.AreEqual(output1.[2], 3)
    Assert.AreEqual(output2.[0], 4)
    Assert.AreEqual(output2.[1], 5)
    Assert.AreEqual(output2.[2], 6)
