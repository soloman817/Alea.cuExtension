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
            dmems.[i] <- dmem

        // sobol
        for i = 0 to ns - 1 do
            let offset = i * vectors
            do! PCalc.stream streams.[i]
            do! sobolIter offset dmems.[i]

        // reduce
        for i = 0 to ns - 1 do
            do! PCalc.stream streams.[i]
            let! dresult = reduce dmems.[i]
            dresults.[i] <- dresult

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
        do! sobolIter 0 dmem
        let! dresult = reduce dmem
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
    
