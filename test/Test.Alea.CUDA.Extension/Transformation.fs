module Test.Alea.CUDA.Extension.Transformation

open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

[<Test>]
let test () =
    let worker = getDefaultWorker()

    let pfunct = cuda {
        let! map1 = PArray.map <@ fun x -> x + 1.0 @>
        let! map2 = PArray.map <@ fun x -> x - 1.0 @>
        let! map3 = PArray.map <@ fun x -> x + 10.0 @>

        return PFunc(fun (m:Module) (i:float[]) ->
            let worker = m.Worker
            let map1 = map1.Apply m
            let map2 = map2.Apply m
            let map3 = map3.Apply m
            pcalc {
                let! i = DArray.CreateInBlob(worker, i)

                let! o1 = i |> map1
                let! o2 = i |> map2
                let! o3 = i |> map3

                let! o1 = o1.ToHost()
                let! o2 = o2.ToHost()
                let! o3 = o3.ToHost()

                return o1, o2, o3 } ) }

    let pfunc = worker.LoadPModule(pfunct).Invoke

    let i = Array.init (1<<<22) (fun i -> float(i))
    let o1, o2, o3 = pfunc i |> PCalc.runWithDiagnoser (Diagnoser.DefaultAll)
//    printfn "%A" i
//    printfn "%A" o1
//    printfn "%A" o2
//    printfn "%A" o3

    let _, loggers = pfunc i |> PCalc.runWithTimingLogger
    loggers.["default"].Dump()

    let _, loggers = pfunc i |> PCalc.runInWorkerWithTimingLogger worker
    loggers.["default"].Dump()

    let _, timing = pfunc i |> PCalc.runWithTiming 10
    printfn "%.3f ms" timing

    let _, timing = pfunc i |> PCalc.runInWorkerWithTiming worker 10
    printfn "%.3f ms" timing

    let _, tc = pfunc i |> PCalc.runWithKernelTiming 10
    printfn "%A" tc

    let _, tc = pfunc i |> PCalc.runInWorkerWithKernelTiming worker 10
    printfn "%A" tc


//[<Test>]
//let ``transform (x:float) -> log x``() =
//    let worker = getDefaultWorker()
//    use transformm = worker.LoadPModule(PArray.transform <@ log @>)
//    let transform = transformm.Invoke
//    let transform (input:float[]) =
//         use data = PArray.Create(worker, input)
//         transform data data
//         data.ToHost()
//
//    let n = 1 <<< 22
//    let input = Array.init n (fun _ -> rng.NextDouble())
//    let hOutput = input |> Array.map log
//    let dOutput = input |> transform
//    (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(1e-10)))
//
//[<Test>]
//let ``transform (x:float32) -> log x``() =
//    let worker = getDefaultWorker()
//    use transformm = worker.LoadPModule(PArray.transform <@ log @>)
//    let transform = transformm.Invoke
//    let transform (input:float32[]) =
//        use data = PArray.Create(worker, input)
//        transform data data
//        data.ToHost()
//
//    let n = 1 <<< 22
//    let input = Array.init n (fun _ -> rng.NextDouble() |> float32)
//    let hOutput = input |> Array.map log
//    let dOutput = input |> transform
//    (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(1e-5)))
//
//[<Test>]
//let ``transform (x:float) -> float32(log x)``() =
//    let worker = getDefaultWorker()
//    use transformm = worker.LoadPModule(PArray.transform <@ fun x -> float32(log x) @>)
//    let transform = transformm.Invoke
//    let transform (input:float[]) =
//        use input = PArray.Create(worker, input)
//        use output = PArray.Create<float32>(worker, input.Length)
//        transform input output
//        output.ToHost()
//
//    let n = 1 <<< 22
//    let input = Array.init n (fun _ -> rng.NextDouble())
//    let hOutput = input |> Array.map log |> Array.map float32
//    let dOutput = input |> transform
//    (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(1e-10)))
//
//[<Test>]
//let ``sequence``() =
//    let worker = getDefaultWorker()
//    use sequencem = worker.LoadPModule(PArray.transformi <@ fun i _ -> i @>)
//    let sequence = sequencem.Invoke
//    let sequence (n:int) =
//        use data = PArray.Create<int>(worker, n)
//        sequence data data
//        data.ToHost()
//
//    let n = 1 <<< 22
//    let hOutput = Array.init n (fun i -> i)
//    let dOutput = sequence n
//    (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))
//
//[<Test>]
//let ``transform (x:float) (y:float32) -> x + float(y)``() =
//    let worker = getDefaultWorker()
//    use transform2m = worker.LoadPModule(PArray.transform2 <@ fun x y -> x + float(y) @>)
//    let transform2 = transform2m.Invoke
//    let transform2 (input1:float[]) (input2:float32[]) =
//        use input1 = PArray.Create(worker, input1)
//        use input2 = PArray.Create(worker, input2)
//        transform2 input1 input2 input1
//        input1.ToHost()
//
//    let n = 1 <<< 22
//    let input1 = Array.init n (fun _ -> rng.NextDouble())
//    let input2 = Array.init n (fun _ -> rng.NextDouble() |> float32)
//    let hOutput = (input1, input2) ||> Array.map2 (fun x y -> x + float(y))
//    let dOutput = (input1, input2) ||> transform2
//    (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(1e-10)))
//
