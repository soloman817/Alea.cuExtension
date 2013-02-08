module Test.Alea.CUDA.Extension.Transformation

open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

[<Test>]
let ``transform (x:float) -> log x``() =
    let worker = getDefaultWorker()
    use transformm = worker.LoadPModule(PArray.transform <@ log @>)
    let transform = transformm.Invoke
    let transform (input:float[]) =
         use data = PArray.Create(worker, input)
         transform data data
         data.ToHost()

    let n = 1 <<< 22
    let input = Array.init n (fun _ -> rng.NextDouble())
    let hOutput = input |> Array.map log
    let dOutput = input |> transform
    (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(1e-10)))

[<Test>]
let ``transform (x:float32) -> log x``() =
    let worker = getDefaultWorker()
    use transformm = worker.LoadPModule(PArray.transform <@ log @>)
    let transform = transformm.Invoke
    let transform (input:float32[]) =
        use data = PArray.Create(worker, input)
        transform data data
        data.ToHost()

    let n = 1 <<< 22
    let input = Array.init n (fun _ -> rng.NextDouble() |> float32)
    let hOutput = input |> Array.map log
    let dOutput = input |> transform
    (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(1e-5)))

[<Test>]
let ``transform (x:float) -> float32(log x)``() =
    let worker = getDefaultWorker()
    use transformm = worker.LoadPModule(PArray.transform <@ fun x -> float32(log x) @>)
    let transform = transformm.Invoke
    let transform (input:float[]) =
        use input = PArray.Create(worker, input)
        use output = PArray.Create<float32>(worker, input.Length)
        transform input output
        output.ToHost()

    let n = 1 <<< 22
    let input = Array.init n (fun _ -> rng.NextDouble())
    let hOutput = input |> Array.map log |> Array.map float32
    let dOutput = input |> transform
    (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(1e-10)))

[<Test>]
let ``sequence``() =
    let worker = getDefaultWorker()
    use sequencem = worker.LoadPModule(PArray.transformi <@ fun i _ -> i @>)
    let sequence = sequencem.Invoke
    let sequence (n:int) =
        use data = PArray.Create<int>(worker, n)
        sequence data data
        data.ToHost()

    let n = 1 <<< 22
    let hOutput = Array.init n (fun i -> i)
    let dOutput = sequence n
    (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))

[<Test>]
let ``transform (x:float) (y:float32) -> x + float(y)``() =
    let worker = getDefaultWorker()
    use transform2m = worker.LoadPModule(PArray.transform2 <@ fun x y -> x + float(y) @>)
    let transform2 = transform2m.Invoke
    let transform2 (input1:float[]) (input2:float32[]) =
        use input1 = PArray.Create(worker, input1)
        use input2 = PArray.Create(worker, input2)
        transform2 input1 input2 input1
        input1.ToHost()

    let n = 1 <<< 22
    let input1 = Array.init n (fun _ -> rng.NextDouble())
    let input2 = Array.init n (fun _ -> rng.NextDouble() |> float32)
    let hOutput = (input1, input2) ||> Array.map2 (fun x y -> x + float(y))
    let dOutput = (input1, input2) ||> transform2
    (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(1e-10)))

