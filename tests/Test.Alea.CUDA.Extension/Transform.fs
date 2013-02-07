module Test.Alea.CUDA.Extension.Transform

open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

let rng = System.Random()

let gpuMap transform =
    let worker = Engine.workers.DefaultWorker
    let mapper = worker.LoadPModule(Transform.map transform).Invoke
    fun input -> mapper.Map(input)

[<Test>]
let ``map log<float>``() =
    let gpuMapLog = gpuMap <@ log @>
    let n = 1 <<< 22
    let input = Array.init n (fun _ -> rng.NextDouble())
    let hOutput = input |> Array.map log
    let dOutput = input |> gpuMapLog
    (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(1e-10)))

[<Test>]
let ``map log<float32>``() =
    let gpuMapLog = gpuMap <@ log @>
    let n = 1 <<< 22
    let input = Array.init n (fun _ -> rng.NextDouble() |> float32)
    let hOutput = input |> Array.map log
    let dOutput = input |> gpuMapLog
    (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(1e-6)))



