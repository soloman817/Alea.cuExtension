module Test.Alea.CUDA.Extension.NormalDistribution

open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.NormalDistribution

let transform (worker:DeviceWorker) (transform:PArray<float> -> PArray<float> -> unit) (input:float[]) =
    use data = PArray.Create(worker, input)
    transform data data
    data.ToHost()

[<Test>]
let ``inverse normal cdf with Shaw Brickman algorithm`` () =
    let worker = getDefaultWorker()
    use transformm = worker.LoadPModule(PArray.transform <@ ShawBrickman.inverseNormalCdf @>)
    let transform = transform worker transformm.Invoke

    let x = [| 0.01..0.01..0.99 |]
    let h = x |> Array.map ShawBrickman.inverseNormalCdf
    let d = transform x

    (h, d) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(1e-14)))

[<Test>]
let ``inverse normal cdf accuracy comparison`` () =
    let worker = getDefaultWorker()

    use transformm1 = worker.LoadPModule(PArray.transform <@ ShawBrickmanExtended.inverseNormalCdf @>)
    use transformm2 = worker.LoadPModule(PArray.transform <@ ShawBrickman.inverseNormalCdf @>)
    use transformm3 = worker.LoadPModule(PArray.transform <@ Acklam.inverseNormalCdf @>)
    use transformm4 = worker.LoadPModule(PArray.transform <@ fun (x:float) -> float(ShawBrickman32.inverseNormalCdf (float32 x)) @>)

    let transform1 = transform worker transformm1.Invoke
    let transform2 = transform worker transformm2.Invoke
    let transform3 = transform worker transformm3.Invoke
    let transform4 = transform worker transformm4.Invoke

    let x = [| 0.001..0.001..0.999 |]
    let d1 = transform1 x
    let d2 = transform2 x
    let d3 = transform3 x
    let d4 = transform4 x

    let error a b = abs (a-b)
    let e1 = Array.map2 (error) d1 d2 |> Array.max
    let e2 = Array.map2 (error) d1 d3 |> Array.max
    let e3 = Array.map2 (error) d1 d4 |> Array.max

    //Array.iteri (fun i xi -> printfn "%.4f (%.4f): d1=%.14f d2=%.14f diff = %.10e" xi xf.[i] d1.[i] d4.[i] (abs (d1.[i] - d4.[i]))) x

    printfn "ShawBrickmanExtended vs ShawBrickman = %.4e" e1
    printfn "ShawBrickmanExtended vs Acklam = %.4e" e2
    printfn "ShawBrickmanExtended vs ShawBrickman32 = %.4e" e3

    Assert.Less(e1, 5e-014)
    Assert.Less(e2, 3e-014)
    Assert.Less(e3, 4.7e-6)

[<Test>]
let ``normal cdf with Abramowitz Stegun algorithm`` () =
    let worker = getDefaultWorker()
    use transformm = worker.LoadPModule(PArray.transform <@ AbramowitzStegun.normalCdf @>)
    let transform = transform worker transformm.Invoke

    let x = [| -5.0..0.01..5.0 |]
    let h = x |> Array.map (AbramowitzStegun.normalCdf)
    let d = transform x

    (h, d) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(1e-14)))
