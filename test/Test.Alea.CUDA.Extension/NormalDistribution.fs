module Test.Alea.CUDA.Extension.NormalDistribution

open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.NormalDistribution

let worker = getDefaultWorker()

[<Test>]
let ``inverse normal cdf with Shaw Brickman algorithm`` () =
    let map = worker.LoadPModule(PArray.map <@ ShawBrickman.inverseNormalCdf @>).Invoke
    let calc = pcalc {
        let x = [| 0.01..0.01..0.99 |]
        let h = x |> Array.map ShawBrickman.inverseNormalCdf
        let! x = DArray.scatterInBlob worker x
        let! d = map x
        let! d = d.Gather()
        (h, d) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(1e-14))) }
    
    calc |> PCalc.run
    calc |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = calc |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>]
let ``inverse normal cdf accuracy comparison`` () =
    let map1 = worker.LoadPModule(PArray.map <@ ShawBrickmanExtended.inverseNormalCdf @>).Invoke
    let map2 = worker.LoadPModule(PArray.map <@ ShawBrickman.inverseNormalCdf @>).Invoke
    let map3 = worker.LoadPModule(PArray.map <@ Acklam.inverseNormalCdf @>).Invoke
    let map4 = worker.LoadPModule(PArray.map <@ fun (x:float) -> float(ShawBrickman32.inverseNormalCdf (float32 x)) @>).Invoke

    let calc = pcalc {
        let x = [| 0.001..0.001..0.999 |]
        let! x = DArray.scatterInBlob worker x

        let! d1 = map1 x
        let! d2 = map2 x
        let! d3 = map3 x
        let! d4 = map4 x

        let! d1 = d1.Gather()
        let! d2 = d2.Gather()
        let! d3 = d3.Gather()
        let! d4 = d4.Gather()

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
        Assert.Less(e3, 4.7e-6) }

    calc |> PCalc.run
    calc |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = calc |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>]
let ``inverse normal cdf accuracy comparison (in one module)`` () =
    let pfunct = cuda {
        let! map1 = PArray.map <@ ShawBrickmanExtended.inverseNormalCdf @>
        let! map2 = PArray.map <@ ShawBrickman.inverseNormalCdf @>
        let! map3 = PArray.map <@ Acklam.inverseNormalCdf @>
        let! map4 = PArray.map <@ fun (x:float) -> float(ShawBrickman32.inverseNormalCdf (float32 x)) @>

        let calc (m:Module) = pcalc {
            let map1 = map1.Apply m
            let map2 = map2.Apply m
            let map3 = map3.Apply m
            let map4 = map4.Apply m

            let x = [| 0.001..0.001..0.999 |]
            let! x = DArray.scatterInBlob worker x

            let! d1 = map1 x
            let! d2 = map2 x
            let! d3 = map3 x
            let! d4 = map4 x

            let! d1 = d1.Gather()
            let! d2 = d2.Gather()
            let! d3 = d3.Gather()
            let! d4 = d4.Gather()

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
            Assert.Less(e3, 4.7e-6) }

        return PFunc(calc) }

    let calc = worker.LoadPModule(pfunct).Invoke

    calc |> PCalc.run
    calc |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = calc |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

[<Test>]
let ``normal cdf with Abramowitz Stegun algorithm`` () =
    let map = worker.LoadPModule(PArray.map <@ AbramowitzStegun.normalCdf @>).Invoke
    let calc = pcalc {
        let x = [| -5.0..0.01..5.0 |]
        let h = x |> Array.map AbramowitzStegun.normalCdf
        let! x = DArray.scatterInBlob worker x
        let! d = map x
        let! d = d.Gather()
        (h, d) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(1e-14))) }

    calc |> PCalc.run
    calc |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
    let _, loggers = calc |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()

