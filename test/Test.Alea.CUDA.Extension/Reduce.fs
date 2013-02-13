module Test.Alea.CUDA.Extension.Reduce

open System
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Reduce

let rng = System.Random()

let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608; 33554432]

[<Test>]
let ``reduce sum<int>`` () =
    
    let worker = Engine.workers.DefaultWorker
    let reducer:IReduce<int> = worker.LoadPModule(reduce plan32).Invoke
    let reduce input = reducer.Reduce(input)

    let test init n = 
        let values = Array.init n (init)
        let total = reduce values
        let expected:int = Array.sum values
        printfn "size %d, total = %d expected = %d" n total expected
        Assert.AreEqual(expected, total) 

    sizes |> Seq.iter (test (fun _ -> 1))
    sizes |> Seq.iter (test (fun _ -> rng.Next(-100, 100)))

[<Test>]
let ``reduce sum<float>`` () =
    let worker = Engine.workers.DefaultWorker
    let reducer:IReduce<float> = worker.LoadPModule(reduce plan64).Invoke
    let reduce input = reducer.Reduce(input)

    let test init n = 
        let values = Array.init n (init)
        let total = reduce values
        let expected:float = Array.sum values
        let relErr = abs (total - expected)/expected
        printfn "size %d, total = %f expected = %f, rel err = %f" n total expected relErr
        Assert.That(relErr < 1e-10)

    sizes |> Seq.iter (test (fun _ -> 1.0))
    sizes |> Seq.iter (test (fun _ -> rng.NextDouble()))

[<Test>]
let ``reduce generic sum<float>`` () =
    let worker = Engine.workers.DefaultWorker
    let reducer = worker.LoadPModule(genericReduce plan64 <@ fun () -> 0.0 @> <@ (+) @> <@ fun x -> x @>).Invoke
    let reduce input = reducer.Reduce(input)

    let test init n = 
        let values = Array.init n (init)
        let total = reduce values
        let expected = Array.sum values
        let relErr = abs (total - expected)/expected
        printfn "size %d, total = %f expected = %f, rel err = %f" n total expected relErr
        Assert.That(relErr < 1e-10)

    sizes |> Seq.iter (test (fun _ -> 1.0))
    sizes |> Seq.iter (test (fun _ -> rng.NextDouble()))

[<Test>]
let ``reduce generic sum square<float>`` () =
    let worker = Engine.workers.DefaultWorker
    let reducer = worker.LoadPModule(genericReduce plan64 <@ fun () -> 0.0 @> <@ (+) @> <@ fun x -> x*x @>).Invoke
    let reduce input = reducer.Reduce(input)
   
    let test init n = 
        let values = Array.init n (init)
        let total = reduce values
        let expected = values |> Array.map (fun x -> x*x) |> Array.sum
        let relErr = abs (total - expected)/expected
        printfn "size %d, total = %f expected = %f, rel err = %f" n total expected relErr
        Assert.That(relErr < 1e-10)

    sizes |> Seq.iter (test (fun _ -> rng.NextDouble()))

[<Test>]
let ``reduce generic max<float>`` () =
    let worker = Engine.workers.DefaultWorker
    let reducer = worker.LoadPModule(genericReduce plan64 <@ fun () -> Double.NegativeInfinity @> <@ max @> <@ fun x -> x @>).Invoke
    let reduce input = reducer.Reduce(input)

    let test init n = 
        let values = Array.init n (init)
        let total = reduce values
        let expected = Array.max values
        let relErr = abs (total - expected)/expected
        printfn "values = %A" values
        printfn "size %d, total = %f expected = %f, rel err = %f" n total expected relErr
        Assert.That(relErr < 1e-10)

    sizes |> Seq.iter (test (fun _ -> rng.NextDouble()))

[<Test>]
let ``reduce generic min<float>`` () =
    let worker = Engine.workers.DefaultWorker
    let reducer = worker.LoadPModule(genericReduce plan64 <@ fun () -> Double.PositiveInfinity @> <@ min @> <@ fun x -> x @>).Invoke
    let reduce input = reducer.Reduce(input)

    let test init n = 
        let values = Array.init n (init)
        let total = reduce values
        let expected = Array.min values
        let relErr = abs (total - expected)/expected
        printfn "values = %A" values
        printfn "size %d, total = %f expected = %f, rel err = %f" n total expected relErr
        Assert.That(relErr < 1e-10)

    sizes |> Seq.iter (test (fun _ -> rng.NextDouble()))



