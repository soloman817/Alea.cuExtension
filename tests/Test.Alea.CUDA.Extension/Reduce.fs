module Test.Alea.CUDA.Extension.Reduce

open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

let rng = System.Random()

let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608; 33554432]

[<Test>]
let ``reduce sum<int>`` () =
    let reduce plan =
        let worker = Engine.workers.DefaultWorker
        let reducer = worker.LoadPModule(Reduce.Sum.reduce plan).Invoke
        fun input -> reducer.Reduce(input)
    
    let reduce = reduce Reduce.plan32

    let test init n = 
        let values = Array.init n (init)
        let total = reduce values
        let expected:int = Array.sum values
        printfn "size %d, total = %d expected = %d" n total expected
        Assert.AreEqual(expected, total) 

    sizes |> Seq.iter (test (fun _ -> 1))
    sizes |> Seq.iter (test (fun _ -> rng.Next(-100, 100)))


[<Test>]
let ``reduce generic sum<float>`` () =
    let reduce plan =
        let worker = Engine.workers.DefaultWorker
        let reducer = worker.LoadPModule(Reduce.Generic.reduce plan <@ fun () -> 0.0 @> <@ (+) @> <@ fun x -> x @>).Invoke
        fun input -> reducer.Reduce(input)
    
    let reduce = reduce Reduce.plan32

    let test init n = 
        let values = Array.init n (init)
        let total = reduce values
        let expected = Array.sum values
        printfn "size %d, total = %f expected = %f" n total expected
        Assert.That(total, Is.EqualTo(expected).Within(1e-12))

    [12; 128; 512; 1024] |> Seq.iter (test (fun _ -> 1.0))
    [12; 128; 512; 1024] |> Seq.iter (test (fun _ -> rng.NextDouble()))


// TODO why does this not work?
//[<Test>]
//let ``reduce sum<float>`` () =
//
//    let worker = Engine.workers.DefaultWorker
//    let f = Reduce.Sum.reduce 
//    let reducer = worker.LoadPModule(f Reduce.plan64).Invoke
//    let reduce input = reducer.Reduce(input)
//
//    let test init n = 
//        let values = Array.init n (init)
//        let total = reduce values
//        let expected:int = Array.sum values
//        printfn "size %d, total = %d expected = %d" n total expected
//        Assert.AreEqual(expected, total) 
//
//    sizes |> Seq.iter (test (fun _ -> 1.0))
//    sizes |> Seq.iter (test (fun _ -> rng.NextDouble(-100.0, 100.0)))


