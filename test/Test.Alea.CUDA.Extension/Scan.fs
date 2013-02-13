module Test.Alea.CUDA.Extension.Scan

open System
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Reduce
open Alea.CUDA.Extension.Scan

let rng = System.Random()

let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608; 33554432]

[<Test>]
let ``scan sum<int>`` () =
    let worker = getDefaultWorker()
    let api = worker.LoadPModule(Sum.scan plan32).Invoke

    let test init n =
        let values = Array.init n (init)        
        printfn "size = %d, values = %A" n values
        let expected = Array.sub (values |> Array.scan (+) 0) 0 values.Length
        let scan = api.Invoke(values, 0)
        printfn "expected = %A" expected
        printfn "scan = %A" scan
        expected |> Array.iteri (fun i e -> Assert.AreEqual(e, scan.[i]))

    [33554432] |> Seq.iter (test (fun _ -> 1))
    //sizes |> Seq.iter (test (fun _ -> rng.Next(-100, 100)))

[<Test>]
let ``scan generic sum<int>`` () =
    let worker = getDefaultWorker()

//    let reduce = worker.LoadPModule(Reduce.reduce plan32 <@ fun () -> 0.0 @> <@ (+) @> <@ fun x -> x @>).Invoke
//
//    let test init n = 
//        let values = Array.init n (init)
//        let total = reduce.Reduce values
//        let expected = Array.sum values
//        let relErr = abs (total - expected)/expected
//        printfn "reduce size %d, total = %f expected = %f, rel err = %f" n total expected relErr
//        Assert.That(relErr < 1e-10)
//
//    [16000] |> Seq.iter (test (fun _ -> 1.0))

    let api = worker.LoadPModule(scan plan32 <@ fun () -> 0 @> <@ (+) @> <@ fun x -> x @>).Invoke

    let test init n =
        let values = Array.init n (init)        
        printfn "size = %d, values = %A" n values
        let expected = Array.sub (values |> Array.scan (+) 0) 0 values.Length
        let scan = api.Invoke(values, 0)
        printfn "expected = %A" expected
        printfn "scan = %A" scan
        expected |> Array.iteri (fun i e -> Assert.AreEqual(e, scan.[i]))

    [33554432] |> Seq.iter (test (fun _ -> 1))
    //sizes |> Seq.iter (test (fun _ -> rng.Next(-100, 100)))


    