module Test.Alea.CUDA.Extension.SegmentedScan

open System
open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Reduce
open Alea.CUDA.Extension.SegmentedScan

let rng = System.Random()

let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608; 33554432]

[<Test>]  
let ``segmented scan reduce test max<int>`` () =
    let worker = getDefaultWorker()   
    let test = worker.LoadPModule(Sum.reduceTest <@(fun () -> -10)@> <@(max)@>).Invoke

    let n = plan32.numThreads
    let v = Array.init n (fun _ -> rng.Next(-5, 5))
    let d = test v
    let expected = Array.max v

    printfn "v = %A" v
    printfn "d = %A" d
    printfn "expected = %A" expected

[<Test>]
let ``segmented scan reduce test sum<int>`` () =
    let worker = getDefaultWorker()
    let test = worker.LoadPModule(Sum.reduceTest <@(fun () -> 0)@> <@(+)@>).Invoke

    let n = plan32.numThreads
    let v = Array.init n (fun _ -> rng.Next(-5, 5))
    let d = test v
    let expected = Array.sum v

    printfn "v = %A" v
    printfn "d = %A" d
    printfn "expected = %A" expected

[<Test>]
let ``segmented scan sum<int>`` () =
    let worker = getDefaultWorker()
    let scan = worker.LoadPModule(segScan ()).Invoke

    let n = 200
    let values = Array.init n (fun _ -> 1)
    let flags = Array.zeroCreate n
    flags.[0] <- 1
    flags.[50] <- 1
    flags.[100] <- 1
    flags.[150] <- 1

    let segScan = scan values flags false

    printfn "segScan = %A" segScan

[<Test>]
let ``segmented scan upsweep`` () =
    let worker = getDefaultWorker()
    let scan = worker.LoadPModule(segScan ()).Invoke

    let n = 5*1024
    let values = Array.init n (fun _ -> 1)
    let flags = Array.zeroCreate n
    flags.[0] <- 1
    flags.[512] <- 1
    flags.[1024] <- 1
    flags.[2000] <- 1
    flags.[3000] <- 1

    let segScan = scan values flags false

    printfn "segScan = %A" segScan




