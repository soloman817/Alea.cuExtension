module Test.Alea.CUDA.Extension.MovingAverage

open System
open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MovingAverage

/// CPU version to calculate moving average
let movingAverage n (series:seq<float>) =
    series    
    |> Seq.windowed n
    |> Seq.map Array.sum
    |> Seq.map (fun a -> a / float n)    

/// Fast CPU version based on arrays to calculate moving average
let movingAverageScan windowSize (series:array<float>) =
    let sums = Array.scan (fun s x -> s + x) 0.0 series
    let ma = Array.zeroCreate (sums.Length - windowSize)
    for i = windowSize to sums.Length - 1 do
        ma.[i - windowSize] <- (sums.[i] - sums.[i - windowSize])/float(windowSize)   
    ma   

let rng = Random(2)
 
let inline maxErr (b:'T[]) (b':'T[]) =
    Array.map2 (fun bi bi' -> abs (bi - bi')) b b' |> Array.max
         
[<Test>]
let ``moving average`` () =
    let worker = getDefaultWorker()
    let movingAverager = worker.LoadPModule(movingAverager ()).Invoke
    let test verify eps windowSize (hValues:float[]) = pcalc {
        // this is a temporary fix until we know how to append a zero in device memory
        let! dValues = DArray.scatterInBlob worker (Array.append hValues [|0.0|])
        let! dResult = movingAverager windowSize dValues 
        let! hResult = dResult.Gather()

        if verify then
            let expected = movingAverageScan windowSize hValues
            (expected, hResult) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
    }
    
    let values n = let rgn = System.Random(2) in Array.init n (fun i -> rng.NextDouble())
    let windowSizes = [2; 3; 10]
    let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608]
    sizes |> Seq.iter (fun n -> windowSizes |> Seq.iter (fun w -> test true 1e-10 w (values n) |> PCalc.run))

[<Test>]
let ``moving average direct`` () =
    let worker = getDefaultWorker()
    let movingAverager = worker.LoadPModule(MovingAvDirect.movingAverager ()).Invoke
    let test verify eps windowSize (hValues:float[]) = pcalc {
        let! dValues = DArray.scatterInBlob worker hValues  
        let! dResult = movingAverager windowSize dValues 
        let! hResult = dResult.Gather()

        if verify then
            let expected = movingAverageScan windowSize hValues
            let hResultCut = Array.sub hResult (windowSize - 1) (hValues.Length - windowSize + 1)
            (expected, hResultCut) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
    }

    let values n = let rgn = System.Random(2) in Array.init n (fun i -> rng.NextDouble())
    let windowSizes = [2; 3; 10]
    let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193]  
    sizes |> Seq.iter (fun n -> windowSizes |> Seq.iter (fun w -> test true 1e-10 w (values n) |> PCalc.run))

   