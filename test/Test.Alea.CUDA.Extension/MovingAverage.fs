module Test.Alea.CUDA.Extension.MovingAverage

open System
open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

/// CPU version to calculate moving average
let movingAverage n (series:seq<float>) =
    series    
    |> Seq.windowed n
    |> Seq.map Array.sum
    |> Seq.map (fun a -> a / float n)    

/// Fast CPU version based on arrays to calculate moving average
let movingAverageFast n (series:array<float>) =
    let sums = Array.scan (fun s x -> s + x) 0.0 series
    let ma = Array.zeroCreate (sums.Length - n)
    for i = n to sums.Length - 1 do
        ma.[i - n] <- (sums.[i] - sums.[i - n])/float(n)   
    ma   

let rng = Random(2)
 
let inline maxErr (b:'T[]) (b':'T[]) =
    Array.map2 (fun bi bi' -> abs (bi - bi')) b b' |> Array.max
         
[<Test>]
let ``moving average`` () =

    ()

