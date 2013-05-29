module Test.Alea.CUDA.Extension.MGPU.Scan

open System
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Scan
open Alea.CUDA.Extension.MGPU.CTAScan
open NUnit.Framework
open Test.Alea.CUDA.Extension.TestUtilities
open Test.Alea.CUDA.Extension.TestUtilities.MGPU


let totalAtEnd = 1

let sizes1A = [|16; 32; 64; 128|]
let sizes1L = sizes1A |> Array.toList

let sizes2A = [|512; 1024; 2048; 3000; 6000; 12000; 24000; 100000; 1000000 |] 
let sizes2L = sizes2A |> Array.toList

       
[<Test>]
let ``simple scan with stats``() =
    let values = Array.init 16 (fun i -> i)
    let op = scanOp ScanOpTypeAdd 0
    let test = testScanStats op totalAtEnd
    test values
    
[<Test>]
let ``simple scan with output display & verify`` () =
    let values = Array.init 16 (fun i -> i)
    let op = scanOp ScanOpTypeAdd 0
    let gold data = data |> Array.scan (+) 0
    let test = testScanVerify op totalAtEnd true gold verify
    test values

[<Test>]
let ``simple scan, past cutoff (n > 20000)`` () =
    let values = Array.init 21000 (fun i -> i)
    let op = scanOp ScanOpTypeAdd 0
    let test = testScanStats op totalAtEnd
    test values

[<Test>]
let ``simple scan with size iter & stats`` () =
    let op = scanOp ScanOpTypeAdd 0
    let test = testScanStats op totalAtEnd
    sizes1L |> Seq.iter (fun count ->  
        test (Array.init count (fun i -> i)))

[<Test>]
let ``simple scan with size iter, display, & verify`` () =
    let op = scanOp ScanOpTypeAdd 0
    let gold data = data |> Array.scan (+) 0
    let test = testScanVerify op totalAtEnd true gold verify
    sizes1L |> Seq.iter (fun count ->  
        test (Array.init count (fun i -> i)))

[<Test>]
let ``scan with big size iter & stats`` () =
    let op = scanOp ScanOpTypeAdd 0
    let test = testScanStats op totalAtEnd
    sizes2L |> Seq.iter (fun count ->  
        test (Array.init count (fun i -> i)))


