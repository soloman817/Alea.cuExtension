module Test.Alea.CUDA.Extension.MGPU.Scan

open System
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Scan
open Alea.CUDA.Extension.MGPU.CTAScan
open NUnit.Framework

let worker = Engine.workers.DefaultWorker

let testScan (op:IScanOp<'TI, 'TV, 'TR>) =
    let scan = worker.LoadPModule(MGPU.PArray.scan op).Invoke

    fun (gold:'TV[] -> 'TV[]) (verify:'TV[] -> 'TV[] -> unit) (data:'TI[]) ->
        let calc = pcalc {
            let! data = DArray.scatterInBlob worker data
            let! result = scan data
            return! result.Value }

        let hOutput = gold data
        let dOutput = PCalc.run calc
        printfn "count(%d) h(%A) (d:%A)" data.Length hOutput dOutput
        verify hOutput dOutput

let istates = [| 2 |]
let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193]//; 9000; 10000; 2097152; 8388608; 33554432; 33554431; 33554433]

[<Test>]
let ``sum int``() =
    let op = scanOp ScanOpTypeAdd 1
    let gold data = Array.scan  (fun e i -> 1 * i) 1 data
    let eps = 1e-8
    let verify (h:int[]) (d:int[]) = 
        for i = 0 to h.Length - 1 do
            Assert.That(d.[i], Is.EqualTo(h.[i]).Within(eps))

    let test = testScan op gold verify

    test (Array.init 12 (fun i -> i))
//    sizes |> Seq.iter (fun count ->
//        (test (Array.init count (fun i -> i))) )