module Test.Alea.CUDA.Extension.MGPU.Scan

open System
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Scan
open Alea.CUDA.Extension.MGPU.CTAScan
open NUnit.Framework

let worker = Engine.workers.DefaultWorker

let displayHandD (h:'T[]) (d:'T[]) =
    printfn "*********HOST************"
    printfn "hCount = (%d)" h.Length
    printfn "h = (%A)" h
    printfn "********DEVICE***********"
    printfn "dCount = (%d)" d.Length
    printfn "d = (%A)" d

let eps = 1e-8

// need to figure out why device array is 1 smaller than host???
let verify (h:int[]) (d:int[]) = 
        for i = 0 to h.Length - 2 do
            Assert.That(d.[i], Is.EqualTo(h.[i]).Within(eps))

let testScan (op:IScanOp<'TI, 'TV, 'TR>) =
    let scan = worker.LoadPModule(MGPU.PArray.scan op).Invoke

    fun (gold:'TV[] -> 'TV[]) (verify:'TV[] -> 'TV[] -> unit) (data:'TI[]) ->
        let calc = pcalc {
            let! data = DArray.scatterInBlob worker data
            let! result = scan data
            return! result.Value }

        let hOutput = gold data
        let dOutput = PCalc.run calc
        verify hOutput dOutput

//let sizes = [|12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193|]//; 9000; 10000; 2097152; 8388608; 33554432; 33554431; 33554433]
let sizes = [|16; 32; 64; 128; 21000|] 

[<Test>]
let ``simple scan``() =
    let op = scanOp ScanOpTypeAdd 0
    let gold data = data |> Array.scan (+) 0
    let test = testScan op gold verify
    sizes |> Seq.iter (fun count -> (test (Array.init count (fun i -> i))) )
        

//let testScan2 =
//    let test (op:IScanOp<'TI, 'TV, 'TR>) =
//        let scan = worker.LoadPModule(MGPU.PArray.scan op).Invoke
//        fun (gold:'TV[] -> 'TV[]) (verify:'TV[] -> 'TV[] -> unit) (data:'TI[]) ->
//            let calc = pcalc {
//                let! data = DArray.scatterInBlob worker data
//                let! result = scan data
//                return! result.Value }
//            let hOutput = gold data
//            let dOutput = PCalc.run calc
//            verify hOutput dOutput
//
//    let values = sizes |> Array.map (fun n -> Array.init n (fun _ -> 1)) |> Array.concat
//    
//    
//    let op = scanOp ScanOpTypeAdd 0
//    //let derp = Array.scan 
//    let gold data = data |> Array.scan (+) 0
//    //let gold = gold values
//    let test = test op gold verify values
//
//
//    let _, loggers = PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
//    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
//    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

//[<Test>]
//let ``scan test`` () =
//    let op = scanOp ScanOpTypeAdd 0
//    let scan = worker.LoadPModule(MGPU.PArray.scan op).Invoke
//    let data = (Array.init 20 (fun i -> i))
//    let calc = pcalc {
//        let! data = DArray.scatterInBlob worker data
//        let! result = scan data
//        return! result.Value }
//
//    let hOutput = Array.scan (+) 1 data
//    let dOutput = PCalc.run calc
//    printfn "count(%d) h(%A) (d:%A)" data.Length hOutput dOutput