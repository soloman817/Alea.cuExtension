[<AutoOpen>]
module Test.Alea.cuBase.MGPU.Util

open NUnit.Framework
open Alea.CUDA
open Alea.cuBase
open Alea.cuBase.IO.Util
open Alea.cuBase.IO.CSV.Output
open Alea.cuBase.IO.Excel.Output


let eps = 1e-8

let getDefaultWorker() =
    if Device.Count = 0 then Assert.Inconclusive("We need at least one device of compute capability 2.0 or greater.")
    Engine.workers.DefaultWorker


type OutputType =
    | OutputTypeBoth
    | OutputTypeCSV
    | OutputTypeExcel
    | OutputTypeNone    

let benchmarkOutput (opt:OutputType) (workingPath:string) (bms:BenchmarkStats) =
    match opt with
    | OutputTypeBoth ->
        benchmarkCSVOutput bms workingPath
        benchmarkExcelOutput bms
    | OutputTypeCSV ->
        benchmarkCSVOutput bms workingPath
    | OutputTypeExcel ->
        benchmarkExcelOutput bms
    | OutputTypeNone ->
        ()

let rng = System.Random()
//////////////////////////////
// see http://stackoverflow.com/questions/17002632/how-to-generate-an-array-with-a-dynamic-type-in-f
type RngOverloads = RngOverloads with
    static member ($) (RngOverloads, fake:int) = fun (x:int) -> int x
    static member ($) (RngOverloads, fake:float32) = fun (x:int) -> float32 x
    static member ($) (RngOverloads, fake:float) = fun (x:int) -> float x
    static member ($) (RngOverloads, fake:int64) = fun (x:int) -> int64 x

// generate an array of random 'T values
let inline rngGenericArray sCount : 'T[] =
    let convert = (RngOverloads $ Unchecked.defaultof<'T>)
    let genValue() = rng.Next() |> convert
    let source = Array.init sCount (fun _ -> genValue())
    source

let inline rngGenericArrayBounded sCount b : 'T[] =
    let convert = (RngOverloads $ Unchecked.defaultof<'T>)
    let genValue() = rng.Next(b) |> convert
    let source = Array.init sCount (fun _ -> genValue())
    source


// generate an array of random 'T values along with a sorted array of random indices
// that are within the bounds of the source array
// example: let (r : float[] * _) = rngGenericArray 10 10
let inline rngGenericArrayI sCount iCount : 'T[] * int[] =
    let convert = (RngOverloads $ Unchecked.defaultof<'T>)
    let genValue() = rng.Next() |> convert
    let source = Array.init sCount (fun _ -> genValue())
    let indices = Array.init iCount (fun _ -> rng.Next sCount) |> Seq.distinct |> Seq.toArray |> Array.sort
    source, indices


// generate (A,I,B) where A is aCount random elements, I is aCount random indices constrained by bCount,
// and B is bCount random elements.  This is used for inserting A into B where I are the random places to insert
let inline rngGenericArrayAIB aCount bCount : ('T[] * int[] * 'T[]) =
    let convert = (RngOverloads $ Unchecked.defaultof<'T>)
    let genValue() = rng.Next() |> convert
    let hA = Array.init aCount (fun _ -> genValue())
    let hI = Array.init aCount (fun _ -> rng.Next(bCount)) |> Array.sort
    let hB = Array.init bCount (fun _ -> genValue())
    hA, hI, hB


let displayHandD (h:'T[]) (d:'T[]) =
    printfn "*********HOST************"
    printfn "COUNT = ( %d )" h.Length
    printfn "DATA = (%A)" h
    printfn "*************************"
    printfn ""
    printfn "********DEVICE***********"
    printfn "COUNT = ( %d )" d.Length
    printfn "DATA = (%A)" d
    printfn "*************************"

let inline verify (h:'T[]) (d:'T[]) = 
        for i = 0 to h.Length - 1 do
            Assert.That(d.[i], Is.EqualTo(h.[i]).Within(eps))

type Verifier<'T>(?eps:float) =
    member v.Verify (h:'T[]) (d:'T[]) = 
        match eps with
        | Some eps -> for i = 0 to h.Length - 1 do
                        Assert.That(d.[i], Is.EqualTo(h.[i]).Within(eps))
        | None -> let eps = 1e-8
                  for i = 0 to h.Length - 1 do
                        Assert.That(d.[i], Is.EqualTo(h.[i]).Within(eps))

let runForStats (pc:PCalc<'T[]>) =
    let _, loggers = pc |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = pc |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    pc |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))






//type Stats(numTests:int) =
//    val TheirStats : float * float list
//    val mutable MyStats : List.Empty()
//    member s.Count i = int(s.MyStats.[i].[0])

/////////////////////////////////////////////////////////////////////////////////////////////
//                                  SCAN
//module ScanUtils =
//    let testScanStats (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) =
//            let scan = worker.LoadPModule(PArray.scan mgpuScanType op totalAtEnd).Invoke
//        
//            fun (data:'TI[]) ->
//                let calc = pcalc {
//                    let! data = DArray.scatterInBlob worker data
//                    let! result = scan data
//                    return! result.Value }
//                runForStats calc |> ignore            
//
//    let testScanVerify (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) (displayOutput:bool) =
//        let scan = worker.LoadPModule(PArray.scan mgpuScanType op totalAtEnd).Invoke
//        fun (gold:'TI[] -> 'TV[]) (verify: 'TV[] -> 'TV[] -> unit) (data:'TI[]) ->
//            let calc = pcalc {
//                let! data = DArray.scatterInBlob worker data
//                let! result = scan data
//                return! result.Value }
//            let hOutput = gold data
//            let dOutput = PCalc.run calc
//            if displayOutput then displayHandD hOutput dOutput
//            verify hOutput dOutput
//
//    let getScanResult (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) = 
//        let scan = worker.LoadPModule(PArray.scan mgpuScanType op totalAtEnd).Invoke
//        fun (data:'TI[]) ->
//            let calc = pcalc {
//                let! data = DArray.scatterInBlob worker data
//                let! result = scan data
//                return! result.Value }        
//            let dResult = PCalc.run calc
//            dResult
//
//    let getExclusiveAndInclusiveResults (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) =
//        let excScan = worker.LoadPModule(PArray.scan ExclusiveScan op totalAtEnd).Invoke
//        let incScan = worker.LoadPModule(PArray.scan InclusiveScan op totalAtEnd).Invoke
//        fun (data:'TI[]) ->
//            let excCalc = pcalc {
//                let! excData = DArray.scatterInBlob worker data
//                let! excResult = excScan excData
//                return! excResult.Value }
//            let incCalc = pcalc {
//                let! incData = DArray.scatterInBlob worker data
//                let! incResult = incScan incData
//                return! incResult.Value }
//            let excResult = PCalc.run excCalc
//            let incResult = PCalc.run incCalc
//
//            excResult, incResult
//
//    let exclusiveScanResults (n:int) =
//        fun (scannedData:'TI[]) ->
//            let esr = Array.sub scannedData 0 n
//            esr
//
//    let inclusiveScanResults (n:int) =
//        fun (scannedData:'TI[]) ->
//            let isr = Array.sub scannedData 1 n
//            isr
//
//    let getHostScanResult (mgpuScanType:int) (n:int) =
//        fun (scannedData:'TI[]) ->
//            let sr = 
//                if mgpuScanType = ExclusiveScan then
//                    exclusiveScanResults n scannedData
//                else
//                    inclusiveScanResults n scannedData
//            sr
//
//    let scanTypeString (mgpuScanType:int) =
//        if mgpuScanType = ExclusiveScan then
//            "Exclusive"
//        else
//            "Inclusive"
//
//    let printScanType (mgpuScanType:int) =
//        let sts = scanTypeString mgpuScanType
//        printfn "Scan Type: %s" sts
//
//    let getHostExcAndIncScanResults (n:int) =
//        fun (scannedData:'TI[]) ->
//            let hostExcScanResult = exclusiveScanResults n scannedData
//            let hostIncScanResult = inclusiveScanResults n scannedData
//            hostExcScanResult, hostIncScanResult
//
//
///////////////////////////////////////////////////////////////////////////////////////////////
////                               BULK REMOVE
////module BulkRemoveUtils = 
////    let bulkRemove =
////        let br = worker.LoadPModule(MGPU.PArray.bulkRemove).Invoke
////        fun (data:'TI[]) (indices:int[]) ->
////            let calc = pcalc {
////                let! data = DArray.scatterInBlob worker data
////                let! indices = DArray.scatterInBlob worker indices
////                let! result = br data indices
////                return! result.Value }
////            let dResult = PCalc.run calc
////            dResult