﻿module Test.Alea.CUDA.Extension.MGPU.Scan

open System
open System.IO
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Scan
open Alea.CUDA.Extension.MGPU.CTAScan
open Test.Alea.CUDA.Extension.MGPU.Util
open Test.Alea.CUDA.Extension.MGPU.BenchmarkStats
open Alea.CUDA.Extension.Output.Util
open Alea.CUDA.Extension.MGPU.PArray
open Alea.CUDA.Extension.Output.CSV
open Alea.CUDA.Extension.Output.Excel
open NUnit.Framework

////////////////////////////
// set this to your device or add your device's C++ output to BenchmarkStats.fs
open Test.Alea.CUDA.Extension.MGPU.BenchmarkStats.TeslaK20c
// in the future maybe we try to get the C++ to interop somehow
/////////////////////////////
open ModernGPU.ScanStats


let pScanner = new PScan()

let totalAtEnd = 1
let totalNotAtEnd = 0

let defaultScanType = ExclusiveScan

let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608]

let worker = getDefaultWorker()

let algName = "Scan"
let scanKernelsUsed = [| "kernelParallelScan"; "kernelScanDownsweep" |] 
let scanBMS4 = new BenchmarkStats4("Scan", scanKernelsUsed, worker.Device.Name, "MGPU", sourceCounts, nIterations)


// we can probably organize this a lot better, but for now, if you just change
// what module you open above all of this should adjust accordingly
let oIntTP, oIntBW = int32_stats |> List.unzip
let oInt64TP, oInt64BW = int64_stats |> List.unzip

for i = 0 to sourceCounts.Length - 1 do
    // this is setting the opponent (MGPU) stats for the int type
    scanBMS4.Int32s.OpponentThroughput.[i].Value <- oIntTP.[i]
    scanBMS4.Int32s.OpponentBandwidth.[i].Value <- oIntBW.[i]
    // set opponent stats for int64
    scanBMS4.Int64s.OpponentThroughput.[i].Value <- oInt64TP.[i]
    scanBMS4.Int64s.OpponentBandwidth.[i].Value <- oInt64BW.[i]
    // dont have the other types yet


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                              IMPORTANT                                                       //
//                                      Choose an Output Type                                                   // 
// This is a switch for all tests, and can do a lot of extra work.  Make sure you turn it off if you just       //
// want to see the console prInt32s.                                                                            //
let outputType = OutputTypeNone // Choices are CSV, Excel, Both, or None. Set to None for doing kernel timing   //
// only one path, we aren't auto-saving excel stuff yet                                                         //
let workingPath = (getWorkingOutputPaths deviceFolderName algName).CSV                                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////



let hostScan (mgpuScanType:int) (n:int) =
    fun (scannedData:'TI[]) ->
        if mgpuScanType = ExclusiveScan then
            Array.sub scannedData 0 n
        else
            Array.sub scannedData 1 n

let testScan () =
    let test verify eps (data:float[]) = pcalc {
        let scan = worker.LoadPModule(pScanner.Scan(ExclusiveScan, (scanOp ScanOpTypeAdd 0.0), totalAtEnd)).Invoke
        let n = data.Length
        printfn "Testing size %d..." n

        let! dSource = DArray.scatterInBlob worker data
        let! total, scanned = scan dSource
        let! results = scanned.Gather()


        if verify then
            let hResults = Array.scan (+) 0.0 data |> hostScan ExclusiveScan n
            let! dResults = scanned.Gather()
            (Verifier(eps)).Verify hResults dResults
            return results
        else 
            do! PCalc.force()
            return results 
            }

    let eps = 1e-10
    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = Array.init n (fun _ -> -1.0)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)

    sizes |> Seq.iter (fun n -> let test = test true eps (values1 n)
                                test |> PCalc.run |> ignore )

    sizes |> Seq.iter (fun n -> let test = test true eps (values2 n) 
                                test |> PCalc.run |> ignore)

    sizes |> Seq.iter (fun n -> let test = test true eps (values3 n)
                                test |> PCalc.run |> ignore)
             
    let n = 2097152
    let test = test false eps (values1 n)

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))


let benchmarkScan (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) (numIt:int) (data:'TI[]) (testIdx:int) =
    let scanner = worker.LoadPModule(pScanner.ScanFunc(mgpuScanType, op, totalAtEnd)).Invoke
    let count = data.Length
    
    
    let calc = pcalc {
        let! dSource = DArray.scatterInBlob worker data
        let! dScanned = DArray.createInBlob worker count
        let! dTotal = DArray.createInBlob worker 1
        let! scan = scanner count

        // warm up
        do! scan dSource dScanned dTotal

        let! dStopwatch = DStopwatch.startNew worker
        for i = 1 to numIt do
            do! scan dSource dScanned dTotal
        do! dStopwatch.Stop()
        
        let! results = dScanned.Gather()
        let! timing = dStopwatch.ElapsedMilliseconds

        return results, timing }
                
    let hResults, timing' = calc |> PCalc.runInWorker worker
    let timing = timing' / 1000.0 // convert to milliseconds
    let bytes = (2 * sizeof<'TI> + sizeof<'TV>) * count |> float
    let throughput = (float count) * (float numIt) / timing
    let bandwidth = bytes * (float numIt) / timing
    
    printfn "%9d: %9.3f M/s %9.3f GB/s %6.3f ms x %4d = %7.3f ms"
        count
        (throughput / 1.0e6)
        (bandwidth / 1.0e9)
        (timing' / (float numIt))
        numIt
        timing'
    
    match typeof<'TI> with
    | x when x = typeof<int> -> scanBMS4.Int32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | x when x = typeof<int64> -> scanBMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | x when x = typeof<float32> -> scanBMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | x when x = typeof<float> -> scanBMS4.Float64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | _ -> ()

let prInt32scanType (mgpuScanType:int) =
    if mgpuScanType = ExclusiveScan then
        printfn "Scan Type: Exclusive"
    else
        printfn "Scan Type: Inclusive"


//need to fix, they are no different
[<Test>]
let ``compare totalAtEnd & totalNotAtEnd`` () =
    let hValues = Array.init 20 (fun _ -> 20)
    printfn "Initial values: %A" hValues
    let hResult = Array.scan (+) 0 hValues
    printfn "Host Scan Result ==> Count: (%d),  %A" hResult.Length hResult
    let op = scanOp ScanOpTypeAdd 0
    let calc (tAE:int) = pcalc {
                                        let scan = worker.LoadPModule(pScanner.Scan(0, op, tAE)).Invoke
                                        let n = hValues.Length
                                        let! dValues = DArray.scatterInBlob worker hValues
                                        let! dResults = DArray.createInBlob<int> worker n
                                        let! dTotal = DArray.createInBlob<int> worker 1
                                        let! total, scanned = scan dValues
                                        return! scanned.Gather()}
                                
    let dResult = (calc totalAtEnd) |> PCalc.run
    printfn "Device Scan Result (Total At End) ==> Count: (%d),  %A" dResult.Length dResult
    let dResult = (calc totalNotAtEnd) |> PCalc.run
    printfn "Device Scan Result (Total Not At End) ==> Count: (%d),  %A" dResult.Length dResult


[<Test>]
let ``exact mgpu website example`` () =
    let hValues = [| 1; 7; 4; 0; 9; 4; 8; 8; 2; 4; 5; 5; 1; 7; 1; 1; 5; 2; 7; 6 |]
    let n = hValues.Length
    let exclusiveResult = [| 0; 1; 8; 12; 12; 21; 25; 33; 41; 43; 47; 52; 57; 58; 65; 66; 67; 72; 74; 81 |]
    let inclusiveResult = [| 1; 8; 12; 12; 21; 25; 33; 41; 43; 47; 52; 57; 58; 65; 66; 67; 72; 74; 81; 87 |]
    printfn "Initial values: %A" hValues
    let hExcResult, hIncResult = 
        (hostScan ExclusiveScan n) (Array.scan (+) 0 hValues), (hostScan InclusiveScan n) (Array.scan (+) 0 hValues) 
    printfn "Host Exclusive Scan Result ==> Count: (%d), %A" hExcResult.Length hExcResult
    printfn "Host Inclusive Scan Result ==> Count: (%d), %A" hIncResult.Length hIncResult
        
    let op = scanOp ScanOpTypeAdd 0
    let calc (tAe:int) (stype:int) = pcalc {  
                                let scan = worker.LoadPModule(pScanner.Scan(stype, op, tAe)).Invoke
                                let n = hValues.Length
                                let! dValues = DArray.scatterInBlob worker hValues
                                let! dResults = DArray.createInBlob<int> worker n
                                let! dTotal = DArray.createInBlob<int> worker 1
                                let! total, scanned = scan dValues
                                return! scanned.Gather()}
    let dExcResult, dIncResult = 
        (calc totalNotAtEnd ExclusiveScan) |> PCalc.run,
        (calc totalNotAtEnd InclusiveScan) |> PCalc.run

    printfn "Device Exclusive Scan Result ==>  Count: (%d), %A" dExcResult.Length dExcResult
    printfn "Device Inclusive Scan Result ==>  Count: (%d), %A" dIncResult.Length dIncResult

    printfn "Checking that device results = host results"
    (hExcResult, dExcResult) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
    (hIncResult, dIncResult) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
    printfn "Checking that device results = moderngpu website example results"
    (dExcResult, exclusiveResult) ||> Array.iter2 (fun d e -> Assert.That(e, Is.EqualTo(d).Within(eps)))
    (dIncResult, inclusiveResult) ||> Array.iter2 (fun d e -> Assert.That(e, Is.EqualTo(d).Within(eps)))


[<Test>]
let ``Scan 3 value test`` () =
    testScan()



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  BENCHMARKING                                                                                                //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

[<Test>]
let ``Scan moderngpu benchmark : int32`` () =
    (sourceCounts, nIterations) ||> List.iteri2 (fun i ns ni ->
        let (source:int[]) = rngGenericArray ns
        benchmarkScan ExclusiveScan (scanOp ScanOpTypeAdd 0) totalNotAtEnd ni source i)
    
    benchmarkOutput outputType workingPath scanBMS4.Int32s


[<Test>]
let ``Scan moderngpu benchmark : int64`` () =
    (sourceCounts, nIterations) ||> List.iteri2 (fun i ns ni ->
        let (source:int64[]) = rngGenericArray ns
        benchmarkScan ExclusiveScan (scanOp ScanOpTypeAdd 0L) totalNotAtEnd ni source i)

    benchmarkOutput outputType workingPath scanBMS4.Int64s


[<Test>]
let ``Scan moderngpu benchmark : float32`` () =
    (sourceCounts, nIterations) ||> List.iteri2 (fun i ns ni ->
        let (source:float32[]) = rngGenericArray ns
        benchmarkScan ExclusiveScan (scanOp ScanOpTypeAdd 0.f) totalNotAtEnd ni source i)

    benchmarkOutput outputType workingPath scanBMS4.Float32s


[<Test>]
let ``Scan moderngpu benchmark : float64`` () =
    (sourceCounts, nIterations) ||> List.iteri2 (fun i ns ni ->
        let (source:float[]) = rngGenericArray ns
        benchmarkScan ExclusiveScan (scanOp ScanOpTypeAdd 0.0) totalNotAtEnd ni source i)

    benchmarkOutput outputType workingPath scanBMS4.Float64s


[<Test>] // above 4 tests, done in sequence (to make output easier)
let ``Scan moderngpu benchmark : 4 type`` () =
    // INT
    printfn "Running Scan moderngpu benchmark : Int32"
    (sourceCounts, nIterations) ||> List.iteri2 (fun i ns ni ->
        let (source:int[]) = rngGenericArray ns
        benchmarkScan ExclusiveScan (scanOp ScanOpTypeAdd 0) totalNotAtEnd ni source i)    
    benchmarkOutput outputType workingPath scanBMS4.Int32s

    // INT64
    printfn "\nRunning Scan moderngpu benchmark : Int64"
    (sourceCounts, nIterations) ||> List.iteri2 (fun i ns ni ->
        let (source:int64[]) = rngGenericArray ns
        benchmarkScan ExclusiveScan (scanOp ScanOpTypeAdd 0L) totalNotAtEnd ni source i)
    benchmarkOutput outputType workingPath scanBMS4.Int64s

    // FLOAT32
    printfn "\nRunning Scan moderngpu benchmark : Float32"
    (sourceCounts, nIterations) ||> List.iteri2 (fun i ns ni ->
        let (source:float32[]) = rngGenericArray ns
        benchmarkScan ExclusiveScan (scanOp ScanOpTypeAdd 0.f) totalNotAtEnd ni source i)
    benchmarkOutput outputType workingPath scanBMS4.Float32s

    // FLOAT64
    printfn "\nRunning Scan moderngpu benchmark : Float64"
    (sourceCounts, nIterations) ||> List.iteri2 (fun i ns ni ->
        let (source:float[]) = rngGenericArray ns
        benchmarkScan ExclusiveScan (scanOp ScanOpTypeAdd 0.0) totalNotAtEnd ni source i)
    benchmarkOutput outputType workingPath scanBMS4.Float64s


[<Test>]
let ``big scan test`` () =
    let scan = worker.LoadPModule(pScanner.Scan()).Invoke

    let N = 25000

    let hData = Array.init N (fun i -> i)
    let hResult = Array.scan (+) 0 hData |> hostScan ExclusiveScan N
    let total, scanned = pcalc {
        let! dData = DArray.scatterInBlob worker hData
        return! scan dData } |> PCalc.run

    printfn "scanned:\n%A" scanned
    printfn "total:\n%A" total
    printfn "h scanned:\n%A" hResult