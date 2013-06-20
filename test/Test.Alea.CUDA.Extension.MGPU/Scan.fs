module Test.Alea.CUDA.Extension.MGPU.Scan

open System
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Scan
open Alea.CUDA.Extension.MGPU.CTAScan
open Test.Alea.CUDA.Extension.MGPU.Util
open NUnit.Framework
open Test.Alea.CUDA.Extension.TestUtilities
//open Test.Alea.CUDA.Extension.TestUtilities.MGPU.ScanUtils

let totalAtEnd = 1
let totalNotAtEnd = 0

let defaultScanType = ExclusiveScan


let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608]

let sizes1A = [|16; 32; 64; 128|]
let sizes1L = sizes1A |> Array.toList

let sizes2A = [|512; 1024; 2048; 3000; 6000; 12000; 24000; 100000; 1000000 |] 
let sizes2L = sizes2A |> Array.toList

let sourceCounts = [10000; 50000; 100000; 200000; 500000; 1000000; 200000; 500000; 10000000; 20000000]
let nIterations = [1000; 1000; 1000; 500; 200; 200; 200; 200; 100; 100]

let worker = getDefaultWorker()

//let testScanStats (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) =
//        let scan = worker.LoadPModule(PArray.scan mgpuScanType op totalAtEnd).Invoke
//        fun (data:'TI[]) ->
//            let calc = pcalc {
//                let! data = DArray.scatterInBlob worker data
//                let! result = scan data
//                return! result.Value }
//            runForStats calc |> ignore            
//
//let testScanVerify (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) (displayOutput:bool) =
//    let scan = worker.LoadPModule(PArray.scan mgpuScanType op totalAtEnd).Invoke
//    fun (gold:'TI[] -> 'TV[]) (verify: 'TV[] -> 'TV[] -> unit) (data:'TI[]) ->
//        let calc = pcalc {
//            let! data = DArray.scatterInBlob worker data
//            let! result = scan data
//            return! result.Value }
//        let hOutput = gold data
//        let dOutput = PCalc.run calc
//        if displayOutput then displayHandD hOutput dOutput
//        verify hOutput dOutput
//
//let getDeviceScanResult (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) = 
//    let scan = worker.LoadPModule(PArray.scan mgpuScanType op totalAtEnd).Invoke
//    fun (data:'TI[]) ->
//        let calc = pcalc {
//            let! data = DArray.scatterInBlob worker data
//            let! result = scan data
//            return! result.Value }        
//        let dResult = PCalc.run calc
//        dResult

let hostScan (mgpuScanType:int) (n:int) =
    fun (scannedData:'TI[]) ->
        if mgpuScanType = ExclusiveScan then
            Array.sub scannedData 0 n
        else
            Array.sub scannedData 1 n

//let testScan (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) =
//    let scan = worker.LoadPModule(MGPU.PArray.scan mgpuScanType op totalAtEnd).Invoke
//    let test verify eps = 
//        fun (data:'TI[]) ->
//            pcalc {
//                let n = data.Length
//                printfn "Testing size %d..." n
//
//                let! dValues = DArray.scatterInBlob worker data
//                let! dResults = DArray.createInBlob<'TR> worker n
//                let! scanner, scanned = scan dValues
//                do! scanner.Value
//                let! results = scanned.Gather()
//
//
//                if verify then
//                    let hResults = Array.scan (+) 0.0 data |> hostScan mgpuScanType n
//                    let! dResults = scanned.Gather()
//                    (General.Verifier(eps)).Verify hResults dResults
//                    return results
//                else 
//                    do! PCalc.force()
//                    return results 
//                    }
//
//    let eps = 1e-10
//    let values1 n = Array.init n (fun _ -> 1.0)
//    let values2 n = Array.init n (fun _ -> -1.0)
//    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)
//
//    sizes |> Seq.iter (fun n -> let test = test true eps
//                                values1 n |> test |> PCalc.run |> ignore )
//
//    sizes |> Seq.iter (fun n -> let test = test true eps
//                                values2 n |> test |> PCalc.run |> ignore)
//
//    sizes |> Seq.iter (fun n -> let test = test true eps
//                                values3 n |> test |> PCalc.run |> ignore)
//             
//    let n = 2097152
//    let test = values1 n |> test false eps
//
//    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
//    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
//    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))


let benchmarkScan (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) =
    let scan = worker.LoadPModule(MGPU.PArray.scan mgpuScanType op totalAtEnd).Invoke
    
    fun (data:'TI[]) (numIt:int) ->
        let count = data.Length
        let calc = pcalc {
            let! dValues = DArray.scatterInBlob worker data
            //let! scanner, scanned = scan dValues
            let! scanner, _ = scan dValues
            let! stopwatch = DStopwatch.startNew worker
            for i = 1 to numIt do
                do! scanner.Value
            do! stopwatch.Stop()

            //let! results = scanned.Gather()
            let! timing = stopwatch.ElapsedMilliseconds

            return timing / 1000.0 }
                
        let timing = calc |> PCalc.run
        let bytes = (2 * sizeof<'TI> + sizeof<'TV>) * count |> float
        let throughput = (float count) * (float numIt) / timing
        let bandwidth = bytes * (float numIt) / timing
        printf "Alea.cuBase:\tN=(%d)\t%9.3f M/s  %7.3f GB/s\t timing: %9.3f\n" count (throughput / 1.0e6) (bandwidth / 1.0e9) timing


let printScanType (mgpuScanType:int) =
    if mgpuScanType = ExclusiveScan then
        printfn "Scan Type: Exclusive"
    else
        printfn "Scan Type: Inclusive"

      
//[<Test>]
//let ``simple scan with stats``() =
//    let values = Array.init 16 (fun i -> i)
//    let op = scanOp ScanOpTypeAdd 0
//    let test = testScanStats defaultScanType op totalAtEnd
//    test values
//    
//[<Test>]
//let ``simple scan with output display & verify`` () =
//    let values = Array.init 16 (fun i -> i)
//    let op = scanOp ScanOpTypeAdd 0
//    let gold data = data |> Array.scan (+) 0
//    let test = testScanVerify defaultScanType op totalAtEnd true gold verify
//    test values

//need to fix, they are no different
[<Test>]
let ``compare totalAtEnd & totalNotAtEnd`` () =
    let hValues = Array.init 20 (fun _ -> 20.0)
    printfn "Initial values: %A" hValues
    let hResult = Array.scan (+) 0.0 hValues
    printfn "Host Scan Result ==> Count: (%d),  %A" hResult.Length hResult
    let op = scanOp ScanOpTypeAdd 0.0
    let calc (tAe:int) = pcalc {  
                                let scan = worker.LoadPModule(MGPU.PArray.scan defaultScanType op tAe).Invoke
                                let n = hValues.Length
                                let! dValues = DArray.scatterInBlob worker hValues
                                let! dResults = DArray.createInBlob<float> worker n
                                let! scanner, scanned = scan dValues
                                do! scanner.Value
                                return! scanned.Gather()}
    let dResult = calc totalAtEnd |> PCalc.run
    printfn "Device Scan Result (Total At End) ==> Count: (%d),  %A" dResult.Length dResult
    let dResult = calc totalNotAtEnd |> PCalc.run
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
                                let scan = worker.LoadPModule(MGPU.PArray.scan stype op tAe).Invoke
                                let n = hValues.Length
                                let! dValues = DArray.scatterInBlob worker hValues
                                let! dResults = DArray.createInBlob<int> worker n
                                let! scanner, scanned = scan dValues
                                do! scanner.Value
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
//

//let sprintTB tp bw = sprintf "\t%9.3f M/s\t%7.3f GB/s\t" (tp / 1e6) (bw / 1e9)
let sprintTB tp bw = sprintf "\t%9.3f M/s\t%7.3f GB/s\t" tp bw
let printfnMgpuThrustTB ((mtp, mbw), (ttp, tbw)) =
    printfn "%s\t%s\t%s\t%s" "MGPU: " (sprintTB mtp mbw) "Thrust: " (sprintTB ttp tbw)


[<Test>]
let ``benchmark scan int`` () =
// Sample moderngpu lib results
//Benchmarking scan on type int
///         mgpu thru       mgpu band       thrust thru     thrust band
//   10K:   522.294 M/s    6.268 GB/s      26.547 M/s    0.319 GB/s
//   50K:  1409.874 M/s   16.918 GB/s     123.302 M/s    1.480 GB/s
//  100K:  2793.189 M/s   33.518 GB/s     531.724 M/s    6.381 GB/s
//  200K:  5016.043 M/s   60.193 GB/s     971.429 M/s   11.657 GB/s
//  500K:  7157.139 M/s   85.886 GB/s    1889.905 M/s   22.679 GB/s
//    1M:  7842.271 M/s   94.107 GB/s    2313.313 M/s   27.760 GB/s
//    2M:  8115.966 M/s   97.392 GB/s    3633.622 M/s   43.603 GB/s
//    5M:  8301.897 M/s   99.623 GB/s    4492.701 M/s   53.912 GB/s
//   10M:  8325.225 M/s   99.903 GB/s    4710.093 M/s   56.521 GB/s
//   20M:  8427.884 M/s  101.135 GB/s    5088.089 M/s   61.057 GB/s

//Benchmarking scan on type int
///                       mgpu thru M/s  mgpu band GB/s  thrust thru M/s  thrust band GB/s
    let mgpuStats_Int = [   ( 522.294,     6.268),     (  26.547,     0.319);
                            (1409.874,    16.918),     ( 123.302,     1.480); 
                            (2793.189,    33.518),     ( 531.724,     6.381); 
                            (5016.043,    60.193),     ( 971.429,    11.657);
                            (7157.139,    85.886),     (1889.905,    22.679);
                            (7842.271,    94.107),     (2313.313,    27.760);
                            (8115.966,    97.392),     (3633.622,    43.603);
                            (8301.897,    99.623),     (4492.701,    53.912);
                            (8325.225,    99.903),     (4710.093,    56.521);
                            (8427.884,   101.135),     (5088.089,    61.057)]
    
    let scan = benchmarkScan ExclusiveScan (scanOp ScanOpTypeAdd 0) totalNotAtEnd
    let genData n = rngGenericArray n
    (sourceCounts, nIterations) ||> List.iteri2 (fun i sc ni -> scan (genData sc) ni
                                                                i |> List.nth mgpuStats_Int |> printfnMgpuThrustTB)

[<Test>]
let ``benchmark scan int64`` () =
// Sample moderngpu lib results
//Benchmarking scan on type int64
///         mgpu thru       mgpu band       thrust thru     thrust band
//   10K:   364.793 M/s    8.755 GB/s      62.907 M/s    1.510 GB/s
//   50K:  1494.888 M/s   35.877 GB/s     264.719 M/s    6.353 GB/s
//  100K:  2588.645 M/s   62.127 GB/s     467.517 M/s   11.220 GB/s
//  200K:  3098.852 M/s   74.372 GB/s     823.708 M/s   19.769 GB/s
//  500K:  3877.090 M/s   93.050 GB/s    1534.737 M/s   36.834 GB/s
//    1M:  4041.277 M/s   96.991 GB/s    2158.156 M/s   51.796 GB/s
//    2M:  4131.842 M/s   99.164 GB/s    2669.093 M/s   64.058 GB/s
//    5M:  4193.033 M/s  100.633 GB/s    3088.460 M/s   74.123 GB/s
//   10M:  4218.722 M/s  101.249 GB/s    3264.636 M/s   78.351 GB/s
//   20M:  4229.460 M/s  101.507 GB/s    3169.477 M/s   76.067 GB/s

//Benchmarking scan on type int64
    let mgpuStats_Int64 = [ ( 364.793,     8.755),  (  62.907,     1.510); 
                            (1494.888,    35.877),  ( 264.719,     6.353); 
                            (2588.645,    62.127),  ( 467.517,    11.220); 
                            (3098.852,    74.372),  ( 823.708,    19.769); 
                            (3877.090,    93.050),  (1534.737,    36.834); 
                            (4041.277,    96.991),  (2158.156,    51.796); 
                            (4131.842,    99.164),  (2669.093,    64.058); 
                            (4193.033,   100.633),  (3088.460,    74.123); 
                            (4218.722,   101.249),  (3264.636,    78.351); 
                            (4229.460,   101.507),  (3169.477,    76.067) ]

    let scan = benchmarkScan defaultScanType (scanOp ScanOpTypeAdd 0L) totalNotAtEnd
    let genData n = rngGenericArray n
    (sourceCounts, nIterations) ||> List.iteri2 (fun i sc ni -> scan (genData sc) ni
                                                                i |> List.nth mgpuStats_Int64 |> printfnMgpuThrustTB)

[<Test>]
let ``benchmark scan float32`` () =
    let scan = benchmarkScan defaultScanType (scanOp ScanOpTypeAdd 0.f) totalNotAtEnd
    let genData n = rngGenericArray n
    (sourceCounts, nIterations) ||> List.iter2 (fun s i -> scan (genData s) i)

[<Test>]
let ``benchmark scan float`` () =
    let scan = benchmarkScan defaultScanType (scanOp ScanOpTypeAdd 0.0) totalNotAtEnd
    let genData n = rngGenericArray n
    (sourceCounts, nIterations) ||> List.iter2 (fun s i -> scan (genData s) i)

//[<Test>]
//let ``simple scan, past cutoff (n > 20000)`` () =
//    let values = Array.init 21000 (fun i -> i)
//    let op = scanOp ScanOpTypeAdd 0
//    let test = testScanStats defaultScanType op totalAtEnd
//    test values
//
//[<Test>]
//let ``simple scan with size iter & stats`` () =
//    let op = scanOp ScanOpTypeAdd 0
//    let test = testScanStats defaultScanType op totalAtEnd
//    sizes1L |> Seq.iter (fun count ->  
//        test (Array.init count (fun i -> i)))
//
//[<Test>]
//let ``simple scan with size iter, display, & verify`` () =
//    let op = scanOp ScanOpTypeAdd 0
//    let gold data = data |> Array.scan (+) 0
//    let test = testScanVerify defaultScanType op totalAtEnd true gold verify
//    sizes1L |> Seq.iter (fun count ->  
//        test (Array.init count (fun i -> i)))
//
//[<Test>]
//let ``scan with big size iter & stats`` () =
//    let op = scanOp ScanOpTypeAdd 0
//    let test = testScanStats defaultScanType op totalAtEnd
//    sizes2L |> Seq.iter (fun count ->  
//        test (Array.init count (fun i -> i)))


//[<Test>]
//let ``compare mpgu scan and alea.extension scan`` () =
//    let scan_extension = worker.LoadPModule(Extension.PArray.sumscan()).Invoke
//    let scan_mgpu = worker.LoadPModule(MGPU.PArray.scan defaultScanType (scanOp ScanOpTypeAdd 0) totalAtEnd).Invoke
//
//    let test verify (hValues:int[]) = pcalc {
//        let n = hValues.Length
//        printfn "Testing size %d..." n
//
//        let! dValues = DArray.scatterInBlob worker hValues
//        let! dResult_extension = scan_extension false dValues
//        
//        // need to fix this set-up... "getScanResult" is just temporary
//        let dResult_mgpu = getScanResult defaultScanType (scanOp ScanOpTypeAdd 0) totalAtEnd hValues
//
//        if not verify then
//            let hResult = hValues |> Array.scan (+) 0 |> getHostScanResult defaultScanType n
//
//            let! dResult_extension = dResult_extension.Gather()
//            (hResult, dResult_extension) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))
//
//            //let dResults_mgpu = (dResults_mgpu.Value)
//            (hResult, dResult_mgpu) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))
//            (dResult_extension, dResult_mgpu) ||> Array.iter2 (fun d1 d2 -> Assert.AreEqual(d1,d2))
//
//        else do! PCalc.force() }
//
//    let values1 n = Array.init n (fun _ -> 1)
//    let values2 n = Array.init n (fun _ -> -1)
//    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.Next(-100, 100))
//
//    sizes1L |> Seq.iter (fun n -> values1 n |> test true |> PCalc.run)
//    sizes1L |> Seq.iter (fun n -> values1 n |> test true |> PCalc.run)
//    sizes1L |> Seq.iter (fun n -> values1 n |> test true |> PCalc.run)
//
//    let n = 2097152
//    let test = values1 n |> test false
//
//    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.[scanTypeString defaultScanType].DumpLogs()
//    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
//    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))





///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////
///////////////// Sample Scan Benchmark Stats from moderngpu lib
//GeForce GTX 560 Ti : 1700.000 Mhz   (Ordinal 0)
//8 SMs enabled. Compute Capability sm_21
//FreeMem:    548MB   TotalMem:   1024MB.
//Mem Clock: 2004.000 Mhz x 256 bits   (128.256 GB/s)
//ECC Disabled
//
//Benchmarking scan on type int
///         mgpu thru       mgpu band       thrust thru     thrust band
//   10K:   522.294 M/s    6.268 GB/s      26.547 M/s    0.319 GB/s
//   50K:  1409.874 M/s   16.918 GB/s     123.302 M/s    1.480 GB/s
//  100K:  2793.189 M/s   33.518 GB/s     531.724 M/s    6.381 GB/s
//  200K:  5016.043 M/s   60.193 GB/s     971.429 M/s   11.657 GB/s
//  500K:  7157.139 M/s   85.886 GB/s    1889.905 M/s   22.679 GB/s
//    1M:  7842.271 M/s   94.107 GB/s    2313.313 M/s   27.760 GB/s
//    2M:  8115.966 M/s   97.392 GB/s    3633.622 M/s   43.603 GB/s
//    5M:  8301.897 M/s   99.623 GB/s    4492.701 M/s   53.912 GB/s
//   10M:  8325.225 M/s   99.903 GB/s    4710.093 M/s   56.521 GB/s
//   20M:  8427.884 M/s  101.135 GB/s    5088.089 M/s   61.057 GB/s
//
//Benchmarking scan on type int64
///         mgpu thru       mgpu band       thrust thru     thrust band
//   10K:   364.793 M/s    8.755 GB/s      62.907 M/s    1.510 GB/s
//   50K:  1494.888 M/s   35.877 GB/s     264.719 M/s    6.353 GB/s
//  100K:  2588.645 M/s   62.127 GB/s     467.517 M/s   11.220 GB/s
//  200K:  3098.852 M/s   74.372 GB/s     823.708 M/s   19.769 GB/s
//  500K:  3877.090 M/s   93.050 GB/s    1534.737 M/s   36.834 GB/s
//    1M:  4041.277 M/s   96.991 GB/s    2158.156 M/s   51.796 GB/s
//    2M:  4131.842 M/s   99.164 GB/s    2669.093 M/s   64.058 GB/s
//    5M:  4193.033 M/s  100.633 GB/s    3088.460 M/s   74.123 GB/s
//   10M:  4218.722 M/s  101.249 GB/s    3264.636 M/s   78.351 GB/s
//   20M:  4229.460 M/s  101.507 GB/s    3169.477 M/s   76.067 GB/s
//
//Benchmarking max-index on type int
//          mgpu thru       mgpu band
//   10K:   132.254 M/s    0.529 GB/s
//   50K:   662.364 M/s    2.649 GB/s
//  100K:  1249.290 M/s    4.997 GB/s
//  200K:  2174.394 M/s    8.698 GB/s
//  500K:  4471.465 M/s   17.886 GB/s
//    1M:  7270.891 M/s   29.084 GB/s
//    2M: 10585.918 M/s   42.344 GB/s
//    5M: 14421.989 M/s   57.688 GB/s
//   10M: 16428.942 M/s   65.716 GB/s
//   20M: 14933.223 M/s   59.733 GB/s
//
//Benchmarking max-index on type int64
//          mgpu thru       mgpu band
//   10K:    97.126 M/s    0.777 GB/s
//   50K:   509.054 M/s    4.072 GB/s
//  100K:   906.303 M/s    7.250 GB/s
//  200K:  1436.067 M/s   11.489 GB/s
//  500K:  2689.148 M/s   21.513 GB/s
//    1M:  4018.140 M/s   32.145 GB/s
//    2M:  4928.488 M/s   39.428 GB/s
//    5M:  5692.572 M/s   45.541 GB/s
//   10M:  6000.538 M/s   48.004 GB/s
//   20M:  6195.764 M/s   49.566 GB/s
