module Test.Alea.CUDA.Extension.MGPU.Scan

open System
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Scan
open Alea.CUDA.Extension.MGPU.CTAScan
open Test.Alea.CUDA.Extension.MGPU.Util
open NUnit.Framework

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
    let hValues = Array.init 20 (fun _ -> 20)
    printfn "Initial values: %A" hValues
    let hResult = Array.scan (+) 0 hValues
    printfn "Host Scan Result ==> Count: (%d),  %A" hResult.Length hResult
    let op = scanOp ScanOpTypeAdd 0
    let calc (tAE:int) = pcalc {
                                        let scan = worker.LoadPModule(MGPU.PArray.scan 0 op tAE).Invoke
                                        let n = hValues.Length
                                        let! dValues = DArray.scatterInBlob worker hValues
                                        let! dResults = DArray.createInBlob<int> worker n
                                        let! scanner, scanned = scan dValues
                                        do! scanner.Value
                                        return! scanned.Gather()}
                                
    //let dResult : int[] = calc |> PCalc.run
    //printfn "Device result (%A)" dResult
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



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  BENCHMARKING
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//let sprintTB tp bw = sprintf "\t%9.3f M/s\t%7.3f GB/s\t" (tp / 1e6) (bw / 1e9)
let sprintTB tp bw = sprintf "\t%9.3f M/s\t%7.3f GB/s\t" tp bw
let printfnMgpuThrustTB ((mtp, mbw), (ttp, tbw)) =
    printfn "%s\t%s\t%s\t%s" "MGPU: " (sprintTB mtp mbw) "Thrust: " (sprintTB ttp tbw)


[<Test>]
let ``benchmark scan int`` () =
    let scan = benchmarkScan ExclusiveScan (scanOp ScanOpTypeAdd 0) totalNotAtEnd
    let genData n = rngGenericArray n
    (sourceCounts, nIterations) ||> List.iteri2 (fun i sc ni -> scan (genData sc) ni
                                                                i |> List.nth mgpuStats_Int |> printfnMgpuThrustTB)


//[<Test>]
//let ``benchmark scan int64`` () =
//    let scan = benchmarkScan defaultScanType (scanOp ScanOpTypeAdd 0L) totalNotAtEnd
//    let genData n = rngGenericArray n
//    (sourceCounts, nIterations) ||> List.iteri2 (fun i sc ni -> scan (genData sc) ni
//                                                                i |> List.nth mgpuStats_Int64 |> printfnMgpuThrustTB)
//
//[<Test>]
//let ``benchmark scan float32`` () =
//    let scan = benchmarkScan defaultScanType (scanOp ScanOpTypeAdd 0.f) totalNotAtEnd
//    let genData n = rngGenericArray n
//    (sourceCounts, nIterations) ||> List.iter2 (fun s i -> scan (genData s) i)
//
//[<Test>]
//let ``benchmark scan float`` () =
//    let scan = benchmarkScan defaultScanType (scanOp ScanOpTypeAdd 0.0) totalNotAtEnd
//    let genData n = rngGenericArray n
//    (sourceCounts, nIterations) ||> List.iter2 (fun s i -> scan (genData s) i)

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