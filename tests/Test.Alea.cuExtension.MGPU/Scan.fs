module Test.Alea.cuBase.MGPU.Scan

open System
open Alea.CUDA
open Alea.cuBase
open Alea.cuBase.MGPU
open NUnit.Framework


let worker = getDefaultWorker()
let pScanner = new PScan()


let totalAtEnd = 1
let totalNotAtEnd = 0

let defaultScanType = ExclusiveScan

let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608]


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