module Test.Alea.CUDA.Extension.MGPU.Scan

open System
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Scan
open Alea.CUDA.Extension.MGPU.CTAScan
open NUnit.Framework
open Test.Alea.CUDA.Extension.TestUtilities
open Test.Alea.CUDA.Extension.TestUtilities.MGPU.ScanUtils


let totalAtEnd = 1
let totalNotAtEnd = 0

let defaultScanType = ExclusiveScan

let sizes1A = [|16; 32; 64; 128|]
let sizes1L = sizes1A |> Array.toList

let sizes2A = [|512; 1024; 2048; 3000; 6000; 12000; 24000; 100000; 1000000 |] 
let sizes2L = sizes2A |> Array.toList

    
      
[<Test>]
let ``simple scan with stats``() =
    let values = Array.init 16 (fun i -> i)
    let op = scanOp ScanOpTypeAdd 0
    let test = testScanStats defaultScanType op totalAtEnd
    test values
    
[<Test>]
let ``simple scan with output display & verify`` () =
    let values = Array.init 16 (fun i -> i)
    let op = scanOp ScanOpTypeAdd 0
    let gold data = data |> Array.scan (+) 0
    let test = testScanVerify defaultScanType op totalAtEnd true gold verify
    test values

//need to fix, they are no different
[<Test>]
let ``compare totalAtEnd & totalNotAtEnd`` () =
    let hValues = Array.init 8 (fun _ -> 8)
    printfn "Initial values: %A" hValues
    let hResult = Array.scan (+) 0 hValues
    printfn "Host Scan Result ==> Count: (%d),  %A" hResult.Length hResult
    let op = scanOp ScanOpTypeAdd 0
    let dResult = getScanResult defaultScanType op totalAtEnd hValues
    printfn "Device Scan Result (Total At End) ==> Count: (%d),  %A" dResult.Length dResult
    let dResult = getScanResult defaultScanType op totalNotAtEnd hValues
    printfn "Device Scan Result (Total Not At End) ==> Count: (%d),  %A" dResult.Length dResult

[<Test>]
let ``exact mgpu website example`` () =
    let hValues = [| 1; 7; 4; 0; 9; 4; 8; 8; 2; 4; 5; 5; 1; 7; 1; 1; 5; 2; 7; 6 |]
    let n = hValues.Length
    let exclusiveResult = [| 0; 1; 8; 12; 12; 21; 25; 33; 41; 43; 47; 52; 57; 58; 65; 66; 67; 72; 74; 81 |]
    let inclusiveResult = [| 1; 8; 12; 12; 21; 25; 33; 41; 43; 47; 52; 57; 58; 65; 66; 67; 72; 74; 81; 87 |]
    printfn "Initial values: %A" hValues
    let hExcResult, hIncResult = (Array.scan (+) 0 hValues) |> getHostExcAndIncScanResults n
    printfn "Host Exclusive Scan Result ==> Count: (%d), %A" hExcResult.Length hExcResult
    printfn "Host Inclusive Scan Result ==> Count: (%d), %A" hIncResult.Length hIncResult
        
    let op = scanOp ScanOpTypeAdd 0
    let dExcResult, dIncResult = getExclusiveAndInclusiveResults op totalNotAtEnd hValues
    printfn "Device Exclusive Scan Result ==>  Count: (%d), %A" dExcResult.Length dExcResult
    printfn "Device Inclusive Scan Result ==>  Count: (%d), %A" dIncResult.Length dIncResult

    printfn "Checking that device results = host results"
    (hExcResult, dExcResult) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
    (hIncResult, dIncResult) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
    printfn "Checking that device results = moderngpu website example results"
    (dExcResult, exclusiveResult) ||> Array.iter2 (fun d e -> Assert.That(e, Is.EqualTo(d).Within(eps)))
    (dIncResult, inclusiveResult) ||> Array.iter2 (fun d e -> Assert.That(e, Is.EqualTo(d).Within(eps)))

    

[<Test>]
let ``simple scan, past cutoff (n > 20000)`` () =
    let values = Array.init 21000 (fun i -> i)
    let op = scanOp ScanOpTypeAdd 0
    let test = testScanStats defaultScanType op totalAtEnd
    test values

[<Test>]
let ``simple scan with size iter & stats`` () =
    let op = scanOp ScanOpTypeAdd 0
    let test = testScanStats defaultScanType op totalAtEnd
    sizes1L |> Seq.iter (fun count ->  
        test (Array.init count (fun i -> i)))

[<Test>]
let ``simple scan with size iter, display, & verify`` () =
    let op = scanOp ScanOpTypeAdd 0
    let gold data = data |> Array.scan (+) 0
    let test = testScanVerify defaultScanType op totalAtEnd true gold verify
    sizes1L |> Seq.iter (fun count ->  
        test (Array.init count (fun i -> i)))

[<Test>]
let ``scan with big size iter & stats`` () =
    let op = scanOp ScanOpTypeAdd 0
    let test = testScanStats defaultScanType op totalAtEnd
    sizes2L |> Seq.iter (fun count ->  
        test (Array.init count (fun i -> i)))


[<Test>]
let ``compare mpgu scan and alea.extension scan`` () =
    let scan_extension = worker.LoadPModule(Extension.PArray.sumscan()).Invoke
    //let scan_mgpu = worker.LoadPModule(MGPU.PArray.scan (scanOp ScanOpTypeAdd 0) totalAtEnd).Invoke

    let test verify (hValues:int[]) = pcalc {
        let n = hValues.Length
        printfn "Testing size %d..." n

        let! dValues = DArray.scatterInBlob worker hValues
        let! dResult_extension = scan_extension false dValues
        
        // need to fix this set-up... "getScanResult" is just temporary
        let dResult_mgpu = getScanResult defaultScanType (scanOp ScanOpTypeAdd 0) totalAtEnd hValues

        if not verify then
            let hResult = hValues |> Array.scan (+) 0 |> getHostScanResult defaultScanType n

            let! dResult_extension = dResult_extension.Gather()
            (hResult, dResult_extension) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))

            //let dResults_mgpu = (dResults_mgpu.Value)
            (hResult, dResult_mgpu) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))

            (dResult_extension, dResult_mgpu) ||> Array.iter2 (fun d1 d2 -> Assert.AreEqual(d1,d2))

        else do! PCalc.force() }

    let values1 n = Array.init n (fun _ -> 1)
    let values2 n = Array.init n (fun _ -> -1)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.Next(-100, 100))

    sizes1L |> Seq.iter (fun n -> values1 n |> test true |> PCalc.run)
    sizes1L |> Seq.iter (fun n -> values1 n |> test true |> PCalc.run)
    sizes1L |> Seq.iter (fun n -> values1 n |> test true |> PCalc.run)

    let n = 2097152
    let test = values1 n |> test false

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.[scanTypeString defaultScanType].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

