module Test.Alea.CUDA.Extension.MGPU.IntervalMove

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU

open Alea.CUDA.Extension.Output.Util
open Alea.CUDA.Extension.Output.CSV
open Alea.CUDA.Extension.Output.Excel

open Test.Alea.CUDA.Extension.MGPU.BenchmarkStats

open NUnit.Framework

////////////////////////////
// set this to your device or add your device's C++ output to BenchmarkStats.fs
open Test.Alea.CUDA.Extension.MGPU.BenchmarkStats.GF560Ti
// in the future maybe we try to get the C++ to interop somehow
/////////////////////////////
open ModernGPU.IntervalExpandStats
open ModernGPU.IntervalMoveStats


let worker = getDefaultWorker()
let pScanner = new PScan()
let pIntervalMover = new PIntervalMove()
let op = (CTAScan.scanOp ScanOpTypeAdd 0)

let iexp_A_4typeStatsList = ModernGPU.IntervalExpandStats.AvgSegLength25.fourTypeStatsList
let iexp_B_4typeStatsList = ModernGPU.IntervalExpandStats.ConstCountChangingExpandRate.fourTypeStatsList

let imv_A_4typeStatsList = ModernGPU.IntervalMoveStats.AvgSegLength25.fourTypeStatsList
let imv_B_4typeStatsList = ModernGPU.IntervalMoveStats.ConstCountChangingExpandRate.fourTypeStatsList

let algName_iexp_A = "Interval Expand - Avg Seg Length 25 elements"
let iexp_A_KernelsUsed = [| "kernelIntervalExpand"; "kernelMergePartition" |]
let iexp_A_BMS4 = getFilledBMS4Object algName_iexp_A iexp_A_KernelsUsed worker.Device.Name "MGPU" sourceCounts nIterations iexp_A_4typeStatsList

let algName_iexp_B = "Interval Expand - Const Count 10M - Changing Expand Rate"
let iexp_B_KernelsUsed = [| "kernelIntervalExpand"; "kernelMergePartition" |]
let iexp_B_BMS4 = getFilledBMS4Object algName_iexp_B iexp_B_KernelsUsed worker.Device.Name "MGPU" constCounts constIterations iexp_B_4typeStatsList

let algName_imv_A = "Interval Move - Avg Seg Length 25 elements"
let imv_A_KernelsUsed = [| "kernelIntervalMove"; "kernelMergePartition" |]
let imv_A_BMS4 = getFilledBMS4Object algName_imv_A imv_A_KernelsUsed worker.Device.Name "MGPU" sourceCounts nIterations imv_A_4typeStatsList

let algName_imv_B = "Interval Move - Const Count 10M - Changing Expand Rate"
let imv_B_KernelsUsed = [| "kernelIntervalMove"; "kernelMergePartition" |]
let imv_B_BMS4 = getFilledBMS4Object algName_imv_B imv_B_KernelsUsed worker.Device.Name "MGPU" constCounts constIterations imv_B_4typeStatsList



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                              IMPORTANT                                                       //
//                                      Choose an Output Type                                                   // 
// This is a switch for all tests, and can do a lot of extra work.  Make sure you turn it off if you just       //
// want to see the console prInt32s.                                                                            //
let outputType = OutputTypeNone // Choices are CSV, Excel, Both, or None. Set to None for doing kernel timing   //
// only one path, we aren't auto-saving excel stuff yet                                                         //
let workingPathExpA = (getWorkingOutputPaths deviceFolderName algName_iexp_A).CSV                               //
let workingPathExpB = (getWorkingOutputPaths deviceFolderName algName_iexp_B).CSV                               //
let workingPathMovA = (getWorkingOutputPaths deviceFolderName algName_imv_A).CSV                                //
let workingPathMovB = (getWorkingOutputPaths deviceFolderName algName_imv_B).CSV                                //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////




let genRandomIntervals (count:int) (numTerms:int) (terms:int[]) =
    let dquot, drem = (count / numTerms), (count % numTerms)
    for i = 0 to numTerms - 1 do
        Array.set terms i (dquot + (if i < drem then 1 else 0))
    for i = 0 to numTerms - 2 do
        let r = rng.Next(numTerms - i - 1)
        let x = min terms.[r] terms.[i + r]
        let r2 = rng.Next(-x,x)
        Array.set terms r (terms.[r] - r2)
        Array.set terms (i + r) (terms.[i + r] + r2)
        swap terms.[r] terms.[i + r]

let benchmarkIntervalExpand (version:char) (hSource:'T[]) (count:int) (numIt:int) (numTerms:int) (testIdx:int) =
    let scan = worker.LoadPModule(pScanner.Scan()).Invoke
    let expand = worker.LoadPModule(pIntervalMover.IntervalExpandFunc()).Invoke
    
    let terms = Array.zeroCreate numTerms
    genRandomIntervals count numTerms terms

    let total, scannedCounts =
        pcalc { let! dCounts = DArray.scatterInBlob worker terms
                return! scan dCounts} |> PCalc.run
        
    let calc = pcalc {
        let! dCounts = DArray.scatterInBlob worker scannedCounts
        
        let! expand = expand total numTerms
        let! dData = DArray.scatterInBlob worker hSource
        let! dDest = DArray.createInBlob worker total
        
        // warm up
        do! expand dCounts dData dDest

        let! dStopwatch = DStopwatch.startNew worker
        for i = 1 to numIt do
            do! expand dCounts dData dDest
        do! dStopwatch.Stop()

        let! results = dDest.Gather()
        let! timing = dStopwatch.ElapsedMilliseconds

        return results, timing }

    let hResults, timing' = calc |> PCalc.runInWorker worker
    let timing = timing' / 1000.0
    let bytes = sizeof<'T> * (count + numTerms) + sizeof<int> * numTerms |> float
    let throughput = (float count) * (float numIt) / timing
    let bandwidth = bytes * (float numIt) / timing
    printfn "%9d: %9.3f M/s %9.3f GB/s %6.3f ms x %4d = %7.3f ms"
        count
        (throughput / 1e6)
        (bandwidth / 1e9)
        (timing' / (float numIt))
        numIt
        timing'

    match version with
    | 'A' ->
        match typeof<'T> with
        | x when x = typeof<int> -> iexp_A_BMS4.Int32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<int64> -> iexp_A_BMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float32> -> iexp_A_BMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float> -> iexp_A_BMS4.Float64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | _ -> ()
    | 'B' ->
        match typeof<'T> with
        | x when x = typeof<int> -> iexp_B_BMS4.Int32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<int64> -> iexp_B_BMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float32> -> iexp_B_BMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float> -> iexp_B_BMS4.Float64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | _ -> ()
    | _ -> ()


let benchmarkIntervalMove (version:char) (hSource:'T[]) (count:int) (numIt:int) (numTerms:int) (testIdx:int) =
    let scan = worker.LoadPModule(pScanner.Scan()).Invoke    
    let move = worker.LoadPModule(pIntervalMover.IntervalMoveFunc()).Invoke

    let permGather, permScatter = (Array.init numTerms (fun i -> i)), 
                                  (Array.init numTerms (fun i -> i))

    for i = 0 to numTerms - 2 do
        let r1 = rng.Next(numTerms - i - 1)
        let r2 = rng.Next(numTerms - i - 1)
        swap permGather.[i] permGather.[i + r1]
        swap permScatter.[i] permScatter.[i + r2]

    let terms = Array.zeroCreate numTerms
    genRandomIntervals count numTerms terms

    let gather, scatter = (Array.zeroCreate numTerms), (Array.zeroCreate numTerms)

    for i = 0 to numTerms - 1 do
        Array.set gather permGather.[i] terms.[i]
        Array.set scatter permScatter.[i] terms.[i]

    let mutable x, y = 0, 0
    for i = 0 to numTerms - 1 do
        let x2, y2 = gather.[i], scatter.[i]
        Array.set gather i x
        Array.set scatter i y
        x <- x + x2
        y <- y + y2

    let gatherHost, scatterHost = (Array.zeroCreate numTerms), (Array.zeroCreate numTerms)
    for i = 0 to numTerms - 1 do
        Array.set gatherHost i (gather.[permGather.[i]])
        Array.set scatterHost i (scatter.[permScatter.[i]])

    let total, scannedCounts =
        pcalc { let! dCounts = DArray.scatterInBlob worker terms
                return! scan dCounts } |> PCalc.run    

    let calc = pcalc {
        let! dCounts = DArray.scatterInBlob worker scannedCounts
        let! dGather = DArray.scatterInBlob worker gatherHost
        let! dScatter = DArray.scatterInBlob worker scatterHost
        let! dSource = DArray.scatterInBlob worker hSource
        let! dDest = DArray.createInBlob<'T> worker count

        let! move = move count numTerms

        // warm up
        do! move dSource dGather dScatter dCounts  dDest

        let! dStopwatch = DStopwatch.startNew worker
        for i = 1 to numIt do
            do! move dSource dGather dScatter dCounts dDest
        do! dStopwatch.Stop()

        let! results = dDest.Gather()
        let! timing = dStopwatch.ElapsedMilliseconds

        return results, timing }

    let hResults, timing' = calc |> PCalc.runInWorker worker
    let timing = timing' / 1000.0
    let bytes = 3 * sizeof<int> * numTerms + 2 * sizeof<'T> * count |> float
    let throughput = (float count) * (float numIt) / timing
    let bandwidth = bytes * (float numIt) / timing
    printfn "%9d: %9.3f M/s %9.3f GB/s %6.3f ms x %4d = %7.3f ms"
        count
        (throughput / 1e6)
        (bandwidth / 1e9)
        (timing' / (float numIt))
        numIt
        timing'

    match version with
    | 'A' ->
        match typeof<'T> with
        | x when x = typeof<int> -> imv_A_BMS4.Int32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<int64> -> imv_A_BMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float32> -> imv_A_BMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float> -> imv_A_BMS4.Float64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | _ -> ()
    | 'B' ->
        match typeof<'T> with
        | x when x = typeof<int> -> imv_B_BMS4.Int32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<int64> -> imv_B_BMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float32> -> imv_B_BMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float> -> imv_B_BMS4.Float64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | _ -> ()
    | _ -> ()


[<Test>]
let ``Demo Interval Expand`` () =
    let expand = worker.LoadPModule(pIntervalMover.IntervalExpand()).Invoke
    let scan = worker.LoadPModule(pScanner.Scan(op)).Invoke

    printfn "INTERVAL-EXPAND DEMONSTRATION:\n"

    let numInputs = 20
    printfn "Expand counts (n = %d):" numInputs
    let hCounts = [| 2;    5;    7;   16;    0;    1;    0;    0;   14;   10;
                     3;   14;    2;    1;   11;    2;    1;    0;    5;    6 |]
    printfn "%A\n" hCounts

    printfn "Expand values (n = %d):" numInputs
    let hInputs = [|  1;    1;    2;    3;    5;    8;   13;   21;   34;   55;
                     89;  144;  233;  377;  610;  987; 1597; 2584; 4181; 6765 |]
    printfn "%A\n" hInputs

    printfn "Expanded data (MGPU result):"
    let hAnswer = [|     1;    1;    1;    1;    1;    1;    1;    2;    2;    2;
                        2;    2;    2;    2;    3;    3;    3;    3;    3;    3;
                        3;    3;    3;    3;    3;    3;    3;    3;    3;    3;
                        8;   34;   34;   34;   34;   34;   34;   34;   34;   34;
                       34;   34;   34;   34;   34;   55;   55;   55;   55;   55;
                       55;   55;   55;   55;   55;   89;   89;   89;  144;  144;
                      144;  144;  144;  144;  144;  144;  144;  144;  144;  144;
                      144;  144;  233;  233;  377;  610;  610;  610;  610;  610;
                      610;  610;  610;  610;  610;  610;  987;  987; 1597; 4181;
                     4181; 4181; 4181; 4181; 6765; 6765; 6765; 6765; 6765; 6765 |]
    printfn "%A\n" hAnswer

    let dScanResults = pcalc {
        let! dCounts = DArray.scatterInBlob worker hCounts
        let! total, scanned = scan dCounts
        let! scanned = scanned.Gather()
        let! total = total.Gather()
        return total, scanned } |> PCalc.run

    let hTotal, hCounts_Scanned = dScanResults
    
    let total = hTotal.[0]
    let hCounts = hCounts_Scanned
    printfn "total: %A" total
    let dExpandResult = 
        pcalc {
            let! dCounts = DArray.scatterInBlob worker hCounts
            let! dInputs = DArray.scatterInBlob worker hInputs
            let! result = expand total dCounts dInputs
            let! result = result.Gather()
        return result } |> PCalc.run

    printfn "Expanded data (Alea.cuBase result):"
    printfn "%A\n" dExpandResult

    (hAnswer, dExpandResult) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
    printfn "\nInterval Expand Passed\n"

[<Test>]
let ``Demo Interval Expand Func`` () =
    // using the "in place" function from pIntervalMover (the one used for benchmarking)
    let expand = worker.LoadPModule(pIntervalMover.IntervalExpandFunc()).Invoke
    let scan = worker.LoadPModule(pScanner.ScanFuncReturnTotal(op)).Invoke

    printfn "INTERVAL-EXPAND DEMONSTRATION:\n"

    let numInputs = 20
    printfn "Expand counts (n = %d):" numInputs
    let hCounts = [| 2;    5;    7;   16;    0;    1;    0;    0;   14;   10;
                     3;   14;    2;    1;   11;    2;    1;    0;    5;    6 |]
    printfn "%A\n" hCounts

    printfn "Expand values (n = %d):" numInputs
    let hInputs = [|  1;    1;    2;    3;    5;    8;   13;   21;   34;   55;
                     89;  144;  233;  377;  610;  987; 1597; 2584; 4181; 6765 |]
    printfn "%A\n" hInputs

    printfn "Expanded data (MGPU result):"
    let hAnswer = [|     1;    1;    1;    1;    1;    1;    1;    2;    2;    2;
                        2;    2;    2;    2;    3;    3;    3;    3;    3;    3;
                        3;    3;    3;    3;    3;    3;    3;    3;    3;    3;
                        8;   34;   34;   34;   34;   34;   34;   34;   34;   34;
                       34;   34;   34;   34;   34;   55;   55;   55;   55;   55;
                       55;   55;   55;   55;   55;   89;   89;   89;  144;  144;
                      144;  144;  144;  144;  144;  144;  144;  144;  144;  144;
                      144;  144;  233;  233;  377;  610;  610;  610;  610;  610;
                      610;  610;  610;  610;  610;  610;  987;  987; 1597; 4181;
                     4181; 4181; 4181; 4181; 6765; 6765; 6765; 6765; 6765; 6765 |]
    printfn "%A\n" hAnswer
    
    let dExpandResult = 
        pcalc {
            // scan
            let! dCounts = DArray.scatterInBlob worker hCounts
            let! dScannedCounts = DArray.createInBlob worker numInputs
            let! scan = scan dCounts.Length
            let! total = scan dCounts dScannedCounts

            // expand
            let! expand = expand total numInputs
            let! dInputs = DArray.scatterInBlob worker hInputs
            let! dDest = DArray.createInBlob worker total
            
            do! expand dScannedCounts dInputs dDest            
            let! result = dDest.Gather()
        return result } |> PCalc.run

    printfn "Expanded data (Alea.cuBase result):"
    printfn "%A\n" dExpandResult
    (hAnswer, dExpandResult) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))


[<Test>]
let ``Demo Interval Expand Func 2`` () =
    let expand = worker.LoadPModule(pIntervalMover.IntervalExpandFunc()).Invoke
    let scan = worker.LoadPModule(pScanner.ScanFuncReturnTotal(op)).Invoke
    let numInputs = 50
    let count = numInputs * 25
    let terms = Array.zeroCreate numInputs
    genRandomIntervals count numInputs terms
    printfn "terms:\n%A" terms
    let hInputs = Array.init numInputs (fun i -> i)     

    let dExpandResult = 
        pcalc {
            let! dCounts = DArray.scatterInBlob worker terms
            let! dScannedCounts = DArray.createInBlob worker numInputs
            
            let! scan = scan dCounts.Length
            let! total = scan dCounts dScannedCounts

            let! expand = expand total numInputs

            let! dInputs = DArray.scatterInBlob worker hInputs
            let! dDest = DArray.createInBlob worker total
                        
            do! expand dScannedCounts dInputs dDest            
            let! result = dDest.Gather()
        return result } |> PCalc.run

    printfn "Expanded data (Alea.cuBase result):"
    printfn "%A\n" dExpandResult


[<Test>]
let ``Demo Interval Move`` () =
    let scan = worker.LoadPModule(pScanner.Scan(op)).Invoke
    let move = worker.LoadPModule(pIntervalMover.IntervalMove()).Invoke

    printfn "INTERVAL-MOVE DEMONSTRATION:\n"

    let numInputs = 20
    printfn "Interval counts:"
    let hCounts = [|    3;    9;    1;    9;    8;    5;   10;    2;    5;    2;
                        8;    6;    5;    2;    4;    0;    8;    2;    5;    6 |]
    printfn "%A\n" hCounts

    printfn "Interval gather:"
    let hGather = [|    75;   86;   17;    2;   67;   24;   37;   11;   95;   35;
                        52;   18;   47;    0;   13;   75;   78;   60;   62;   29 |]
    printfn "%A\n" hGather

    printfn "Interval scatter:"
    let hScatter = [|   10;   80;   99;   27;   41;   71;   15;    0;   36;   13;
                        89;   49;   66;   97;   76;   76;    2;   25;   61;   55 |]
    printfn "%A\n" hScatter

    printfn "Moved data (MGPU result):"
    let hAnswer = [| 11;   12;   78;   79;   80;   81;   82;   83;   84;   85;
                    75;   76;   77;   35;   36;   37;   38;   39;   40;   41;
                    42;   43;   44;   45;   46;   60;   61;    2;    3;    4;
                     5;    6;    7;    8;    9;   10;   95;   96;   97;   98;
                    99;   67;   68;   69;   70;   71;   72;   73;   74;   18;
                    19;   20;   21;   22;   23;   29;   30;   31;   32;   33;
                    34;   62;   63;   64;   65;   66;   47;   48;   49;   50;
                    51;   24;   25;   26;   27;   28;   13;   14;   15;   16;
                    86;   87;   88;   89;   90;   91;   92;   93;   94;   52;
                    53;   54;   55;   56;   57;   58;   59;    0;    1;   17 |]
    printfn "%A\n" hAnswer


    let dScanResults = 
        pcalc {
            let! dCounts = DArray.scatterInBlob worker hCounts
            let! total, scanned = scan dCounts
            let! scanned = scanned.Gather()
            let! total = total.Gather()
        return total, scanned } |> PCalc.run

    let hTotal, hCounts_Scanned = dScanResults
    
    let total = hTotal.[0]
    let hCounts = hCounts_Scanned

    let dMoveResult = 
        pcalc {
            let! dGather = DArray.scatterInBlob worker hGather
            let! dScatter = DArray.scatterInBlob worker hScatter
            let! dCounts = DArray.scatterInBlob worker hCounts
            let! dInput = DArray.scatterInBlob worker ([| 0..total |])          
            let! result = move total dGather dScatter dCounts dInput
            let! result = result.Gather()
        return result } |> PCalc.run

    printfn "Moved data (Alea.cuBase result):"
    printfn "%A\n" dMoveResult

    (hAnswer, dMoveResult) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
    printfn "\nInterval Move Passed\n"




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  BENCHMARKING                                                                                                //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
[<Test>]
let ``IntervalExpand moderngpu benchmark (A) : 4type`` () =
    printfn "\nInterval Expand, Avg Seg Length = 25, Int32"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : int[] = Array.init (ns / 25) (fun i -> i)        
        benchmarkIntervalExpand 'A' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathExpA iexp_A_BMS4.Int32s

    printfn "\nInterval Expand, Avg Seg Length = 25, Int64"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : int64[] = Array.init ns (fun i -> int64 i)
        benchmarkIntervalExpand 'A' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathExpA iexp_A_BMS4.Int64s

    printfn "\nInterval Expand, Avg Seg Length = 25, Float32"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : float32[] = Array.init ns (fun i -> float32 i)
        benchmarkIntervalExpand 'A' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathExpA iexp_A_BMS4.Float32s

    printfn "\nInterval Expand, Avg Seg Length = 25, Float64"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : float[] = Array.init ns (fun i -> float i)
        benchmarkIntervalExpand 'A' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathExpA iexp_A_BMS4.Float64s


[<Test>]
let ``IntervalExpand moderngpu benchmark (B) : 4type`` () =
    printfn "\nInterval Expand, 10M Count & Changing Expand Rate, Int32"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : int[] = Array.init ns (fun i -> i)
        benchmarkIntervalExpand 'B' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathExpB iexp_B_BMS4.Int32s

    printfn "\nInterval Expand, 10M Count & Changing Expand Rate, Int64"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : int64[] = Array.init ns (fun i -> int64 i)
        benchmarkIntervalExpand 'B' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathExpB iexp_B_BMS4.Int64s

    printfn "\nInterval Expand, 10M Count & Changing Expand Rate, Float32"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : float32[] = Array.init ns (fun i -> float32 i)
        benchmarkIntervalExpand 'B' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathExpB iexp_B_BMS4.Float32s

    printfn "\nInterval Expand, 10M Count & Changing Expand Rate, Float64"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : float[] = Array.init ns (fun i -> float i)
        benchmarkIntervalExpand 'B' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathExpB iexp_B_BMS4.Float64s



[<Test>]
let ``IntervalMove moderngpu benchmark (A) : 4type`` () =
    printfn "\nInterval Move, Avg Seg Length = 25, Int32"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : int[] = Array.init ns (fun i -> i)
        benchmarkIntervalMove 'A' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathMovA imv_A_BMS4.Int32s

    printfn "\nInterval Move, Avg Seg Length = 25, Int64"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : int64[] = Array.init ns (fun i -> int64 i)
        benchmarkIntervalMove 'A' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathMovA imv_A_BMS4.Int64s

    printfn "\nInterval Move, Avg Seg Length = 25, Float32"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : float32[] = Array.init ns (fun i -> float32 i)
        benchmarkIntervalMove 'A' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathMovA imv_A_BMS4.Float32s

    printfn "\nInterval Move, Avg Seg Length = 25, Float64"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : float[] = Array.init ns (fun i -> float i)
        benchmarkIntervalMove 'A' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathMovA imv_A_BMS4.Float64s


[<Test>]
let ``IntervalMove moderngpu benchmark (B) : 4type`` () =
    printfn "\nInterval Move, 10M Count & Changing Expand Rate, Int32"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : int[] = Array.init ns (fun i -> i)
        benchmarkIntervalMove 'B' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathMovB imv_B_BMS4.Int32s

    printfn "\nInterval Move, 10M Count & Changing Expand Rate, Int64"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : int64[] = Array.init ns (fun i -> int64 i)
        benchmarkIntervalMove 'B' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathMovB imv_B_BMS4.Int64s

    printfn "\nInterval Move, 10M Count & Changing Expand Rate, Float32"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : float32[] = Array.init ns (fun i -> float32 i)
        benchmarkIntervalMove 'B' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathMovB imv_B_BMS4.Float32s

    printfn "\nInterval Move, 10M Count & Changing Expand Rate, Float64"
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let input : float[] = Array.init ns (fun i -> float i)
        benchmarkIntervalMove 'B' input ns ni (ns / 25) i)
    benchmarkOutput outputType workingPathMovB imv_B_BMS4.Float64s

