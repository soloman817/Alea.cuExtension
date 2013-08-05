module Test.Alea.CUDA.Extension.MGPU.LoadBalance

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU

open Test.Alea.CUDA.Extension.MGPU.Util
open Test.Alea.CUDA.Extension.MGPU.BenchmarkStats

open Alea.CUDA.Extension.Output.Util
open Alea.CUDA.Extension.Output.CSV
open Alea.CUDA.Extension.Output.Excel


open NUnit.Framework

////////////////////////////
// set this to your device or add your device's C++ output to BenchmarkStats.fs
open Test.Alea.CUDA.Extension.MGPU.BenchmarkStats.GF560Ti
// in the future maybe we try to get the C++ to interop somehow
/////////////////////////////
open ModernGPU.LoadBalanceStats


let worker = getDefaultWorker()
let rng = System.Random()
let pScanner = new PScan()
let pLBS = new PLoadBalanceSearch()

let op = (CTAScan.scanOp ScanOpTypeAdd 0)
let scanner = pScanner.Scan(op) |> worker.LoadPModule


let algNameA, algNameB = "Load Balance (A)", "Load Balance (B)"
let lbKernelsUsed = [| "kernelLoadBalance"; "kernelMergePartition" |]
let lbBMS_A = new BenchmarkStats(algNameA, lbKernelsUsed, worker.Device.Name, "Int32", "MGPU", sourceCounts, nIterations)
let lbBMS_B = new BenchmarkStats(algNameB, lbKernelsUsed, worker.Device.Name, "Int32", "MGPU", (List.init 11 (fun _ -> 10000000)), (List.init 11 (fun _ -> 300)))

// we can probably organize this a lot better, but for now, if you just change
// what module you open above and all of this should adjust accordingly
let oIntTP_A, oIntBW_A = Regular.int32_stats |> List.unzip
let oIntTP_B, oIntBW_B = ChangingRatio.int32_stats |> List.unzip


for i = 0 to sourceCounts.Length - 1 do
    // this is setting the opponent (MGPU) stats for the int type
    lbBMS_A.OpponentThroughput.[i].Value <- oIntTP_A.[i]
    lbBMS_A.OpponentBandwidth.[i].Value <- oIntBW_A.[i]

for i = 0 to 9 do
    lbBMS_B.OpponentThroughput.[i].Value <- oIntTP_B.[i]
    lbBMS_B.OpponentBandwidth.[i].Value <- oIntBW_B.[i]

    

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                              IMPORTANT                                                       //
//                                      Choose an Output Type                                                   // 
// This is a switch for all tests, and can do a lot of extra work.  Make sure you turn it off if you just       //
// want to see the console prInt32s.                                                                            //
let outputType = OutputTypeNone // Choices are CSV, Excel, Both, or None. Set to None for doing kernel timing   //
// only one path, we aren't auto-saving excel stuff yet                                                         //
let workingPathA = (getWorkingOutputPaths deviceFolderName algNameA).CSV
let workingPathB = (getWorkingOutputPaths deviceFolderName algNameB).CSV                                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


let benchmarkLoadBalance (version:char) (total:int) (numIt:int) (percentTerms:float) (testIdx:int) =
    let loadBalanceSearch = worker.LoadPModule(pLBS.SearchFunc()).Invoke
    let scan = scanner.Invoke

    let count = int(1.0 - percentTerms) * total
    let numTerms = total - count
    let dquot, drem = (count / numTerms), (count % numTerms)
    let mutable terms = Array.zeroCreate numTerms
    for i = 0 to numTerms - 1 do
        Array.set terms i (dquot + (if i < drem then 1 else 0))

    for i = 0 to numTerms - 2 do
        let r = rng.Next(numTerms - i - 1)
        let x = min terms.[r] terms.[i + r]
        let r2 = rng.Next(-x,x)
        terms.[r] <- terms.[r] - r2
        terms.[i + r] <- terms.[i + r] + r2
        swap terms.[r] terms.[i + r]

    let terms = terms

    let dScanResult = pcalc {
        let! dCounts = DArray.scatterInBlob worker terms
        let! total, scanned = scan dCounts
        let! scanned = scanned.Gather()
        let! total = total.Gather()        
        return total, scanned } |> PCalc.run

    let dTotal, dScannedCounts = dScanResult
    let scanTotal = dTotal.[0]

    let calc = pcalc {
        let! dCounts = DArray.scatterInBlob worker dScannedCounts
        let! dIndex = DArray.createInBlob worker count
        
        let! loadBalanceSearch = loadBalanceSearch count numTerms
        // warm up
        do! loadBalanceSearch dCounts dIndex

        let! dStopwatch = DStopwatch.startNew worker
        for i = 1 to numIt do
            do! loadBalanceSearch dCounts dIndex
        do! dStopwatch.Stop()

        let! results = dIndex.Gather()
        let! timing = dStopwatch.ElapsedMilliseconds

        return results, timing }

    let hResults, timing' = calc |> PCalc.runInWorker worker
    let timing = timing' / 1000.0
    let bytes = sizeof<int> * total |> float
    let throughput = (float total) * (float numIt) / timing
    let bandwidth = bytes * (float numIt) / timing

    printfn "%9d: %9.3f M/s %9.3f GB/s %6.3f ms x %4d = %7.3f ms"
        count
        (throughput / 1e6)
        (bandwidth / 1e9)
        (timing' / (float numIt))
        numIt
        timing'
    
    match version with
    | 'A' -> lbBMS_A.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | 'B' -> lbBMS_B.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | _ -> ()

[<Test>]
let ``Load Balance Search simple`` () =
//LOAD-BALANCING SEARCH DEMONSTRATION:
    let scan = scanner.Invoke

    let search = worker.LoadPModule(pLBS.Search()).Invoke

    //Object hCounts
    let hCounts = [|    4;    0;    5;    5;    0;    5;    5;    1;    3;    1;
                       0;    3;    1;    1;    3;    5;    5;    5;    5;    5;
                       0;    4;    5;    5;    5;    0;    2;    4;    4;    1;
                       0;    0;    2;    0;    5;    3;    4;    5;    5;    3;
                       3;    4;    0;    2;    5;    1;    5;    4;    4;    2 |]
    
    printfn "Object hCounts:\n%A" hCounts
        
    let dResult = pcalc {
        let! dCounts = DArray.scatterInBlob worker hCounts        
        let! total, scanned = scan dCounts
        let! scanned = scanned.Gather()
        let! total = total.Gather()        
        return total, scanned } |> PCalc.run

    let dTotal, dScannedCounts = dResult
    let dTotal = dTotal.[0]

    //Scan of object counts:
    let hScannedCounts = [|   0;    4;    4;    9;   14;   14;   19;   24;   25;   28;
                            29;   29;   32;   33;   34;   37;   42;   47;   52;   57;
                            62;   62;   66;   71;   76;   81;   81;   83;   87;   91;
                            92;   92;   92;   94;   94;   99;  102;  106;  111;  116;
                           119;  122;  126;  126;  128;  133;  134;  139;  143;  147 |]
    //Total:  149
    let hTotal = 149

    printfn "\nScan of object counts:\n%A" dScannedCounts
    printfn "\nTotal: %A" dTotal
    
    (hScannedCounts, dScannedCounts) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
    printfn "Scan Passed"
    if hTotal = dTotal then printfn "Total Passed"

    
    //Object references:
    let hObjRefs = [|   0;    0;    0;    0;    2;    2;    2;    2;    2;    3;
                        3;    3;    3;    3;    5;    5;    5;    5;    5;    6;
                        6;    6;    6;    6;    7;    8;    8;    8;    9;   11;
                       11;   11;   12;   13;   14;   14;   14;   15;   15;   15;
                       15;   15;   16;   16;   16;   16;   16;   17;   17;   17;
                       17;   17;   18;   18;   18;   18;   18;   19;   19;   19;
                       19;   19;   21;   21;   21;   21;   22;   22;   22;   22;
                       22;   23;   23;   23;   23;   23;   24;   24;   24;   24;
                       24;   26;   26;   27;   27;   27;   27;   28;   28;   28;
                       28;   29;   32;   32;   34;   34;   34;   34;   34;   35;
                       35;   35;   36;   36;   36;   36;   37;   37;   37;   37;
                       37;   38;   38;   38;   38;   38;   39;   39;   39;   40;
                       40;   40;   41;   41;   41;   41;   43;   43;   44;   44;
                       44;   44;   44;   45;   46;   46;   46;   46;   46;   47;
                       47;   47;   47;   48;   48;   48;   48;   49;   49      |]
    
    let dObjRefs = 
        pcalc {
            let! dCounts = DArray.scatterInBlob worker hScannedCounts            
            let! results = search dTotal dCounts
            let! results = results.Gather()            
        return results } |> PCalc.run
            
    printfn "\nObject References:\n%A" dObjRefs

    (hObjRefs, dObjRefs) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
    printfn "\nLoad Balance Search Passed\n"
         


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  BENCHMARKING                                                                                                //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
[<Test>]
let ``LoadBalance moderngpu benchmark (A) : int32`` () =
    let percentTerms = 0.25
    
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        printfn "ns = (%d); ni = (%d); i = (%d)" ns ni i
        benchmarkLoadBalance 'A' ns ni percentTerms i )
    benchmarkOutput outputType workingPathA lbBMS_A


[<Test>]
let ``LoadBalance moderngpu benchmark (B) : int32`` () =
    for test = 0 to 9 do
        let ratio = 0.05 + 0.10 * (float test)
        benchmarkLoadBalance 'B' 10000000 300 ratio test
    benchmarkOutput outputType workingPathB lbBMS_B