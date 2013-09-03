module Test.Alea.CUDA.Extension.MGPU.Benchmark.LoadBalance

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Test.Alea.CUDA.Extension.MGPU.Util
open Alea.CUDA.Extension.IO.Util
open NUnit.Framework

////////////////////////////
// set this to your device or add your device's C++ output to BenchmarkStats.fs
open GF560Ti
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
    let scan = worker.LoadPModule(pScanner.Scan()).Invoke

    let count = int((1.0 - percentTerms) * (float total))
    let numTerms = total - count
    let dquot, drem = (count / numTerms), (count % numTerms)
    let terms = Array.zeroCreate numTerms
    for i = 0 to numTerms - 1 do
        Array.set terms i (dquot + (if i < drem then 1 else 0))

    for i = 0 to numTerms - 2 do
        let r = rng.Next(0, numTerms - i - 1)
        let x = min terms.[r] terms.[i + r]
        let r2 = rng.Next(x) * (if rng.Next(1) = 1 then 1 else -1)
        terms.[r] <- terms.[r] - r2
        terms.[i + r] <- terms.[i + r] + r2
        swap terms.[r] terms.[i + r]    
    
    let total, scannedCounts =
        pcalc { let! dCounts = DArray.scatterInBlob worker terms
                return! scan dCounts } |> PCalc.run    

    let calc = pcalc {
        let! dCounts = DArray.scatterInBlob worker scannedCounts
        let! dIndex = DArray.createInBlob worker total
                
        let! lbSearch = loadBalanceSearch total numTerms
        // warm up
        do! lbSearch dCounts dIndex

        let! dStopwatch = DStopwatch.startNew worker
        for i = 1 to numIt do
            do! lbSearch dCounts dIndex
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
let ``LoadBalance moderngpu benchmark (A) : int32`` () =
    let percentTerms = 0.25
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->        
        benchmarkLoadBalance 'A' ns ni percentTerms i )
    benchmarkOutput outputType workingPathA lbBMS_A


[<Test>]
let ``LoadBalance moderngpu benchmark (B) : int32`` () =
    for test = 0 to 9 do
        let ratio = 0.05 + 0.10 * (float test)
        benchmarkLoadBalance 'B' 10000000 300 ratio test
    benchmarkOutput outputType workingPathB lbBMS_B