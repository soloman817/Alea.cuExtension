module Test.Alea.CUDA.Extension.MGPU.Benchmark.Scan

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.IO
open Test.Alea.CUDA.Extension.MGPU.Util
open NUnit.Framework

////////////////////////////
// set this to your device or add your device's C++ output to BenchmarkStats.fs
open GF560Ti
// in the future maybe we try to get the C++ to interop somehow
/////////////////////////////
open ModernGPU.ScanStats

let worker = getDefaultWorker()
let pScanner = new PScan()

let totalNotAtEnd = 0

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



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  SCAN BENCHMARKING                                                                                                //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
