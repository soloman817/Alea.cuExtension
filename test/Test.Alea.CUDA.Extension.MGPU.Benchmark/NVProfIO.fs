module Test.Alea.CUDA.Extension.MGPU.Benchmark.NVProfIO

open System
open System.IO
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.IO
open Alea.CUDA.Extension.IO.CSV
open Alea.CUDA.Extension.IO.NVProfilerTools
open Test.Alea.CUDA.Extension.MGPU

open NUnit.Framework

open Test.Alea.CUDA.Extension.MGPU.BenchmarkStats.GF560Ti
open ModernGPU

let sourceCounts = BulkInsertStats.sourceCounts
let nIterations = BulkInsertStats.nIterations

let workingDir = @"X:\dev\GitHub\moderngpu\Release\"
//
//[<Test>]
//let ``nvprofiler parser test: bulk insert (int32)`` () =
//    let outfileName = "benchmarkinsert_nvprof_output.txt"
//    let outfilePath = workingDir + outfileName
//    if File.Exists(outfilePath) then
//        File.Delete(outfilePath)
//
//    let nvpdg = new NVProfDataGenerator(workingDir, outfilePath)
//
//    let nvprofArgs = "--csv --normalized-time-unit us "
//    let programName = "benchmarkinsert.exe "
//
//    (sourceCounts, nIterations) ||> List.iteri2 (fun i ns ni ->
//        let ns,ni = ns.ToString(), ni.ToString()
//        let programArgs = "BulkInsert Int32 ns ni"
//        nvpdg.Execute nvprofArgs programName programArgs |> ignore )
//
//    //let outfilePath = workingDir + nvpdg.outfile
//    //let outfile = workingDir + "benchmarkinsert_nvprof_out.txt"
//    //printfn "%A" outfile
//    let nvprofSDC = new NVProfSummaryDataCollector(outfilePath, 2, sourceCounts, nIterations)
//        
//    let avgklt = nvprofSDC.GetAverageKernelLaunchTimings()
//    
//    avgklt |> Array.iter (fun x ->
//        let kn = (fst x)
//        let avgdurs = (snd x)
//        printfn "Average Kernel Launch Times for %s" kn
//        printfn "num elem\tavg time (us)"
//        (sourceCounts |> Array.ofList, avgdurs) ||> Array.iter2 (fun n d ->
//            printfn "%d\t\t%f" n d)
//        printfn "\n" )


[<Test>]
let ``nvprofiler parser test: bulk remove (int32) loop`` () =
    let outfileName = "benchmarkinsert_nvprof_output.txt"
    let outfilePath = workingDir + outfileName
    if File.Exists(outfilePath) then
        File.Delete(outfilePath)
    let nvpdg = new NVProfDataGenerator(workingDir, outfilePath)
    let nvprofArgs = "--csv --normalized-time-unit us --print-gpu-trace "
    let programName = "benchmarkinsert.exe "
    nvpdg.Execute nvprofArgs programName ""
    printfn "done executing"
    //let sourceCounts = [10000; 50000; 100000; 200000; 500000]
    //let nIterations = [2000; 2000; 2000; 1000; 500]
    let nvprofgputdc = new NVProfGPUTraceDataCollector(outfilePath, sourceCounts, nIterations)
    let stopwatch = new System.Diagnostics.Stopwatch()
    stopwatch.Start()
    let klt = nvprofgputdc.GetAverageKernelLaunchTimings 2 2 1 "us"
    stopwatch.Stop()
    printfn "get avg took %d ms" stopwatch.ElapsedMilliseconds
    klt |> Array.iter (fun x ->
        x |> Array.iter (fun y ->
            let knames, results = y
            knames |> Array.iteri (fun i kn ->        
            printfn "Average Kernel Launch Times (us) for %s" kn
            let avgs = results.[i] |> List.ofArray
            (sourceCounts, avgs) ||> List.iter2 (fun n avgdur -> printfn "%d\t\t%9.3f" n avgdur)
            printfn "\n" ))
            )

[<Test>]
let ``nvprfiler parser test : no data gen`` () =
    let outfileName = "benchmarkinsert_nvprof_output.txt"
    let outfilePath = workingDir + outfileName
    let nvprofgputdc = new NVProfGPUTraceDataCollector(outfilePath, sourceCounts, nIterations)
    let stopwatch = new System.Diagnostics.Stopwatch()
    stopwatch.Start()
    let klt = nvprofgputdc.GetAverageKernelLaunchTimings 2 2 4 "us"
    stopwatch.Stop()
    printfn "get avg took %d ms" stopwatch.ElapsedMilliseconds
    klt |> Array.iter (fun x ->
        x |> Array.iter (fun y ->
            let knames, results = y
            knames |> Array.iteri (fun i kn ->        
            printfn "Average Kernel Launch Times (us) for %s" kn
            let avgs = results.[i] |> List.ofArray
            (sourceCounts, avgs) ||> List.iter2 (fun n avgdur -> printfn "%d\t\t%9.3f" n avgdur)
            printfn "\n" ))
            )

