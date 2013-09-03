module Test.Alea.CUDA.Extension.MGPU.Benchmark.Insert

open System
open System.IO
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.IO.Util
open Test.Alea.CUDA.Extension.MGPU
open NUnit.Framework
////////////////////////////
// set this to your device or add your device's C++ output to BenchmarkStats.fs
open BenchmarkStats.GF560Ti
// in the future maybe we try to get the C++ to interop somehow
/////////////////////////////



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  BULK INSERT BENCHMARKING                                                                                    //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
module BulkInsert =
    open ModernGPU.BulkInsertStats

    let worker = getDefaultWorker()
    let pfuncts = new PBulkInsert()
    let rng = System.Random()
    
    let aib count =
        let aCount = count / 2
        let bCount = count - aCount
        aCount,bCount

    let aibCounts = sourceCounts |> List.map (fun x -> aib x)
    let aCounts, bCounts = aibCounts |> List.unzip

    let algName = "Bulk Insert"
    let biKernelsUsed = [| "kernelBulkInsert"; "kernelMergePartition" |]    
    let biBMS4 = getFilledBMS4Object algName biKernelsUsed worker.Device.Name "MGPU" sourceCounts nIterations fourTypeStatsList
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                              IMPORTANT                                                       //
    //                                      Choose an Output Type                                                   // 
    // This is a switch for all tests, and can do a lot of extra work.  Make sure you turn it off if you just       //
    // want to see the console prInt32s.                                                                            //
    let outputType = OutputTypeNone // Choices are CSV, Excel, Both, or None. Set to None for doing kernel timing   //
    //let overwrite = false               // overwrite old data or create new data?                                 //
    // only one path, we aren't auto-saving excel stuff yet                                                         //
    let workingPath = (getWorkingOutputPaths deviceFolderName algName).CSV                                          //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    let benchmarkBulkInsert (dataA:'T[]) (indices:int[]) (dataB:'T[]) (numIt:int) (testIdx:int) =
        let inserter = worker.LoadPModule(pfuncts.BulkInsertFunc()).Invoke

        let calc = pcalc {
            let! dA = DArray.scatterInBlob worker dataA
            let! dI = DArray.scatterInBlob worker indices
            let! dB = DArray.scatterInBlob worker dataB
            let! dR = DArray.createInBlob worker (dataA.Length + dataB.Length)

            let! insert = inserter dataA.Length dataB.Length

            // warm up
            do! insert dA dI dB dR

            let! dStopwatch = DStopwatch.startNew worker
            for i = 1 to numIt do
                do! insert dA dI dB dR
            do! dStopwatch.Stop()

            let! results = dR.Gather()
            let! timing = dStopwatch.ElapsedMilliseconds

            return results, timing }

        let count, aCount, bCount = (dataA.Length + dataB.Length), dataA.Length, dataB.Length
        // I use runInWorker to avoid thread switching.
        let hResults, timing' = calc |> PCalc.runInWorker worker
        let timing = timing' / 1000.0 // timing (in second), timing' (in millisecond)
        let bytes = (sizeof<int> + 2 * sizeof<'T>) * aCount + 2 * sizeof<'T> * bCount |> float
        let throughput = (float count) * (float numIt) / timing
        let bandwidth = bytes * (float numIt) / timing
    
        printfn "%9d: %9.3f M/s %9.3f GB/s %6.3f ms x %4d = %7.3f ms"
            count
            (throughput / 1e6)
            (bandwidth / 1e9)
            (timing' / (float numIt))
            numIt
            timing'

        match typeof<'T> with
        | x when x = typeof<int> -> biBMS4.Int32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<int64> -> biBMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float32> -> biBMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float> -> biBMS4.Float64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | _ -> ()


    [<Test>]
    let ``BulkInsert moderngpu benchmark : int32`` () =    
        (aCounts, bCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (na, nb, ni) ->
            let (r:int[]*int[]*int[]) = rngGenericArrayAIB na nb
            let hA,hI,hB = r
            benchmarkBulkInsert hA hI hB ni i  )
    
        benchmarkOutput outputType workingPath biBMS4.Int32s


    [<Test>]
    let ``BulkInsert moderngpu benchmark : int64`` () =    
        (aCounts, bCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (na, nb, ni) ->
            let (r:int64[]*int[]*int64[]) = rngGenericArrayAIB na nb
            let hA,hI,hB = r
            benchmarkBulkInsert hA hI hB ni i  )

        benchmarkOutput outputType workingPath biBMS4.Int64s
    

    [<Test>]
    let ``BulkInsert moderngpu benchmark : float32`` () =    
        (aCounts, bCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (na, nb, ni) ->
            let (r:float32[]*int[]*float32[]) = rngGenericArrayAIB na nb
            let hA,hI,hB = r
            benchmarkBulkInsert hA hI hB ni i  )

        benchmarkOutput outputType workingPath biBMS4.Float32s

    [<Test>]
    let ``BulkInsert moderngpu benchmark : float64`` () =    
        (aCounts, bCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (na, nb, ni) ->
            let (r:float[]*int[]*float[]) = rngGenericArrayAIB na nb
            let hA,hI,hB = r
            benchmarkBulkInsert hA hI hB ni i  )

        benchmarkOutput outputType workingPath biBMS4.Float64s


    [<Test>] // above 4 tests, done in sequence (to make output easier)
    let ``BulkInsert moderngpu benchmark : 4 type`` () =    
        // INT
        printfn "Running BulkInsert moderngpu benchmark : Int"
        (aCounts, bCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (na, nb, ni) ->
            let (r:int[]*int[]*int[]) = rngGenericArrayAIB na nb
            let hA,hI,hB = r
            benchmarkBulkInsert hA hI hB ni i  )    
        benchmarkOutput outputType workingPath biBMS4.Int32s

        // INT64
        printfn "\nRunning BulkInsert moderngpu benchmark : Int64"
        (aCounts, bCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (na, nb, ni) ->
            let (r:int64[]*int[]*int64[]) = rngGenericArrayAIB na nb
            let hA,hI,hB = r
            benchmarkBulkInsert hA hI hB ni i  )
        benchmarkOutput outputType workingPath biBMS4.Int64s
    
        // FLOAT32
        printfn "\nRunning BulkInsert moderngpu benchmark : Float32"
        (aCounts, bCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (na, nb, ni) ->
            let (r:float32[]*int[]*float32[]) = rngGenericArrayAIB na nb
            let hA,hI,hB = r
            benchmarkBulkInsert hA hI hB ni i  )
        benchmarkOutput outputType workingPath biBMS4.Float32s

        // FLOAT64
        printfn "\nRunning BulkInsert moderngpu benchmark : Float64"
        (aCounts, bCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (na, nb, ni) ->
            let (r:float[]*int[]*float[]) = rngGenericArrayAIB na nb
            let hA,hI,hB = r
            benchmarkBulkInsert hA hI hB ni i  )
        benchmarkOutput outputType workingPath biBMS4.Float64s



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  BULK REMOVE BENCHMARKING                                                                                    //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
module BulkRemove =
    open ModernGPU.BulkRemoveStats

    let worker = getDefaultWorker()
    let pfuncts = new PBulkRemove()
    let rng = System.Random()

    let algName = "Bulk Remove"
    let brKernelsUsed = [| "kernelBulkRemove"; "binary search partitions" |]
    let brBMS4 = getFilledBMS4Object algName brKernelsUsed worker.Device.Name "MGPU" sourceCounts nIterations fourTypeStatsList
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                              IMPORTANT                                                       //
    //                                      Choose an Output Type                                                   // 
    // This is a switch for all tests, and can do a lot of extra work.  Make sure you turn it off if you just       //
    // want to see the console prInt32s.                                                                            //
    let outputType = OutputTypeNone // Choices are CSV, Excel, Both, or None. Set to None for doing kernel timing   //
    // only one path, we aren't auto-saving excel stuff yet                                                         //
    let workingPath = (getWorkingOutputPaths deviceFolderName algName).CSV                                          //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    let removeAmount = 2 //half
    let removeCount c = c / removeAmount
    let removeCounts = sourceCounts |> List.map (fun x -> removeCount x)
    
    let benchmarkBulkRemove (data:'T[]) (indices:int[]) (numIt:int) (testIdx:int) =
        let remover = worker.LoadPModule(pfuncts.BulkRemoveFunc()).Invoke

        let calc = pcalc {
            let! dSource = DArray.scatterInBlob worker data
            let! dIndices = DArray.scatterInBlob worker indices
            let! dRemoved = DArray.createInBlob worker (data.Length - indices.Length)

            let! remove = remover data.Length

            // warm up
            do! remove dSource dIndices dRemoved

            let! dStopwatch = DStopwatch.startNew worker
            for i = 1 to numIt do
                do! remove dSource dIndices dRemoved
            do! dStopwatch.Stop()

            let! results = dRemoved.Gather()
            let! timing = dStopwatch.ElapsedMilliseconds

            return results, timing }

        let count, removeCount, keepCount = data.Length, indices.Length, (data.Length - indices.Length)
        // I use runInWorker to avoid thread switching.
        let hResults, timing' = calc |> PCalc.runInWorker worker
        let timing = timing' / 1000.0 // timing (in second), timing' (in millisecond)
        let bytes = (sizeof<'T> * count + keepCount * sizeof<'T> + removeCount * sizeof<'T>) |> float
        let throughput = (float count) * (float numIt) / timing
        let bandwidth = bytes * (float numIt) / timing
        printfn "%9d: %9.3f M/s %9.3f GB/s %6.3f ms x %4d = %7.3f ms"
            count
            (throughput / 1e6)
            (bandwidth / 1e9)
            (timing' / (float numIt))
            numIt
            timing'

        match typeof<'T> with
        | x when x = typeof<int> -> brBMS4.Int32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<int64> -> brBMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float32> -> brBMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float> -> brBMS4.Float64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | _ -> ()
    


    [<Test>]
    let ``BulkRemove moderngpu benchmark : int32`` () =    
        (sourceCounts, nIterations, removeCounts) |||> List.zip3 |> List.iteri (fun i (ns, ni, nr) ->
            let (source:int[]), indices = rngGenericArrayI ns nr
            benchmarkBulkRemove source indices ni i  )    
        benchmarkOutput outputType workingPath brBMS4.Int32s    


    [<Test>]
    let ``BulkRemove moderngpu benchmark : int64`` () =    
        (sourceCounts, nIterations, removeCounts) |||> List.zip3 |> List.iteri (fun i (ns, ni, nr) ->
            let (source:int64[]), indices = rngGenericArrayI ns nr
            benchmarkBulkRemove source indices ni i   )
        benchmarkOutput outputType workingPath brBMS4.Int64s


    [<Test>]
    let ``BulkRemove moderngpu benchmark : float32`` () =    
        (sourceCounts, nIterations, removeCounts) |||> List.zip3 |> List.iteri (fun i (ns, ni, nr) ->
            let (source:float32[]), indices = rngGenericArrayI ns nr
            benchmarkBulkRemove source indices ni i   )
        benchmarkOutput outputType workingPath brBMS4.Float32s


    [<Test>]
    let ``BulkRemove moderngpu benchmark : float64`` () =    
        (sourceCounts, nIterations, removeCounts) |||> List.zip3 |> List.iteri (fun i (ns, ni, nr) ->
            let (source:float[]), indices = rngGenericArrayI ns nr
            benchmarkBulkRemove source indices ni i   )
        benchmarkOutput outputType workingPath brBMS4.Float64s



    [<Test>] // above 4 tests, done in sequence (to make output easier)
    let ``BulkRemove moderngpu benchmark : 4 type`` () =
        // INT
        printfn "Running BulkRemove moderngpu benchmark : Int"    
        (sourceCounts, nIterations, removeCounts) |||> List.zip3 |> List.iteri (fun i (ns, ni, nr) ->
            let (source:int[]), indices = rngGenericArrayI ns nr
            benchmarkBulkRemove source indices ni i  )    
        benchmarkOutput outputType workingPath brBMS4.Int32s    


        // INT64
        printfn "\nRunning BulkRemove moderngpu benchmark : Int64"    
        (sourceCounts, nIterations, removeCounts) |||> List.zip3 |> List.iteri (fun i (ns, ni, nr) ->
            let (source:int64[]), indices = rngGenericArrayI ns nr
            benchmarkBulkRemove source indices ni i   )
        benchmarkOutput outputType workingPath brBMS4.Int64s


        // FLOAT32
        printfn "\nRunning BulkRemove moderngpu benchmark : Float32"
        (sourceCounts, nIterations, removeCounts) |||> List.zip3 |> List.iteri (fun i (ns, ni, nr) ->
            let (source:float32[]), indices = rngGenericArrayI ns nr
            benchmarkBulkRemove source indices ni i   )
        benchmarkOutput outputType workingPath brBMS4.Float32s


        // FLOAT64
        printfn "\nRunning BulkRemove moderngpu benchmark : Float64"    
        (sourceCounts, nIterations, removeCounts) |||> List.zip3 |> List.iteri (fun i (ns, ni, nr) ->
            let (source:float[]), indices = rngGenericArrayI ns nr
            benchmarkBulkRemove source indices ni i   )
        benchmarkOutput outputType workingPath brBMS4.Float64s


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  MGPU C++ Profiling                                                                                          //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
module Profiling =
    open Extension.IO.NVProfilerTools
    let sourceCounts = BenchmarkStats.bulkInsert.SourceCounts
    let nIterations = BenchmarkStats.bulkInsert.Iterations
    let workingDir = @"X:\dev\GitHub\moderngpu\Release\"
    let numberAlgs = 2 // bulkRemove & bulkInsert
    let kernelsPerAlg = 2 // bulkRemove has binary search & bulkRemove, bulkInsert has merge path partitions & bulkInsert
    let typesPerAlg = 4 // int32, int64, float32, float64
    let timeUnit = "us" // microseconds

    [<Test>]
    let ``nvprofiler full 4type test: BulkRemove & BulkInsert`` () =
        let outfileName = "benchmarkinsert_nvprof_output.txt"
        let outfilePath = workingDir + outfileName
        if File.Exists(outfilePath) then
            File.Delete(outfilePath)
        let nvpdg = new NVProfDataGenerator(workingDir, outfilePath)
        let nvprofArgs = "--csv --normalized-time-unit " + timeUnit + " --print-gpu-trace "
        let programName = "benchmarkinsert.exe "
        nvpdg.Execute nvprofArgs programName ""
        printfn "done executing"
        let nvprofgputdc = new NVProfGPUTraceDataCollector(outfilePath, sourceCounts, nIterations)
        nvprofgputdc.DisplayAverageKernelLaunchTimings numberAlgs kernelsPerAlg typesPerAlg timeUnit