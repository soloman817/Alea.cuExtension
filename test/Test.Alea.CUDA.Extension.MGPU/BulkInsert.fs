module Test.Alea.CUDA.Extension.MGPU.BulkInsert

open System
open System.IO
open System.Diagnostics
open System.Collections.Generic
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.DerivedPatterns
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.BulkInsert
open Test.Alea.CUDA.Extension.MGPU.Util
open Test.Alea.CUDA.Extension.MGPU.BenchmarkStats
open Alea.CUDA.Extension.Output.Util
open Alea.CUDA.Extension.Output.CSV
open Alea.CUDA.Extension.Output.Excel

open NUnit.Framework


////////////////////////////
// set this to your device or add your device's C++ output to BenchmarkStats.fs
open Test.Alea.CUDA.Extension.MGPU.BenchmarkStats.TeslaK20c
// in the future maybe we try to get the C++ to interop somehow
/////////////////////////////


let worker = Engine.workers.DefaultWorker
let rng = System.Random()

let sourceCounts = BenchmarkStats.sourceCounts
let nIterations = BenchmarkStats.bulkInsertIterations

let aib count =
    let aCount = count / 2
    let bCount = count - aCount
    aCount,bCount

let aibCounts = sourceCounts |> List.map (fun x -> aib x)

let sourceCounts2 = [512; 1024; 2048; 3000; 6000; 12000; 24000; 100000; 1000000]
let aibCounts2 = sourceCounts2 |> List.map (fun x -> aib x)



let aCounts, bCounts = aibCounts |> List.unzip

let algName = "Bulk Insert"
let biKernelsUsed = [| "kernelBulkInsert"; "kernelMergePartition" |]
let biBMS4 = new BenchmarkStats4(algName, biKernelsUsed, worker.Device.Name, "MGPU", sourceCounts, nIterations)
 

// we can probably organize this a lot better, but for now, if you just change
// what module you open above and all of this should adjust accordingly
let oIntTP, oIntBW = moderngpu_bulkInsertStats_int |> List.unzip
let oInt64TP, oInt64BW = moderngpu_bulkInsertStats_int64 |> List.unzip
let oFloat32TP, oFloat32BW = moderngpu_bulkInsertStats_float32 |> List.unzip
let oFloat64TP, oFloat64BW = moderngpu_bulkInsertStats_float64 |> List.unzip


for i = 0 to sourceCounts.Length - 1 do
    // this is setting the opponent (MGPU) stats for the int type
    biBMS4.Int32s.OpponentThroughput.[i].Value <- oIntTP.[i]
    biBMS4.Int32s.OpponentBandwidth.[i].Value <- oIntBW.[i]
    // set opponent stats for int64
    biBMS4.Int64s.OpponentThroughput.[i].Value <- oInt64TP.[i]
    biBMS4.Int64s.OpponentBandwidth.[i].Value <- oInt64BW.[i]
    // set oppenent stats for float32
    biBMS4.Float32s.OpponentThroughput.[i].Value <- oFloat32TP.[i]
    biBMS4.Float32s.OpponentBandwidth.[i].Value <- oFloat32BW.[i]
    // set oppenent stats for float64
    biBMS4.Float64s.OpponentThroughput.[i].Value <- oFloat64TP.[i]
    biBMS4.Float64s.OpponentBandwidth.[i].Value <- oFloat64BW.[i]


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


let hostBulkInsert (dataA:'T[]) (indices:int[]) (dataB:'T[]) =
    let result : 'T[] = Array.zeroCreate (dataA.Length + dataB.Length)
    Array.blit dataB 0 result 0 indices.[0]
    Array.set result indices.[0] dataA.[0]
    for i = 1 to indices.Length - 1 do
        Array.blit dataB indices.[i - 1] result (indices.[i - 1] + i) (indices.[i] - indices.[i - 1])
        Array.set result (indices.[i] + i) dataA.[i]
    let i = indices.Length - 1
    Array.blit dataB indices.[i] result (indices.[i] + i + 1) (result.Length - (indices.[i] + i + 1))
    result

let testBulkInsert() =
    let test verify eps (dataA:'T[]) (indices:int[]) (dataB:'T[]) = pcalc {
        let bulkin = worker.LoadPModule(MGPU.PArray.bulkInsert()).Invoke
    
        let aCount = dataA.Length
        let bCount = dataB.Length
        printfn "Testing %d items inserted into %d..." aCount bCount

        let! dA = DArray.scatterInBlob worker dataA     // elements to insert
        let! dI = DArray.scatterInBlob worker indices   // where to insert them
        let! dB = DArray.scatterInBlob worker dataB     // collection they are inserted into
        let! dR = bulkin dA dI dB

        if verify then
            let hResults = hostBulkInsert dataA indices dataB
            let! dResults = dR.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
                    
        else 
            do! PCalc.force() }

    let eps = 1e-10
    let values1 na nb = 
        let (r:int[]*int[]*int[]) = rngGenericArrayAIB na nb
        let hA,hI,hB = r
        hA,hI,hB

    let values2 na nb = 
        let (r:int[]*int[]*int[]) = rngGenericArrayAIB na nb
        let hA,hI,hB = r
        hA,hI,hB

    let values3 na nb = 
        let (r:float[]*int[]*float[]) = rngGenericArrayAIB na nb
        let hA,hI,hB = r
        hA,hI,hB  
        

    aibCounts2 |> List.iter (fun (na,nb) -> let test = test true eps
                                            values1 na nb |||> test |> PCalc.run)

    aibCounts2 |> List.iter (fun (na,nb) -> let test = test true eps
                                            values2 na nb |||> test |> PCalc.run)

    aibCounts2 |> List.iter (fun (na,nb) -> let test = test true eps
                                            values3 na nb |||> test |> PCalc.run)
        
    let n = 2097152
    let na,nb = aib n
    let test = (values1 na nb) |||> test false eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
        

let benchmarkBulkInsert (dataA:'T[]) (indices:int[]) (dataB:'T[]) (numIt:int) (testIdx:int) =
    let inserter = worker.LoadPModule(PArray.bulkInsertInPlace()).Invoke

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
let ``bulkInsert simple example`` () =
    let hB = Array.init 20 (fun i -> i)     // source data
    let hI = [| 3; 7; 11; 14; 19 |]         // indices of insertions
    let hA = [| 93; 97; 911; 914; 919 |]    // data to insert into hB

    let pfunct = MGPU.PArray.bulkInsert()
    let bulkin = worker.LoadPModule(pfunct).Invoke

    let dResult = pcalc {
        let! dA = DArray.scatterInBlob worker hA
        let! dB = DArray.scatterInBlob worker hB
        let! dI = DArray.scatterInBlob worker hI
        let! dR = bulkin dA dI dB
        let! results = dR.Gather()
        return results } |> PCalc.run

    printfn "%A" dResult


[<Test>]
let ``bulkInsert moderngpu website example`` () =
    let hI = [|2..5..100|]
    let aCount, bCount = hI.Length, 100
    
    let hA = [|1000..10..((aCount*10+1000)-10)|]
    let hB = Array.init bCount int
    
    let answer = [|     0;  1; 1000;  2;     3;  4;    5;  6; 1010;  7;
                        8;  9;   10; 11;  1020; 12;   13; 14;   15; 16;
                     1030; 17;   18; 19;    20; 21; 1040; 22;   23; 24;
                       25; 26; 1050; 27;    28; 29;   30; 31; 1060; 32;
                       33; 34;   35; 36;  1070; 37;   38; 39;   40; 41;
                     1080; 42;   43; 44;    45; 46; 1090; 47;   48; 49;
                       50; 51; 1100; 52;    53; 54;   55; 56; 1110; 57;
                       58; 59;   60; 61;  1120; 62;   63; 64;   65; 66;
                     1130; 67;   68; 69;    70; 71; 1140; 72;   73; 74;
                       75; 76; 1150; 77;    78; 79;   80; 81; 1160; 82;
                       83; 84;   85; 86;  1170; 87;   88; 89;   90; 91;
                     1180; 92;   93; 94;    95; 96; 1190; 97;   98; 99 |]
    
    let hResult = hostBulkInsert hA hI hB
    (hResult, answer) ||> Array.iter2 (fun h a -> Assert.AreEqual(h, a))

    let pfunct = MGPU.PArray.bulkInsert()
    let bulkin = worker.LoadPModule(pfunct).Invoke

    let dResult = pcalc {
        let! dA = DArray.scatterInBlob worker hA
        let! dB = DArray.scatterInBlob worker hB
        let! dI = DArray.scatterInBlob worker hI
        let! dR = bulkin dA dI dB
        let! results = dR.Gather()
        return results } |> PCalc.run
    
    (hResult, dResult) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))


    printfn "%A" dResult



[<Test>]
let ``BulkInsert 3 value test`` () =
    testBulkInsert()



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  BENCHMARKING                                                                                                //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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



//[<Test>]
//let ``bulkInsert moderngpu website example 2`` () =
//    let aCount, bCount = 100, 400  // insert 100 elements into a 400 element array
//    let hA = Array.init aCount (fun _ -> 9999) // what to insert
//    let hB = Array.init bCount (fun i -> i)
//    
//    let hI = [|   1;   12;   13;   14;   14;   18;   20;   38;   39;   44;
//                 45;   50;   50;   50;   54;   56;   59;   63;   68;   69;
//                 74;   75;   84;   84;   88;  111;  111;  119;  121;  123;
//                126;  127;  144;  153;  157;  159;  163;  169;  169;  175;
//                178;  183;  190;  194;  195;  196;  196;  201;  219;  219;
//                253;  256;  259;  262;  262;  266;  272;  273;  278;  283;
//                284;  291;  296;  297;  302;  303;  306;  306;  317;  318;
//                318;  319;  319;  320;  320;  323;  326;  329;  330;  334;
//                340;  349;  352;  363;  366;  367;  369;  374;  381;  383;
//                383;  384;  386;  388;  388;  389;  393;  398;  398;  399 |]
//    
//    let pfunct = MGPU.PArray.bulkInsert()
//    let bulkin = worker.LoadPModule(pfunct).Invoke
//    
//    let dResult = pcalc {
//        let! dA = DArray.scatterInBlob worker hA
//        let! dB = DArray.scatterInBlob worker hB
//        let! dI = DArray.scatterInBlob worker hI
//        let! dR = bulkin dA dI dB
//        let! results = dR.Gather()
//        return results } |> PCalc.run
//
//    printfn "x"