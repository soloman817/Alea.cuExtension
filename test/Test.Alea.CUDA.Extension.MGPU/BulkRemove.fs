module Test.Alea.CUDA.Extension.MGPU.BulkRemove

open System
open System.IO
open System.Diagnostics
open System.Collections.Generic
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.DerivedPatterns
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.BulkRemove
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
let nIterations = BenchmarkStats.bulkRemoveIterations

let algName = "Bulk Remove"
let brKernelsUsed = [| "kernelBulkRemove"; "binary search partitions" |]
let brBMS4 = new BenchmarkStats4(algName, brKernelsUsed, worker.Device.Name, "MGPU", sourceCounts, nIterations)

// we can probably organize this a lot better, but for now, if you just change
// what module you open above and all of this should adjust accordingly
let oIntTP, oIntBW = moderngpu_bulkInsertStats_int |> List.unzip
let oInt64TP, oInt64BW = moderngpu_bulkInsertStats_int64 |> List.unzip
let oFloat32TP, oFloat32BW = moderngpu_bulkInsertStats_float32 |> List.unzip
let oFloat64TP, oFloat64BW = moderngpu_bulkInsertStats_float64 |> List.unzip


for i = 0 to sourceCounts.Length - 1 do
    // this is setting the opponent (MGPU) stats for the int type
    brBMS4.Ints.OpponentThroughput.[i].Value <- oIntTP.[i]
    brBMS4.Ints.OpponentBandwidth.[i].Value <- oIntBW.[i]
    // set opponent stats for int64
    brBMS4.Int64s.OpponentThroughput.[i].Value <- oInt64TP.[i]
    brBMS4.Int64s.OpponentBandwidth.[i].Value <- oInt64BW.[i]
    // set oppenent stats for float32
    brBMS4.Float32s.OpponentThroughput.[i].Value <- oFloat32TP.[i]
    brBMS4.Float32s.OpponentBandwidth.[i].Value <- oFloat32BW.[i]
    // set oppenent stats for float64
    brBMS4.Floats.OpponentThroughput.[i].Value <- oFloat64TP.[i]
    brBMS4.Floats.OpponentBandwidth.[i].Value <- oFloat64BW.[i]


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                              IMPORTANT                                                       //
//                                      Choose an Output Type                                                   // 
// This is a switch for all tests, and can do a lot of extra work.  Make sure you turn it off if you just       //
// want to see the console prints.                                                                              //
let outputType = OutputTypeCSV    // Choices are CSV, Excel, Both, or None                                     //
// only one path, we aren't auto-saving excel stuff yet                                                         //
let workingPath = (getWorkingOutputPaths deviceFolderName algName).CSV                                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


let removeAmount = 2 //half
let removeCount c = c / removeAmount
let removeCounts = sourceCounts |> List.map (fun x -> removeCount x)

let sourceCounts2 = [512; 1024; 2048; 3000; 6000; 12000; 24000; 100000; 1000000]
let removeAmount2 = 2
let removeCount2 c2 = c2 / removeAmount2
let removeCounts2 = sourceCounts2 |> List.map (fun x -> removeCount2 x)


let verifyCount = 3

let percentDiff x y = abs(x - y) / ((x + y) / 2.0)

let hostBulkRemove (data:'T[]) (indices:int[]) =
    let result = List<'T>()
    let indices = indices |> Set.ofArray
    data |> Array.iteri (fun i x -> if not (indices |> Set.contains i) then result.Add(x))
    result.ToArray()

// @COMMENTS@: index we assume always be int type
let testBulkRemove() =
    let test verify eps (data:'T[]) (indices:int[]) = pcalc {
        let bulkrem = worker.LoadPModule(MGPU.PArray.bulkRemove()).Invoke
    
        let n = data.Length
        printfn "Testing size %d..." n

        let! dSource = DArray.scatterInBlob worker data
        let! dRemoveIndices = DArray.scatterInBlob worker indices
        let! dRemoved = bulkrem dSource dRemoveIndices

        if verify then
            let hResults = hostBulkRemove data indices
            let! dResults = dRemoved.Gather()
            // @COMMENTS@ : cause we don't change the data (we just removed something), so it should equal
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
                    
        else 
            do! PCalc.force() }

    let eps = 1e-10
    let values1 n r = 
        //let source = Array.init n (fun i -> i)
        let (r:int[]*_) = rngGenericArrayI n r
        let source, indices = (fst r), (snd r)
        source, indices

    let values2 n r = 
        //let source = Array.init n (fun i -> -i)
        let (r:int[]*_) = rngGenericArrayI n r
        let source, indices = (fst r), (snd r)
        source, indices

    let values3 n r = 
        let source = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)
        let (r:float[]*_) = rngGenericArrayI n r
        let indices = (snd r)
        source, indices  
        

    (sourceCounts2, removeCounts2) ||> Seq.iter2 (fun ns nr -> let test = test true eps  
                                                               values1 ns nr ||> test |> PCalc.run)
    (sourceCounts2, removeCounts2) ||> Seq.iter2 (fun ns nr -> let test = test true eps  
                                                               values2 ns nr ||> test |> PCalc.run)
    (sourceCounts2, removeCounts2) ||> Seq.iter2 (fun ns nr -> let test = test true eps  
                                                               values3 ns nr ||> test |> PCalc.run)
         
    let n = 2097152
    let test = values1 n (removeCount2 n) ||> test false eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
   

let inline verifyAll (h:'T[] list) (d:'T[] list) =
    (h, d) ||> List.iteri2 (fun i hi di -> if i < verifyCount then Util.verify hi di)


let benchmarkBulkRemove (data:'T[]) (indices:int[]) (numIt:int) (testIdx:int) =
    let remover = worker.LoadPModule(PArray.bulkRemoveInPlace()).Invoke

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
    // @COMMENTS@ should we divid timing by numIt
    //let newstat = [|(float count); (throughput / 1.0e6); (bandwidth / 1.0e9); timing|]            
    //let newstat = [| (float count); (throughput / 1e6); (bandwidth / 1e9); (timing' / (float numIt)) |]
    //stats.AddStat newstat
    printfn "%9d: %9.3f M/s %9.3f GB/s %6.3f ms x %4d = %7.3f ms"
        count
        (throughput / 1e6)
        (bandwidth / 1e9)
        (timing' / (float numIt))
        numIt
        timing'

    match typeof<'T> with
    | x when x = typeof<int> -> brBMS4.Ints.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | x when x = typeof<int64> -> brBMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | x when x = typeof<float32> -> brBMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | x when x = typeof<float> -> brBMS4.Floats.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | _ -> ()
 
 
[<Test>]
let ``bulkRemove moderngpu web example : float`` () =
    let hValues = Array.init 100 float
    let hIndices = [|  1;  4;  5;  7; 10; 14; 15; 16; 18; 19;
                      27; 29; 31; 32; 33; 36; 37; 39; 50; 59;
                      60; 61; 66; 78; 81; 83; 85; 90; 91; 96;
                      97; 98; 99 |]
    let answer = [| 0;  2;  3;  6;  8;  9; 11; 12; 13; 17; 
                   20; 21; 22; 23; 24; 25; 26; 28; 30; 34;
                   35; 38; 40; 41; 42; 43; 44; 45; 46; 47;
                   48; 49; 51; 52; 53; 54; 55; 56; 57; 58;
                   62; 63; 64; 65; 67; 68; 69; 70; 71; 72;
                   73; 74; 75; 76; 77; 79; 80; 82; 84; 86;
                   87; 88; 89; 92; 93; 94; 95 |] |> Array.map float

    let hResult = hostBulkRemove hValues hIndices
    (hResult, answer) ||> Array.iter2 (fun h a -> Assert.AreEqual(h, a))

    let pfunct = MGPU.PArray.bulkRemove()
    let br = worker.LoadPModule(pfunct).Invoke

    let dResult = pcalc {
        let! data = DArray.scatterInBlob worker hValues
        let! indices = DArray.scatterInBlob worker hIndices
        let! removed = br data indices
        let! results = removed.Gather()
        return results } |> PCalc.run
    (hResult, dResult) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

[<Test>]
let ``bulkRemove moderngpu web example : int`` () =
    let hValues = Array.init 100 int
    let hIndices = [|  1;  4;  5;  7; 10; 14; 15; 16; 18; 19;
                      27; 29; 31; 32; 33; 36; 37; 39; 50; 59;
                      60; 61; 66; 78; 81; 83; 85; 90; 91; 96;
                      97; 98; 99 |]
    let answer = [| 0;  2;  3;  6;  8;  9; 11; 12; 13; 17; 
                   20; 21; 22; 23; 24; 25; 26; 28; 30; 34;
                   35; 38; 40; 41; 42; 43; 44; 45; 46; 47;
                   48; 49; 51; 52; 53; 54; 55; 56; 57; 58;
                   62; 63; 64; 65; 67; 68; 69; 70; 71; 72;
                   73; 74; 75; 76; 77; 79; 80; 82; 84; 86;
                   87; 88; 89; 92; 93; 94; 95 |] |> Array.map int

    let hResult = hostBulkRemove hValues hIndices
    (hResult, answer) ||> Array.iter2 (fun h a -> Assert.AreEqual(h, a))

    let pfunct = MGPU.PArray.bulkRemove()
    let br = worker.LoadPModule(pfunct).Invoke

    let dResult = pcalc {
        let! data = DArray.scatterInBlob worker hValues
        let! indices = DArray.scatterInBlob worker hIndices
        let! removed = br data indices
        let! results = removed.Gather()
        return results } |> PCalc.run
    (hResult, dResult) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))


[<Test>]
let ``bulkRemove moderngpu web example : float32`` () =
    let hValues = Array.init 100 float32
    let hIndices = [|  1;  4;  5;  7; 10; 14; 15; 16; 18; 19;
                      27; 29; 31; 32; 33; 36; 37; 39; 50; 59;
                      60; 61; 66; 78; 81; 83; 85; 90; 91; 96;
                      97; 98; 99 |]
    let answer = [| 0;  2;  3;  6;  8;  9; 11; 12; 13; 17; 
                   20; 21; 22; 23; 24; 25; 26; 28; 30; 34;
                   35; 38; 40; 41; 42; 43; 44; 45; 46; 47;
                   48; 49; 51; 52; 53; 54; 55; 56; 57; 58;
                   62; 63; 64; 65; 67; 68; 69; 70; 71; 72;
                   73; 74; 75; 76; 77; 79; 80; 82; 84; 86;
                   87; 88; 89; 92; 93; 94; 95 |] |> Array.map float32

    let hResult = hostBulkRemove hValues hIndices
    (hResult, answer) ||> Array.iter2 (fun h a -> Assert.AreEqual(h, a))

    let pfunct = MGPU.PArray.bulkRemove()
    let br = worker.LoadPModule(pfunct).Invoke

    let dResult = pcalc {
        let! data = DArray.scatterInBlob worker hValues
        let! indices = DArray.scatterInBlob worker hIndices
        let! removed = br data indices
        let! results = removed.Gather()
        return results } |> PCalc.run
    (hResult, dResult) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))


[<Test>]
let ``BulkRemove 3 value test`` () =
    testBulkRemove()


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  BENCHMARKING                                                                                                //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

[<Test>]
let ``BulkRemove moderngpu benchmark : int`` () =    
    (sourceCounts, nIterations, removeCounts) |||> List.zip3 |> List.iteri (fun i (ns, ni, nr) ->
        let (source:int[]), indices = rngGenericArrayI ns nr
        benchmarkBulkRemove source indices ni i  )
    
    benchmarkOutput outputType workingPath brBMS4.Ints    


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
let ``BulkRemove moderngpu benchmark : float`` () =    
    (sourceCounts, nIterations, removeCounts) |||> List.zip3 |> List.iteri (fun i (ns, ni, nr) ->
        let (source:float[]), indices = rngGenericArrayI ns nr
        benchmarkBulkRemove source indices ni i   )

    benchmarkOutput outputType workingPath brBMS4.Floats



[<Test>] // above 4 tests, done in sequence (to make output easier)
let ``BulkRemove moderngpu benchmark : 4 type`` () =
    // INT
    printfn "Running BulkRemove moderngpu benchmark : Int"    
    (sourceCounts, nIterations, removeCounts) |||> List.zip3 |> List.iteri (fun i (ns, ni, nr) ->
        let (source:int[]), indices = rngGenericArrayI ns nr
        benchmarkBulkRemove source indices ni i  )    
    benchmarkOutput outputType workingPath brBMS4.Ints    


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
    benchmarkOutput outputType workingPath brBMS4.Floats