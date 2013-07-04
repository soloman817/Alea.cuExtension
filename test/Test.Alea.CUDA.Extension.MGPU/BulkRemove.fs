module Test.Alea.CUDA.Extension.MGPU.BulkRemove

open System
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

open NUnit.Framework

let worker = Engine.workers.DefaultWorker
let rng = System.Random()

let sourceCounts = BenchmarkStats.sourceCounts
let nIterations = BenchmarkStats.bulkRemoveIterations

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
        let brp = worker.LoadPModule(MGPU.PArray.bulkRemove()).Invoke
    
        let n = data.Length
        printfn "Testing size %d..." n

        let! dSource = DArray.scatterInBlob worker data
        let! dRemoveIndices = DArray.scatterInBlob worker indices
        let! dRemoved = brp dSource dRemoveIndices

        if verify then
            let hResults = hostBulkRemove data indices
            let! dResults = dRemoved.Gather()
            // @COMMENTS@ : cause we don't change the data (we just removed something), so it should equal
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
                    
        else 
            do! PCalc.force() }

    let eps = 1e-10
    let values1 n r = 
        let source = Array.init n (fun _ -> 1)
        let (r:int[]*_) = rngGenericArrayI n r
        let indices = (snd r)
        source, indices

    let values2 n r = 
        let source = Array.init n (fun _ -> -1)
        let (r:int[]*_) = rngGenericArrayI n r
        let indices = (snd r)
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


let benchmarkBulkRemove() =
    let bulkRemove = worker.LoadPModule(PArray.bulkRemove () ).Invoke
    fun (data:'T[]) (indices:'T[]) (numIt:'T) ->
        
        let calc = pcalc {
            let! dSource = DArray.scatterInBlob worker data
            let! dRemoveIndices = DArray.scatterInBlob worker indices            
            let! stopwatch = DStopwatch.startNew worker
            //let! remover, removed = bulkRemove dSource dRemoveIndices
            let! removed = bulkRemove dSource dRemoveIndices

//            for i = 1 to numIt do
//                do! remover.Value
            do! stopwatch.Stop()
            
            let! results = removed.Gather()
            let! timing = stopwatch.ElapsedMilliseconds

            return results, timing / 1000.0 }

        let count, removeCount, keepCount = data.Length, indices.Length, (data.Length - indices.Length)
        let hResults, timing = calc |> PCalc.run
        let bytes = (sizeof<'T> * count + keepCount * sizeof<'T> + removeCount * sizeof<'T>) |> float
        let throughput = (float count) * (float numIt) / timing
        let bandwidth = bytes * (float numIt) / timing
        let newstat = [|(float count); (throughput / 1.0e6); (bandwidth / 1.0e9); timing|]            
        //stats.AddStat newstat
        printfn "(%d): %9.3f M/s \t %7.3f GB/s \t Time: %7.3f s" (int newstat.[0]) newstat.[1] newstat.[2] newstat.[3]
 
 
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


[<Test>]
let ``BulkRemove moderngpu benchmark int`` () =    
                
    (sourceCounts, nIterations, removeCounts) |||> List.zip3 |> List.iter
        (fun test -> let ns,ni,nr = test 
                     //let bulkRemove = benchmarkBulkRemove<int> ns nr
                     let (x:int[] * _) = rngGenericArrayI ns nr
                     let source, indices = (fst x),(snd x)
                     benchmarkBulkRemove () source indices ni    )
    //printfn "My Stats: %A" myStats

//    intStats.CompareResults
//    printfn "Verified?"
//    intStats.GetResults ||> verifyAll
//    printfn "yes"

//[<Test>]
//let ``BulkRemove moderngpu benchmark int64`` () =

//
//    let intStats = Stats(sourceCounts.Length, mgpuStats_Int64)
//    let int64benchmarkBulkRemove = benchmarkBulkRemove intStats 0L
//    
//    (sourceCounts, nIterations) ||> List.iter2 (fun s i -> 
//                                                    stopwatch.Reset()
//                                                    benchmarkBulkRemove s i)
//    intStats.CompareResults




////////////////////////////////////////////////////////////////////////////////////////////////////////
///  Saving Old Code for Now
////////////////////////////////////////////////////////////////////////////////////////////////////////
//type Stats( numTests, mgpuStats : (float * float) list) =
//    let mutable myStats = 
//        let mutable r = []
//        for i = 0 to numTests - 1 do
//            let mutable a = Array.zeroCreate<float> 4
//            r <- r @ [a]
//        r
//
//    let resultSizes = sourceCounts |> List.collect (fun s -> [s - s/removeAmount])
//
//    let mutable myDeviceResults : 'T[] list = 
//        resultSizes |> List.collect (fun n -> [Array.zeroCreate<'T> n])       
//
//    let mutable myHostResults : 'T[] list = 
//        resultSizes |> List.collect (fun n -> [Array.zeroCreate<'T> n])     
//
//    let mutable dResultIdx = 0
//    let mutable hResultIdx = 0
//    let mutable statCount = 0
//
//    member s.AddStat (stat:float[]) =
//        if statCount < numTests then
//            for i = 0 to myStats.[statCount].Length - 1 do
//                myStats.[statCount].[i] <- stat.[i]
//                
//    member s.AddDeviceResult (result:'T[]) =
//        for i = 0 to myDeviceResults.[dResultIdx].Length - 1 do
//            myDeviceResults.[dResultIdx].[i] <- result.[i]
//        dResultIdx <- dResultIdx + 1
//
//    member s.AddHostResult (result:'T[]) =
//        for i = 0 to myHostResults.[hResultIdx].Length - 1 do
//            myHostResults.[hResultIdx].[i] <- result.[i]
//        hResultIdx <- hResultIdx + 1
//    
//    member s.GetResultIdx = hResultIdx
//
//    member s.GetResults : ('T[] list * 'T[] list) = myHostResults, myDeviceResults
//
//    member s.ithArrSize (i:int) = int(myStats.[i].[0])
//    member s.ithThrouputs (i:int) = fst mgpuStats.[i], myStats.[i].[1]
//    member s.ithBandwidths (i:int) = snd mgpuStats.[i], myStats.[i].[2]
//    member s.ithTiming (i:int) = myStats.[i].[3]
//
//    member s.CompareResults = 
//        printfn "\n****************** Comparison of Results to ModernGPU Library ********************"
//        printfn "COUNT\t\tMGPU Throughput\tMy Throughput\t\tMGPU Bandwidth\t\tMy Bandwidth"
//        for i = 0 to sourceCounts.Length - 1 do
//            let count = s.ithArrSize(i)
//            let mgpuTp, myTp = s.ithThrouputs(i)
//            let mgpuBw, myBw = s.ithBandwidths(i)
//            printfn "(%d)\t\t(%9.3f)\t\t(%9.3f)\t\t(%7.3f)\t\t(%7.3f)" count mgpuTp myTp mgpuBw myBw
//        printfn "\nCOUNT\tThroughput Percent Difference\t\tBandwidth Percent Difference"
//        for i = 0 to sourceCounts.Length - 1 do
//            let count = s.ithArrSize (i)
//            let thruDiff = s.ithThrouputs(i) ||> percentDiff
//            let bandDiff = s.ithBandwidths(i) ||> percentDiff
//            printfn "(%d)\t\t(%5.1f)\t\t\t\t(%5.1f)" count thruDiff bandDiff
//        printfn "\nCOUNT\t\tTIMING(Me Only)"
//        for i = 0 to sourceCounts.Length - 1 do
//            let count = s.ithArrSize(i)
//            let time = s.ithTiming(i)
//            printfn "(%d)\t\t(%7.3f s)" count time