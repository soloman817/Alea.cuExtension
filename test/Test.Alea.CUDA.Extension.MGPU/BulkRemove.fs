module Test.Alea.CUDA.Extension.MGPU.BulkRemove

open System
open System.Diagnostics
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.DerivedPatterns
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.BulkRemove
open Test.Alea.CUDA.Extension.TestUtilities
//open Test.Alea.CUDA.Extension.TestUtilities.MGPU.BulkRemoveUtils
open NUnit.Framework

let worker = Engine.workers.DefaultWorker
let rng = System.Random()
let stopwatch = Stopwatch()



//let sourceCounts = [10e3; 50e3; 100e3; 200e3; 500e3; 1e6; 2e6; 5e6; 10e6; 20e6]
let sourceCounts = [10000; 50000; 100000; 200000; 500000; 1000000; 2000000; 5000000; 10000000; 20000000]
let nIterations = [2000; 2000; 2000; 1000; 500; 400; 400; 400; 300; 300]
let srcItPairs = List.zip sourceCounts nIterations

let percentDiff x y = abs(x - y) / ((x + y) / 2.0)

let genData sCount rCount =
    let source = Array.init sCount (fun _ -> rng.Next())
    let indices = Array.init rCount (fun _ -> rng.Next sCount) |> Array.sort
    source, indices

let bulkRemove =
    let br = worker.LoadPModule(MGPU.PArray.bulkRemove).Invoke
    fun (data:'TI[]) (indices:int[]) ->
        let calc = pcalc {
            let! data = DArray.scatterInBlob worker data
            let! indices = DArray.scatterInBlob worker indices
            let! result = br data indices
            return! result.Value }
        let dResult = PCalc.run calc
        dResult

let inline bulkRemove2<'TI when 'TI : unmanaged> = cuda {
    let! api = BulkRemove.bulkRemove
    
    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m
        
        fun (data:DArray<'TI>) (indices:DArray<int>) (dest:DArray<'TI>) (numIt:int) ->
            pcalc {
                let sourceCount = data.Length
                //printfn "sourceCount = (%d)" sourceCount
                let indicesCount = indices.Length
                //printfn "indicesCount = (%d)" indicesCount
                let api = api sourceCount indices.Ptr indicesCount
                //let! stopwatch = DStopwatch.startNew worker
                //do! PCalc.action (fun _ -> stopwatch.Start())
                //do stopwatch.Reset()
                
//                let pLoop = pcalc { let stopwatch = new Stopwatch()
//                                    stopwatch.Start()
                for i = 0 to numIt - 1 do
                    do! PCalc.action (fun hint -> api.Action hint data.Ptr dest.Ptr)
//                                    let time = stopwatch.Elapsed.TotalMilliseconds / 1000.0
//                                    stopwatch.Reset()
//                                    return time }
                    
                    
                    
                //let _, ktc = pLoop |> PCalc.runWithKernelTiming numIt

                //let! timing = stopwatch.ElapsedMilliseconds
                //do! PCalc.action (fun _ -> stopwatch.Elapsed.TotalMilliseconds)
                //stopwatch.Stop()
                //let! removed = dest.Gather()
                //printfn "gather"
                //printfn "timing"
                //return removed, timing / 1000.0 } ) }
                (*return! pLoop *)} ) }
let mutable myStats = []

let benchmarkBulkRemove =
    fun (count:int) (numIt:int) (genData : int -> int -> ('T[] * int[] )) ->
        //let bulkRemove = worker.LoadPModule(bulkRemove2).Invoke
        let removeCount = count / 2
        let keepCount = count - removeCount
        let source, indices = genData count removeCount
        
        let data = DArray.scatterInBlob worker source |> PCalc.run
        let indices = DArray.scatterInBlob worker indices |> PCalc.run
        let removed = DArray.createInBlob worker removeCount |> PCalc.run

//        let bulkRemove = pcalc {
//            //let! data = DArray.scatterInBlob worker source
//            //let! indices = DArray.scatterInBlob worker indices
//            //let! removed = DArray.createInBlob worker removeCount
//            return! worker.LoadPModule(bulkRemove2).Invoke data indices removed numIt }
        
        let timing = 
            stopwatch.Start()
            (worker.LoadPModule(bulkRemove2).Invoke data indices removed numIt) |> PCalc.run
            stopwatch.Stop()
            (stopwatch.Elapsed.TotalSeconds)
        
        //do stopwatch.Reset()
        let bytes = (sizeof<'T> * count + keepCount * sizeof<'T> + removeCount * sizeof<'T>) |> float
        let throughput = (float count) * (float numIt) / timing
        let bandwidth = bytes * (float numIt) / timing
        let stats = 
            let newstat = [|(float count); (throughput / 1.0e6); (bandwidth / 1.0e9); timing|]
            myStats <- myStats @ [newstat]
            newstat
        printfn "(%d): %9.3f M/s \t %7.3f GB/s \t Time: %7.3f s" (int stats.[0]) stats.[1] stats.[2] stats.[3]
        

[<Test>]
let ``BulkRemove moderngpu benchmark int`` () =
                            // throughtput, bandwidth
    let mgpuStats_Int = [   (371.468, 2.972);
                            (1597.495, 12.78);
                            (3348.861, 26.791);
                            (5039.794, 40.318);
                            (7327.432, 58.619);
                            (8625.687, 69.005);
                            (9446.528, 75.572);
                            (9877.425, 79.019);
                            (9974.556, 79.796);
                            (10060.556, 80.484)]
    
    let genData sCount rCount =
        let source = Array.init sCount (fun _ -> rng.Next())
        let indices = Array.init rCount (fun _ -> rng.Next sCount) |> Array.sort
        source, indices

    //let myStats = [||] // count; throughput; bandwidth; timing
    //let addStat s = Array.append myStats s
    //stopwatch.Reset()
    (sourceCounts, nIterations) ||> List.iter2 (fun s i -> 
                                                    stopwatch.Reset()
                                                    benchmarkBulkRemove s i genData)
    //printfn "My Stats: %A" myStats

    let n i = int(myStats.[i].[0])
    let tp i = fst mgpuStats_Int.[i], myStats.[i].[1]
    let bw i = snd mgpuStats_Int.[i], myStats.[i].[2]
    let t i = myStats.[i].[3]


    let compareResults = 
        printfn "\n****************** Comparison of Results to ModernGPU Library ********************"
        printfn "COUNT\t\tMGPU Throughput\tMy Throughput\t\tMGPU Bandwidth\t\tMy Bandwidth"
        for i = 0 to sourceCounts.Length - 1 do
            let count = n i
            let mgpuTp, myTp = tp i
            let mgpuBw, myBw = bw i
            printfn "(%d)\t\t(%9.3f)\t\t(%9.3f)\t\t(%7.3f)\t\t(%7.3f)" count mgpuTp myTp mgpuBw myBw
        printfn "\nCOUNT\tThroughput Percent Difference\t\tBandwidth Percent Difference"
        for i = 0 to sourceCounts.Length - 1 do
            let count = n i
            let thruDiff = tp i ||> percentDiff
            let bandDiff = bw i ||> percentDiff
            printfn "(%d)\t\t(%5.1f)\t\t\t\t(%5.1f)" count thruDiff bandDiff
        printfn "\nCOUNT\t\tTIMING(Me Only)"
        for i = 0 to sourceCounts.Length - 1 do
            let count = n i
            let time = t i
            printfn "(%d)\t\t(%7.3f s)" count time
    compareResults

//[<Test>]
//let ``BulkRemove moderngpu benchmark int64`` () =
//    let mgpuStats_Int64 = [ (328.193, 4.595);
//                            (1670.632, 23.389);
//                            (2898.674, 40.581);
//                            (3851.190, 53.917);
//                            (5057.443, 70.804);
//                            (5661.127, 79.256);
//                            (6052.202, 84.731);
//                            (6232.150, 87.250);
//                            (6273.645, 87.831);
//                            (6311.973, 88.638)]
//[<Test>]
//let ``simple bulkRemove`` () =
//    let hValues = Array.init 1000 (fun i -> i)
//    printfn "Initial Array:  %A" hValues
//    let hIndices = [| 2; 3; 8; 11; 13; 14 |]
//    printfn "Indices to remove: %A" hIndices
//    let hResult = Set.difference (hValues |> Set.ofArray) (hIndices |> Set.ofArray) |> Set.toArray
//    printfn "Host Result After Removal:  %A" hResult
//    let dResult = bulkRemove hValues hIndices
//    printfn "Device Result After Removal:  %A" dResult
//
//    displayHandD hResult dResult
//


[<Test>]
let ``bulkRemove moderngpu web example`` () =
    let hValues = Array.init 100 (fun i -> i)
    //printfn "Initial Array:  %A" hValues
    let hRemoveIndices = [|  1;  4;  5;  7; 10; 14; 15; 16; 18; 19;
                            27; 29; 31; 32; 33; 36; 37; 39; 50; 59;
                            60; 61; 66; 78; 81; 83; 85; 90; 91; 96;
                            97; 98; 99 |]

    //printfn "Indices to remove: %A" hRemoveIndices
    let answer = [| 0;  2;  3;  6;  8;  9; 11; 12; 13; 17; 
                   20; 21; 22; 23; 24; 25; 26; 28; 30; 34;
                   35; 38; 40; 41; 42; 43; 44; 45; 46; 47;
                   48; 49; 51; 52; 53; 54; 55; 56; 57; 58;
                   62; 63; 64; 65; 67; 68; 69; 70; 71; 72;
                   73; 74; 75; 76; 77; 79; 80; 82; 84; 86;
                   87; 88; 89; 92; 93; 94; 95 |]


    let hResult = Set.difference (hValues |> Set.ofArray) (hRemoveIndices |> Set.ofArray) |> Set.toArray
    //printfn "Host Result After Removal:  %A" hResult
    let dResult = bulkRemove hValues hRemoveIndices
    //printfn "Device Result After Removal:  %A" dResult
    displayHandD hResult dResult

    printfn "hResult == answer?"
    verify hResult answer
    printfn "yes"
    printfn "dResult == hResult?"
    verify hResult dResult
    printfn "yes"

    
    
//type ConvertType =
//    | AsInts
//    | AsFloat32s
//    | AsFloats
//    | AsInt64s
//
//type InputType =
//    | Ints of int[]
//    | Float32s of float32[]
//    | Floats of float[]
//    | Int64s of int64[]
//
//let genData : int -> int -> ConvertType -> InputType * int[] =
//    fun (sCount:int) (rCount:int) (ct:ConvertType) ->
//        let source = 
//            match ct with
//            | AsInts -> Array.init sCount (fun _ -> rng.Next()) |> Array.map (fun e -> int e) |> Ints
//            | AsFloat32s -> Array.init sCount (fun _ -> rng.Next()) |> Array.map (fun e -> float32 e) |> Float32s
//            | AsFloats -> Array.init sCount (fun _ -> rng.Next()) |> Array.map (fun e -> float e) |> Floats
//            | AsInt64s -> Array.init sCount (fun _ -> rng.Next()) |> Array.map (fun e -> int64 e) |> Int64s
//        let indices = Array.init rCount (fun _ -> rng.Next sCount) |> Array.sort
//        source, indices


//let bulkRemove =
//    let br = worker.LoadPModule(MGPU.PArray.bulkRemove).Invoke
//    fun (data:'TI[]) (indices:int[]) ->
//        let calc = pcalc {
//            let! data = DArray.scatterInBlob worker data
//            let! indices = DArray.scatterInBlob worker indices
//            let! result = br data indices
//            return! result.Value }
//        let dResult = PCalc.run calc
//        dResult



// My output from moderngpu benchmarkbulkinsert run
//GeForce GTX 560 Ti : 1700.000 Mhz   (Ordinal 0)
//8 SMs enabled. Compute Capability sm_21
//FreeMem:    760MB   TotalMem:   1024MB.
//Mem Clock: 2004.000 Mhz x 256 bits   (128.256 GB/s)
//ECC Disabled
//
//Benchmarking BulkRemove on type int.
//   10K:   371.468 M/s      2.972 GB/s
//   50K:  1597.495 M/s     12.780 GB/s
//  100K:  3348.861 M/s     26.791 GB/s
//  200K:  5039.794 M/s     40.318 GB/s
//  500K:  7327.432 M/s     58.619 GB/s
//    1M:  8625.687 M/s     69.005 GB/s
//    2M:  9446.528 M/s     75.572 GB/s
//    5M:  9877.425 M/s     79.019 GB/s
//   10M:  9974.556 M/s     79.796 GB/s
//   20M: 10060.556 M/s     80.484 GB/s
//Benchmarking BulkRemove on type int64.
//   10K:   328.193 M/s      4.595 GB/s
//   50K:  1670.632 M/s     23.389 GB/s
//  100K:  2898.674 M/s     40.581 GB/s
//  200K:  3851.190 M/s     53.917 GB/s
//  500K:  5057.443 M/s     70.804 GB/s
//    1M:  5661.127 M/s     79.256 GB/s
//    2M:  6052.202 M/s     84.731 GB/s
//    5M:  6232.150 M/s     87.250 GB/s
//   10M:  6273.645 M/s     87.831 GB/s
//   20M:  6311.973 M/s     88.368 GB/s
//Benchmarking BulkInsert on type int.
//   10K:   327.200 M/s      3.272 GB/s
//   50K:  1486.284 M/s     14.863 GB/s
//  100K:  2971.904 M/s     29.719 GB/s
//  200K:  4160.540 M/s     41.605 GB/s
//  500K:  5837.133 M/s     58.371 GB/s
//    1M:  6666.386 M/s     66.664 GB/s
//    2M:  6963.040 M/s     69.630 GB/s
//    5M:  7010.378 M/s     70.104 GB/s
//   10M:  7012.604 M/s     70.126 GB/s
//   20M:  6974.831 M/s     69.748 GB/s
//Benchmarking BulkInsert on type int64.
//   10K:   324.171 M/s      5.835 GB/s
//   50K:  1617.291 M/s     29.111 GB/s
//  100K:  2467.732 M/s     44.419 GB/s
//  200K:  3296.471 M/s     59.336 GB/s
//  500K:  4251.011 M/s     76.518 GB/s
//    1M:  4668.734 M/s     84.037 GB/s
//    2M:  4819.431 M/s     86.750 GB/s
//    5M:  4837.521 M/s     87.075 GB/s
//   10M:  4848.496 M/s     87.273 GB/s
//   20M:  4827.018 M/s     86.886 GB/s
