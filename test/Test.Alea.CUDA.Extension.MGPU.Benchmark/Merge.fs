module Test.Alea.CUDA.Extension.MGPU.Benchmark.Merge

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.IO.Util
open Test.Alea.CUDA.Extension.MGPU.Util
open NUnit.Framework

////////////////////////////
// set this to your device or add your device's C++ output to BenchmarkStats.fs
open GF560Ti
// in the future maybe we try to get the C++ to interop somehow
/////////////////////////////



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  MERGE KEYS BENCHMARKING                                                                                     //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
module MergeKeys =
    open ModernGPU.MergeStats
    open ModernGPU.MergeStats.Keys

    let worker = getDefaultWorker()
    let pfuncts = new PMerge()

    let algName = "Merge Keys"
    let mkkernelsUsed = [| "kernelMerge"; "kernelMergePartitions" |]
    let mkBMS4 = new BenchmarkStats4(algName, mkkernelsUsed, worker.Device.Name, "MGPU", sourceCounts, nIterations)
    
    // we can probably organize this a lot better, but for now, if you just change
    // what module you open above and all of this should adjust accordingly
    let oIntTP, oIntBW = int32_stats |> List.unzip
    let oInt64TP, oInt64BW = int64_stats |> List.unzip
    
    for i = 0 to sourceCounts.Length - 1 do
        // this is setting the opponent (MGPU) stats for the int type
        mkBMS4.Int32s.OpponentThroughput.[i].Value <- oIntTP.[i]
        mkBMS4.Int32s.OpponentBandwidth.[i].Value <- oIntBW.[i]
        // set opponent stats for int64
        mkBMS4.Int64s.OpponentThroughput.[i].Value <- oInt64TP.[i]
        mkBMS4.Int64s.OpponentBandwidth.[i].Value <- oInt64BW.[i]   


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                              IMPORTANT                                                       //
    //                                      Choose an Output Type                                                   // 
    // This is a switch for all tests, and can do a lot of extra work.  Make sure you turn it off if you just       //
    // want to see the console prInt32s.                                                                            //
    let outputType = OutputTypeBoth // Choices are CSV, Excel, Both, or None. Set to None for doing kernel timing   //
    // only one path, we aren't auto-saving excel stuff yet                                                         //
    let workingPath = (getWorkingOutputPaths deviceFolderName algName).CSV                                        //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    let aib count =
        let aCount = count / 2
        let bCount = count - aCount
        aCount,bCount

    let aibCounts = sourceCounts |> List.map (fun x -> aib x)

    let benchmarkMergeKeys (count:int) (aData:'T[]) (aCount:int) (bData:'T[]) (bCount:int) (compOp:IComp<'T>) (numIt:int) (testIdx:int) =
        let merge = worker.LoadPModule(pfuncts.MergeKeys(compOp)).Invoke
        let aData = aData |> Array.sort
        let bData = bData |> Array.sort

        let calc = pcalc {
            let! dA = DArray.scatterInBlob worker aData        
            let! dB = DArray.scatterInBlob worker bData
            let! dC = DArray.createInBlob worker (aCount + bCount)
            let! merge = merge aCount bCount

            // warm up
            do! merge dA dB dC

            let! dStopwatch = DStopwatch.startNew worker
            for i = 1 to numIt do
                do! merge dA dB dC
            do! dStopwatch.Stop()

            let! results = dC.Gather()
            let! timing = dStopwatch.ElapsedMilliseconds

            return results, timing }

        let hResults, timing' = calc |> PCalc.run
        let timing = timing' / 1000.0
        let bytes = (2 * sizeof<'T> * count) |> float
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
        | x when x = typeof<int> -> mkBMS4.Int32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<int64> -> mkBMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float32> -> mkBMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float> -> mkBMS4.Float64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | _ -> ()
 


    [<Test>]
    let ``MergeKeys moderngpu benchmark : int32`` () =
        let compOp = comp CompTypeLess 0
        (sourceCounts, aibCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (ns, (na, nb), ni) ->
            let (a:int[]) = rngGenericArrayBounded na (ns - 1)
            let (b:int[]) = rngGenericArrayBounded nb (ns - 1)
            benchmarkMergeKeys ns a na b nb compOp ni i )
        benchmarkOutput outputType workingPath mkBMS4.Int32s

    [<Test>]
    let ``MergeKeys moderngpu benchmark : int64`` () =
        let compOp = comp CompTypeLess 0L
        (sourceCounts, aibCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (ns, (na, nb), ni) ->
            let (a:int64[]) = rngGenericArrayBounded na (ns - 1)
            let (b:int64[]) = rngGenericArrayBounded nb (ns - 1)
            benchmarkMergeKeys ns a na b nb compOp ni i )
        benchmarkOutput outputType workingPath mkBMS4.Int64s




    [<Test>] // above 2 tests, done in sequence (to make output easier)
    let ``MergeKeys moderngpu benchmark : 2 type`` () =    
        // INT
        printfn "Running Merge Keys moderngpu benchmark : Int32"
        let compOp = comp CompTypeLess 0
        (sourceCounts, aibCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (ns, (na, nb), ni) ->
            let (a:int[]) = rngGenericArrayBounded na (ns - 1)
            let (b:int[]) = rngGenericArrayBounded nb (ns - 1)
            benchmarkMergeKeys ns a na b nb compOp ni i )
        benchmarkOutput outputType workingPath mkBMS4.Int32s


        // INT64
        printfn "Running Merge Keys moderngpu benchmark : Int64"
        let compOp = comp CompTypeLess 0L
        (sourceCounts, aibCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (ns, (na, nb), ni) ->
            let (a:int64[]) = rngGenericArrayBounded na (ns - 1)
            let (b:int64[]) = rngGenericArrayBounded nb (ns - 1)
            benchmarkMergeKeys ns a na b nb compOp ni i )
        benchmarkOutput outputType workingPath mkBMS4.Int64s



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  MERGE PAIRS BENCHMARKING                                                                                    //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
module MergePairs =
    open ModernGPU.MergeStats
    open ModernGPU.MergeStats.Pairs

    let worker = getDefaultWorker()
    let pfuncts = new PMerge()
        
    let algName = "Merge Pairs"
    let mpkernelsUsed = [| "kernelMerge"; "kernelMergePartitions" |]
    let mpBMS4 = new BenchmarkStats4(algName, mpkernelsUsed, worker.Device.Name, "MGPU", sourceCounts, nIterations)

    // we can probably organize this a lot better, but for now, if you just change
    // what module you open above and all of this should adjust accordingly
    let oIntTP2, oIntBW2 = Pairs.int32_stats |> List.unzip
    let oInt64TP2, oInt64BW2 = Pairs.int64_stats |> List.unzip
       
    for i = 0 to sourceCounts.Length - 1 do
        // this is setting the opponent (MGPU) stats for the int type
        mpBMS4.Int32s.OpponentThroughput.[i].Value <- oIntTP2.[i]
        mpBMS4.Int32s.OpponentBandwidth.[i].Value <- oIntBW2.[i]
        // set opponent stats for int64
        mpBMS4.Int64s.OpponentThroughput.[i].Value <- oInt64TP2.[i]
        mpBMS4.Int64s.OpponentBandwidth.[i].Value <- oInt64BW2.[i]
    

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                              IMPORTANT                                                       //
    //                                      Choose an Output Type                                                   // 
    // This is a switch for all tests, and can do a lot of extra work.  Make sure you turn it off if you just       //
    // want to see the console prInt32s.                                                                            //
    let outputType = OutputTypeBoth // Choices are CSV, Excel, Both, or None. Set to None for doing kernel timing   //
    // only one path, we aren't auto-saving excel stuff yet                                                         //
    let workingPath = (getWorkingOutputPaths deviceFolderName algName).CSV                                        //    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    let aib count =
        let aCount = count / 2
        let bCount = count - aCount
        aCount,bCount

    let aibCounts = sourceCounts |> List.map (fun x -> aib x)

    let benchmarkMergePairs (count:int) (aKeys:'T[]) (aVals:'T[]) (aCount:int) (bKeys:'T[]) (bVals:'T[]) (bCount:int) (compOp:IComp<'T>) (numIt:int) (testIdx:int) =
        let merge = worker.LoadPModule(pfuncts.MergePairs(compOp)).Invoke
        let aKeys = aKeys |> Array.sort
        let bKeys = bKeys |> Array.sort

        let calc = pcalc {
            let! dAk = DArray.scatterInBlob worker aKeys
            let! dBk = DArray.scatterInBlob worker bKeys
            let! dAv = DArray.scatterInBlob worker aVals
            let! dBv = DArray.scatterInBlob worker bVals
            let! merge = merge aCount bCount
            let! dCk = DArray.createInBlob worker (count)
            let! dCv = DArray.createInBlob worker (count)

            // warm up
            do! merge dAk dAv dBk dBv dCk dCv

            let! dStopwatch = DStopwatch.startNew worker
            for i = 1 to numIt do
                do! merge dAk dAv dBk dBv dCk dCv
            do! dStopwatch.Stop()

            let! results = pcalc { let! dCk = dCk.Gather()
                                   let! dCv = dCv.Gather() 
                                   return dCk, dCv}

            let! timing = dStopwatch.ElapsedMilliseconds

            return results, timing }

        let hResults, timing' = calc |> PCalc.runInWorker worker
        let timing = timing' / 1000.0
        let bytes = (2 * (2 * sizeof<'T>) * count) |> float
        let bandwidth = bytes * (float numIt) / timing
        let throughput = (float count) * (float numIt) / timing
        printfn "%9d: %9.3f M/s %9.3f GB/s %6.3f ms x %4d = %7.3f ms"
            count
            (throughput / 1e6)
            (bandwidth / 1e9)
            (timing' / (float numIt))
            numIt
            timing'

        match typeof<'T> with
        | x when x = typeof<int> -> mpBMS4.Int32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<int64> -> mpBMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float32> -> mpBMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | x when x = typeof<float> -> mpBMS4.Float64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
        | _ -> ()


    [<Test>]
    let ``MergePairs moderngpu benchmark : int32`` () =
        let compOp = comp CompTypeLess 0
        (sourceCounts, aibCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (ns, (na, nb), ni) ->
            let (aK:int[]) = rngGenericArrayBounded na (ns - 1)
            let aV = Array.init<int> na (fun i -> i)
            let (bK:int[]) = rngGenericArrayBounded nb (ns - 1)
            let bV = Array.init<int> nb (fun i -> i + na)
            benchmarkMergePairs ns aK aV na bK bV nb compOp ni i )
        benchmarkOutput outputType workingPath mpBMS4.Int32s

    [<Test>]
    let ``MergePairs moderngpu benchmark : int64`` () =
        let compOp = comp CompTypeLess 0L
        (sourceCounts, aibCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (ns, (na, nb), ni) ->
            let (aK:int64[]) = rngGenericArrayBounded na (ns - 1)
            let aV = Array.init<int64> na (fun i -> int64 i)
            let (bK:int64[]) = rngGenericArrayBounded nb (ns - 1)
            let bV = Array.init<int64> nb (fun i -> int64(i + na))
            benchmarkMergePairs ns aK aV na bK bV nb compOp ni i )
        benchmarkOutput outputType workingPath mpBMS4.Int64s


    [<Test>] // above 2 tests, done in sequence (to make output easier)
    let ``MergePairs moderngpu benchmark : 2 type`` () =    
        // INT
        printfn "Running Merge Pairs moderngpu benchmark : Int32"
        let compOp = comp CompTypeLess 0
        (sourceCounts, aibCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (ns, (na, nb), ni) ->
            let (aK:int[]) = rngGenericArrayBounded na (ns - 1)
            let aV = Array.init<int> na (fun i -> i)
            let (bK:int[]) = rngGenericArrayBounded nb (ns - 1)
            let bV = Array.init<int> nb (fun i -> i + na)
            benchmarkMergePairs ns aK aV na bK bV nb compOp ni i )
        benchmarkOutput outputType workingPath mpBMS4.Int32s


        // INT64
        printfn "Running Merge Pairs moderngpu benchmark : Int64"
        let compOp = comp CompTypeLess 0L
        (sourceCounts, aibCounts, nIterations) |||> List.zip3 |> List.iteri (fun i (ns, (na, nb), ni) ->
            let (aK:int64[]) = rngGenericArrayBounded na (ns - 1)
            let aV = Array.init<int64> na (fun i -> int64 i)
            let (bK:int64[]) = rngGenericArrayBounded nb (ns - 1)
            let bV = Array.init<int64> nb (fun i -> int64(i + na))
            benchmarkMergePairs ns aK aV na bK bV nb compOp ni i )
        benchmarkOutput outputType workingPath mpBMS4.Int64s