module Test.Alea.CUDA.Extension.MGPU.Benchmark.IntervalMove

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.Output.Util
open Test.Alea.CUDA.Extension.MGPU.Util
open NUnit.Framework

////////////////////////////
// set this to your device or add your device's C++ output to BenchmarkStats.fs
open GF560Ti
// in the future maybe we try to get the C++ to interop somehow
/////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  INTERVAL EXPAND BENCHMARKING                                                                                                //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
module IntervalExpand =
    open ModernGPU.IntervalExpandStats
    
    let worker = getDefaultWorker()
    let pScanner = new PScan()
    let pIntervalMover = new PIntervalMove()
    let op = (CTAScan.scanOp ScanOpTypeAdd 0)

    let iexp_A_4typeStatsList = ModernGPU.IntervalExpandStats.AvgSegLength25.fourTypeStatsList
    let iexp_B_4typeStatsList = ModernGPU.IntervalExpandStats.ConstCountChangingExpandRate.fourTypeStatsList
    
    let algName_iexp_A = "Interval Expand - Avg Seg Length 25 elements"
    let iexp_A_KernelsUsed = [| "kernelIntervalExpand"; "kernelMergePartition" |]
    let iexp_A_BMS4 = getFilledBMS4Object algName_iexp_A iexp_A_KernelsUsed worker.Device.Name "MGPU" sourceCounts nIterations iexp_A_4typeStatsList

    let algName_iexp_B = "Interval Expand - Const Count 10M - Changing Expand Rate"
    let iexp_B_KernelsUsed = [| "kernelIntervalExpand"; "kernelMergePartition" |]
    let iexp_B_BMS4 = getFilledBMS4Object algName_iexp_B iexp_B_KernelsUsed worker.Device.Name "MGPU" constCounts constIterations iexp_B_4typeStatsList


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                              IMPORTANT                                                       //
    //                                      Choose an Output Type                                                   // 
    // This is a switch for all tests, and can do a lot of extra work.  Make sure you turn it off if you just       //
    // want to see the console prInt32s.                                                                            //
    let outputType = OutputTypeNone // Choices are CSV, Excel, Both, or None. Set to None for doing kernel timing   //
    // only one path, we aren't auto-saving excel stuff yet                                                         //
    let workingPathExpA = (getWorkingOutputPaths deviceFolderName algName_iexp_A).CSV                               //
    let workingPathExpB = (getWorkingOutputPaths deviceFolderName algName_iexp_B).CSV                               //    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    let genRandomIntervals (count:int) (numTerms:int) (terms:int[]) =
        let dquot, drem = (count / numTerms), (count % numTerms)
        for i = 0 to numTerms - 1 do
            Array.set terms i (dquot + (if i < drem then 1 else 0))
        for i = 0 to numTerms - 2 do
            let r = rng.Next(numTerms - i - 1)
            let x = min terms.[r] terms.[i + r]
            let r2 = rng.Next(-x,x)
            Array.set terms r (terms.[r] - r2)
            Array.set terms (i + r) (terms.[i + r] + r2)
            swap terms.[r] terms.[i + r]

    let benchmarkIntervalExpand (version:char) (hSource:'T[]) (count:int) (numIt:int) (numTerms:int) (testIdx:int) =
        let scan = worker.LoadPModule(pScanner.Scan()).Invoke
        let expand = worker.LoadPModule(pIntervalMover.IntervalExpandFunc()).Invoke
    
        let terms = Array.zeroCreate numTerms
        genRandomIntervals count numTerms terms

        let total, scannedCounts =
            pcalc { let! dCounts = DArray.scatterInBlob worker terms
                    return! scan dCounts} |> PCalc.run
        
        let calc = pcalc {
            let! dCounts = DArray.scatterInBlob worker scannedCounts
        
            let! expand = expand total numTerms
            let! dData = DArray.scatterInBlob worker hSource
            let! dDest = DArray.createInBlob worker total
        
            // warm up
            do! expand dCounts dData dDest

            let! dStopwatch = DStopwatch.startNew worker
            for i = 1 to numIt do
                do! expand dCounts dData dDest
            do! dStopwatch.Stop()

            let! results = dDest.Gather()
            let! timing = dStopwatch.ElapsedMilliseconds

            return results, timing }

        let hResults, timing' = calc |> PCalc.runInWorker worker
        let timing = timing' / 1000.0
        let bytes = sizeof<'T> * (count + numTerms) + sizeof<int> * numTerms |> float
        let throughput = (float count) * (float numIt) / timing
        let bandwidth = bytes * (float numIt) / timing
        printfn "%9d: %9.3f M/s %9.3f GB/s %6.3f ms x %4d = %7.3f ms"
            count
            (throughput / 1e6)
            (bandwidth / 1e9)
            (timing' / (float numIt))
            numIt
            timing'

        match version with
        | 'A' ->
            match typeof<'T> with
            | x when x = typeof<int> -> iexp_A_BMS4.Int32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | x when x = typeof<int64> -> iexp_A_BMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | x when x = typeof<float32> -> iexp_A_BMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | x when x = typeof<float> -> iexp_A_BMS4.Float64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | _ -> ()
        | 'B' ->
            match typeof<'T> with
            | x when x = typeof<int> -> iexp_B_BMS4.Int32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | x when x = typeof<int64> -> iexp_B_BMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | x when x = typeof<float32> -> iexp_B_BMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | x when x = typeof<float> -> iexp_B_BMS4.Float64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | _ -> ()
        | _ -> ()

    
    [<Test>]
    let ``IntervalExpand moderngpu benchmark (A) : 4type`` () =
        printfn "\nInterval Expand, Avg Seg Length = 25, Int32"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : int[] = Array.init (ns / 25) (fun i -> i)        
            benchmarkIntervalExpand 'A' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathExpA iexp_A_BMS4.Int32s

        printfn "\nInterval Expand, Avg Seg Length = 25, Int64"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : int64[] = Array.init ns (fun i -> int64 i)
            benchmarkIntervalExpand 'A' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathExpA iexp_A_BMS4.Int64s

        printfn "\nInterval Expand, Avg Seg Length = 25, Float32"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : float32[] = Array.init ns (fun i -> float32 i)
            benchmarkIntervalExpand 'A' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathExpA iexp_A_BMS4.Float32s

        printfn "\nInterval Expand, Avg Seg Length = 25, Float64"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : float[] = Array.init ns (fun i -> float i)
            benchmarkIntervalExpand 'A' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathExpA iexp_A_BMS4.Float64s


    [<Test>]
    let ``IntervalExpand moderngpu benchmark (B) : 4type`` () =
        printfn "\nInterval Expand, 10M Count & Changing Expand Rate, Int32"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : int[] = Array.init ns (fun i -> i)
            benchmarkIntervalExpand 'B' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathExpB iexp_B_BMS4.Int32s

        printfn "\nInterval Expand, 10M Count & Changing Expand Rate, Int64"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : int64[] = Array.init ns (fun i -> int64 i)
            benchmarkIntervalExpand 'B' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathExpB iexp_B_BMS4.Int64s

        printfn "\nInterval Expand, 10M Count & Changing Expand Rate, Float32"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : float32[] = Array.init ns (fun i -> float32 i)
            benchmarkIntervalExpand 'B' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathExpB iexp_B_BMS4.Float32s

        printfn "\nInterval Expand, 10M Count & Changing Expand Rate, Float64"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : float[] = Array.init ns (fun i -> float i)
            benchmarkIntervalExpand 'B' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathExpB iexp_B_BMS4.Float64s



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  INTERVAL MOVE BENCHMARKING                                                                                  //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
module IntervalMove =
    open ModernGPU.IntervalMoveStats
    let worker = getDefaultWorker()
    let pScanner = new PScan()
    let pIntervalMover = new PIntervalMove()
    let op = (CTAScan.scanOp ScanOpTypeAdd 0)
    
    let imv_A_4typeStatsList = ModernGPU.IntervalMoveStats.AvgSegLength25.fourTypeStatsList
    let imv_B_4typeStatsList = ModernGPU.IntervalMoveStats.ConstCountChangingExpandRate.fourTypeStatsList
        
    let algName_imv_A = "Interval Move - Avg Seg Length 25 elements"
    let imv_A_KernelsUsed = [| "kernelIntervalMove"; "kernelMergePartition" |]
    let imv_A_BMS4 = getFilledBMS4Object algName_imv_A imv_A_KernelsUsed worker.Device.Name "MGPU" sourceCounts nIterations imv_A_4typeStatsList

    let algName_imv_B = "Interval Move - Const Count 10M - Changing Expand Rate"
    let imv_B_KernelsUsed = [| "kernelIntervalMove"; "kernelMergePartition" |]
    let imv_B_BMS4 = getFilledBMS4Object algName_imv_B imv_B_KernelsUsed worker.Device.Name "MGPU" constCounts constIterations imv_B_4typeStatsList
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                              IMPORTANT                                                       //
    //                                      Choose an Output Type                                                   // 
    // This is a switch for all tests, and can do a lot of extra work.  Make sure you turn it off if you just       //
    // want to see the console prInt32s.                                                                            //
    let outputType = OutputTypeNone // Choices are CSV, Excel, Both, or None. Set to None for doing kernel timing   //
    // only one path, we aren't auto-saving excel stuff yet                                                         //    
    let workingPathMovA = (getWorkingOutputPaths deviceFolderName algName_imv_A).CSV                                //
    let workingPathMovB = (getWorkingOutputPaths deviceFolderName algName_imv_B).CSV                                //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    let genRandomIntervals (count:int) (numTerms:int) (terms:int[]) =
        let dquot, drem = (count / numTerms), (count % numTerms)
        for i = 0 to numTerms - 1 do
            Array.set terms i (dquot + (if i < drem then 1 else 0))
        for i = 0 to numTerms - 2 do
            let r = rng.Next(numTerms - i - 1)
            let x = min terms.[r] terms.[i + r]
            let r2 = rng.Next(-x,x)
            Array.set terms r (terms.[r] - r2)
            Array.set terms (i + r) (terms.[i + r] + r2)
            swap terms.[r] terms.[i + r]

    let benchmarkIntervalMove (version:char) (hSource:'T[]) (count:int) (numIt:int) (numTerms:int) (testIdx:int) =
        let scan = worker.LoadPModule(pScanner.Scan()).Invoke    
        let move = worker.LoadPModule(pIntervalMover.IntervalMoveFunc()).Invoke

        let permGather, permScatter = (Array.init numTerms (fun i -> i)), 
                                      (Array.init numTerms (fun i -> i))

        for i = 0 to numTerms - 2 do
            let r1 = rng.Next(numTerms - i - 1)
            let r2 = rng.Next(numTerms - i - 1)
            swap permGather.[i] permGather.[i + r1]
            swap permScatter.[i] permScatter.[i + r2]

        let terms = Array.zeroCreate numTerms
        genRandomIntervals count numTerms terms

        let gather, scatter = (Array.zeroCreate numTerms), (Array.zeroCreate numTerms)

        for i = 0 to numTerms - 1 do
            Array.set gather permGather.[i] terms.[i]
            Array.set scatter permScatter.[i] terms.[i]

        let mutable x, y = 0, 0
        for i = 0 to numTerms - 1 do
            let x2, y2 = gather.[i], scatter.[i]
            Array.set gather i x
            Array.set scatter i y
            x <- x + x2
            y <- y + y2

        let gatherHost, scatterHost = (Array.zeroCreate numTerms), (Array.zeroCreate numTerms)
        for i = 0 to numTerms - 1 do
            Array.set gatherHost i (gather.[permGather.[i]])
            Array.set scatterHost i (scatter.[permScatter.[i]])

        let total, scannedCounts =
            pcalc { let! dCounts = DArray.scatterInBlob worker terms
                    return! scan dCounts } |> PCalc.run    

        let calc = pcalc {
            let! dCounts = DArray.scatterInBlob worker scannedCounts
            let! dGather = DArray.scatterInBlob worker gatherHost
            let! dScatter = DArray.scatterInBlob worker scatterHost
            let! dSource = DArray.scatterInBlob worker hSource
            let! dDest = DArray.createInBlob<'T> worker count

            let! move = move count numTerms

            // warm up
            do! move dSource dGather dScatter dCounts  dDest

            let! dStopwatch = DStopwatch.startNew worker
            for i = 1 to numIt do
                do! move dSource dGather dScatter dCounts dDest
            do! dStopwatch.Stop()

            let! results = dDest.Gather()
            let! timing = dStopwatch.ElapsedMilliseconds

            return results, timing }

        let hResults, timing' = calc |> PCalc.runInWorker worker
        let timing = timing' / 1000.0
        let bytes = 3 * sizeof<int> * numTerms + 2 * sizeof<'T> * count |> float
        let throughput = (float count) * (float numIt) / timing
        let bandwidth = bytes * (float numIt) / timing
        printfn "%9d: %9.3f M/s %9.3f GB/s %6.3f ms x %4d = %7.3f ms"
            count
            (throughput / 1e6)
            (bandwidth / 1e9)
            (timing' / (float numIt))
            numIt
            timing'

        match version with
        | 'A' ->
            match typeof<'T> with
            | x when x = typeof<int> -> imv_A_BMS4.Int32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | x when x = typeof<int64> -> imv_A_BMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | x when x = typeof<float32> -> imv_A_BMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | x when x = typeof<float> -> imv_A_BMS4.Float64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | _ -> ()
        | 'B' ->
            match typeof<'T> with
            | x when x = typeof<int> -> imv_B_BMS4.Int32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | x when x = typeof<int64> -> imv_B_BMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | x when x = typeof<float32> -> imv_B_BMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | x when x = typeof<float> -> imv_B_BMS4.Float64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
            | _ -> ()
        | _ -> ()


    [<Test>]
    let ``IntervalMove moderngpu benchmark (A) : 4type`` () =
        printfn "\nInterval Move, Avg Seg Length = 25, Int32"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : int[] = Array.init ns (fun i -> i)
            benchmarkIntervalMove 'A' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathMovA imv_A_BMS4.Int32s

        printfn "\nInterval Move, Avg Seg Length = 25, Int64"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : int64[] = Array.init ns (fun i -> int64 i)
            benchmarkIntervalMove 'A' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathMovA imv_A_BMS4.Int64s

        printfn "\nInterval Move, Avg Seg Length = 25, Float32"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : float32[] = Array.init ns (fun i -> float32 i)
            benchmarkIntervalMove 'A' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathMovA imv_A_BMS4.Float32s

        printfn "\nInterval Move, Avg Seg Length = 25, Float64"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : float[] = Array.init ns (fun i -> float i)
            benchmarkIntervalMove 'A' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathMovA imv_A_BMS4.Float64s


    [<Test>]
    let ``IntervalMove moderngpu benchmark (B) : 4type`` () =
        printfn "\nInterval Move, 10M Count & Changing Expand Rate, Int32"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : int[] = Array.init ns (fun i -> i)
            benchmarkIntervalMove 'B' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathMovB imv_B_BMS4.Int32s

        printfn "\nInterval Move, 10M Count & Changing Expand Rate, Int64"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : int64[] = Array.init ns (fun i -> int64 i)
            benchmarkIntervalMove 'B' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathMovB imv_B_BMS4.Int64s

        printfn "\nInterval Move, 10M Count & Changing Expand Rate, Float32"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : float32[] = Array.init ns (fun i -> float32 i)
            benchmarkIntervalMove 'B' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathMovB imv_B_BMS4.Float32s

        printfn "\nInterval Move, 10M Count & Changing Expand Rate, Float64"
        (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
            let input : float[] = Array.init ns (fun i -> float i)
            benchmarkIntervalMove 'B' input ns ni (ns / 25) i)
        benchmarkOutput outputType workingPathMovB imv_B_BMS4.Float64s