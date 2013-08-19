module Test.Alea.CUDA.Extension.MGPU.SortedSearch

open System
open System.IO
open System.Diagnostics
open System.Collections.Generic
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.DerivedPatterns
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.SortedSearch
open Test.Alea.CUDA.Extension.MGPU.Util
open Test.Alea.CUDA.Extension.MGPU.BenchmarkStats
open Alea.CUDA.Extension.IO.Util
open Alea.CUDA.Extension.IO.CSV
open Alea.CUDA.Extension.IO.Excel

open NUnit.Framework

////////////////////////////
// set this to your device or add your device's C++ output to BenchmarkStats.fs
open Test.Alea.CUDA.Extension.MGPU.BenchmarkStats.GF560Ti
// in the future maybe we try to get the C++ to interop somehow
/////////////////////////////
open ModernGPU.SortedSearchStats

let pSortedSearch = MGPU.PArray.PSortedSearch()
let worker = Engine.workers.DefaultWorker



// This is the weight used in the mgpu benchmark, make sure the data in 
// BenchmarkStats.fs used this same weight or the comparisons will be off
let mgpuWeight = 0.25  

let algName1 = "Sorted Search 1"
let ss1kernelsUsed = [| "kernelSortedSearch"; "kernelMergePartitions" |]
let ss1BMS4 = new BenchmarkStats4(algName1, ss1kernelsUsed, worker.Device.Name, "MGPU", sourceCounts, nIterations)
let algName2 = "Sorted Search 2"
let ss2kernelsUsed = [| "kernelSortedSearch"; "kernelMergePartitions" |]
let ss2BMS4 = new BenchmarkStats4(algName2, ss1kernelsUsed, worker.Device.Name, "MGPU", sourceCounts, nIterations)

// we can probably organize this a lot better, but for now, if you just change
// what module you open above and all of this should adjust accordingly
let oIntTP, oIntBW = SS1.int32_stats |> List.unzip
let oInt64TP, oInt64BW = SS1.int64_stats |> List.unzip
let oFloat32TP, oFloat32BW = SS1.float32_stats |> List.unzip
let oFloat64TP, oFloat64BW = SS1.float64_stats |> List.unzip

let oIntTP2, oIntBW2 = SS2.int32_stats |> List.unzip
let oInt64TP2, oInt64BW2 = SS2.int64_stats |> List.unzip
let oFloat32TP2, oFloat32BW2 = SS2.float32_stats |> List.unzip
let oFloat64TP2, oFloat64BW2 = SS2.float64_stats |> List.unzip

for i = 0 to sourceCounts.Length - 1 do
    // this is setting the opponent (MGPU) stats for the int type
    ss1BMS4.Int32s.OpponentThroughput.[i].Value <- oIntTP.[i]
    ss1BMS4.Int32s.OpponentBandwidth.[i].Value <- oIntBW.[i]
    // set opponent stats for int64
    ss1BMS4.Int64s.OpponentThroughput.[i].Value <- oInt64TP.[i]
    ss1BMS4.Int64s.OpponentBandwidth.[i].Value <- oInt64BW.[i]
    // set oppenent stats for float32
    ss1BMS4.Float32s.OpponentThroughput.[i].Value <- oFloat32TP.[i]
    ss1BMS4.Float32s.OpponentBandwidth.[i].Value <- oFloat32BW.[i]
    // set oppenent stats for float64
    ss1BMS4.Float64s.OpponentThroughput.[i].Value <- oFloat64TP.[i]
    ss1BMS4.Float64s.OpponentBandwidth.[i].Value <- oFloat64BW.[i]

for i = 0 to sourceCounts.Length - 1 do
    // this is setting the opponent (MGPU) stats for the int type
    ss2BMS4.Int32s.OpponentThroughput.[i].Value <- oIntTP2.[i]
    ss2BMS4.Int32s.OpponentBandwidth.[i].Value <- oIntBW2.[i]
    // set opponent stats for int64
    ss2BMS4.Int64s.OpponentThroughput.[i].Value <- oInt64TP2.[i]
    ss2BMS4.Int64s.OpponentBandwidth.[i].Value <- oInt64BW2.[i]
    // set oppenent stats for float32
    ss2BMS4.Float32s.OpponentThroughput.[i].Value <- oFloat32TP2.[i]
    ss2BMS4.Float32s.OpponentBandwidth.[i].Value <- oFloat32BW2.[i]
    // set oppenent stats for float64
    ss2BMS4.Float64s.OpponentThroughput.[i].Value <- oFloat64TP2.[i]
    ss2BMS4.Float64s.OpponentBandwidth.[i].Value <- oFloat64BW2.[i]


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                              IMPORTANT                                                       //
//                                      Choose an Output Type                                                   // 
// This is a switch for all tests, and can do a lot of extra work.  Make sure you turn it off if you just       //
// want to see the console prInt32s.                                                                            //
let outputType = OutputTypeBoth // Choices are CSV, Excel, Both, or None. Set to None for doing kernel timing   //
// only one path, we aren't auto-saving excel stuff yet                                                         //
let workingPath1 = (getWorkingOutputPaths deviceFolderName algName1).CSV                                        //
let workingPath2 = (getWorkingOutputPaths deviceFolderName algName2).CSV                                        //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

let haystack =  [|   0;    5;    5;    7;    7;    7;    7;    8;    9;    9;
                    10;   11;   12;   14;   15;   15;   16;   17;   19;   19;
                    20;   24;   25;   28;   28;   29;   31;   33;   36;   36;
                    37;   38;   40;   42;   42;   43;   45;   46;   49;   50;
                    51;   51;   51;   52;   53;   55;   56;   57;   60;   60;
                    61;   61;   62;   62;   64;   66;   68;   69;   73;   74;
                    79;   81;   82;   84;   85;   88;   90;   90;   95;   97;
                    99;  101;  105;  108;  108;  111;  115;  118;  118;  119;
                    119;  119;  119;  122;  122;  123;  125;  126;  126;  130;
                    133;  133;  135;  135;  139;  140;  143;  145;  145;  146;
                    147;  149;  149;  149;  154;  158;  160;  161;  165;  166;
                    168;  169;  170;  172;  172;  174;  174;  174;  175;  175;
                    175;  177;  179;  182;  183;  184;  186;  187;  188;  190;
                    192;  193;  194;  196;  198;  199;  199;  205;  205;  208;
                    209;  215;  217;  218;  218;  218;  220;  220;  221;  221;
                    223;  224;  225;  230;  234;  234;  235;  240;  240;  243;
                    244;  249;  250;  251;  252;  253;  253;  254;  255;  255;
                    255;  257;  258;  258;  259;  262;  263;  265;  267;  270;
                    270;  274;  278;  278;  278;  279;  280;  281;  284;  284;
                    284;  285;  285;  292;  294;  295;  296;  296;  296;  298 |]

//Needles array:
let needles = [|    3;    3;   12;   16;   16;   17;   17;   19;   20;   21;
                    24;   27;   27;   28;   30;   31;   35;   39;   40;   42;
                    52;   52;   53;   53;   54;   55;   57;   58;   62;   63;
                    72;   75;   83;   86;   86;   89;   92;   95;   98;   98;
                    99;   99;   99;  100;  104;  105;  107;  109;  110;  111;
                    112;  117;  118;  121;  124;  126;  129;  132;  133;  139;
                    140;  148;  156;  160;  161;  167;  168;  173;  179;  186;
                    191;  198;  202;  202;  212;  212;  214;  220;  223;  229;
                    233;  239;  245;  254;  256;  256;  260;  268;  269;  269;
                    271;  271;  272;  273;  277;  285;  296;  296;  299;  299 |]

//Lower bound array: (this is the answer)
let lowerBoundArray = [|    1;    1;   12;   16;   16;   17;   17;   18;   20;   21;
                            21;   23;   23;   23;   26;   26;   28;   32;   32;   33;
                            43;   43;   44;   44;   45;   45;   47;   48;   52;   54;
                            58;   60;   63;   65;   65;   66;   68;   68;   70;   70;
                            70;   70;   70;   71;   72;   72;   73;   75;   75;   75;
                            76;   77;   77;   83;   86;   87;   89;   90;   90;   94;
                            95;  101;  105;  106;  107;  110;  110;  115;  122;  126;
                            130;  134;  137;  137;  141;  141;  141;  146;  150;  153;
                            154;  157;  161;  167;  171;  171;  175;  179;  179;  179;
                            181;  181;  181;  181;  182;  191;  196;  196;  200;  200 |]

[<Test>]
let ``SortedSearch mgpu website example 1`` () =
    let sortedSearch = worker.LoadPModule(pSortedSearch.SortedSearch(MgpuBoundsLower, 0)).Invoke

    let dResult = pcalc {
        let! dNeedles = DArray.scatterInBlob worker needles
        let! dHaystack = DArray.scatterInBlob worker haystack
        
        let needlesSize, haystackSize = needles.Length, haystack.Length
                        
        let! sortedSearch = sortedSearch needlesSize haystackSize
        
        do! sortedSearch dNeedles dHaystack dNeedles 
        
        let! results = dNeedles.Gather()
        return results } |> PCalc.run

    printfn "%A" dResult

[<Test>]
let ``SortedSearch mgpu website example : float64`` () =
    let fneedles = needles |> Array.map (fun x -> float x)
    let haystack = haystack |> Array.map (fun x -> float x)

    let sortedSearch = worker.LoadPModule(pSortedSearch.SortedSearch(MgpuBoundsLower, 0.0)).Invoke

    let dResult = pcalc {
        let! dfNeedles = DArray.scatterInBlob worker fneedles
        let! dNeedles = DArray.scatterInBlob worker needles
        let! dHaystack = DArray.scatterInBlob worker haystack
        
        let needlesSize, haystackSize = needles.Length, haystack.Length
                        
        let! sortedSearch = sortedSearch needlesSize haystackSize
        
        do! sortedSearch dfNeedles dHaystack dNeedles 
        
        let! result = pcalc {   let! dN = dNeedles.Gather()
                                let! dfN = dfNeedles.Gather()
                                return dN, dfN }
        return result } |> PCalc.runInWorker worker

    let hN, hfN = dResult
    printfn "hN: %A" hN
    printfn "hfN: %A" hfN


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  BENCHMARKING : Sorted Search (1)                                                                            //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

let benchmarkSortedSearch (count:int) (aData:'T[]) (aCount:int) (bData:'T[]) (bCount:int) (compOp:IComp<'T>) (numIt:int) (testIdx:int) =
    let search = worker.LoadPModule(pSortedSearch.SortedSearch(MgpuBoundsLower, compOp)).Invoke
    let aData = aData |> Array.sort
    let bData = bData |> Array.sort
    //let aIndices = aData |> Array.map (fun x -> int x) |> Array.sort

    let calc = pcalc {
        let! dA = DArray.scatterInBlob worker aData
        let! daI = DArray.createInBlob<int> worker aCount
        let! dB = DArray.scatterInBlob worker bData
        let! search = search aCount bCount

        // warm up
        do! search dA dB daI

        let! dStopwatch = DStopwatch.startNew worker
        for i = 1 to numIt do
            do! search dA dB daI
        do! dStopwatch.Stop()

        let! results = daI.Gather()
        let! timing = dStopwatch.ElapsedMilliseconds

        return results, timing }

    let hResults, timing' = calc |> PCalc.run
    let timing = timing' / 1000.0
    let bytes = (sizeof<'T> * count + sizeof<int> * aCount) |> float
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
    | x when x = typeof<int> -> ss1BMS4.Int32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | x when x = typeof<int64> -> ss1BMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | x when x = typeof<float32> -> ss1BMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | x when x = typeof<float> -> ss1BMS4.Float64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | _ -> ()
 


[<Test>]
let ``SortedSearch1 moderngpu benchmark : int32`` () =
    let compOp = comp CompTypeLess 0
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount
        let (a:int[]) = rngGenericArrayBounded aCount (ns - 1)
        let (b:int[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch ns a aCount b bCount compOp ni i )
    benchmarkOutput outputType workingPath1 ss1BMS4.Int32s

[<Test>]
let ``SortedSearch1 moderngpu benchmark : int64`` () =
    let compOp = comp CompTypeLess 0L
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount        
        let (a:int64[]) = rngGenericArrayBounded aCount (ns - 1)        
        let (b:int64[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch ns a aCount b bCount compOp ni i )
//    benchmarkOutput outputType workingPath1 ss1BMS4.Int64s

[<Test>]
let ``SortedSearch1 moderngpu benchmark : float32`` () =
    let compOp = comp CompTypeLess 0.f
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount
        let (a:float32[]) = rngGenericArrayBounded aCount (ns - 1)
        let (b:float32[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch ns a aCount b bCount compOp ni i )
    benchmarkOutput outputType workingPath1 ss1BMS4.Float32s

[<Test>]
let ``SortedSearch1 moderngpu benchmark : float64`` () =
    let compOp = comp CompTypeLess 0.0
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount
        let (a:float[]) = rngGenericArrayBounded aCount (ns - 1)
        let (b:float[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch ns a aCount b bCount compOp ni i )
    benchmarkOutput outputType workingPath1 ss1BMS4.Float64s


[<Test>] // above 4 tests, done in sequence (to make output easier)
let ``SortedSearch1 moderngpu benchmark : 4 type`` () =    
    // INT
    printfn "Running Sorted Search (1) moderngpu benchmark : Int32"
    let compOp = comp CompTypeLess 0
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount
        let (a:int[]) = rngGenericArrayBounded aCount (ns - 1)
        let (b:int[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch ns a aCount b bCount compOp ni i )
    benchmarkOutput outputType workingPath1 ss1BMS4.Int32s


    // INT64
    printfn "Running Sorted Search (1) moderngpu benchmark : Int64"
    let compOp = comp CompTypeLess 0L
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount
        let (a:int64[]) = rngGenericArrayBounded aCount (ns - 1)
        let (b:int64[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch ns a aCount b bCount compOp ni i )
    benchmarkOutput outputType workingPath1 ss1BMS4.Int64s


    // FLOAT32
    printfn "Running Sorted Search (1) moderngpu benchmark : Float32"
    let compOp = comp CompTypeLess 0.f
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount
        let (a:float32[]) = rngGenericArrayBounded aCount (ns - 1)
        let (b:float32[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch ns a aCount b bCount compOp ni i )
    benchmarkOutput outputType workingPath1 ss1BMS4.Float32s

    // FLOAT64
    printfn "Running Sorted Search (1) moderngpu benchmark : Float64"
    let compOp = comp CompTypeLess 0.0
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount
        let (a:float[]) = rngGenericArrayBounded aCount (ns - 1)
        let (b:float[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch ns a aCount b bCount compOp ni i )
    benchmarkOutput outputType workingPath1 ss1BMS4.Float64s


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  BENCHMARKING : Sorted Search (2)                                                                            //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

let benchmarkSortedSearch2 (count:int) (aData:'T[]) (aCount:int) (bData:'T[]) (bCount:int) (compOp:IComp<'T>) (numIt:int) (testIdx:int) =
    let search = worker.LoadPModule(pSortedSearch.SortedSearch(MgpuBoundsLower, MgpuSearchTypeIndexMatch, MgpuSearchTypeIndexMatch, compOp)).Invoke
    let aData = aData |> Array.sort
    let bData = bData |> Array.sort

    let calc = pcalc {
        let! dA = DArray.scatterInBlob worker aData
        let! daI = DArray.createInBlob worker aCount
        let! dbI = DArray.createInBlob worker bCount
        let! dB = DArray.scatterInBlob worker bData
        let! search = search aCount bCount

        // warm up
        do! search dA dB daI dbI

        let! dStopwatch = DStopwatch.startNew worker
        for i = 1 to numIt do
            do! search dA dB daI dbI
        do! dStopwatch.Stop()

        let! results = dbI.Gather()
        let! timing = dStopwatch.ElapsedMilliseconds

        return results, timing }

    let hResults, timing' = calc |> PCalc.runInWorker worker
    let timing = timing' / 1000.0
    let bytes = ( sizeof<'T> * count + sizeof<int> * aCount ) |> float
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
    | x when x = typeof<int> -> ss2BMS4.Int32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | x when x = typeof<int64> -> ss2BMS4.Int64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | x when x = typeof<float32> -> ss2BMS4.Float32s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | x when x = typeof<float> -> ss2BMS4.Float64s.NewEntry_My3 testIdx (throughput / 1.0e6) (bandwidth / 1.0e9) timing'
    | _ -> ()


[<Test>]
let ``SortedSearch2 moderngpu benchmark : int32`` () =
    let compOp = comp CompTypeLess 0
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount
        let (a:int[]) = rngGenericArrayBounded aCount (ns - 1)
        let (b:int[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch2 ns a aCount b bCount compOp ni i )
    benchmarkOutput outputType workingPath2 ss2BMS4.Int32s

[<Test>]
let ``SortedSearch2 moderngpu benchmark : int64`` () =
    let compOp = comp CompTypeLess 0L
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount
        let (a:int64[]) = rngGenericArrayBounded aCount (ns - 1)
        let (b:int64[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch2 ns a aCount b bCount compOp ni i )
    benchmarkOutput outputType workingPath2 ss2BMS4.Int64s

[<Test>]
let ``SortedSearch2 moderngpu benchmark : float32`` () =
    let compOp = comp CompTypeLess 0.f
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount
        let (a:float32[]) = rngGenericArrayBounded aCount (ns - 1)
        let (b:float32[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch2 ns a aCount b bCount compOp ni i )
    benchmarkOutput outputType workingPath2 ss2BMS4.Float32s

[<Test>]
let ``SortedSearch2 moderngpu benchmark : float64`` () =
    let compOp = comp CompTypeLess 0.0
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount
        let (a:float[]) = rngGenericArrayBounded aCount (ns - 1)
        let (b:float[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch2 ns a aCount b bCount compOp ni i )
    benchmarkOutput outputType workingPath2 ss2BMS4.Float64s


[<Test>] // above 4 tests, done in sequence (to make output easier)
let ``SortedSearch2 moderngpu benchmark : 4 type`` () =    
    // INT
    printfn "Running Sorted Search (2) moderngpu benchmark : Int32"
    let compOp = comp CompTypeLess 0
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount
        let (a:int[]) = rngGenericArrayBounded aCount (ns - 1)
        let (b:int[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch2 ns a aCount b bCount compOp ni i )
    benchmarkOutput outputType workingPath2 ss2BMS4.Int32s


    // INT64
    printfn "Running Sorted Search (2) moderngpu benchmark : Int64"
    let compOp = comp CompTypeLess 0L
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount
        let (a:int64[]) = rngGenericArrayBounded aCount (ns - 1)
        let (b:int64[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch2 ns a aCount b bCount compOp ni i )
    benchmarkOutput outputType workingPath2 ss2BMS4.Int64s


    // FLOAT32
    printfn "Running Sorted Search (2) moderngpu benchmark : Float32"
    let compOp = comp CompTypeLess 0.f
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount
        let (a:float32[]) = rngGenericArrayBounded aCount (ns - 1)
        let (b:float32[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch2 ns a aCount b bCount compOp ni i )
    benchmarkOutput outputType workingPath2 ss2BMS4.Float32s

    // FLOAT64
    printfn "Running Sorted Search (2) moderngpu benchmark : Float64"
    let compOp = comp CompTypeLess 0.0
    (sourceCounts, nIterations) ||> List.zip |> List.iteri (fun i (ns, ni) ->
        let aCount = int(mgpuWeight * (float ns))
        let bCount = ns - aCount
        let (a:float[]) = rngGenericArrayBounded aCount (ns - 1)
        let (b:float[]) = rngGenericArrayBounded bCount (ns - 1)
        benchmarkSortedSearch2 ns a aCount b bCount compOp ni i )
    benchmarkOutput outputType workingPath2 ss2BMS4.Float64s