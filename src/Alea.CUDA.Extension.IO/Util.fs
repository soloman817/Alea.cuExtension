[<AutoOpen>]
module Alea.CUDA.Extension.IO.Util

open System
open System.IO
open System.Text.RegularExpressions
open Alea.CUDA.Extension.Util


let splitAt n xs = (Seq.truncate n xs, if Seq.length xs < n then Seq.empty else Seq.skip n xs)
let rec chunk n xs =
    if Seq.isEmpty xs then Seq.empty
    else
        let (ys,zs) = splitAt n xs
        Seq.append (Seq.singleton ys) (chunk n zs)

let swatchStopPrintReset (stopwatch:System.Diagnostics.Stopwatch) (nameOfThingBeingTimed:string) =
    stopwatch.Stop()
    printfn "Elapsed Time for (%s)  ==>  %8.2f (s)" nameOfThingBeingTimed ((float stopwatch.ElapsedMilliseconds) / 1000.0)
    stopwatch.Reset()

[<AutoOpen>]
module Input =
    // http://www2.imm.dtu.dk/~mire/FSharpBook/
    type parser<'a> = string -> int -> ('a * int) list

    let captureSingle (ma:Match) (n:int) =
        ma.Groups.[n].Captures.[0].Value

    let token (reg:Regex) (conv:string -> 'a) : parser<'a> =
        fun str pos ->
            let ma = reg.Match(str, pos)
            match ma.Success with
                | false -> []
                | _ ->
                    let pos2 = pos + ma.Length
                    [( conv(captureSingle ma 1), pos2)]

    let emptyToken (reg:Regex) : parser<unit> =
        fun str pos ->
            let ma = reg.Match(str,pos)
            match ma.Success with
            | false -> []
            | _     -> let pos2 = pos + ma.Length
                       [((), pos2)]

    type ParserClass() =
        member t.Bind(p:parser<'a>, f:'a -> parser<'b>):parser<'b> =
            fun str pos ->
                List.collect (fun (a,apos) -> f a str apos) (p str pos)
        member bld.Zero() = (fun _ _ -> []):parser<'a>
        member bld.Return a = (fun str pos -> [(a, pos)]):parser<'a>
        member bld.ReturnFrom (p:parser<'a>) = p

    let parser = new ParserClass()

    let pairOf p1 p2 = parser {
        let! x1 = p1
        let! x2 = p2
        return (x1,x2)}

    let (<|>) (p1:parser<'a>) (p2:parser<'a>) =
        (fun str pos -> (p1 str pos) @ (p2 str pos)):parser<'a>

    let rec listOf p = parser {return []}
                       <|> parser {let! x = p
                                   let! xs = listOf p
                                   return x::xs}
    // end http://www2.imm.dtu.dk/~mire/FSharpBook/


    let kernelNameReg = Regex @"(Kernel[\w]+)"
    let durationReg = Regex @",(\d+\.\d+)"
    let eosReg        = Regex @"(\n*)"
    
    let kernel = token kernelNameReg id
    let duration = token durationReg float
    let eos    = emptyToken eosReg
 
//    let kTiming = pairOf kernel duration
//    let kData = listOf kTiming
//
//    let stuff = parser { let! k = kData
//                         let! _ = eos
//                         return k }
//
//    printfn "%A" (stuff info 0)


[<AutoOpen>]
module Output =
    type WorkingOutputPaths =
        {
            Excel : string
            CSV : string
        }

    let getWorkingPathCSV (deviceFolderName:string) (algorithmName:string) (*overwrite:bool*) = 
        let perfDataDir = Directory.CreateDirectory("../../New Performance Data")
        let mutable workingPath = ""    
        //if overwrite then
        let benchCsvDir = perfDataDir.CreateSubdirectory("Benchmark_CSV")
        let deviceDir = benchCsvDir.CreateSubdirectory(deviceFolderName)
        let workingDir = deviceDir.CreateSubdirectory(algorithmName)
        workingPath <- workingDir.FullName + "/"
    //    else 
    //        let newPerfData = perfDataDir.CreateSubdirectory("new")
    //        let benchCsvDir = newPerfData.CreateSubdirectory("Benchmerk_CSV")
    //        let deviceDir = benchCsvDir.CreateSubdirectory(deviceFolderName)
    //        let workingDir = deviceDir.CreateSubdirectory(algorithmName)
    //        workingPath <- workingDir.FullName + "/"    
        workingPath


    let getWorkingPathExcel (deviceFolderName:string) (algorithmName:string) = 
        let perfDataDir = Directory.CreateDirectory("../../New Performance Data")
        let mutable workingPath = ""    
        //if overwrite then
        let benchCsvDir = perfDataDir.CreateSubdirectory("Benchmark_Excel")
        let deviceDir = benchCsvDir.CreateSubdirectory(deviceFolderName)
        let workingDir = deviceDir.CreateSubdirectory(algorithmName)
        workingPath <- workingDir.FullName + "/"
    //    else 
    //        let newPerfData = perfDataDir.CreateSubdirectory("new")
    //        let benchCsvDir = newPerfData.CreateSubdirectory("Benchmerk_Excel")
    //        let deviceDir = benchCsvDir.CreateSubdirectory(deviceFolderName)
    //        let workingDir = deviceDir.CreateSubdirectory(algorithmName)
    //        workingPath <- workingDir.FullName + "/"    
        workingPath


    let getWorkingOutputPaths (deviceFolderName:string) (algorithmName:string) (*overwrite:bool*) = 
        let wop = 
            { Excel = getWorkingPathExcel deviceFolderName algorithmName (*overwrite*)
              CSV = getWorkingPathCSV deviceFolderName algorithmName (*overwrite*) }
        wop


    type IHeaders = 
        abstract member HList : List<string> with get, set

    type IOutputColumn<'T> =
        abstract member Label : string with get, set
        abstract member Values : List<'T> with get, set


    type Entry =
        val mutable Kernel : string     // which kernel this came from
        val mutable RowID : int
        val mutable StatKind : string
        val mutable ElementCount : int
        val mutable Iterations : int
        val mutable Value : float
        new (k, rId, sK, ns, ni, v) = {Kernel = k; RowID = rId; StatKind = sK; ElementCount = ns; Iterations = ni; Value = v }
    
        member e.CountIterPair = e.ElementCount, e.Iterations

        static member Zero = new Entry("",0,"",0,0,0.0)
    //
    //    member e.DisplayWith = 
    //        printfn "%9.3f\t%d" e.Value e.Count
    //
    //type IBenchmarkStats =
    //    abstract AlgorithmName : string
    //    abstract DeviceName : string
    //    abstract TestedType : string
    //    abstract Opponent : string
    //    
    //    abstract MyThroughput : List<Entry>
    //    abstract MyBandwidth : List<Entry>
    //    abstract MyTiming : List<Entry>
    //
    //    abstract OpponentThroughput : List<Entry>
    //    abstract OpponentBandwidth : List<Entry>
    //    abstract OpponentTiming : List<Entry>
    //
    //    abstract NewEntry_MyTP : (int -> float -> unit)
    //    abstract NewEntry_MyBW : (int -> float -> unit)
    //    abstract NewEntry_MyT : (int -> float -> unit)
    //
    //    abstract NewEntry_OppTP : (int -> float -> unit)
    //    abstract NewEntry_OppBW : (int -> float -> unit)
    //    abstract NewEntry_OppT : (int -> float -> unit)


    type BenchmarkStats (algorithmName : string, kernelsUsed:string[], deviceName : string, testedType : string, opponent : string, sourceCounts : int list, nIterations: int list) =
        let percentDifference a b = (abs(a - b) / ((a + b) / 2.0)) * 100.0
    
        let mutable myTP, myBW, myT = // my throughput, bandwidth, & timing
            let mutable tp, bw, t = 
                Array.init sourceCounts.Length (fun _ -> Entry.Zero),
                Array.init sourceCounts.Length (fun _ -> Entry.Zero),
                Array.init sourceCounts.Length (fun _ -> Entry.Zero)

            for i = 0 to sourceCounts.Length - 1 do
                let ns, ni = sourceCounts.[i], nIterations.[i] 
                (tp, i, (new Entry("Alea.cuBase", i, "Throughput", ns, ni, 0.0))) |||> Array.set
                (bw, i, (new Entry("Alea.cuBase", i, "Bandwidth", ns, ni, 0.0))) |||> Array.set
                (t, i, (new Entry("Alea.cuBase", i, "Timing", ns, ni, 0.0))) |||> Array.set
            tp, bw, t

        let mutable oppTP, oppBW, oppT = // opponents throughput, bandwidth, & timing
            let mutable tp, bw, t = 
                Array.init sourceCounts.Length (fun _ -> Entry.Zero),
                Array.init sourceCounts.Length (fun _ -> Entry.Zero),
                Array.init sourceCounts.Length (fun _ -> Entry.Zero)
            for i = 0 to sourceCounts.Length - 1 do
                let ns, ni = sourceCounts.[i], nIterations.[i]
                (tp, i, (new Entry(opponent, i, "Throughput", ns, ni, 0.0))) |||> Array.set
                (bw, i, (new Entry(opponent, i, "Bandwidth", ns, ni, 0.0))) |||> Array.set
                (t, i, (new Entry(opponent, i, "Timing", ns, ni, 0.0))) |||> Array.set
            tp, bw, t
    
        member bs.AlgorithmName = algorithmName
        member bs.KernelsUsed = kernelsUsed
        member bs.SourceCounts = sourceCounts
        member bs.NumIterations = nIterations
        member bs.DeviceName = deviceName
        member bs.TestedType = testedType
        member bs.Opponent = opponent
    
        member bs.MyThroughput with get() = myTP and set(tp:Entry[]) = myTP <- tp
        member bs.MyBandwidth with get() = myBW and set(bw:Entry[]) = myBW <- bw
        member bs.MyTiming with get() = myT and set(t:Entry[]) = myT <- t

        member bs.OpponentThroughput with get() = oppTP and set(tp:Entry[]) = oppTP <- tp
        member bs.OpponentBandwidth with get() = oppBW and set(bw:Entry[]) = oppBW <- bw
        member bs.OpponentTiming with get() = oppT and set(t:Entry[]) = oppT <- t

        member bs.NewEntry_MyTP (rowId:int) (tp:float)  = myTP.[rowId].Value <- tp
        member bs.NewEntry_MyBW (rowId:int) (bw:float) = myBW.[rowId].Value <- bw
        member bs.NewEntry_MyT (rowId:int) (t:float) = myT.[rowId].Value <- t

        member bs.NewEntry_My3 (rowId:int) (tPut:float) (bWidth:float) (time:float) =
            bs.NewEntry_MyTP rowId tPut
            bs.NewEntry_MyBW rowId bWidth
            bs.NewEntry_MyT rowId time

        member bs.NewEntry_OppTP (rowId:int) (tp:float)  = oppTP.[rowId].Value <- tp
        member bs.NewEntry_OppBW (rowId:int) (bw:float) = oppBW.[rowId].Value <- bw
        member bs.NewEntry_OppT (rowId:int) (t:float) = oppT.[rowId].Value <- t

        member bs.NewEntry_Opp3 (rowId:int) (tPut:float) (bWidth:float) (time:float) =
            bs.NewEntry_OppTP rowId tPut
            bs.NewEntry_OppBW rowId bWidth
            bs.NewEntry_OppT rowId time

        member bs.ThroughputEntryPair (elementCount:int) =
            let idx = sourceCounts |> List.findIndex (fun ns -> ns = elementCount)
            ((bs.MyThroughput).[idx]).Value,
            ((bs.OpponentThroughput).[idx]).Value
        
        
        member bs.BandwidthEntryPair (elementCount:int) =
            let idx = sourceCounts |> List.findIndex (fun ns -> ns = elementCount)
            ((bs.MyBandwidth).[idx]).Value,
            ((bs.OpponentBandwidth).[idx]).Value
   
        member bs.TimingEntryPair (elementCount:int) = 
            let idx = sourceCounts |> List.findIndex (fun ns -> ns = elementCount)
            ((bs.MyTiming).[idx]).Value,
            ((bs.OpponentTiming).[idx]).Value
            

        member bs.CompareThroughput (elementCount:int) =
            let pair = bs.ThroughputEntryPair elementCount
            let mytp, otp = (fst pair),(snd pair)
            let pd = percentDifference mytp otp
            printfn "Throughput @ %d:\t %s: %7.3f\t Alea.cuBase: %7.3f\t Diff: %5.2f percent" elementCount opponent otp mytp pd


        member bs.CompareBandwidth (elementCount:int) =
            let pair = bs.BandwidthEntryPair elementCount
            let mytp, otp = (fst pair),(snd pair)
            let pd = percentDifference mytp otp
            printfn "Bandwidth @ %d:\t %s: %7.3f\t Alea.cuBase: %7.3f\t Diff: %5.2f percent" elementCount opponent otp mytp pd

        member bs.CompareTiming (elementCount:int) =
           let pair = bs.TimingEntryPair elementCount
           let mytp, otp = (fst pair),(snd pair)
           let pd = percentDifference mytp otp
           printfn "Timing @ %d:\t %s: %7.3f\t Alea.cuBase: %7.3f\t Diff: %5.2f percent" elementCount opponent otp mytp pd

        member bs.ShowKernelsUsed =
            printfn "Kernels used by the %s algorithm: %A" bs.AlgorithmName bs.KernelsUsed


    type BenchmarkStats4 =
        val mutable Int32s : BenchmarkStats
        val mutable Int64s : BenchmarkStats
        val mutable Float32s : BenchmarkStats
        val mutable Float64s : BenchmarkStats
    
        new(algorithmName : string, kernelsUsed : string[], deviceName : string, opponent : string, sourceCounts : int list, nIterations: int list) =
            { Int32s = new BenchmarkStats(algorithmName, kernelsUsed, deviceName, "Int32", opponent, sourceCounts, nIterations);
              Int64s = new BenchmarkStats(algorithmName, kernelsUsed, deviceName, "Int64", opponent, sourceCounts, nIterations);
              Float32s = new BenchmarkStats(algorithmName, kernelsUsed, deviceName, "Float32", opponent, sourceCounts, nIterations);
              Float64s = new BenchmarkStats(algorithmName, kernelsUsed, deviceName, "Float64", opponent, sourceCounts, nIterations) }


    let getFilledBMS4Object 
            (algName:string) 
            (kernelsUsed:string[]) 
            (deviceName:string)
            (opponentName:string)
            (sourceCounts: list<int>)
            (nIterations: list<int>)
            (fourTypeStats: list<list<float * float>>) =
        let bms4obj = new BenchmarkStats4(algName, kernelsUsed, deviceName, opponentName, sourceCounts, nIterations)
        let i32stats, i64stats, f32stats, f64stats = fourTypeStats.[0],fourTypeStats.[1],fourTypeStats.[2],fourTypeStats.[3]
        i32stats |> List.iteri (fun i (tp, bw) ->
            bms4obj.Int32s.OpponentThroughput.[i].Value <- tp
            bms4obj.Int32s.OpponentBandwidth.[i].Value <- bw )
        i64stats |> List.iteri (fun i (tp, bw) ->
            bms4obj.Int64s.OpponentThroughput.[i].Value <- tp
            bms4obj.Int64s.OpponentBandwidth.[i].Value <- bw )
        f32stats |> List.iteri (fun i (tp, bw) ->
            bms4obj.Float32s.OpponentThroughput.[i].Value <- tp
            bms4obj.Float32s.OpponentBandwidth.[i].Value <- bw )
        f64stats |> List.iteri (fun i (tp, bw) ->
            bms4obj.Float64s.OpponentThroughput.[i].Value <- tp
            bms4obj.Float64s.OpponentBandwidth.[i].Value <- bw )

        bms4obj


module CodeMetrics = 
    let dirPath = "X:\dev\GitHub\\Alea.cuExtension"
    let outputFileName = dirPath + "\CodeMetricsOutput.txt"
    let save = true
    let print = true

    let lines(file) =
        use reader = File.OpenText(file)
        let lineSequence(file) =
        
            Seq.unfold(fun line ->
                if line = null then 
                    reader.Close()
                    None
                else
                    Some(line, reader.ReadLine())) (reader.ReadLine())

        let calculateStats(lines:seq<string>) =
            lines |> Seq.fold(
                            fun (lineCount, typ, dfault, err) line -> match line with
                                                                       | _ when line.Contains("type") -> (lineCount+1, typ+1, dfault, err)
                                                                       | _ when line.Contains("default") -> (lineCount+1, typ, dfault+1, err)
                                                                       | _ when line.Contains("*ERROR*") -> (lineCount+1, typ, dfault, err+1)
                                                                       | _ -> (lineCount+1, typ, dfault, err)) (0,0,0,0)
        calculateStats(lineSequence(file))

    let rec visitor dir filter =
        seq{ yield! Directory.GetFiles(dir, filter)
             for subdir in Directory.GetDirectories(dir) do yield! visitor subdir filter}

    let ArrayOfAllFiles =
        visitor dirPath "*"
        |> Array.ofSeq

    let ArrayOf_fs_Files =
        visitor dirPath "*.fs"
        |> Array.ofSeq

    let ArrayOf_fsx_Files =
        visitor dirPath "*.fsx"
        |> Array.ofSeq

    let LineCount arrayOfFilePaths =
        arrayOfFilePaths
        |> Array.map (fun x -> lines(x))
        |> Array.map (fun (x,_,_,_) -> x)
        |> Array.sum

    let totalLineCount_fs = 
        ArrayOf_fs_Files |> LineCount
    
    let totalLineCount_fsx = 
        ArrayOf_fsx_Files |> LineCount

    let totalLineCount =
        totalLineCount_fs + totalLineCount_fsx
    
    let typeCount =
        ArrayOf_fs_Files 
        |> Array.map (fun x -> lines(x))
        |> Array.map (fun (_,x,_,_) -> x)
        |> Array.sum

    let stats = 
        ["************** Code Metrics *******************"] @
        [sprintf "Total Line Count:\t\t%d" totalLineCount] @
        [sprintf "Line Count of .fs files:\t%d" totalLineCount_fs] @
        [sprintf "Line Count of .fsx files:\t%d" totalLineCount_fsx] @
        [sprintf "Total Type Count:\t\t%d" typeCount]

    let printStats = (List.map (fun (x:string) -> Console.WriteLine(x)) stats) |> ignore
    let saveStats = File.WriteAllLines(outputFileName, Array.ofList stats)

    if print then do
        printStats
    else if save then do
        saveStats