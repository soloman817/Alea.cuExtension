module Alea.CUDA.Extension.IO.CSV

open System
open System.IO
open System.Text.RegularExpressions
//open Alea.CUDA.Extension.IO.Util
open FileHelpers

let defaultOutputPath = "."

type IHeaders = 
    abstract member HList : List<string> with get, set


[<AutoOpen>]
module Input =    
    open Input
    let save = true
    let print = true
    

//    type ProfOutputParser() =
//        member pop.Go(csvFile) =
//            let engine = new FileHelperEngine(typeof<ProfiledKernelLaunch2>)
//            let res = engine.ReadFile(csvFile)
//            let downcast_PKL_Array = Array.map (fun (a:obj) -> a :?> ProfiledKernelLaunch)
//            let res_PKL = downcast_PKL_Array res
//            ()
//            //do printfn "%A" res_PKL
//
//
//    type NVProfOutputParser(csvFile:string, nKernels, sourceCounts:int list, nIterations:int list) =
//        let engine = new FileHelperEngine(typeof<ProfiledKernelLaunch2>)
//        let res = engine.ReadFile(csvFile)
//        let downcast_PKL_Array = Array.map (fun (a:obj) -> a :?> ProfiledKernelLaunch)
//        
//        let launches = downcast_PKL_Array res
//
//
//        let mutable kNames = Array.init nKernels (fun _ -> "")
//        
//        let mutable startedKernels = false
//        let kreg = @"(Kernel[\w]+)"
//        
//        let mutable durations = List.init kNames.Length (fun _ -> Array.zeroCreate<float> nIterations.Length)
//        
//        let getKernelNames() =
//            let mutable i = 0
//            let mutable finished = false
//            let mutable knames = []
//            while not finished do
//                let kn = launches.[i].Name
//                let m = Regex(kreg).Match(kn)
//                let kn = if m.Success then knames <- knames @ [m.Groups.[1].Captures.[0].Value]
//                if knames.Length < nKernels then i <- i + 1 else finished <- true
//            knames |> List.iteri (fun i x -> Array.set kNames i x)
//        //do printfn "got kernel names"
//        
//        let getDurations() =
//            let dur = Array.zeroCreate<float> launches.Length 
//            launches |> Array.iteri (fun i x -> Array.set dur i x.Duration)
//            dur
//        let allDurations = getDurations()
//
//        let sumDurations() =
//            let mutable nItrIndex = 0
//            let mutable kNameIndex = 0
//            
//            let mutable i = 0
//            let mutable itrGroup = nIterations.[nItrIndex]
//            let mutable numLaunches = 0
//
//            while i < launches.Length do
//                //printfn "i, nItrIndex, kNameIndex, itrGroup = %d, %d, %d, %d" i nItrIndex kNameIndex itrGroup
//                if launches.[i].Name.ToCharArray().[0] <> '[' then
//                    numLaunches <- numLaunches + 1
//                    durations.[kNameIndex].[nItrIndex] <- durations.[kNameIndex].[nItrIndex] + allDurations.[i]
//                    if kNameIndex < nKernels - 1 then kNameIndex <- kNameIndex + 1 else kNameIndex <- 0
//                
//                    if (i > itrGroup * nKernels) then 
//                        nItrIndex <- nItrIndex + 1
//                        itrGroup <- itrGroup + nIterations.[nItrIndex]
//                    
//                    i <- i + 1
//
//            if numLaunches = (nKernels * 2 * 2000) then
//                printfn "sum is correct"
//                
//
////                let dt,gx,gy,gz,kn = 
////                    float(tokens.[1]),   // duration
//////                    int(tokens.[2]), int(tokens.[3]), int(tokens.[4]), // grid x,y,z
//////                    (Regex(kreg).Match(tokens.[16])).Groups.[1].Captures.[0].Value // kernel name
////            lines |> Seq.mapi (fun i l ->
////                if
////                let tokens = Regex(",").Split(l)
////
////                l <- tokens.[1] + "," + tokens.[2] + "," + tokens.[3] + "," + tokens.[4]
//
//        member nvp.GetAverageKernelLaunchTimings() =
//            do getKernelNames()
//            do sumDurations()
//            durations |> List.iter (fun e ->
//                            e |> Array.iteri (fun i dur -> Array.set e i (dur / float nIterations.[i])) )
//            (kNames |> List.ofArray, durations) ||> List.zip
           



    

[<AutoOpen>]
module Output =
    open Output
    //type ICSVOutput =
    //    abstract MainDirectory : (Directory -> unit)
    //    //abstract SubDirectories : (List<Directory> -> unit)
    //    //abstract CurrentOutputDirectory : (Directory -> unit) with get, set
    //    //abstract CurrentOutputFile : File 
    //    abstract CSVStreamWriter : StreamWriter
    //    abstract AppendEntry : (Entry -> File -> unit)
    //    abstract AppendEntryPair : (Entry -> Entry -> File -> unit)


    //let writeHeading (bms:BenchmarkStats) (w:StreamWriter) =
    //    w.WriteLine(System.DateTime.Now.ToShortDateString())
    //    w.WriteLine("Algorithm:\t" + bms.AlgorithmName)
    //    w.WriteLine("Tested Type:\t" + bms.TestedType)
    //    w.WriteLine("Device Used:\t" + bms.DeviceName)
    //    w.WriteLine("")

    let benchmarkCSVOutput (bms:BenchmarkStats) (outPath:string) =
        let fname = bms.AlgorithmName + "_" + bms.TestedType + ".txt"
        let csvFStream = File.Create((outPath + fname))
        let csvWriter = new StreamWriter(csvFStream)
    
        let firstRow = "Algorithm,Tested Type,Device Used,Compared Against,"
        let mainHeaders = bms.AlgorithmName + "," + bms.TestedType + "," + bms.DeviceName + "," + bms.Opponent + ","
        //writeHeading bms csvWriter
        csvWriter.WriteLine(firstRow)
        csvWriter.WriteLine(mainHeaders)
        csvWriter.WriteLine("")

        let tpHeaders = "Iterations,Elements,Alea.cuBase," + bms.Opponent + ","
        let bwHeaders = tpHeaders
        let ktTitles = bms.KernelsUsed
        let ktHeaders = "Elements,Alea.cuBase," + bms.Opponent + "," + "Difference,"

        let counts, nItrs = bms.SourceCounts, bms.NumIterations
        let throughputCSV =
            let mutable tpcsv = []
            let mytp, otp = bms.MyThroughput, bms.OpponentThroughput
            for i = 0 to counts.Length - 1 do
                let ln = sprintf "%d,%d,%f,%f," nItrs.[i] counts.[i] mytp.[i].Value otp.[i].Value
                tpcsv <- tpcsv @ [ln]
            tpcsv

        let bandwidthCSV =
            let mutable bwcsv = []
            let mybw, obw = bms.MyBandwidth, bms.OpponentBandwidth
            for i = 0 to counts.Length - 1 do
                let ln = sprintf "%d,%d,%f,%f," nItrs.[i] counts.[i] mybw.[i].Value obw.[i].Value
                bwcsv <- bwcsv @ [ln]
            bwcsv

        let kernelTimingCSV = List.init counts.Length (fun i -> (sprintf "%d,%f,%f,%f," counts.[i] 0.0 0.0 0.0) )
    
        csvWriter.WriteLine("Throughput")
        csvWriter.WriteLine(tpHeaders)
        for i = 0 to counts.Length - 1 do
            csvWriter.WriteLine(throughputCSV.[i])
        csvWriter.WriteLine("")
    
        csvWriter.WriteLine("Bandwidth")
        csvWriter.WriteLine(bwHeaders)
        for i = 0 to counts.Length - 1 do
            csvWriter.WriteLine(bandwidthCSV.[i])
        csvWriter.WriteLine("")

        for i = 0 to ktTitles.Length - 1 do
            csvWriter.WriteLine(ktTitles.[i])
            csvWriter.WriteLine(ktHeaders)
            kernelTimingCSV |> List.iter (fun x -> csvWriter.WriteLine(x))
            csvWriter.WriteLine("")

        csvWriter.Close()
        csvFStream.Close()


    let benchmarkCSVOutput4 (bms4:BenchmarkStats4) =
        let mainDir = Directory.CreateDirectory("../../BenchmarkOutput_CSV")
        while not mainDir.Exists do
        let algName = bms4.Int32s.AlgorithmName
        let workingDir = Directory.CreateDirectory(mainDir.ToString() + "/" + algName)
        while not mainDir.Exists do
        let workingPath = workingDir.ToString()
        benchmarkCSVOutput bms4.Int32s workingPath
        benchmarkCSVOutput bms4.Int64s workingPath
        benchmarkCSVOutput bms4.Float32s workingPath
        benchmarkCSVOutput bms4.Float64s workingPath








//
//    type NVProfOutputParser(csvFile:string, nKernels, sourceCounts:int list, nIterations:int list) =        
//        let engine = new FileHelperEngine(typeof<ProfiledKernelLaunch2>)
//        let res = engine.ReadFile(csvFile)
//        let downcast_PKL_Array = Array.map (fun (a:obj) -> a :?> ProfiledKernelLaunch)
//        
//        let launches = downcast_PKL_Array res
//
//
//        let mutable kNames = Array.init nKernels (fun _ -> "")
//        
//        let mutable startedKernels = false
//        let kreg = @"(Kernel[\w]+)"
//        
//        let mutable durations = List.init kNames.Length (fun _ -> Array.zeroCreate<float> nIterations.Length)
//        
//        let getKernelNames() =
//            let mutable i = 0
//            let mutable finished = false
//            let mutable knames = []
//            while not finished do
//                let kn = launches.[i].Name
//                let m = Regex(kreg).Match(kn)
//                let kn = if m.Success then knames <- knames @ [m.Groups.[1].Captures.[0].Value]
//                if knames.Length < nKernels then i <- i + 1 else finished <- true
//            knames |> List.iteri (fun i x -> Array.set kNames i x)
//        //do printfn "got kernel names"
//        
//        let getDurations() =
//            let dur = Array.zeroCreate<float> launches.Length 
//            launches |> Array.iteri (fun i x -> Array.set dur i x.Duration)
//            dur
//        let allDurations = getDurations()
//
//        let sumDurations() =
//            let mutable nItrIndex = 0
//            let mutable kNameIndex = 0
//            
//            let mutable i = 0
//            let mutable itrGroup = nIterations.[nItrIndex]
//            let mutable numLaunches = 0
//
//            while i < launches.Length do
//                //printfn "i, nItrIndex, kNameIndex, itrGroup = %d, %d, %d, %d" i nItrIndex kNameIndex itrGroup
//                if launches.[i].Name.ToCharArray().[0] <> '[' then
//                    numLaunches <- numLaunches + 1
//                    durations.[kNameIndex].[nItrIndex] <- durations.[kNameIndex].[nItrIndex] + allDurations.[i]
//                    if kNameIndex < nKernels - 1 then kNameIndex <- kNameIndex + 1 else kNameIndex <- 0
//                
//                    if (i > itrGroup * nKernels) then 
//                        nItrIndex <- nItrIndex + 1
//                        itrGroup <- itrGroup + nIterations.[nItrIndex]
//                    
//                    i <- i + 1
//
//            if numLaunches = (nKernels * 2 * 2000) then
//                printfn "sum is correct"
//                
//
////                let dt,gx,gy,gz,kn = 
////                    float(tokens.[1]),   // duration
//////                    int(tokens.[2]), int(tokens.[3]), int(tokens.[4]), // grid x,y,z
//////                    (Regex(kreg).Match(tokens.[16])).Groups.[1].Captures.[0].Value // kernel name
////            lines |> Seq.mapi (fun i l ->
////                if
////                let tokens = Regex(",").Split(l)
////
////                l <- tokens.[1] + "," + tokens.[2] + "," + tokens.[3] + "," + tokens.[4]
//
//        member nvp.GetAverageKernelLaunchTimings() =
//            do getKernelNames()
//            do sumDurations()
//            durations |> List.iter (fun e ->
//                            e |> Array.iteri (fun i dur -> Array.set e i (dur / float nIterations.[i])) )
//            (kNames |> List.ofArray, durations) ||> List.zip




//        let getLines() = 
//            use reader = File.OpenText(csvFile)
//            let lines =
//                Seq.unfold(fun line ->
//                    if line = null then 
//                        reader.Close()
//                        None
//                    else
//                        Some(line, reader.ReadLine())) (reader.ReadLine()) 
//            //let lines = lines |> Seq.takeWhile (fun l -> (Regex(",").Split(l)).Length >= 16) |> Array.ofSeq
//            lines |> Array.ofSeq
//
//        let mutable lines = getLines()
//        do printfn "got lines"
//
//        let mutable firstKernelRow = 0
//
//        let findFirstKernelRow() =
//            let mutable i = 0            
//            let mutable finished = false
//            while not finished do
//                let l = lines.[i]
//                let tokens = Regex(",").Split(l)
//                if tokens.Length >= 16 then 
//                    let ca = tokens.[16].ToCharArray()
//                    if ca.[0] <> '[' then
//                        firstKernelRow <- i
//                        finished <- true
//            i <- i + 1
//        do findFirstKernelRow()
//        do printfn "get first kernel row"
//
//        let getKernelNames() =
//            let mutable i = firstKernelRow
//            let mutable finished = false
//            let mutable knames = []
//            while not finished do
//                let kn = (Regex(",").Split(lines.[i])).[16]
//                let m = Regex(kreg).Match(kn)
//                let kn = if m.Success then knames <- knames @ [m.Groups.[1].Captures.[0].Value]
//                if knames.Length < nKernels then i <- i + 1 else finished <- true
//            knames |> List.iteri (fun i x -> Array.set kNames i x)
//        //do printfn "got kernel names"
//
//        let getDurations() =
//            let dur = Array.zeroCreate<float> lines.Length 
//            lines |> Array.iteri (fun i l -> Array.set dur i (float((Regex(",").Split(l)).[1])))
//            dur
//        let allDurations = getDurations()
//        do printfn "got all durations"
//               
//        let sumDurations() =
//            let mutable nItrIndex = 0
//            let mutable kNameIndex = 0
//            let mutable currGridX = 0
//            let mutable currGridY = 0
//            let mutable currGridZ = 0
//            
//            let mutable i = firstKernelRow
//            let mutable itrGroup = nIterations.[nItrIndex]
//
//            while i < lines.Length do
//                //let tokens = Regex(",").Split(lines.[i])
//                //let dt = float(tokens.[1])
//                durations.[kNameIndex].[nItrIndex] <- durations.[kNameIndex].[nItrIndex] + allDurations.[i]
//                if kNameIndex < nKernels then kNameIndex <- kNameIndex + 1 else kNameIndex <- 0
//                
//                if (i >= itrGroup) then 
//                    nItrIndex <- nItrIndex + 1
//                    itrGroup <- nIterations.[nItrIndex]                    
//                i <- i + 1                
//
////                let dt,gx,gy,gz,kn = 
////                    float(tokens.[1]),   // duration
//////                    int(tokens.[2]), int(tokens.[3]), int(tokens.[4]), // grid x,y,z
//////                    (Regex(kreg).Match(tokens.[16])).Groups.[1].Captures.[0].Value // kernel name
////            lines |> Seq.mapi (fun i l ->
////                if
////                let tokens = Regex(",").Split(l)
////
////                l <- tokens.[1] + "," + tokens.[2] + "," + tokens.[3] + "," + tokens.[4]
//
//        member nvp.GetAverageKernelLaunchTimings() =
//            do getKernelNames()
//            do sumDurations()
//            durations |> List.iter (fun e ->
//                            e |> Array.iteri (fun i dur -> Array.set e i (dur / float nIterations.[i])) )
//            (kNames |> List.ofArray, durations) ||> List.zip