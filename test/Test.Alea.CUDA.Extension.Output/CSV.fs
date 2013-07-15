module Test.Alea.CUDA.Extension.Output.CSV

open System.IO
open Microsoft.Office.Interop.Excel
open Test.Alea.CUDA.Extension.Output.Util


let defaultOutputPath = "."



type IHeaders = 
    abstract member HList : List<string> with get, set


//type ICSVOutput =
//    abstract MainDirectory : (Directory -> unit)
//    //abstract SubDirectories : (List<Directory> -> unit)
//    //abstract CurrentOutputDirectory : (Directory -> unit) with get, set
//    //abstract CurrentOutputFile : File 
//    abstract CSVStreamWriter : StreamWriter
//    abstract AppendEntry : (Entry -> File -> unit)
//    abstract AppendEntryPair : (Entry -> Entry -> File -> unit)

let writeHeading (bms:BenchmarkStats) (w:StreamWriter) =
    w.WriteLine(System.DateTime.Now.ToShortDateString())
    w.WriteLine("Algorithm:\t" + bms.AlgorithmName)
    w.WriteLine("Tested Type:\t" + bms.TestedType)
    w.WriteLine("Device Used:\t" + bms.DeviceName)
    w.WriteLine("")

let benchmarkCSVOutput (bms:BenchmarkStats) (outPath:string) =
    let fname = bms.AlgorithmName + "_" + bms.TestedType + ".txt"
    let csvFStream = File.Create((outPath + fname))
    let csvWriter = new StreamWriter(csvFStream)
    writeHeading bms csvWriter
    
    let tpHeaders = "# Iterations,# Elements,Alea.cuBase," + bms.Opponent
    let bwHeaders = tpHeaders

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

    csvWriter.Close()
    csvFStream.Close()


let benchmarkCSVOutput4 (bms4:BenchmarkStats4) =
    let mainDir = Directory.CreateDirectory("../../BenchmarkOutput_CSV")
    while not mainDir.Exists do
    let algName = bms4.Ints.AlgorithmName
    let workingDir = Directory.CreateDirectory(mainDir.ToString() + "/" + algName)
    while not mainDir.Exists do
    let workingPath = workingDir.ToString()
    benchmarkCSVOutput bms4.Ints workingPath
    benchmarkCSVOutput bms4.Int64s workingPath
    benchmarkCSVOutput bms4.Float32s workingPath
    benchmarkCSVOutput bms4.Floats workingPath
