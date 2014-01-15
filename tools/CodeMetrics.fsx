#I @"..\packages\FSPowerPack.Core.Community.3.0.0.0\Lib\Net40"

#r "FSharp.PowerPack.dll"
#r "FSharp.PowerPack.Metadata.dll"

open System
open System.IO
open System.Text
open Microsoft.FSharp.Metadata

type OutputFileType =
    | MGPU
    | CUB


type Options private () =
    static let instance = Options()
    let mutable solutionFolder : string option = @"..\" |> Some
    let mutable destinationFolder : string option = None
    //let mutable mgpuMetricsFile : string option = None
    //let mutable cubMetricsFile : string option = None


    member this.SolutionFolder
        with get() = match solutionFolder with Some folder -> folder | None -> failwith "Solution folder not set."
        and set folder = solutionFolder <- Some folder

    member this.DestinationFolder 
        with get() = match destinationFolder with Some folder -> folder | None -> failwith "Destination folder not set."
        and set folder = destinationFolder <- Some folder

//    member this.MGPUMetricsFile
//        with get() = match mgpuMetricsFile with Some folder -> folder | None -> failwith "Assembly folder not set."
//        and set folder = mgpuMetricsFile <- Some folder


    static member Instance = instance

let (@@) a b = Path.Combine(a, b)


let rootDir = Path.Combine("../", __SOURCE_DIRECTORY__)
let outputDir = Path.Combine(rootDir, "deploy")
let outputFilePath outputFileType = 
    match outputFileType with
    | OutputFileType.MGPU ->
        Path.Combine(outputDir, "\CodeMetrics_MGPU.txt")
    | OutputFileType.CUB ->
        Path.Combine(outputDir, "\CodeMetrics_CUB.txt")

let outputFile = outputFilePath OutputFileType.MGPU    

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
                                                                    // can change these later and use to count methods, especially with CUB
                                                                   | _ when line.Contains("type") -> (lineCount+1, typ+1, dfault, err)
                                                                   | _ when line.Contains("default") -> (lineCount+1, typ, dfault+1, err)
                                                                   | _ when line.Contains("*ERROR*") -> (lineCount+1, typ, dfault, err+1)
                                                                   | _ -> (lineCount+1, typ, dfault, err)) (0,0,0,0)

    calculateStats(lineSequence(file))


let rec visitor dir filter =
    seq{ yield! Directory.GetFiles(dir, filter)
         for subdir in Directory.GetDirectories(dir) do yield! visitor subdir filter}

let ArrayOfAllFiles =
    visitor rootDir "*"
    |> Array.ofSeq

let ArrayOf_source_Files =
    visitor rootDir "*.cpp"
    |> Array.ofSeq

let ArrayOf_header_Files =
    visitor rootDir "*.h"
    |> Array.ofSeq


let LineCount arrayOfFilePaths =
    arrayOfFilePaths
    |> Array.map (fun x -> lines(x))
    |> Array.map (fun (x,_,_,_) -> x)
    |> Array.sum

let totalLineCount_cpp = 
    ArrayOf_source_Files |> LineCount
    
let totalLineCount_h = 
    ArrayOf_header_Files |> LineCount

let totalLineCount =
    totalLineCount_cpp + totalLineCount_h
    
let typeCount =
    ArrayOf_source_Files 
    |> Array.map (fun x -> lines(x))
    |> Array.map (fun (_,x,_,_) -> x)
    |> Array.sum

let emptyFolder (folder:string) =
    try Directory.Delete(folder, true)
    with _ -> ()
    Directory.CreateDirectory(folder) |> ignore

let createDocument() = Options.Instance.SolutionFolder

let specs =
    [
        "-s", ArgType.String(fun folder -> Options.Instance.SolutionFolder <- folder), "set solution folder"
        "-d", ArgType.String(fun folder -> Options.Instance.DestinationFolder <- folder), "set destination folder"        
    ] |> List.map (fun (sh, ty, desc) -> ArgInfo(sh, ty, desc))

ArgParser.Parse(specs)

let build() =
    emptyFolder(Options.Instance.DestinationFolder)
    //File.Copy(Options.Instance.SolutionFolder @@ "")
    createDocument()

let stats = 
    ["************** Code Metrics *******************"] @
    [sprintf "Total Line Count:\t\t%d" totalLineCount] @
    [sprintf "Line Count of .fs files:\t%d" totalLineCount_cpp] @
    [sprintf "Line Count of .fsx files:\t%d" totalLineCount_h] @
    [sprintf "Total Type Count:\t\t%d" typeCount]

let printStats = (List.map (fun (x:string) -> Console.WriteLine(x)) stats) |> ignore
let saveStats = File.WriteAllLines(outputFile, Array.ofList stats)

if print then do
    printStats
else if save then do
    saveStats
