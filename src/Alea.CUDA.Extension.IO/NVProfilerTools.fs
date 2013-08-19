module Alea.CUDA.Extension.IO.NVProfilerTools

open System
open System.IO
open System.Diagnostics
open System.Text
open System.Text.RegularExpressions
open FileHelpers



type CollectorType =
    | MgpuBenchmark

type NVProfDataGenerator =
    val workingDir : string
    val outfilePath : string
    new (wd, ofp) = {   workingDir = wd; 
                        outfilePath = ofp }    
    
    
    
    member nvp.Execute (nvprofArgs:string) (programName:string) (programArgs:string) =
        let nvprof = new Process()
//        Args <- Args + " " + programName + " " + "0"
//        programArgs |> List.iter (fun arg -> Args <- Args + " " + arg)        
        let nvprofInfo = new ProcessStartInfo("nvprof", nvprofArgs + programName + programArgs)
        nvprofInfo.WorkingDirectory <- nvp.workingDir
        nvprofInfo.UseShellExecute <- false
        nvprofInfo.RedirectStandardOutput <- true
        nvprofInfo.CreateNoWindow <- true
        nvprof.StartInfo <- nvprofInfo
        nvprof.Start() |> ignore
        
        let reader = nvprof.StandardOutput
//        let mutable allOutput = new StringBuilder()
//        while not nvprof.HasExited do
//            allOutput <- allOutput.Append(reader.ReadToEnd())
        let ao = reader.ReadToEnd()
        
        nvprof.WaitForExit()
        nvprof.Close()
        //let ao = allOutput.ToString()
        //printfn "%A" ao
        File.AppendAllText(nvp.outfilePath, ao)
//        let tokens = Regex(@"[\n\r]+").Split(ao)
//        tokens |> Array.iter (fun x -> if x.Contains("Kernel") then
//                                        File.AppendAllLines(nvp.outfilePath, seq { yield x }) )
//        nvprof.WaitForInputIdle() |> ignore
//        nvprof.StandardInput.WriteLine()
        
        
        //Args <- DefaultArgs


type KernelNameConverter() =
    inherit ConverterBase()
    let kReg = Regex(@"(Kernel[\w]+)")
    override snc.StringToField(from) =
            let m = kReg.Match(from)
            let from = m.Groups.[1].Captures.[0].Value
            from :> obj
    override snc.FieldToString(fieldValue:obj) =
            fieldValue.ToString()


type SciNotationConverter() =
    inherit ConverterBase()
    let sciReg = Regex(@"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)")
    override snc.StringToField(from) =
            let m = sciReg.Match(from)
            let from = m.Groups.[1].Captures.[0].Value
            Convert.ToDouble(Double.Parse(from)) :> obj
    override snc.FieldToString(fieldValue:obj) =            
            fieldValue.ToString()
    
[< DelimitedRecord(",") >] [< IgnoreEmptyLines >] [<IgnoreFirst(6)>]
type ProfiledKernelLaunch_summary () =
    [<DefaultValue>] val mutable Time_percent : float
    [<DefaultValue>] [<FieldConverter(typeof<SciNotationConverter>)>] val mutable Time : float
    [<DefaultValue>] val mutable Calls : int
    [<DefaultValue>] [<FieldConverter(typeof<SciNotationConverter>)>] val mutable Avg : float
    [<DefaultValue>] [<FieldConverter(typeof<SciNotationConverter>)>] val mutable Min : float
    [<DefaultValue>] [<FieldConverter(typeof<SciNotationConverter>)>] val mutable Max : float
    [<DefaultValue>] [<FieldQuoted()>] [<FieldConverter(typeof<KernelNameConverter>)>] val mutable Name : string
    
//[< DelimitedRecord(",") >] [< IgnoreEmptyLines >]
//type ProfiledKernelLaunch_apiTrace () =

[< DelimitedRecord(",") >] [< IgnoreEmptyLines >] [<IgnoreFirst(6)>]
type ProfiledKernelLaunch_gpuTrace () =
    //[<FieldConverter(typeof<SciNotationConverter>)>]
    [<DefaultValue>] [<FieldConverter(typeof<SciNotationConverter>)>] val mutable Start : float
    [<DefaultValue>] [<FieldConverter(typeof<SciNotationConverter>)>] val mutable Duration : float
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable GridX : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable GridY : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable GridZ : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable BlockX : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable BlockY : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable BlockZ : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable RegPerThread : int
    [<DefaultValue>] [<FieldNullValue(typeof<float>, "0.0")>] val mutable StaticSMem : float
    [<DefaultValue>] [<FieldNullValue(typeof<float>, "0.0")>] val mutable DynamicSMem : float
    [<DefaultValue>] [<FieldNullValue(typeof<float>, "0.0")>] val mutable Size : float
    [<DefaultValue>] [<FieldNullValue(typeof<float>, "0.0")>] val mutable Throughput : float
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable Device : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable Contex : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable Stream : int
    [<DefaultValue>] [<FieldQuoted()>] val mutable Name : string


type NVProfSummaryDataCollector(csvFile:string, nKernels, sourceCounts:int list, nIterations:int list) =
        let engine = new FileHelperEngine(typeof<ProfiledKernelLaunch_summary>)
        let res = engine.ReadFile(csvFile)
        let downcast_PKL_Array = Array.map (fun (a:obj) -> a :?> ProfiledKernelLaunch_summary)
        let launches = downcast_PKL_Array res
        
        //do printfn "launches size: %A" launches.Length
        
        let mutable kNames = Array.init nKernels (fun _ -> "")       
        let kreg = @"(Kernel[\w]+)"
        
        //let mutable avgDurations = List.init kNames.Length (fun _ -> Array.zeroCreate<float> nIterations.Length)
        
        

        let getKernelNames() =
            let mutable i = 0
            let mutable finished = false
            let mutable knames = []
            while not finished do
                let kn = launches.[i].Name
                knames <- knames @ [kn]
                if knames.Length < nKernels then i <- i + 1 else finished <- true
            knames |> List.iteri (fun i x -> Array.set kNames i x)

        member nvp.GetAverageKernelLaunchTimings() =
            let result = Array.init nKernels (fun _ -> ("", Array.zeroCreate<float> nIterations.Length))
            launches |> Seq.ofArray 
            |> Seq.groupBy (fun x -> x.Name) 
            |> List.ofSeq 
            |> List.iteri (fun i x ->
                let e = (snd x) |> Array.ofSeq |> Array.map (fun x -> x.Avg)
                Array.set result i ((fst x), e))
            result


type NVProfGPUTraceDataCollector(csvFile:string, nKernels, sourceCounts:int list, nIterations:int list) =
        let engine = new FileHelperEngine(typeof<ProfiledKernelLaunch_gpuTrace>)
        let res = engine.ReadFile(csvFile)
        let downcast_PKL_Array = Array.map (fun (a:obj) -> a :?> ProfiledKernelLaunch_gpuTrace)
        let launches = downcast_PKL_Array res
        
        //do printfn "launches size: %A" launches.Length
        
        let mutable kNames = Array.init nKernels (fun _ -> "")       
        let kreg = @"(Kernel[\w]+)"
        
        //let mutable avgDurations = List.init kNames.Length (fun _ -> Array.zeroCreate<float> nIterations.Length)
        
        

        let getKernelNames() =
            let mutable i = 0
            let mutable finished = false
            let mutable knames = []
            while not finished do
                let kn = launches.[i].Name
                knames <- knames @ [kn]
                if knames.Length < nKernels then i <- i + 1 else finished <- true
            knames |> List.iteri (fun i x -> Array.set kNames i x)

        member nvp.GetAverageKernelLaunchTimings() =
            let result = Array.init nKernels (fun _ -> ("", Array.zeroCreate<float> nIterations.Length))
            let durationSums = Array.init (nKernels * nIterations.Length) (fun _ -> 0.0)
            let nis = List.scan (fun x e -> x + e) 0 nIterations
            (nIterations, nis) ||> List.iteri2 (fun i ni n ->
                let idx = if i > 0 then (i + nKernels) * n else 0
                Array.sub launches idx (n * nKernels) |> Array.iteri (fun j y ->
                    Array.set durationSums ((i % nKernels) + j) (durationSums.[(i % nKernels) + j] + y.Duration) ) )
            let durationAverages = 
                nIterations |> List.mapi (fun i x ->
                    Array.sub durationSums (i * nKernels) nKernels
                    |> Array.map (fun y -> y / (float x)) )
            
            result