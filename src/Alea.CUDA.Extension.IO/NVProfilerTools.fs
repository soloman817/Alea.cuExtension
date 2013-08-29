module Alea.CUDA.Extension.IO.NVProfilerTools

open System
open System.IO
open System.Diagnostics
open System.Text
open System.Text.RegularExpressions
open FileHelpers


let kNameRegex = Regex(@"(Kernel[\w]+)")
let typeNames = ["Int32"; "Int64"; "Float32"; "Float64"]


type CollectorType =
    | MgpuBenchmark

type NVProfDataGenerator =
    val workingDir : string
    val outfilePath : string
    new (wd, ofp) = {   workingDir = wd; 
                        outfilePath = ofp }

    member nvp.Execute (nvprofArgs:string) (programName:string) (programArgs:string) =
        let nvprof = new Process()   
        let nvprofInfo = new ProcessStartInfo("nvprof", nvprofArgs + programName + programArgs)
        nvprofInfo.WorkingDirectory <- nvp.workingDir
        nvprofInfo.UseShellExecute <- false
        nvprofInfo.RedirectStandardOutput <- true
        nvprofInfo.CreateNoWindow <- true
        nvprof.StartInfo <- nvprofInfo
        nvprof.Start() |> ignore
        
        let reader = nvprof.StandardOutput
        let ao = reader.ReadToEnd()
        
        nvprof.WaitForExit()
        nvprof.Close()
        
        File.AppendAllText(nvp.outfilePath, ao)


type KernelNameConverter() =
    inherit ConverterBase()    
    override snc.StringToField(from) =
            let m = kNameRegex.Match(from)
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

// TODO    
//[< DelimitedRecord(",") >] [< IgnoreEmptyLines >]
//type ProfiledKernelLaunch_apiTrace () =

[<ConditionalRecord(RecordCondition.IncludeIfMatchRegex, @"(Kernel[\w]+)")>]
[< DelimitedRecord(",") >] [< IgnoreEmptyLines >]
type ProfiledKernelLaunch_gpuTrace () =
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
    [<DefaultValue>] [<FieldQuoted()>] [<FieldConverter(typeof<KernelNameConverter>)>] val mutable Name : string


type NVProfSummaryDataCollector(csvFile:string, nkernels, sourceCounts:int list, nIterations:int list) =
        let engine = new FileHelperEngine(typeof<ProfiledKernelLaunch_summary>)
        let res = engine.ReadFile(csvFile)
        let downcast_PKL_Array = Array.map (fun (a:obj) -> a :?> ProfiledKernelLaunch_summary)
        let launches = downcast_PKL_Array res
        
        let mutable kNames = Array.init nkernels (fun _ -> "")       
        
        let getKernelNames() =
            let mutable i = 0
            let mutable finished = false
            let mutable knames = []
            while not finished do
                let kn = launches.[i].Name
                knames <- knames @ [kn]
                if knames.Length < nkernels then i <- i + 1 else finished <- true
            knames |> List.iteri (fun i x -> Array.set kNames i x)

        member nvp.GetAverageKernelLaunchTimings() =
            let result = Array.init nkernels (fun _ -> ("", Array.zeroCreate<float> nIterations.Length))
            launches |> Seq.ofArray 
            |> Seq.groupBy (fun x -> x.Name) 
            |> List.ofSeq 
            |> List.iteri (fun i x ->
                let e = (snd x) |> Array.ofSeq |> Array.map (fun x -> x.Avg)
                Array.set result i ((fst x), e))
            result


type NVProfGPUTraceDataCollector(csvFile:string, sourceCounts:int list, nIterations:int list) =
        let engine = new FileHelperEngine(typeof<ProfiledKernelLaunch_gpuTrace>)
        
        let gatherData (ignoreFirstLines:int) (ignoreLastLines:int) =            
            engine.Options.IgnoreFirstLines <- ignoreFirstLines + 6 // ignores nvprof misc output
            engine.Options.IgnoreLastLines <- ignoreLastLines
            let res = engine.ReadFile(csvFile)
            let downcast_PKL_Array = Array.map (fun (a:obj) -> a :?> ProfiledKernelLaunch_gpuTrace)
            (downcast_PKL_Array res)
                
//        let gatherDurations =
//            gatherData 0 0 |> Array.map (fun x -> x.Duration)

        let getDurationsSeconds (launches:ProfiledKernelLaunch_gpuTrace[]) =
            launches |> Array.map (fun x -> x.Duration)       
        
        let getDurationsMicro (launches:ProfiledKernelLaunch_gpuTrace[]) =
            launches |> Array.map (fun x -> x.Duration * 1000000.0)
        
        let getDurationsMilli (launches:ProfiledKernelLaunch_gpuTrace[]) =
            launches |> Array.map (fun x -> x.Duration * 1000.0)   

        let durations tu l = 
            match tu with
            | "us" -> getDurationsMicro(l)
            | "ms" -> getDurationsMilli(l)
            | "s"  -> getDurationsSeconds(l)
            | _ -> getDurationsMicro(l)

        member nvp.GetAverageKernelLaunchTimings (numberAlgs:int) (kernelsPerAlg:int) (typesPerAlg:int) (timeUnit:string) =
            let launchesPerType = (nIterations |> List.sum) * kernelsPerAlg
            let launchesPerAlg = (nIterations |> List.sum) * kernelsPerAlg * typesPerAlg
            let na, nkpa, ntpa, nlpt, nlpa = numberAlgs, kernelsPerAlg, typesPerAlg, launchesPerType, launchesPerAlg
            let N = na * nlpa
            printfn "N == %d" N
            let allLaunches = gatherData 0 0
            
            
            ([| for a in 0..(na - 1) do
                let algResult = ([| for t in 0..(ntpa - 1) do
                                        let launches = Array.sub allLaunches (a*nlpa + t*nlpt) nlpt
                                        let typeName = typeNames.[t]
                                        let knames = [|for i in 0..(nkpa - 1) do yield (launches.[i].Name + "<" + typeName + ">")|]
                                        let durations = durations timeUnit launches
                                        let nis = List.scan (fun x e -> x + e) 0 nIterations                   
                    
                                        let averages = 
                                            (nIterations |> List.mapi (fun i ni ->
                                                let iD = Array.sub durations (nis.[i]*nkpa) (ni*nkpa)
                                                let kD = [| for a in 1..nkpa..iD.Length do yield iD.[(a - 1)..(a - 1)+(nkpa-1)] |]                                                
                                                [| for i in 0..(nkpa - 1) do
                                                    yield [| for j in 0..(kD.Length - 1) do
                                                                yield kD.[j].[i] |] 
                                                    |> Array.average |]) )
                                        if (a = na - 1) && (t = ntpa - 1) then do printfn "%A" averages
                                        let result = [| for i in 0..(nkpa - 1) do
                                                            yield [| for j in 0..(nIterations.Length - 1) do
                                                                        yield averages.[j].[i] |] 
                                                        |]
                                        yield (knames, result) |])
                yield algResult
                |])

        member nvp.DisplayAverageKernelLaunchTimings (numberAlgs:int) (kernelsPerAlg:int) (typesPerAlg:int) (timeUnit:string) =
            let klt = nvp.GetAverageKernelLaunchTimings numberAlgs kernelsPerAlg typesPerAlg timeUnit            
            klt |> Array.iter (fun x ->
                x |> Array.iter (fun y ->
                    let knames, results = y
                    knames |> Array.iteri (fun i kn ->        
                    printfn "Average Kernel Launch Times<%s> for %s" timeUnit kn
                    let avgs = results.[i] |> List.ofArray
                    (sourceCounts, avgs) ||> List.iter2 (fun n avgdur -> printfn "%d\t\t%9.3f" n avgdur)
                    printfn "\n" ))
                    )
            