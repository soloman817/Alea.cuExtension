module NVProfilerTools.DataGenerators

open System
open System.Diagnostics
open System.IO

type DataGenerator =
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

