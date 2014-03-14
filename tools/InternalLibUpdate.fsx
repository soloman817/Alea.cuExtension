open System
open System.Diagnostics
open System.IO
//these are to go in the Alea.cuExtension/lib
let gitexe          = @"C:\Program Files (x86)\Git\bin\git.exe"
let workingdir      = @"X:\repo\git\Alea.cuExtension\lib"
let output          = @"gitScriptOut.txt"

let cub_url         = @"clone https://github.com/NVlabs/cub.git"
let mgpu_url        = @"clone https://github.com/NVlabs/moderngpu.git"
let fsplot_url      = @"clone https://github.com/TahaHachana/FsPlot.git"
let nvproftools_url = @"clone https://github.com/acbrewbaker/NVProfilerTools.git"
let cudalab_url     = @"clone https://github.com/soloman817/CUDALab.git"

let urlList = [cub_url; mgpu_url; fsplot_url; nvproftools_url; cudalab_url]

let gitUpdate() =
    let git = new Process()
    for url in urlList do
        let gitInfo = new ProcessStartInfo(gitexe, url)
        gitInfo.WorkingDirectory <- workingdir
        gitInfo.UseShellExecute <- false
        gitInfo.RedirectStandardOutput <- true
        gitInfo.CreateNoWindow <- true
        git.StartInfo <- gitInfo
        git.Start() |> ignore

        let reader = git.StandardOutput
        let ao = reader.ReadToEnd()

        git.WaitForExit()
        git.Close()

        File.AppendAllText(output, ao)


gitUpdate()

