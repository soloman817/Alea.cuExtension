open System
open System.IO
open System.Text.RegularExpressions

let path = @"X:\dev\GitHub\moderngpu\Release\benchmarkinsert.exe"


//let exec program =
//    let pro = System.Diagnostics.ProcessStartInfo(program)
//    let p = System.Diagnostics.Process.Start(pro)
//    p.WaitForExit()
//    p.ExitCode
//
//exec path
let args = "--csv benchmarkinsert.exe"

let exec =
    let prog = System.Diagnostics.ProcessStartInfo("nvprof")
    prog.WorkingDirectory <- @"X:\dev\GitHub\moderngpu\Release\"
    //use mutable outfile = if not (File.Exists(prog.WorkingDirectory + "out.txt")) then File.Create(prog.WorkingDirectory + "out.txt")
    
    //use outwriter = new StreamWriter(outfile)
    prog.UseShellExecute <- false
    prog.RedirectStandardOutput <- true
    prog.Arguments <- args
    let p = System.Diagnostics.Process.Start(prog)
    p.WaitForExit()
    //let derp = postream.ReadToEnd()
    //printfn "%A" derp
    let all = ((p.StandardOutput).ReadToEnd())
    let tokens = Regex(@"[\n\r]+").Split(all)
    let mutable out = []
    tokens |> Array.iter (fun x -> if x.Contains("Kernel") then 
                                    printfn "%A" x
                                    let tok = Regex(",").Split(x)
                                    let m = Regex(@"(Kernel[\w]+)").Match(tok.[6])
                                    if m.Success then
                                        let kn = m.Groups.[1].Captures.[0].Value
                                        let l = seq { yield kn + "," + tok.[1] }
                                        File.AppendAllLines(prog.WorkingDirectory + "out.txt", l) )
    p.ExitCode
exec