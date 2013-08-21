open System
open System.IO
open System.Text.RegularExpressions
//
//let path = @"X:\dev\GitHub\moderngpu\Release\benchmarkinsert.exe"
//
//
////let exec program =
////    let pro = System.Diagnostics.ProcessStartInfo(program)
////    let p = System.Diagnostics.Process.Start(pro)
////    p.WaitForExit()
////    p.ExitCode
////
////exec path
//let args = "--csv benchmarkinsert.exe"
//
//let exec =
//    let prog = System.Diagnostics.ProcessStartInfo("nvprof")
//    prog.WorkingDirectory <- @"X:\dev\GitHub\moderngpu\Release\"
//    //use mutable outfile = if not (File.Exists(prog.WorkingDirectory + "out.txt")) then File.Create(prog.WorkingDirectory + "out.txt")
//    
//    //use outwriter = new StreamWriter(outfile)
//    prog.UseShellExecute <- false
//    prog.RedirectStandardOutput <- true
//    prog.Arguments <- args
//    let p = System.Diagnostics.Process.Start(prog)
//    p.WaitForExit()
//    //let derp = postream.ReadToEnd()
//    //printfn "%A" derp
//    let all = ((p.StandardOutput).ReadToEnd())
//    let tokens = Regex(@"[\n\r]+").Split(all)
//    let mutable out = []
//    tokens |> Array.iter (fun x -> if x.Contains("Kernel") then 
//                                    printfn "%A" x
//                                    let tok = Regex(",").Split(x)
//                                    let m = Regex(@"(Kernel[\w]+)").Match(tok.[6])
//                                    if m.Success then
//                                        let kn = m.Groups.[1].Captures.[0].Value
//                                        let l = seq { yield kn + "," + tok.[1] }
//                                        File.AppendAllLines(prog.WorkingDirectory + "out.txt", l) )
//    p.ExitCode
//exec
let rng = System.Random()
let nkernels = 2
let nIterations = [ 5; 3] //; 3; 1; 1]
let nlaunches = nIterations |> List.map (fun x -> x * nkernels) |> List.sum
let kernel1 = [| 0.3;  0.4;  0.9;  0.9;  0.5;  0.4;  0.6;  0.7 |] //;  0.5;  0.8;  0.3;  0.5;  0.6 |] //Array.init (nlaunches / nkernels) (fun _ -> rng.NextDouble())
let kernel2 = [| 0.23; 0.14; 0.19; 0.19; 0.25; 0.14; 0.16; 0.27 |] // 0.25; 0.18; 0.23; 0.15; 0.16 |] //Array.init (nlaunches / nkernels) (fun _ -> rng.NextDouble() * 0.2)
let kernels = [kernel1; kernel2]
printfn "kernel1 launch1 sum = %A" (kernel1.[0..4] |> Array.sum)
printfn "kernel2 launch1 sum = %A" (kernel2.[0..4] |> Array.sum)
let launches = 
    seq { let x = ref 0
          while !x < nlaunches do
            yield kernel1.[(!x / nkernels)]
            yield kernel2.[(!x / nkernels)]
            x := !x + nkernels
            } |> Array.ofSeq

let answerk1 =
    let averages = Array.zeroCreate<float> nIterations.Length
    let xs = List.scan (fun x e -> x + e) 0 nIterations
    let x = ref 0
    while !x < nIterations.Length do
        let sum = Array.sub kernel1 xs.[!x] nIterations.[!x] |> Array.sum        
        let derp = (/) (float nIterations.[!x])
        let avg = Array.sub kernel1 xs.[!x] nIterations.[!x] |> Array.sum |> (*) (1.0 / (float nIterations.[!x]))        
        Array.set averages !x avg
        x := !x + 1
    averages
//printfn "%A" answerk1

let answerk2 =
    let averages = Array.zeroCreate<float> nIterations.Length
    let xs = List.scan (fun x e -> x + e) 0 nIterations
    let x = ref 0
    while !x < nIterations.Length do        
        let avg = Array.sub kernel2 xs.[!x] nIterations.[!x] |> Array.sum |> (*) (1.0 / (float nIterations.[!x]))
        Array.set averages !x avg
        x := !x + 1
    averages
//printfn "%A" answerk2


let answer = [answerk1; answerk2]
()
//printfn "ANSWER = %A" answer




let result = Array.init nkernels (fun _ -> ("", Array.zeroCreate<float> nIterations.Length))
let sums = Array.init (nkernels * nIterations.Length) (fun _ -> 0.0) |> Seq.ofArray
let nis = List.scan (fun x e -> x + e) 0 nIterations
printfn "NIS = %A" nis
let averages =
    nIterations |> List.mapi (fun i ni ->
        let ilaunches = launches |> Array.sub <|| (nis.[i]*nkernels, ni*nkernels)
        //printfn "ilaunches = %A" ilaunches
        //for l in ilaunches do printfn "(%d) %A" i l
        //printfn "%A" [| 0..nkernels..(ilaunches.Length - 1) |]
        let kernels = [| for a in 1..nkernels..ilaunches.Length do yield ilaunches.[(a - 1)..(a - 1)+(nkernels-1)] |]
        printfn "kernels %A" kernels
        [| for i in 0..(nkernels - 1) do
            yield [| for j in 0..(kernels.Length - 1) do
                        yield kernels.[j].[i] |] 
            |> Array.average |]
        
        //(sums |> Array.map (fun x -> x / (float ni)))
        //printfn "kernels = %A" avg
//        let sums = seq { for idx in 0..(nkernels - 1) do
//                            let kd = seq { for k in kernels do yield Seq.head (Seq.skip idx k) }
//                            yield kd |> Seq.sum }
//        printfn "SUMS %A" sums
//        sums |> Seq.map (fun x -> (x / (float ni))) )
            )

printfn "averages = %A" averages
//
////printfn "%A" nis
//printfn "niter = %d, nis = %d" nIterations.Length nis.Length
//nIterations |> List.iteri (fun i ni ->
//    //let n = nis.[i]
//    //let ilaunches = List.init nkernels 
////    let ilaunches = 
////        seq { let x = ref n
////              while !x < (n * nkernels) do
////                yield launches.[!x]
//    //let idx = if i > 0 then (i + nkernels) * n else 0
//    Array.sub launches nis.[i] (ni * nkernels) |> List.ofArray |>
//        List.collect (fun x -> [for i in 1..nkernels ->  
//    Array.iteri (fun j y ->
//        Array.set sums ((i % nkernels) + i) (sums.[(i % nkernels) + i] + y) ) )

//
//printfn "AVERAGES == %A" averages