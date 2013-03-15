open System
open System.Collections.Generic
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.Matlab.Plot

let [<ReflectedDefinition>] pi = System.Math.PI
let [<ReflectedDefinition>] sigma1 = 0.04
let [<ReflectedDefinition>] sigma2 = 0.04
let [<ReflectedDefinition>] sigma3 = 0.04
let initialCondExpr =
    <@ fun t x y -> 1.0/3.0*exp (-((x-0.2)*(x-0.2) + (y-0.2)*(y-0.2))/(2.0*sigma1*sigma1)) / (sigma1*sigma1*2.0*pi) +
                    1.0/3.0*exp (-((x-0.8)*(x-0.8) + (y-0.8)*(y-0.8))/(2.0*sigma2*sigma2)) / (sigma2*sigma2*2.0*pi) +
                    1.0/3.0*exp (-((x-0.8)*(x-0.8) + (y-0.2)*(y-0.2))/(2.0*sigma3*sigma3)) / (sigma3*sigma3*2.0*pi) @>
let boundaryExpr = <@ fun t x y -> 0.0 @>
let sourceFunctionExpr = <@ fun t x y -> 0.0 @>

let worker = Engine.workers.DefaultWorker
let solve = worker.LoadPModule(Heat2dAdi.solve initialCondExpr boundaryExpr sourceFunctionExpr).Invoke
let calc k tstart tstop Lx Ly nx ny dt = pcalc {
    let! x, y, u = solve k tstart tstop Lx Ly nx ny dt
    let! u = u.Gather()
    return x, y, u }

type TimingMethod =
    | None
    | Timing
    | TimingLogger
    | KernelTiming

let heatdist tstop =
    let k = 1.0
    let tstart = 0.0
    let Lx = 1.0
    let Ly = 1.0
    let dt = 0.01

    let nx = 512
    let ny = 512

    let calc = calc k tstart tstop Lx Ly nx ny dt
    // change this for different timing
    let tm = TimingMethod.KernelTiming
    let x, y, u =
        printfn "tstop = %f" tstop
        match tm with
        | TimingMethod.None ->
            calc |> PCalc.run
        | TimingMethod.Timing ->
            let (x, y, u), timings = calc |> PCalc.runWithTiming 3
            printfn "timing = %.6f ms" (timings |> Array.average)
            x, y, u
        | TimingMethod.KernelTiming ->
            let (x, y, u), ktc = calc |> PCalc.runWithKernelTiming 1
            ktc.Dump()
            x, y, u
        | TimingMethod.TimingLogger ->
            let (x, y, u), loggers = calc |> PCalc.runWithTimingLogger
            loggers.["default"].DumpLogs()
            x, y, u

    x, y, u

let plotWithMatlab (results:float * float[] * float[] * float[]) =
    let tstop, x, y, u = results
    plotSurfaceOfArray x y u "x" "y" "heat" (sprintf "Heat 2d ADI t=%f" tstop) ([400.; 200.; 750.; 700.] |> Seq.ofList |> Some)

let results =
    [| 0.0; 0.005; 0.01; 0.02; 0.03; 0.04 |]
    //[| 0.01 |]
    |> Array.map (fun tstop -> let x, y, u = heatdist tstop in tstop, x, y, u)

results |> Array.iter plotWithMatlab

printf "Press Enter to quit..."
System.Console.ReadKey(true) |> ignore
