module Alea.CUDA.Extension.PCalc

open System
open System.Collections.Generic
open Alea.CUDA

let run (calc:PCalc<'T>) =
    let s0 = PCalcState.Create(PCalcDiagnoser.None)
    let r, s1 = calc.Invoke(s0)
    s1.DisposeResources()
    r

let runWithTiming (n:int) (calc:PCalc<'T>) =
    if n < 1 then failwith "n must >= 1"
    let timings = Array.zeroCreate<float> n
    for i = 1 to n - 1 do
        let _, timing = Timing.tictoc (fun () -> run calc)
        timings.[i] <- timing
    let r, timing = Timing.tictoc (fun () -> run calc)
    timings.[0] <- timing
    r, timings

let runWithKernelTiming (n:int) (calc:PCalc<'T>) =
    if n < 1 then failwith "n must >= 1"

    let tc = Timing.TimingCollector()
    let diagnoser = PCalcDiagnoser.None

    for i = 1 to n - 1 do
        let s0 = { PCalcState.Create(diagnoser) with KernelTimingCollector = Some(tc) }
        let _, s1 = calc.Invoke(s0)
        s1.DisposeResources()

    let s0 = { PCalcState.Create(diagnoser) with KernelTimingCollector = Some(tc) }
    let r, s1 = calc.Invoke(s0)
    s1.DisposeResources()

    r, tc    
    
let runWithTimingLogger (calc:PCalc<'T>) =
    let loggers = Dictionary<string, Timing.TimingLogger>(16)
    let s0 = { PCalcState.Create(PCalcDiagnoser.None) with TimingLoggers = Some(loggers) }
    let logger = s0.GetTimingLogger("default")
    let r, s1 = calc.Invoke(s0)
    s1.DisposeResources()
    logger.Finish()
    loggers |> Seq.iter (fun pair -> pair.Value.Finish())
    r, loggers

let runWithDiagnoser (diagnoser:PCalcDiagnoser) (calc:PCalc<'T>) =
    let s0 = PCalcState.Create(diagnoser)
    let r, s1 = calc.Invoke(s0)
    s1.DisposeResources()
    r

let runInWorker (worker:DeviceWorker) (calc:PCalc<'T>) = worker.Eval(fun () -> run calc)
let runInWorkerWithTiming (worker:DeviceWorker) (n:int) (calc:PCalc<'T>) = worker.Eval(fun () -> runWithTiming n calc)
let runInWorkerWithKernelTiming (worker:DeviceWorker) (n:int) (calc:PCalc<'T>) = worker.Eval(fun () -> runWithKernelTiming n calc)
let runInWorkerWithTimingLogger (worker:DeviceWorker) (calc:PCalc<'T>) = worker.Eval(fun () -> runWithTimingLogger calc)
let runInWorkerWithDiagnoser (worker:DeviceWorker) (diagnoser:PCalcDiagnoser) (calc:PCalc<'T>) = worker.Eval(fun () -> runWithDiagnoser diagnoser calc)

let tlogger (name:string) = PCalc(fun s -> s.GetTimingLogger(name), s)
let action (f:unit -> unit) = PCalc(fun s -> s.Actions.Add(f); (), s)

let lpmod () = PCalc(fun s -> (fun lp -> s.AddKernelDiagnoser lp), s)
let lpmods stream = PCalc(fun s -> (fun (lp:LaunchParam) -> lp.SetStream(stream) |> s.AddKernelDiagnoser), s)

