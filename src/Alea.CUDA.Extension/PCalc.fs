module Alea.CUDA.Extension.PCalc

open System
open System.Collections.Generic
open Alea.CUDA

let run (calc:PCalc<'T>) =
    let diagnoser = PCalcDiagnoser.None
    let sp = { Diagnoser = diagnoser; TimingLoggers = None; KernelTimingCollector = None }
    let s0 = PCalcState(sp)
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

    let diagnoser = PCalcDiagnoser.None
    let ktc = Timing.TimingCollector()
    let sp = { Diagnoser = diagnoser; TimingLoggers = None; KernelTimingCollector = Some ktc }

    for i = 1 to n - 1 do
        let s0 = PCalcState(sp)
        let _, s1 = calc.Invoke(s0)
        s1.DisposeResources()

    let s0 = PCalcState(sp)
    let r, s1 = calc.Invoke(s0)
    s1.DisposeResources()

    r, ktc    
    
let runWithTimingLogger (calc:PCalc<'T>) =
    let diagnoser = PCalcDiagnoser.None
    let loggers = Dictionary<string, Timing.TimingLogger>(16)
    let sp = { Diagnoser = diagnoser; TimingLoggers = Some loggers; KernelTimingCollector = None }
    let s0 = PCalcState(sp)
    let logger = s0.TimingLogger("default")
    let r, s1 = calc.Invoke(s0)
    s1.DisposeResources()
    logger.Finish()
    loggers |> Seq.iter (fun pair -> pair.Value.Finish())
    r, loggers

let runWithDiagnoser (diagnoser:PCalcDiagnoser) (calc:PCalc<'T>) =
    let sp = { Diagnoser = diagnoser; TimingLoggers = None; KernelTimingCollector = None }
    let s0 = PCalcState(sp)
    let r, s1 = calc.Invoke(s0)
    s1.DisposeResources()
    r

let runInWorker (worker:DeviceWorker) (calc:PCalc<'T>) = worker.Eval(fun () -> run calc)
let runInWorkerWithTiming (worker:DeviceWorker) (n:int) (calc:PCalc<'T>) = worker.Eval(fun () -> runWithTiming n calc)
let runInWorkerWithKernelTiming (worker:DeviceWorker) (n:int) (calc:PCalc<'T>) = worker.Eval(fun () -> runWithKernelTiming n calc)
let runInWorkerWithTimingLogger (worker:DeviceWorker) (calc:PCalc<'T>) = worker.Eval(fun () -> runWithTimingLogger calc)
let runInWorkerWithDiagnoser (worker:DeviceWorker) (diagnoser:PCalcDiagnoser) (calc:PCalc<'T>) = worker.Eval(fun () -> runWithDiagnoser diagnoser calc)

let tlogger (name:string) = PCalc(fun s -> s.TimingLogger(name), s)

let action (f:LPHint -> unit) = PCalc(fun s -> s.AddAction(f); (), s)

let stream (stream:Stream) = PCalc(fun s -> s.LPHint <- { s.LPHint with Stream = stream }; (), s)
let streamWithHint (stream:Stream) (totalStreams:int) = PCalc(fun s -> s.LPHint <- { s.LPHint with Stream = stream; TotalStreams = Some totalStreams }; (), s)

