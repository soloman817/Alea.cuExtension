module Alea.CUDA.Extension.Timing

open System
open System.Numerics
open System.Diagnostics
open System.Text
open System.Collections.Generic
open Alea.CUDA

let tictoc (f:unit -> 'T) =
    let stopwatch = Stopwatch.StartNew()
    let result = f()
    stopwatch.Stop()
    let timing = stopwatch.Elapsed.TotalMilliseconds
    result, timing

type TimingCollector() =
    let timings = new Dictionary<string, List<float>>()

    member this.Add(name, timing) =
        match timings.ContainsKey(name) with
        | false -> timings.Add(name, List<float>())
        | _ -> ()
        timings.[name].Add(timing)

    member this.AverageTiming(kernel) =
        timings.[kernel] |> Seq.average

    member this.TotalMilliseconds =
        timings |> Seq.fold (fun s pair -> s + this.AverageTiming(pair.Key)) 0.0

    member this.Report =
        let parts = timings |> Seq.fold (fun s pair -> s + sprintf "[%s:%.6f ms], " pair.Key (this.AverageTiming(pair.Key))) ""
        parts + sprintf "[TOTAL:%.6f ms]" this.TotalMilliseconds

    member this.Reset() = timings.Clear()

let collectTiming (collector:TimingCollector) (prefix:string) (name:string) (timing:float) =
    let name = sprintf "%s.%s" prefix name
    collector.Add(name, timing)

type TimingCollectFunc = string -> float -> unit
let tcToDiag (tc:TimingCollectFunc) = fun (name:string) (stats:KernelExecutionStats) -> tc name stats.TimeSpan
