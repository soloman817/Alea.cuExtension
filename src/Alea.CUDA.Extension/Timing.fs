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

    member this.Add(kernel, timing) =
        match timings.ContainsKey(kernel) with
        | false -> timings.Add(kernel, List<float>())
        | _ -> ()
        timings.[kernel].Add(timing)

    member this.AverageTiming(kernel) =
        timings.[kernel] |> Seq.average

    member this.TotalMilliseconds =
        timings |> Seq.fold (fun s pair -> s + this.AverageTiming(pair.Key)) 0.0

    member this.Report =
        let parts = timings |> Seq.fold (fun s pair -> s + sprintf "[%s:%.6f ms], " pair.Key (this.AverageTiming(pair.Key))) ""
        parts + sprintf "[TOTAL:%.6f ms]" this.TotalMilliseconds

    member this.Reset() = timings.Clear()

    override this.ToString() = this.Report

type ITimingLogger =
    abstract Log : string -> unit
    abstract Split : unit -> unit
    abstract Finish : unit -> unit
    abstract Dump : unit -> unit

type TimingLogger(name:string) =
    let splitter = ""
    let timings = new List<string * float>()
    let watch = Stopwatch.StartNew()
    let mutable oldmsg = splitter

    member this.Log(newmsg:string) =
        watch.Stop()
        let timing = watch.Elapsed.TotalMilliseconds
        timings.Add(oldmsg, timing)
        oldmsg <- newmsg
        watch.Restart()

    member this.Split() = this.Log(splitter)

    member this.Finish() =
        watch.Stop()
        let timing = watch.Elapsed.TotalMilliseconds
        timings.Add(oldmsg, timing)
        oldmsg <- splitter

    member this.Dump() =
        let timings = timings |> Array.ofSeq
        let maxMsgLength = timings |> Array.map (fun (msg, _) -> msg |> String.length) |> Array.max
        let spaces = timings |> Array.map (fun (msg, _) -> maxMsgLength - (String.length msg) + 2)
        let total = timings |> Array.fold (fun total (_, timing) -> total + timing) 0.0
        let percentages = timings |> Array.map (fun (_, timing) -> timing / total)
        let title = sprintf "***** %s *****" name

        printfn "%s" title
        for i = 0 to timings.Length - 1 do
            printf "%s" (timings.[i] |> fst)
            for j = 1 to spaces.[i] do printf " "
            printfn "%15.3f ms %6.2f%%" (timings.[i] |> snd) (percentages.[i] * 100.0)
        for i = 1 to (String.length title) do printf "*"
        printfn ""
        printf "TOTAL"
        for i = 1 to maxMsgLength - 5 + 2 do printf " "
        printfn "%15.3f ms %6.2f%%" total ((percentages |> Array.sum) * 100.0)

    interface ITimingLogger with
        member this.Log(msg) = this.Log(msg)
        member this.Split() = this.Split()
        member this.Finish() = this.Finish()
        member this.Dump() = this.Dump()

