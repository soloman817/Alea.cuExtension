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

    member this.Dump() =
        let timings = timings |> Seq.map (fun pair -> pair.Key, pair.Value |> Seq.average, pair.Value.Count) |> Array.ofSeq
        let totalTiming = timings |> Seq.sumBy (fun (_, timing, _) -> timing)
        let totalCount = timings |> Seq.sumBy (fun (_, _, count) -> count)

        let maxMsgLength = timings |> Array.map (fun (msg, _, _) -> msg.Length) |> Array.max
        let space = 2
        let width = maxMsgLength + space + 15 + 3 + 6
        let mkmsg (msg:string) = sprintf "%s%s" msg (String.replicate (maxMsgLength - msg.Length + space) " ")

        printfn "%s" (String.replicate width "=")
        timings |> Array.iter (fun (msg, timing, count) -> printfn "%s%15.6f ms %5d" (mkmsg msg) timing count)
        printfn "%s" (String.replicate width "-")
        printfn "%s%15.6f ms %5d" (mkmsg "TOTAL") totalTiming totalCount
        printfn "%s" (String.replicate width "=")

    override this.ToString() = this.Report

type ITimingLogger =
    abstract Log : string -> unit
    abstract Touch : unit -> unit
    abstract Finish : unit -> unit

let dummyTimingLogger = 
    { new ITimingLogger with
        member this.Log(msg) = ()
        member this.Touch() = ()
        member this.Finish() = () }

type TimingLogger(name:string) =
    let timings = new List<string option * float>()
    let mutable msg : string option = None
    let mutable finished = false
    let watch = Stopwatch.StartNew()

    let stop() =
        watch.Stop()
        let timing = watch.Elapsed.TotalMilliseconds
        timings.Add(msg, timing)
        msg <- None

    let start(msg':string option) =
        msg <- msg'
        watch.Restart()

    member this.Log(msg:string) =
        if finished then failwith "logger is finished"
        stop()
        start(Some msg)

    member this.Touch() =
        if finished then failwith "logger is finished"
        stop()
        start(None)

    member this.Finish() =
        if not finished then
            stop()
            finished <- true

    member this.TotalMilliseconds = timings |> Seq.sumBy snd
    member this.TotalExplicitMilliseconds = timings |> Seq.filter (fun (msg, _) -> msg.IsSome) |> Seq.sumBy snd
    member this.TotalImplicitMilliseconds = timings |> Seq.filter (fun (msg, _) -> msg.IsNone) |> Seq.sumBy snd

    member this.Logs =
        let total = this.TotalMilliseconds
        let total' = this.TotalExplicitMilliseconds
        timings
        |> Seq.choose (fun (msg, timing) -> match msg with Some(msg) -> Some(msg, timing) | None -> None)
        |> Seq.map (fun (msg, timing) -> msg, timing, timing / total', timing / total)
        |> Array.ofSeq

    member this.AllLogs =
        let total = this.TotalMilliseconds
        timings
        |> Seq.map (fun (msg, timing) ->
            match msg with
            | Some(msg) -> msg, timing, timing / total, timing / total
            | None -> "", timing, timing / total, timing / total)
        |> Array.ofSeq

    member this.Dump(logs:(string * float * float * float)[]) =
        let totalMsg = "TOTAL"
        let totalTiming = this.TotalMilliseconds
        let totalPercentage = 1.0

        let totalExplicitMsg = "TOTAL EXPLICIT"
        let totalExplicitTiming = this.TotalExplicitMilliseconds
        let totalExplicitPercentage = totalExplicitTiming / totalTiming

        let totalImplicitMsg = "TOTAL IMPLICIT"
        let totalImplicitTiming = this.TotalImplicitMilliseconds
        let totalImplicitPercentage = totalImplicitTiming / totalTiming

        let total'Msg = "summary"
        let total'Timing = logs |> Array.map (fun (_, timing, _, _) -> timing) |> Array.sum
        let total'Percentage' = logs |> Array.map (fun (_, _, percentage, _) -> percentage) |> Array.sum
        let total'Percentage = logs |> Array.map (fun (_, _, _, percentage) -> percentage) |> Array.sum

        let maxMsgLength = logs |> Array.map (fun (msg, _, _, _) -> msg.Length) |> Array.max
        let maxMsgLength = max maxMsgLength totalMsg.Length
        let maxMsgLength = max maxMsgLength totalExplicitMsg.Length
        let maxMsgLength = max maxMsgLength totalImplicitMsg.Length
        let maxMsgLength = max maxMsgLength total'Msg.Length

        let width space = maxMsgLength + space + 15 + 4 + 6 + 1 + 1 + 6 + 1
        let space = if (width 1 - name.Length) % 2 = 0 then 1 else 2
        let width = width space
        let mkmsg (msg:string) = sprintf "%s%s" msg (String.replicate (maxMsgLength - msg.Length + space) " ")

        printfn "%s" (String.replicate width "=")
        printfn "%s%s" (String.replicate ((width - name.Length) / 2) " ") name
        printfn "%s" (String.replicate width "-")

        // print logs
        logs |> Array.iter (fun (msg, timing, percentage', percentage) ->
            printfn "%s%15.3f ms %6.2f%% %6.2f%%" (mkmsg msg) timing (percentage' * 100.0) (percentage * 100.0))
        printfn "%s" (String.replicate width "-")

        // local total (for verify)
        printfn "%s%15.3f ms %6.2f%% %6.2f%%" (mkmsg total'Msg) total'Timing (total'Percentage' * 100.0) (total'Percentage * 100.0)
        printfn "%s" (String.replicate width "-")

        // total of explict and implicit
        printfn "%s%15.3f ms         %6.2f%%" (mkmsg totalExplicitMsg) totalExplicitTiming (totalExplicitPercentage * 100.0)
        printfn "%s%15.3f ms         %6.2f%%" (mkmsg totalImplicitMsg) totalImplicitTiming (totalImplicitPercentage * 100.0)
        printfn "%s" (String.replicate width "-")

        // print total
        printfn "%s%15.3f ms         %6.2f%%" (mkmsg totalMsg) totalTiming (totalPercentage * 100.0)
        printfn "%s" (String.replicate width "=")

    member this.DumpLogs() = this.Dump(this.Logs)
    member this.DumpAllLogs() = this.Dump(this.AllLogs)

    interface ITimingLogger with
        member this.Log(msg) = this.Log(msg)
        member this.Touch() = this.Touch()
        member this.Finish() = this.Finish()

