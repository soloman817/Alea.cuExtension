[<AutoOpen>]
module Alea.CUDA.Extension.PCalc

open System
open System.Collections.Generic
open Alea.CUDA

type BlobSlot =
    | Extent of DeviceWorker * int
    | FromHost of DeviceWorker * Stream * Array

    member this.Worker =
        match this with
        | Extent(worker, _) -> worker
        | FromHost(worker, _, _) -> worker

    member this.Size =
        match this with
        | Extent(_, size) -> size
        | FromHost(_, _, array) -> Buffer.ByteLength(array)

    member this.Type =
        match this with
        | Extent(_, _) -> "Extent"
        | FromHost(_, stream, _) -> sprintf "FromHost@%A" stream.Handle

type Diagnoser =
    {
        DebugBlob : bool
        DebugResourceDispose : bool
        KernelLaunchDiagnoser : Engine.Diagnoser option
    }

    static member DefaultNone =
        {
            DebugBlob = false
            DebugResourceDispose = false
            KernelLaunchDiagnoser = None
        }

    static member DefaultAll =
        {
            DebugBlob = true
            DebugResourceDispose = true
            KernelLaunchDiagnoser = Some Util.kldiag
        }

type State =
    {
        Diagnoser : Diagnoser
        Blob : List<int * BlobSlot> // blob slots in building
        Blobs : Dictionary<int, DevicePtr<byte>> // blob slots that is freezed!
        Actions : List<Lazy<unit>>
        Resources : List<IDisposable>
        KernelTimingCollector : Timing.TimingCollector option
        TimingLoggers : Dictionary<string, Timing.ITimingLogger> option
    }

    member this.GetTimingLogger (name:string) =
        match this.TimingLoggers with
        | Some(loggers) -> 
            match loggers.ContainsKey(name) with
            | true -> loggers.[name]
            | false ->
                let logger = Timing.TimingLogger(name)
                loggers.Add(name, logger)
                logger :> Timing.ITimingLogger
        | None -> 
            { new Timing.ITimingLogger with
                member this.Log(msg) = ()
                member this.Split() = ()
                member this.Finish() = ()
                member this.Dump() = ()
            }

    member private this._FreezeBlob() =
        let f () =
            let logger = this.GetTimingLogger("default")

            // freeze and malloc blob
            let slots =
                this.Blob
                |> Seq.groupBy (fun (id, slot) -> slot.Worker.WorkerThreadId)
                |> Seq.map (fun (_, slots) ->
                    let slots = slots |> Seq.toArray
                    let worker = (snd slots.[0]).Worker
                    let padding = Util.padding worker.Device.TextureAlignment

                    // calc size
                    let slots =
                        slots
                        |> Array.map (fun (id, slot) ->
                            let size = slot.Size
                            let padding = padding size
                            id, slot, size, padding)

                    // sort slots
                    slots
                    |> Array.sortInPlaceWith (fun (_, lslot, _, _) (_, rslot, _, _) ->
                        match lslot, rslot with
                        | Extent(_, _), FromHost(_, _, _) -> 1
                        | FromHost(_, _, _), Extent(_, _) -> -1
                        | FromHost(_, lstream, _), FromHost(_, rstream, _) ->
                            if lstream.Handle < rstream.Handle then -1
                            elif lstream.Handle > rstream.Handle then 1
                            else 0
                        | _ -> 0)

                    if this.Diagnoser.DebugBlob then
                        printfn "Freezing blob on %s:" worker.Name
                        slots
                        |> Array.iter (fun (id, slot, size, padding) ->
                            printfn "==> Slot[%d] size=%d (%.3f MB) padding=%d type=%s"
                                id
                                size
                                (float(size) / 1024.0 / 1024.0)
                                padding
                                slot.Type)

                    // malloc blob
                    logger.Log("malloc blob")
                    let dmem =
                        let total = slots |> Array.fold (fun total (_, _, size, padding) -> total + size + padding) 0
                        if this.Diagnoser.DebugBlob then
                            printfn "Malloc blob on %s: %d bytes (%.3f MB)"
                                worker.Name
                                total
                                (float(total) / 1024.0 / 1024.0)
                        let dmem = worker.Malloc<byte>(total)
                        this.Resources.Add(dmem)
                        dmem
                    logger.Split()

                    worker, dmem, slots)
                |> Array.ofSeq

            // scatter by streams (async memcpy)
            slots
            |> Array.iter (fun (worker, dmem, slots) ->
                slots
                |> Array.filter (fun (_, slot, size, padding) ->
                    match slot with
                    | BlobSlot.FromHost(_, _, _) -> true
                    | BlobSlot.Extent(_, _) -> false)
                |> Seq.groupBy (fun (_, slot, _, _) ->
                    match slot with
                    | BlobSlot.FromHost(_, stream, _) -> stream
                    | _ -> failwith "BUG!")
                |> Seq.map (fun (stream, slots) -> stream, slots |> Array.ofSeq)
                |> Array.ofSeq
                |> Array.fold (fun (dblob:DevicePtr<byte>) (stream, slots) ->
                    let total = slots |> Array.fold (fun total (_, _, size, padding) -> total + size + padding) 0
                    if total > 0 then
                        if this.Diagnoser.DebugBlob then
                            printfn "Scatter blob on %s.Stream[%A]: %d bytes (%.3f MB)"
                                worker.Name
                                stream.Handle
                                total
                                (float(total) / 1024.0 / 1024.0)
                        logger.Log("create hblob")
                        let hblob = Array.zeroCreate<byte> total
                        slots
                        |> Array.fold (fun (offset:int) (_, slot, size, padding) ->
                            match slot with
                            | BlobSlot.FromHost(_, _, array) ->
                                Buffer.BlockCopy(array, 0, hblob, offset, size)
                                offset + size + padding
                            | _ -> failwith "BUG") 0
                        |> ignore
                        logger.Log("scatter hblob")
                        DevicePtrUtil.Scatter(worker, hblob, dblob, total) //TODO using stream!
                        logger.Split()
                    dblob + total) dmem.Ptr
                |> ignore)

            // fill ptr
            slots
            |> Array.iter (fun (_, dmem, slots) ->
                slots
                |> Array.fold (fun (ptr:DevicePtr<byte>) (id, _, size, padding) ->
                    this.Blobs.Add(id, ptr)
                    ptr + size + padding) dmem.Ptr
                |> ignore)

            this.Blob.Clear()

        if this.Blob.Count > 0 then f()

    member this.AddBlobSlot(slot:BlobSlot) =
        let f () =
            let id = this.Blobs.Count + this.Blob.Count
            this.Blob.Add(id, slot)
            id

        lock this f

    member this.GetBlobSlot(id:int) =
        let f () =
            let freezed = id < this.Blobs.Count
            if not freezed then this._FreezeBlob()
            this.Blobs.[id]

        lock this f

    member this.Free() =
        let f () =
            this.Resources
            |> Seq.iter (fun o ->
                if this.Diagnoser.DebugResourceDispose then printfn "Disposing %A ..." o
                o.Dispose())
            this.Resources.Clear()

        lock this f

    member this.RunActions() =
        let f () =
            let logger = this.GetTimingLogger("default")
            this._FreezeBlob()
            logger.Log("run actions")
            this.Actions |> Seq.iter (fun f -> f.Force())
            logger.Split()
            this.Actions.Clear()

        lock this f

    member this.AddKernelDiagnoser (lp:LaunchParam) =
        match this.Diagnoser.KernelLaunchDiagnoser, this.KernelTimingCollector with
        | None, None -> lp
        | Some(diagnoser), Some(collector) ->
            let diagnoser stat =
                diagnoser stat
                collector.Add(stat.Kernel.Name, stat.TimeSpan)
            lp |> Engine.setDiagnoser diagnoser
        | None, Some(collector) ->
            let diagnoser stat =
                collector.Add(stat.Kernel.Name, stat.TimeSpan)
            lp |> Engine.setDiagnoser diagnoser
        | Some(diagnoser), None -> lp |> Engine.setDiagnoser diagnoser

    static member Create(diagnoser) =
        let capacity = 16
        {
            Diagnoser = diagnoser
            Blob = List<int * BlobSlot>(capacity)
            Blobs = Dictionary<int, DevicePtr<byte>>(capacity)
            Actions = List<Lazy<unit>>(capacity)
            Resources = List<IDisposable>(capacity)
            KernelTimingCollector = None
            TimingLoggers = None
        }
        
type PCalc<'T> =
    | PCalc of (State -> 'T * State)
    member this.Invoke s0 = match this with PCalc(f) -> f(s0)

type PCalcBuilder() =
    member this.Bind(x:PCalc<'a>, res:'a -> PCalc<'b>) =
        PCalc(fun s0 -> let r, s1 = x.Invoke(s0) in res(r).Invoke(s1))
    member this.Return(x:'a) = PCalc(fun s -> x, s)

let pcalc = PCalcBuilder()

let run (calc:PCalc<'T>) =
    let s0 = State.Create(Diagnoser.DefaultNone)
    let r, s1 = calc.Invoke(s0)
    s1.Free()
    r

let runWithTiming (n:int) (calc:PCalc<'T>) =
    let timings = Array.zeroCreate<float> n
    for i = 1 to n - 1 do
        let _, timing = Timing.tictoc (fun () -> run calc)
        timings.[i] <- timing
    let r, timing = Timing.tictoc (fun () -> run calc)
    timings.[0] <- timing
    let timing = timings |> Array.average
    r, timing

let runWithKernelTiming (n:int) (calc:PCalc<'T>) =
    let tc = Timing.TimingCollector()
    let diagnoser = Diagnoser.DefaultNone

    for i = 1 to n - 1 do
        let s0 = { State.Create(diagnoser) with KernelTimingCollector = Some(tc) }
        let r, s1 = calc.Invoke(s0)
        s1.Free()

    let s0 = { State.Create(diagnoser) with KernelTimingCollector = Some(tc) }
    let r, s1 = calc.Invoke(s0)
    s1.Free()

    r, tc    
    
let runWithTimingLogger (calc:PCalc<'T>) =
    let s0 = { State.Create(Diagnoser.DefaultNone) with TimingLoggers = Some(Dictionary<string, Timing.ITimingLogger>(16)) }
    let logger = s0.GetTimingLogger("default")
    let r, s1 = calc.Invoke(s0)
    logger.Log("release resources")
    s1.Free()
    logger.Finish()
    r, s0.TimingLoggers.Value

let runWithDiagnoser (diagnoser:Diagnoser) (calc:PCalc<'T>) =
    let s0 = State.Create(diagnoser)
    let r, s1 = calc.Invoke(s0)
    s1.Free()
    r

let runInWorker (worker:DeviceWorker) (calc:PCalc<'T>) = worker.Eval(fun () -> run calc)
let runInWorkerWithTiming (worker:DeviceWorker) (n:int) (calc:PCalc<'T>) = worker.Eval(fun () -> runWithTiming n calc)
let runInWorkerWithKernelTiming (worker:DeviceWorker) (n:int) (calc:PCalc<'T>) = worker.Eval(fun () -> runWithKernelTiming n calc)
let runInWorkerWithTimingLogger (worker:DeviceWorker) (calc:PCalc<'T>) = worker.Eval(fun () -> runWithTimingLogger calc)
let runInWorkerWithDiagnoser (worker:DeviceWorker) (diagnoser:Diagnoser) (calc:PCalc<'T>) = worker.Eval(fun () -> runWithDiagnoser diagnoser calc)

let action (f:Lazy<unit>) = PCalc(fun s -> s.Actions.Add(f); (), s)

let lpmod () = PCalc(fun s -> (fun lp -> s.AddKernelDiagnoser lp), s)

type DArray<'T when 'T:unmanaged> internal (worker:DeviceWorker, length:int, rawptr:Lazy<DevicePtr<byte>>) =
    let size = length * sizeof<'T>

    member this.Worker = worker
    member this.Length = length
    member this.Size = size
    member this.RawPtr = rawptr.Value
    member this.Ptr = rawptr.Value.Reinterpret<'T>()

    member this.ToHost() =
        PCalc(fun s ->
            s.RunActions()
            let logger = s.GetTimingLogger("default")
            logger.Log("prepare host for gather")
            let host = Array.zeroCreate<'T> length
            logger.Log("gather")
            DevicePtrUtil.Gather(worker, this.Ptr, host, length)
            logger.Split()
            host, s)

type DArray private () =
    static member CreateInBlob(worker:DeviceWorker, n:int) =
        PCalc(fun s -> 
            let id = s.AddBlobSlot(BlobSlot.Extent(worker, n * sizeof<'T>))
            let rawptr = Lazy.Create(fun () -> s.GetBlobSlot(id))
            let darray = DArray<'T>(worker, n, rawptr)
            darray, s)

    static member CreateInBlob(worker:DeviceWorker, array:'T[]) =
        PCalc(fun s ->
            let id = s.AddBlobSlot(BlobSlot.FromHost(worker, Engine.defaultStream, array))
            let rawptr = Lazy.Create(fun () -> s.GetBlobSlot(id))
            let darray = DArray<'T>(worker, array.Length, rawptr)
            darray, s)

    static member CreateInBlob(worker:DeviceWorker, stream:Stream, array:'T[]) =
        PCalc(fun s ->
            let id = s.AddBlobSlot(BlobSlot.FromHost(worker, stream, array))
            let rawptr = Lazy.Create(fun () -> s.GetBlobSlot(id))
            let darray = DArray<'T>(worker, array.Length, rawptr)
            darray, s)


