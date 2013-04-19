[<AutoOpen>]
module Alea.CUDA.Extension.PCalcBuilder

open System
open System.Runtime.InteropServices
open System.Collections.Generic
open Alea.CUDA
open Alea.Interop.CUDA

//typedef void (CUDA_CB *CUstreamCallback)(CUstream hStream, CUresult status, void *userData);
type CUstreamCallback = delegate of CUstream * CUresult * nativeint -> unit

//CUresult CUDAAPI cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags);
[<DllImport("nvcuda.dll", EntryPoint="cuStreamAddCallback", CallingConvention=CallingConvention.StdCall)>]
extern CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, nativeint userData, uint32 flags);

type PinnedArray(harray:Array) =
    let handle = GCHandle.Alloc(harray, GCHandleType.Pinned)
    member this.Free() = handle.Free()
    member this.Ptr = handle.AddrOfPinnedObject()

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
        | FromHost(_, _, array) ->
            let length = array.Length
            match length with
            | 0 -> 0
            | _ ->
                let ty = array.GetValue(0).GetType()
                let size = Marshal.SizeOf(ty)
                size * length

    member this.Type =
        match this with
        | Extent(_, _) -> "Extent"
        | FromHost(_, stream, _) -> sprintf "FromHost@%X" stream.Handle

type PCalcDiagnoser =
    {
        DebugLevel : int
        KernelLaunchDiagnoser : Engine.Diagnoser option
    }

    static member None =
        {
            DebugLevel = 0
            KernelLaunchDiagnoser = None
        }

    static member All(level) =
        {
            DebugLevel = level
            KernelLaunchDiagnoser = Some Util.kldiag
        }

type ActionHint =
    {
        Diagnose : (KernelExecutionStats -> unit) option
        Stream : Stream
        TotalStreams : int option
        DebugLevel : int
    }

    member this.ModifyLaunchParam (lp:LaunchParam) = lp.SetStream(this.Stream).SetDiagnoser(this.Diagnose)

type PCalcStateParam =
    {
        Diagnoser : PCalcDiagnoser
        TimingLoggers : Dictionary<string, Timing.TimingLogger> option
        KernelTimingCollector : Timing.TimingCollector option
    }

type PCalcState internal (param:PCalcStateParam) =
    let capacity = 16

    let blob = List<int * BlobSlot>(capacity)
    let blobs = Dictionary<int, DeviceMemory * DevicePtr<byte>>(capacity)
    let actions = List<ActionHint * (ActionHint -> unit)>(capacity)
    let resources = List<IDisposable>(capacity)

    let mutable hint : ActionHint =
        let diagnose =
            match param.Diagnoser.KernelLaunchDiagnoser, param.KernelTimingCollector with
            | None, None -> None
            | Some(diagnose), None -> Some diagnose
            | Some(diagnose), Some(collector) ->
                fun stats ->
                    diagnose stats
                    let name = sprintf "%d.%X.%s.%X" stats.Kernel.Worker.WorkerThreadId stats.Kernel.Module.Handle stats.Kernel.Name stats.LaunchParam.Stream.Handle
                    collector.Add(name, stats.TimeSpan)
                |> Some
            | None, Some(collector) ->
                fun stats ->
                    let name = sprintf "%d.%X.%s.%X" stats.Kernel.Worker.WorkerThreadId stats.Kernel.Module.Handle stats.Kernel.Name stats.LaunchParam.Stream.Handle
                    collector.Add(name, stats.TimeSpan)
                |> Some
        { Diagnose = diagnose; Stream = Engine.defaultStream; TotalStreams = None; DebugLevel = param.Diagnoser.DebugLevel }

    member this.DebugLevel = param.Diagnoser.DebugLevel

    member this.ActionHint with get() = hint and set(lphint') = hint <- lphint'

    member this.TimingLogger (name:string) =
        match param.TimingLoggers with
        | None -> Timing.dummyTimingLogger
        | Some(loggers) -> 
            if not (loggers.ContainsKey(name)) then loggers.Add(name, Timing.TimingLogger(name))
            loggers.[name] :> Timing.ITimingLogger

    member this.FreezeBlob() =
        let f () =
            let logger = this.TimingLogger("default")

            // freeze and malloc blob
            let slots =
                blob
                |> Seq.groupBy (fun (id, slot) -> slot.Worker.WorkerThreadId)
                |> Seq.map (fun (_, slots) ->
                    let slots = slots |> Seq.toArray
                    let worker = (snd slots.[0]).Worker
                    let padding = Util.padding worker.Device.TextureAlignment

                    // calc size
                    let slots = slots |> Array.map (fun (id, slot) ->
                        let size = slot.Size
                        let padding = padding size
                        id, slot, size, padding)

                    // sort slots
                    slots |> Array.sortInPlaceWith (fun (_, lslot, _, _) (_, rslot, _, _) ->
                        match lslot, rslot with
                        | Extent(_, _), FromHost(_, _, _) -> 1
                        | FromHost(_, _, _), Extent(_, _) -> -1
                        | FromHost(_, lstream, _), FromHost(_, rstream, _) ->
                            if lstream.Handle < rstream.Handle then -1
                            elif lstream.Handle > rstream.Handle then 1
                            else 0
                        | _ -> 0)

                    if this.DebugLevel >= 1 then
                        printfn "Freezing blob on %s:" worker.Name
                        slots |> Array.iter (fun (id, slot, size, padding) ->
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
                        if this.DebugLevel >= 1 then
                            printfn "Malloc blob on %s: %d bytes (%.3f MB)"
                                worker.Name
                                total
                                (float(total) / 1024.0 / 1024.0)
                        let dmem = worker.Malloc<byte>(total)
                        resources.Add(dmem)
                        dmem
                    logger.Touch()

                    worker, dmem, slots)
                |> Array.ofSeq

            // scatter by streams (async memcpy)
            slots |> Array.iter (fun (worker, dmem, slots) ->
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
                    if slots.Length > 0 then
                        logger.Log("memcpy blob")

                        let dblob' = slots |> Array.fold (fun (darray:DevicePtr<byte>) (_, slot, size, padding) ->
                            match slot with
                            | BlobSlot.FromHost(_, _, harray) ->
                                if this.DebugLevel >= 1 then
                                    printfn "Memcpy blob on %s.Stream[%X]: %d bytes (%.3f MB)"
                                        worker.Name
                                        stream.Handle
                                        size
                                        (float(size) / 1024.0 / 1024.0)

                                // experimental async memcpy (only with non-default stream)
                                match stream.IsDefault with
                                | false ->
                                    let pinnedArray = PinnedArray(harray)
                                    
                                    let callback (stream:CUstream) (result:CUresult) (userdata:nativeint) =
                                        pinnedArray.Free()
                                    
                                    let callback = CUstreamCallback(callback)
                                    
                                    fun () ->
                                        cuSafeCall(cuMemcpyHtoDAsync(darray.Handle, pinnedArray.Ptr, nativeint(size), stream.Handle))
                                        cuSafeCall(cuStreamAddCallback(stream.Handle, callback, 0n, 0u))
                                    |> worker.Eval

                                | true ->
                                    let handle = GCHandle.Alloc(harray, GCHandleType.Pinned)
                                    fun () ->
                                        try cuSafeCall(cuMemcpyHtoD(darray.Handle, handle.AddrOfPinnedObject(), nativeint(size)))
                                        finally handle.Free()
                                    |> worker.Eval

                            | _ -> failwith "shouldn't happen"

                            darray + size + padding) dblob

                        logger.Touch()

                        dblob'
                    else dblob) dmem.Ptr
                |> ignore)

            // fill ptr
            slots |> Array.iter (fun (_, dmem, slots) ->
                let dmem' = dmem :> DeviceMemory
                slots
                |> Array.fold (fun (ptr:DevicePtr<byte>) (id, _, size, padding) ->
                    blobs.Add(id, (dmem', ptr))
                    ptr + size + padding) dmem.Ptr
                |> ignore)

            blob.Clear()

        if blob.Count > 0 then f()

    member this.AddBlobSlot(slot:BlobSlot) =
        let id = blobs.Count + blob.Count
        blob.Add(id, slot)
        id

    member this.GetBlobSlot(id:int) =
        let freezed = id < blobs.Count
        if not freezed then this.FreezeBlob()
        blobs.[id]

    member this.DisposeResources() =
        if resources.Count > 0 then
            let logger = this.TimingLogger("default")
            logger.Log("release resources")
            resources |> Seq.iter (fun o ->
                if this.DebugLevel >= 1 then printfn "Disposing %A ..." o
                o.Dispose())
            resources.Clear()
            logger.Touch()

    member this.AddAction(f:ActionHint -> unit) =
        actions.Add(hint, f)

    member this.RunActions() =
        if actions.Count > 0 then
            this.FreezeBlob()
            let logger = this.TimingLogger("default")
            logger.Log(sprintf "run %d actions" actions.Count)
            actions |> Seq.iter (fun (lphint, f) -> f lphint)
            actions.Clear()
            logger.Touch()

    member this.AddResource(resource:IDisposable) = resources.Add(resource)

type PCalc<'T> =
    | PCalc of (PCalcState -> 'T * PCalcState)
    member this.Invoke s0 = match this with PCalc(f) -> f(s0)

[<Sealed>]
type PCalcBuilder() =
    member this.Bind(x:PCalc<'a>, res:'a -> PCalc<'b>) =
        PCalc(fun s0 -> let r, s1 = x.Invoke(s0) in res(r).Invoke(s1))
    member this.Return(x:'a) = PCalc(fun s -> x, s)
    member this.ReturnFrom(x:'a) = x
    member this.Zero() = PCalc(fun s -> (), s)
    member this.For(elements:seq<'a>, forBody:'a -> PCalc<unit>) =
        PCalc(fun s0 -> (), elements |> Seq.fold (fun s0 e -> let _, s1 = (forBody e).Invoke(s0) in s1) s0)
    member this.Combine(partOne:PCalc<unit>, partTwo:PCalc<'a>) =
        PCalc(fun s0 -> let (), s1 = partOne.Invoke(s0) in partTwo.Invoke(s1))
    member this.Delay(restOfComputation:unit -> PCalc<'a>) =
        PCalc(fun s0 -> restOfComputation().Invoke(s0))
    member this.Using<'a, 'b when 'a :> IDisposable>(x:'a, res:'a -> PCalc<'b>) =
        PCalc(fun s0 -> try res(x).Invoke(s0) finally x.Dispose())

let pcalc = PCalcBuilder()

