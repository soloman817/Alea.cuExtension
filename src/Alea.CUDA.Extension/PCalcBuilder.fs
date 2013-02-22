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
        | FromHost(_, _, array) -> Buffer.ByteLength(array)

    member this.Type =
        match this with
        | Extent(_, _) -> "Extent"
        | FromHost(_, stream, _) -> sprintf "FromHost@%A" stream.Handle

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

type PCalcState =
    {
        Diagnoser : PCalcDiagnoser
        Blob : List<int * BlobSlot> // blob slots in building
        Blobs : Dictionary<int, DeviceMemory * DevicePtr<byte>> // blob slots that is freezed!
        Actions : List<unit -> unit>
        Resources : List<IDisposable>
        KernelTimingCollector : Timing.TimingCollector option
        TimingLoggers : Dictionary<string, Timing.TimingLogger> option
    }

    member this.DebugLevel = this.Diagnoser.DebugLevel

    member this.GetTimingLogger (name:string) =
        match this.TimingLoggers with
        | Some(loggers) -> 
            if not (loggers.ContainsKey(name)) then loggers.Add(name, Timing.TimingLogger(name))
            loggers.[name] :> Timing.ITimingLogger
        | None -> Timing.dummyTimingLogger

    member this.FreezeBlob() =
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
                        this.Resources.Add(dmem)
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
                                    printfn "Memcpy blob on %s.Stream[%A]: %d bytes (%.3f MB)"
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

//                    // make hblob first then do one copy
//                    let total = slots |> Array.fold (fun total (_, _, size, padding) -> total + size + padding) 0
//                    if total > 0 then
//                        if this.DebugLevel >= 1 then
//                            printfn "Memcpy blob on %s.Stream[%A]: %d bytes (%.3f MB)"
//                                worker.Name
//                                stream.Handle
//                                total
//                                (float(total) / 1024.0 / 1024.0)
//                        logger.Log("create hblob")
//                        let hblob = Array.zeroCreate<byte> total
//                        slots
//                        |> Array.fold (fun (offset:int) (_, slot, size, padding) ->
//                            match slot with
//                            | BlobSlot.FromHost(_, _, array) ->
//                                Buffer.BlockCopy(array, 0, hblob, offset, size)
//                                offset + size + padding
//                            | _ -> failwith "BUG") 0
//                        |> ignore
//                        logger.Log("memcpy hblob")
//                        DevicePtrUtil.Scatter(worker, hblob, dblob, total) //TODO using stream!
//                        logger.Touch()
//                    dblob + total) dmem.Ptr
                |> ignore)

            // fill ptr
            slots |> Array.iter (fun (_, dmem, slots) ->
                let dmem' = dmem :> DeviceMemory
                slots
                |> Array.fold (fun (ptr:DevicePtr<byte>) (id, _, size, padding) ->
                    this.Blobs.Add(id, (dmem', ptr))
                    ptr + size + padding) dmem.Ptr
                |> ignore)

            this.Blob.Clear()

        if this.Blob.Count > 0 then f()

    member this.AddBlobSlot(slot:BlobSlot) =
        let id = this.Blobs.Count + this.Blob.Count
        this.Blob.Add(id, slot)
        id

    member this.GetBlobSlot(id:int) =
        let freezed = id < this.Blobs.Count
        if not freezed then this.FreezeBlob()
        this.Blobs.[id]

    member this.DisposeResources() =
        if this.Resources.Count > 0 then
            let logger = this.GetTimingLogger("default")
            logger.Log("release resources")
            this.Resources |> Seq.iter (fun o ->
                if this.DebugLevel >= 1 then printfn "Disposing %A ..." o
                o.Dispose())
            this.Resources.Clear()
            logger.Touch()

    member this.RunActions() =
        if this.Actions.Count > 0 then
            this.FreezeBlob()
            let logger = this.GetTimingLogger("default")
            logger.Log("run actions")
            this.Actions |> Seq.iter (fun f -> f())
            this.Actions.Clear()
            logger.Touch()

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
            Blobs = Dictionary<int, DeviceMemory * DevicePtr<byte>>(capacity)
            Actions = List<unit -> unit>(capacity)
            Resources = List<IDisposable>(capacity)
            KernelTimingCollector = None
            TimingLoggers = None
        }
        
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

let pcalc = PCalcBuilder()

type LPModifier = LaunchParam -> LaunchParam

