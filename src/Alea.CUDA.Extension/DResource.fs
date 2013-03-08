﻿namespace Alea.CUDA.Extension

open System
open Alea.CUDA

type DArray<'T when 'T:unmanaged> internal (worker:DeviceWorker, offset:int, length:int, needDispose:bool, dmem:Lazy<DeviceMemory option * DevicePtr<byte>>) =
    inherit DisposableObject()

    let size = length * sizeof<'T>

    member this.DMem = dmem
    member this.Worker = worker
    member this.Length = length
    member this.Offset = offset
    member this.Size = size
    member this.Memory = fst dmem.Value
    member this.Ptr = ((snd dmem.Value) + offset).Reinterpret<'T>()

    member this.Gather() =
        PCalc(fun s ->
            s.RunActions()
            let logger = s.TimingLogger("default")
            logger.Log("gather array")
            let host = Array.zeroCreate<'T> length
            DevicePtrUtil.Gather(worker, this.Ptr, host, length)
            logger.Touch()
            host, s)

    override this.Dispose(disposing) =
        if needDispose && disposing && dmem.IsValueCreated && this.Memory.IsSome then
            this.Memory.Value.Dispose()
        base.Dispose(disposing)

type DScalar<'T when 'T:unmanaged> internal (worker:DeviceWorker, offset:int, needDispose:bool, dmem:Lazy<DeviceMemory option * DevicePtr<byte>>) =
    inherit DisposableObject()

    let size = sizeof<'T>
    let result = Array.zeroCreate<'T> 1

    member this.DMem = dmem
    member this.Worker = worker
    member this.Offset = offset
    member this.Size = size
    member this.Memory = fst dmem.Value
    member this.Ptr = ((snd dmem.Value) + offset).Reinterpret<'T>()

    member this.Gather() =
        PCalc(fun s ->
            s.RunActions()
            let logger = s.TimingLogger("default")
            logger.Log("gather scalar")
            DevicePtrUtil.Gather(worker, this.Ptr, result, 1)
            logger.Touch()
            result.[0], s)

    override this.Dispose(disposing) = 
        if needDispose && disposing && dmem.IsValueCreated && this.Memory.IsSome then
            this.Memory.Value.Dispose()
        base.Dispose(disposing)

type DStopwatch internal (worker:DeviceWorker, stream:Stream option, start:Event, stop:Event) =
    
    let mutable stream = stream
    let mutable elapsed : float option = None
    let mutable stopped = true

    member internal this.Start() = pcalc {
        if not stopped then failwith "this stopwatch already started!"
        if elapsed.IsSome then failwith "this stopwatch already finished!"
        stopped <- false
        let action (hint:ActionHint) =
            let _stream = if stream.IsSome then stream.Value else hint.Stream
            stream <- Some _stream
            start.Record(_stream)
        do! PCalc.action action }

    member this.Stop() = pcalc {
        if stopped then failwith "this stopwatch already finished!"
        if elapsed.IsSome then failwith "this stopwatch already finished!"
        stopped <- true
        let action (hint:ActionHint) =
            stop.Record(stream.Value)
            stop.Synchronize()
            elapsed <- worker.ElapsedMilliseconds(start, stop) |> Some
        do! PCalc.action action }

    member this.ElapsedMilliseconds =
        PCalc(fun s ->
            match elapsed with
            | Some(elapsed) -> elapsed, s
            | None ->
                let s = if not stopped then let _, s = this.Stop().Invoke(s) in s else s
                s.RunActions()
                elapsed.Value, s)

[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module DStopwatch =
    let startNew (worker:DeviceWorker) =
        PCalc(fun s ->
            let start = worker.CreateEvent()
            let stop = worker.CreateEvent()
            s.AddResource(start)
            s.AddResource(stop)
            let stopwatch = DStopwatch(worker, None, start, stop)
            let _, s = stopwatch.Start().Invoke(s)
            stopwatch, s)

[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module DArray =
    let createInBlob<'T when 'T:unmanaged> (worker:DeviceWorker) (n:int) =
        PCalc(fun s -> 
            let id = s.AddBlobSlot(BlobSlot.Extent(worker, n * sizeof<'T>))
            let dmem = Lazy.Create(fun () -> let mem, ptr = s.GetBlobSlot(id) in Some mem, ptr)
            let darray = new DArray<'T>(worker, 0, n, false, dmem)
            darray, s)

    let scatterInBlob (worker:DeviceWorker) (harray:'T[]) =
        PCalc(fun s ->
            let id = s.AddBlobSlot(BlobSlot.FromHost(worker, Engine.defaultStream, harray))
            let dmem = Lazy.Create(fun () -> let mem, ptr = s.GetBlobSlot(id) in Some mem, ptr)
            let darray = new DArray<'T>(worker, 0, harray.Length, false, dmem)
            darray, s)

    let gather (darray:DArray<'T>) = darray.Gather()

    let toScalar (darray:DArray<'T>) (idx:int) = new DScalar<'T>(darray.Worker, darray.Offset + idx * sizeof<'T>, false, darray.DMem)
    let ofScalar (dscalar:DScalar<'T>) = new DArray<'T>(dscalar.Worker, dscalar.Offset, 1, false, dscalar.DMem)

[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module DScalar =
    let createInBlob<'T when 'T:unmanaged> (worker:DeviceWorker) =
        PCalc(fun s ->
            let id = s.AddBlobSlot(BlobSlot.Extent(worker, sizeof<'T>))
            let dmem = Lazy.Create(fun () -> let mem, ptr = s.GetBlobSlot(id) in Some mem, ptr)
            let dscalar = new DScalar<'T>(worker, 0, false, dmem)
            dscalar, s)

    let scatterInBlob (worker:DeviceWorker) (hscalar:'T) =
        PCalc(fun s ->
            let id = s.AddBlobSlot(BlobSlot.FromHost(worker, Engine.defaultStream, [| hscalar |]))
            let dmem = Lazy.Create(fun () -> let mem, ptr = s.GetBlobSlot(id) in Some mem, ptr)
            let dscalar = new DScalar<'T>(worker, 0, false, dmem)
            dscalar, s)

    let gather (dscalar:DScalar<'T>) = dscalar.Gather()

    let toArray (dscalar:DScalar<'T>) = new DArray<'T>(dscalar.Worker, dscalar.Offset, 1, false, dscalar.DMem)
    let ofArray (darray:DArray<'T>) (idx:int) = new DScalar<'T>(darray.Worker, darray.Offset + idx * sizeof<'T>, false, darray.DMem)

module DStream =

    let create (worker:DeviceWorker) =
        PCalc(fun s ->
            let stream = worker.CreateStream()
            s.AddResource(stream)
            stream, s)