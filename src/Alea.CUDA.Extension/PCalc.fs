[<AutoOpen>]
module Alea.CUDA.Extension.PCalc

open System
open System.Collections.Generic
open Alea.CUDA

type BlobSlot =
    | FromHost of DeviceWorker * Array
    | Extent of DeviceWorker * int

    member this.Worker = 
        match this with
        | FromHost(worker, _) -> worker
        | Extent(worker, _) -> worker

    member this.SortKey =
        match this with
        | FromHost(_, _) -> 0
        | Extent(_, _) -> 1

    member this.Size =
        match this with
        | FromHost(_, array) -> Buffer.ByteLength(array)
        | Extent(_, size) -> size

type Diagnoser =
    {
        DebugBlob : bool
        DebugResourceDispose : bool
        KernelLaunchDiagnoser : Engine.Diagnoser option
    }

    static member Default() =
        {
            DebugBlob = false
            DebugResourceDispose = false
            KernelLaunchDiagnoser = None
        }

type State =
    {
        Diagnoser : Diagnoser
        Blob : Dictionary<int, BlobSlot> // blob slots in building
        Blobs : Dictionary<int, DevicePtr<byte>> // blob slots that is freezed!
        Actions : List<unit -> unit>
        Resources : List<IDisposable>
    }

    member this.AddBlobSlot(slot:BlobSlot) =
        let id = this.Blobs.Count + this.Blob.Count
        this.Blob.Add(id, slot)
        id

    member this.GetBlobSlot(id:int) =
        let freezed = id < this.Blobs.Count

        if not freezed then
            this.Blob
            |> Seq.fold (fun (dict:Dictionary<int, List<int * BlobSlot>>) pair ->
                let id = pair.Key
                let slot = pair.Value
                let wid = slot.Worker.WorkerThreadId
                if not (dict.ContainsKey(wid)) then dict.Add(wid, List<int * BlobSlot>())
                dict.[wid].Add(id, slot)
                dict) (Dictionary<int, List<int * BlobSlot>>())
            |> Seq.toArray
            |> Array.map (fun pair -> pair.Value |> Array.ofSeq)
            |> Array.iter (fun slots ->
                let worker = let _, slot = slots.[0] in slot.Worker
                let alignment = worker.Device.Attribute(DeviceAttribute.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT)
                let padding = Util.padding alignment
                if this.Diagnoser.DebugBlob then printfn "Creating blob for %s:" worker.Name

                // sort to move stuff need copy in the front
                let slots = slots |> Array.sortBy (fun (_, slot) -> slot.SortKey)
                let sizes = slots |> Array.map (fun (_, slot) -> let size = slot.Size in size, padding size)
                if this.Diagnoser.DebugBlob then
                    (slots, sizes)
                    ||> Array.iter2 (fun (id, slot) (size, padding) ->
                        printfn "==> Slot[%d] size=%d padding=%d type=%s"
                            id size padding
                            (match slot with BlobSlot.FromHost(_, _) -> "FromHost" | BlobSlot.Extent(_, _) -> "Extent"))

                // malloc
                let totalMallocSize = sizes |> Array.fold (fun total (size, padding) -> total + size + padding) 0
                let dmem = worker.Malloc<byte>(totalMallocSize)
                this.Resources.Add(dmem)
                if this.Diagnoser.DebugBlob then printfn "==> Malloc %d bytes" totalMallocSize

                // memcpy
                let totalMemcpySize =
                    (slots, sizes)
                    ||> Array.fold2 (fun total (_, slot) (size, padding) ->
                        match slot with
                        | BlobSlot.FromHost(_, _) -> total + size + padding
                        | _ -> total) 0
                if totalMemcpySize > 0 then
                    let memcpyBlob = Array.zeroCreate<byte> totalMemcpySize
                    (slots, sizes)
                    ||> Array.fold2 (fun (offset:int) (_, slot) (size, padding) -> 
                        match slot with
                        | BlobSlot.FromHost(_, array) ->
                            Buffer.BlockCopy(array, 0, memcpyBlob, offset, size)
                            offset + size + padding
                        | _ -> offset) 0 |> ignore
                    DevicePtrUtil.Scatter(worker, memcpyBlob, dmem.Ptr, totalMemcpySize)
                    if this.Diagnoser.DebugBlob then printfn "==> Memcpy %d bytes" totalMemcpySize

                // fill freezed blob
                (slots, sizes)
                ||> Array.fold2 (fun (ptr:DevicePtr<byte>) (id, _) (size, padding) ->
                    this.Blobs.Add(id, ptr)
                    ptr + size + padding) dmem.Ptr |> ignore)

            // clear building blob slots
            this.Blob.Clear()

        this.Blobs.[id]

    member this.Free() =
        this.Resources
        |> Seq.iter (fun o ->
            if this.Diagnoser.DebugResourceDispose then printfn "Disposing %A ..." o
            o.Dispose())
        this.Resources.Clear()

    member this.RunActions() = this.Actions |> Seq.iter (fun f -> f()); this.Actions.Clear()

    member this.AddKernelDiagnoser (lp:LaunchParam) =
        match this.Diagnoser.KernelLaunchDiagnoser with
        | Some(diagnoser) -> lp |> Engine.setDiagnoser diagnoser
        | None -> lp

    static member Default() =
        {
            Diagnoser = Diagnoser.Default()
            Blob = Dictionary<int, BlobSlot>()
            Blobs = Dictionary<int, DevicePtr<byte>>()
            Actions = List<unit -> unit>()
            Resources = List<IDisposable>()
        }
        
    static member Default(diagnoser) =
        {
            Diagnoser = diagnoser
            Blob = Dictionary<int, BlobSlot>()
            Blobs = Dictionary<int, DevicePtr<byte>>()
            Actions = List<unit -> unit>()
            Resources = List<IDisposable>()
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
    let s0 = State.Default()
    let r, s1 = calc.Invoke(s0)
    s1.Free()
    r

let runWithDiagnoser (diagnoser:Diagnoser) (calc:PCalc<'T>) =
    let s0 = State.Default(diagnoser)
    let r, s1 = calc.Invoke(s0)
    s1.Free()
    r

let addAction (f:unit -> unit) = PCalc(fun s -> s.Actions.Add(f); (), s)

let getLaunchParamModifier () = PCalc(fun s -> (fun lp -> s.AddKernelDiagnoser lp), s)

let createArray<'T when 'T:unmanaged> (worker:DeviceWorker) (n:int) =
    PCalc(fun s -> 
        let id = s.AddBlobSlot(BlobSlot.Extent(worker, n * sizeof<'T>))
        let rawptr = Lazy.Create(fun () -> s.GetBlobSlot(id))
        let darray = DArray<'T>(worker, n, rawptr)
        darray, s)

let scatterArray<'T when 'T:unmanaged> (worker:DeviceWorker) (array:'T[]) =
    PCalc(fun s ->
        let id = s.AddBlobSlot(BlobSlot.FromHost(worker, array))
        let rawptr = Lazy.Create(fun () -> s.GetBlobSlot(id))
        let darray = DArray<'T>(worker, array.Length, rawptr)
        darray, s)

let gatherArray (data:DArray<'T>) = PCalc(fun s -> s.RunActions(); data.ToHost(), s)

