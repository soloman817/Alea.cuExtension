namespace Alea.CUDA.Extension

open Alea.CUDA

type DArray<'T when 'T:unmanaged> internal (worker:DeviceWorker, length:int, needDispose:bool, dmem:Lazy<DeviceMemory option * DevicePtr<'T>>) =
    inherit DisposableObject()

    let size = length * sizeof<'T>

    member this.DMem = dmem
    member this.Worker = worker
    member this.Length = length
    member this.Size = size
    member this.Memory = fst dmem.Value
    member this.Ptr = snd dmem.Value

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

type DScalar<'T when 'T:unmanaged> internal (worker:DeviceWorker, offset:int, needDispose:bool, dmem:Lazy<DeviceMemory option * DevicePtr<'T>>) =
    inherit DisposableObject()

    let size = sizeof<'T>
    let result = Array.zeroCreate<'T> 1

    member this.DMem = dmem
    member this.Worker = worker
    member this.Size = size
    member this.Memory = fst dmem.Value
    member this.Ptr = snd dmem.Value

    member this.Gather() =
        PCalc(fun s ->
            s.RunActions()
            let logger = s.TimingLogger("default")
            logger.Log("gather scalar")
            DevicePtrUtil.Gather(worker, this.Ptr + offset, result, 1)
            logger.Touch()
            result.[0], s)

    override this.Dispose(disposing) = 
        if needDispose && disposing && dmem.IsValueCreated && this.Memory.IsSome then
            this.Memory.Value.Dispose()
        base.Dispose(disposing)

[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module DArray =
    let createInBlob (worker:DeviceWorker) (n:int) =
        PCalc(fun s -> 
            let id = s.AddBlobSlot(BlobSlot.Extent(worker, n * sizeof<'T>))
            let dmem = Lazy.Create(fun () -> let mem, ptr = s.GetBlobSlot(id) in Some mem, ptr.Reinterpret<'T>())
            let darray = new DArray<'T>(worker, n, false, dmem)
            darray, s)

    let scatterInBlob (worker:DeviceWorker) (harray:'T[]) =
        PCalc(fun s ->
            let id = s.AddBlobSlot(BlobSlot.FromHost(worker, Engine.defaultStream, harray))
            let dmem = Lazy.Create(fun () -> let mem, ptr = s.GetBlobSlot(id) in Some mem, ptr.Reinterpret<'T>())
            let darray = new DArray<'T>(worker, harray.Length, false, dmem)
            darray, s)

    let gather (darray:DArray<'T>) = darray.Gather()

    let toScalar (darray:DArray<'T>) (idx:int) = new DScalar<'T>(darray.Worker, idx, false, darray.DMem)
    let ofScalar (dscalar:DScalar<'T>) = new DArray<'T>(dscalar.Worker, 1, false, dscalar.DMem)

[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module DScalar =

    let createInBlob (worker:DeviceWorker) =
        PCalc(fun s ->
            let id = s.AddBlobSlot(BlobSlot.Extent(worker, sizeof<'T>))
            let dmem = Lazy.Create(fun () -> let mem, ptr = s.GetBlobSlot(id) in Some mem, ptr.Reinterpret<'T>())
            let dscalar = new DScalar<'T>(worker, 0, false, dmem)
            dscalar, s)

    let scatterInBlob (worker:DeviceWorker) (hscalar:'T) =
        PCalc(fun s ->
            let id = s.AddBlobSlot(BlobSlot.FromHost(worker, Engine.defaultStream, [| hscalar |]))
            let dmem = Lazy.Create(fun () -> let mem, ptr = s.GetBlobSlot(id) in Some mem, ptr.Reinterpret<'T>())
            let dscalar = new DScalar<'T>(worker, 0, false, dmem)
            dscalar, s)

    let gather (dscalar:DScalar<'T>) = dscalar.Gather()

    let toArray (dscalar:DScalar<'T>) = new DArray<'T>(dscalar.Worker, 1, false, dscalar.DMem)
    let ofArray (darray:DArray<'T>) (idx:int) = new DScalar<'T>(darray.Worker, idx, false, darray.DMem)

module DStream =

    let create (worker:DeviceWorker) =
        PCalc(fun s ->
            let stream = worker.CreateStream()
            s.AddResource(stream)
            stream, s)