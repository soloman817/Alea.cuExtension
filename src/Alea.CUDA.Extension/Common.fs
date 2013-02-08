namespace Alea.CUDA.Extension

open Alea.CUDA

type PArray internal (memory:DeviceMemory, rawptr:DevicePtr<byte>, size:int, needDispose:bool) =
    inherit DisposableObject()
    member this.Memory = memory
    member this.Worker = memory.Worker
    member this.RawPtr = rawptr
    member this.Size = size

    override this.Dispose(disposing) =
        if needDispose && disposing then memory.Dispose()
        base.Dispose(disposing)

type PArray<'T when 'T:unmanaged> internal (memory:DeviceMemory, ptr:DevicePtr<'T>, length:int, needDispose:bool) =
    inherit PArray(memory, ptr.Reinterpret<byte>(), length * sizeof<'T>, needDispose)
    
    member this.Ptr = ptr
    member this.Length = length
    
    member this.ToHost() =
        let host = Array.zeroCreate<'T> length
        DevicePtrUtil.Gather(memory.Worker, ptr, host, length)
        host

type PArray with
    static member Create(worker:DeviceWorker, host:'T[]) =
        let memory = worker.Malloc(host)
        new PArray<'T>(memory, memory.Ptr, memory.Length, true)

    static member Create<'T when 'T:unmanaged>(worker:DeviceWorker, length:int) =
        let memory = worker.Malloc(length)
        new PArray<'T>(memory, memory.Ptr, memory.Length, true)

