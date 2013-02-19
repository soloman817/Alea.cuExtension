[<AutoOpen>]
module Alea.CUDA.Extension.Memory

open Alea.CUDA

type DArray<'T when 'T:unmanaged> internal (worker:DeviceWorker, length:int, rawptr:Lazy<DevicePtr<byte>>) =
    let size = length * sizeof<'T>

    member this.Worker = worker
    member this.Length = length
    member this.Size = size
    member this.RawPtr = rawptr.Value
    member this.Ptr = rawptr.Value.Reinterpret<'T>()

    member internal this.ToHost() =
        let host = Array.zeroCreate<'T> length
        DevicePtrUtil.Gather(worker, this.Ptr, host, length)
        host