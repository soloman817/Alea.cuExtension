module Alea.CUDA.Extension.MGPU.Intrinsics

// this file maps to intrinsics.cuh. it provides some implementation 
// of some low level device functions.

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension.MGPU.DeviceUtil

// actrually, the DeviceFunction.__brev(x) doesn't provide host code,
// if you call it from host, you will get exception. we can extend this
// later to add host algorithm, but I leave it now, because for now, it 
// is only called from kernel.
let [<ReflectedDefinition>] brev x = DeviceFunction.__brev(x)


let ulonglong_as_uint2 = 
    <@ fun (x:uint64) ->
        let hb = uint32(x &&& 0xffffffffUL)
        let lb = uint32(x >>> 32)
        uint2(hb,lb) @>


let [<ReflectedDefinition>] popc (x:uint32) = 
    let mutable i = 31
    let mutable r = 31u
    while i <> 0 do
        if (uint32(1 <<< i) &&& uint32(x)) <> 0u then
            r <- uint32(31 - i)
        else
            r <- 32u
        i <- i - 1
    r

