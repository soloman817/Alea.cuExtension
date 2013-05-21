module Alea.CUDA.Extension.MGPU.Intrinsics

// this file maps to intrinsics.cuh. it provides some implementation 
// of some low level device functions.

open Alea.CUDA

// actrually, the DeviceFunction.__brev(x) doesn't provide host code,
// if you call it from host, you will get exception. we can extend this
// later to add host algorithm, but I leave it now, because for now, it 
// is only called from kernel.
let [<ReflectedDefinition>] brev x = DeviceFunction.__brev(x)

