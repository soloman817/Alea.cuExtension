module Alea.CUDA.Extension.MGPU.Intrinsics

open Alea.CUDA

let [<ReflectedDefinition>] brev x = DeviceFunction.__brev(x)

