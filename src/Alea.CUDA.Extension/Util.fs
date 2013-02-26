module Alea.CUDA.Extension.Util

open Alea.CUDA

let [<ReflectedDefinition>] WARP_SIZE = 32
let [<ReflectedDefinition>] LOG_WARP_SIZE = 5

let [<ReflectedDefinition>] identity x = x

let divup num den = (num + den - 1) / den

let ispow2 x = x &&& x-1 = 0
  
let nextpow2 i =
    let mutable x = i - 1
    x <- x ||| (x >>> 1)
    x <- x ||| (x >>> 2)
    x <- x ||| (x >>> 4)
    x <- x ||| (x >>> 8)
    x <- x ||| (x >>> 16)
    x + 1
     
let log2 (arg:int) =
    if arg = 0 then failwith "argument cannot be zero"
    let mutable n = arg
    let mutable logValue = 0
    while n > 1 do
        logValue <- logValue + 1
        n <- n >>> 1
    logValue

let padding alignment size =
    match alignment with
    | 0 -> 0
    | alignment -> (divup size alignment) * alignment - size

let dim3str (d:dim3) = sprintf "(%dx%dx%d)" d.x d.y d.z

let kldiag (stats:Engine.KernelExecutionStats) =
    let name = sprintf "%d.%X.%s.%X" stats.Kernel.Worker.WorkerThreadId stats.Kernel.Module.Handle stats.Kernel.Name stats.LaunchParam.Stream.Handle
    printfn "%s: %s %s %6.2f%% %.6f ms"
        name
        (stats.LaunchParam.GridDim |> dim3str)
        (stats.LaunchParam.BlockDim |> dim3str)
        (stats.Occupancy * 100.0)
        stats.TimeSpan

module NumericLiteralG =
    let [<ReflectedDefinition>] inline FromZero() = LanguagePrimitives.GenericZero
    let [<ReflectedDefinition>] inline FromOne() = LanguagePrimitives.GenericOne
    let [<ReflectedDefinition>] inline FromInt32 (i:int) = Alea.CUDA.DevicePrimitive.genericNumberFromInt32 i

