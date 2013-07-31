module Alea.CUDA.Extension.MGPU.DeviceUtil

// This file maps to the deviceutil.cuh file in mgpu, just some utilities.

open Alea.CUDA
open Microsoft.FSharp.Quotations

let [<ReflectedDefinition>] WARP_SIZE = 32
let [<ReflectedDefinition>] LOG_WARP_SIZE = 5

[<Struct;Align(8)>]
type Int2 =
    val mutable x : int
    val mutable y : int
    [<ReflectedDefinition>] // @HERE!!@
    new (x, y) = { x = x; y = y }

type int2 = Int2

[<Struct;Align(8)>]
type UInt2 =
    val mutable x : uint32
    val mutable y : uint32
    [<ReflectedDefinition>]
    new (x, y) = { x = x; y = y }

type uint2 = UInt2


[<Struct;Align(8)>]
type Int3 =
    val mutable x : int
    val mutable y : int
    val mutable z : int
    [<ReflectedDefinition>] // @HERE!!@
    new (x, y, z) = { x = x; y = y; z = z }

type int3 = Int3

[<Struct;Align(8)>]
type Int4 =
    val mutable x : int
    val mutable y : int
    val mutable z : int
    val mutable w : int
    [<ReflectedDefinition>] // @HERE!!@
    new (x, y, z, w) = { x = x; y = y; z = z; w = w }
    
type int4 = Int4


let divideTaskRange (numItems:int) (numWorkers:int) =
    let quot = numItems / numWorkers
    let rem = numItems % numWorkers
    int2(quot, rem)

// this function will be called inside kernel, so it need 
// reflected definition to generate quotation.
[<ReflectedDefinition>]
let computeTaskRange (block:int) (task:int2) =
    let mutable range = int2()
    range.x <- task.x * block
    range.x <- range.x + (min block task.y)
    range.y <- range.x + task.x + (if block < task.y then 1 else 0)
    range

[<ReflectedDefinition>]
let computeTaskRangeEx (block:int) (task:int2) (blockSize:int) (count:int) =
    let mutable range = computeTaskRange block task
    range.x <- range.x * blockSize
    range.y <- min count (range.y * blockSize)
    range


// @COMMENTS@ : as I said, bool just cannot be used as argument of a kernel function,
// but it is quite ok to be used as lambda argument, so here I changed the return type
// to be bool
type IComp<'TC> =
    abstract Identity : 'TC
    abstract Host : ('TC -> 'TC -> bool)
    abstract Device : Expr<'TC -> 'TC -> bool>
    

type CompType =
    | CompTypeLess
    | CompTypeLess_Equal
    | CompTypeGreater
    | CompTypeGreater_Equal

let comp (compType:CompType) (ident:'T) = 
    { new IComp<'T> with
        member this.Identity = ident
        // @COMMENTS@ : after changed the return type to be bool, it is easy to write here
        member this.Host = 
            match compType with
            | CompTypeLess -> fun a b -> a < b
            | CompTypeLess_Equal -> fun a b -> a <= b
            | CompTypeGreater -> fun a b -> a > b
            | CompTypeGreater_Equal -> fun a b -> a >= b
        member this.Device =
            match compType with
            | CompTypeLess -> <@ fun a b -> a < b @>
            | CompTypeLess_Equal -> <@ fun a b -> a <= b @>
            | CompTypeGreater -> <@ fun a b -> a > b @>
            | CompTypeGreater_Equal -> <@ fun a b -> a >= b @> }


let [<ReflectedDefinition>] swap a b =
    let c = a
    let mutable a = a
    let mutable b = b
    a <- b
    b <- c

type MgpuSearchType =
    | MgpuSearchTypeNone
    | MgpuSearchTypeIndex
    | MgpuSearchTypeMatch
    | MgpuSearchTypeIndexMatch