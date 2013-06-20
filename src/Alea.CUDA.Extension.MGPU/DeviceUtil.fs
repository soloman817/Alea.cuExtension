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
    new (x, y) = { x = x; y = y }

type int2 = Int2

[<Struct;Align(8)>]
type Int3 =
    val mutable x : int
    val mutable y : int
    val mutable z : int
    new (x, y, z) = { x = x; y = y; z = z }

type int3 = Int3

[<Struct;Align(8)>]
type Int4 =
    val mutable x : int
    val mutable y : int
    val mutable z : int
    val mutable w : int
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

type IComp<'TC> =
    abstract Identity : 'TC
    abstract Host : ('TC -> 'TC -> int)
    abstract Device : Expr<'TC -> 'TC -> int>
    

type CompType =
    | CompTypeLess
    | CompTypeLess_Equal
    | CompTypeGreater
    | CompTypeGreater_Equal

let inline comp (compType:CompType) (ident:'T) = 
    { new IComp<'T> with
        member this.Identity = ident
        member this.Host = 
            match compType with
            | CompTypeLess -> fun a b -> if a < b then 1 else 0
            | CompTypeLess_Equal -> fun a b -> if a <= b then 1 else 0
            | CompTypeGreater -> fun a b -> if a > b then 1 else 0
            | CompTypeGreater_Equal -> fun a b -> if a >= b then 1 else 0
        member this.Device =
            match compType with
            | CompTypeLess -> <@ fun a b -> if a < b then 1 else 0 @>
            | CompTypeLess_Equal -> <@ fun a b -> if a <= b then 1 else 0 @>
            | CompTypeGreater -> <@ fun a b -> if a > b then 1 else 0 @>
            | CompTypeGreater_Equal -> <@ fun a b -> if a >= b then 1 else 0 @> }