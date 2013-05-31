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


//[<ReflectedDefinition>] let Comp (a:'T) (b:'T) = if a <> b then 0 else 1




//type IComp<'T> =
//    abstract HLess : ('T -> 'T -> int)
//    abstract DLess : Expr<'T -> 'T -> int>
//    abstract HLess_Equal : ('T -> 'T -> int)
//    abstract DLess_Equal : Expr<'T -> 'T -> int>
//    abstract HGreater : ('T -> 'T -> int)
//    abstract DGreater : Expr<'T -> 'T -> int>
//    abstract HGreater_Equal : ('T -> 'T -> int)
//    abstract DGreater_Equal : Expr<'T -> 'T -> int>

type IComp<'T> =
    abstract Host : ('T -> 'T -> int)
    abstract Device : Expr<'T -> 'T -> int>

type CompType =
    | CompTypeLess
    | CompTypeLess_Equal
    | CompTypeGreater
    | CompTypeGreater_Equal

//let inline comp = 
//    { new IComp<'T> with
//        member this.HLess = fun a b -> if a < b then 1 else 0
//        member this.DLess = <@ fun a b -> if a < b then 1 else 0 @>
//        member this.HLess_Equal = fun a b -> if a <= b then 1 else 0
//        member this.DLess_Equal = <@ fun a b -> if a <= b then 1 else 0 @>
//        member this.HGreater = fun a b -> if a > b then 1 else 0
//        member this.DGreater = <@ fun a b -> if a > b then 1 else 0 @>
//        member this.HGreater_Equal = fun a b -> if a >= b then 1 else 0
//        member this.DGreater_Equal = <@ fun a b -> if a >= b then 1 else 0 @> }

let inline comp (compType:CompType) = 
    { new IComp<'T> with
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