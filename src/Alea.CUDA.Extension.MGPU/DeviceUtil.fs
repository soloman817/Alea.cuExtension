module Alea.CUDA.Extension.MGPU.DeviceUtil

// This file maps to the deviceutil.cuh file in mgpu, just some utilities.

open Alea.CUDA

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


// found in mgpudevice.cuh
type MgpuBounds =
    | MgpuBoundsLower
    | MgpuBoundsUpper


type MgpuScanType =
    | MgpuScanTypeExc
    | MgpuScanTypeInc


type MgpuSearchType =
    | MgpuSearchTypeNone
    | MgpuSearchTypeIndex
    | MgpuSearchTypeMatch
    | MgpuSearchTypeIndexMatch


type MgpuJoinKind =
    | MgpuJoinKindInner
    | MgpuJoinKindLeft
    | MgpuJoinKindRight
    | MgpuJoinKindOuter


type MgpuSetOp =
    | MgpuSetOpIntersection
    | MgpuSetOpUnion
    | MgpuSetOpDiff
    | MgpuSetOpSymDiff

