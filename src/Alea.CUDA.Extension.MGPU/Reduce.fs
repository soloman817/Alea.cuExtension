module Alea.CUDA.Extension.MGPU.Reduce

#nowarn "9"
#nowarn "51"

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTAScan

[<Struct;StructLayout(LayoutKind.Sequential, Size=16);Align(16)>]
type Extent16 =
    val dummy : byte

[<Struct;StructLayout(LayoutKind.Sequential, Size=8);Align(8)>]
type Extent8 =
    val dummy : byte

[<Struct;StructLayout(LayoutKind.Sequential, Size=4);Align(4)>]
type Extent4 =
    val dummy : byte

[<Struct;StructLayout(LayoutKind.Sequential, Size=2);Align(2)>]
type Extent2 =
    val dummy : byte

[<Struct;StructLayout(LayoutKind.Sequential, Size=1);Align(1)>]
type Extent1 =
    val dummy : byte

let createSharedExpr (align:int) (size:int) =
    let length = divup size align
    match align with
    | 16 -> <@ __shared__<Extent16>(length).Ptr(0).Reinterpret<byte>() @>
    | 8  -> <@ __shared__<Extent8>(length).Ptr(0).Reinterpret<byte>() @>
    | 4  -> <@ __shared__<Extent4>(length).Ptr(0).Reinterpret<byte>() @>
    | 2  -> <@ __shared__<Extent2>(length).Ptr(0).Reinterpret<byte>() @>
    | 1  -> <@ __shared__<Extent1>(length).Ptr(0).Reinterpret<byte>() @>
    | _ -> failwithf "wrong align of %d" align

type Plan =
    {
        NT : int
        VT: int
    }

let kernelReduce (plan:Plan) (op:IScanOp<'TI, 'TV, 'TR>) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    let capacity, reduce = ctaReduce NT op
    let alignOfTI, sizeOfTI = TypeUtil.cudaAlignOf typeof<'TI>, sizeof<'TI>
    let alignOfTV, sizeOfTV = TypeUtil.cudaAlignOf typeof<'TV>, sizeof<'TV>
    let sharedAlign = max alignOfTI alignOfTV
    let sharedSize = max (sizeOfTI * NV) (sizeOfTV * capacity)
    let createSharedExpr = createSharedExpr sharedAlign sharedSize

    let commutative = op.Commutative
    let identity = op.Identity
    let extract = op.DExtract
    let plus = op.DPlus

    let deviceGlobalToReg = deviceGlobalToReg NT VT

    <@ fun (data_global:DevicePtr<'TI>) (count:int) (task:int2) (reduction_global:DevicePtr<'TV>) ->
        let extract = %extract
        let plus = %plus
        let deviceGlobalToReg = %deviceGlobalToReg
        let reduce = %reduce

        let shared = %(createSharedExpr)
        let sharedReduce = shared.Reinterpret<'TV>()
        let sharedInputs = shared.Reinterpret<'TI>()

        let tid = threadIdx.x
        let block = blockIdx.x
        let first = VT * tid

        let mutable range = computeTaskRangeEx block task NV count

        let mutable total = extract identity -1
        let mutable totalDefined = false

        while range.x < range.y do
            let count2 = min NV (count - range.x)

            let inputs = __local__<'TI>(VT).Ptr(0)
            deviceGlobalToReg count2 (data_global + range.x) tid inputs false

            if commutative then
                for i = 0 to VT - 1 do
                    let index = NT * i + tid
                    if index < count2 then
                        let x = extract inputs.[i] (range.x + index)
                        total <- if i > 0 || totalDefined then plus total x else x
            else 
                // TODO
                ()

            range.x <- range.x + NV
            totalDefined <- true

        if commutative then
            total <- reduce tid total sharedReduce

        if tid = 0 then reduction_global.[block] <- total @>

type IReduce<'TI, 'TV> =
    {
        NumBlocks : int // how may 'TV
        Action : ActionHint -> DevicePtr<'TI> ->DevicePtr<'TV> -> unit
        Result : DevicePtr<'TV> -> 'TV
    }

let reduce (op:IScanOp<'TI, 'TV, 'TR>) = cuda {
    let cutOff = 20000
    let plan1 = { NT = 512; VT = 5 }
    let plan2 = { NT = 128; VT = 9 }
    let! kernelReduce1 = kernelReduce plan1 op |> defineKernelFunc
    let! kernelReduce2 = kernelReduce plan2 op |> defineKernelFunc
    let hplus = op.HPlus

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelReduce1 = kernelReduce1.Apply m
        let kernelReduce2 = kernelReduce2.Apply m

        fun (count:int) ->
            let numBlocks, task, lp, kernelReduce =
                if count < cutOff then
                    let plan = plan1
                    let kernelReduce = kernelReduce1
                    let NV = plan.NT * plan.VT
                    let numTiles = divup count NV
                    let numBlocks = 1
                    let task = int2(numTiles, 1)
                    let lp = LaunchParam(1, plan.NT)
                    numBlocks, task, lp, kernelReduce
                else
                    let plan = plan2
                    let kernelReduce = kernelReduce2
                    let NV = plan.NT * plan.VT
                    let numTiles = divup count NV
                    let numBlocks = min (worker.Device.NumSm * 25) numTiles
                    let task = divideTaskRange numTiles numBlocks
                    let lp = LaunchParam(numBlocks, plan.NT)
                    numBlocks, task, lp, kernelReduce

            let action (hint:ActionHint) (data:DevicePtr<'TI>) (reduction:DevicePtr<'TV>) =
                let lp = lp |> hint.ModifyLaunchParam
                kernelReduce.Launch lp data count task reduction

            let result (reduction:DevicePtr<'TV>) =
                let host = Array.zeroCreate numBlocks
                DevicePtrUtil.Gather(worker, reduction, host, numBlocks)
                host |> Array.reduce hplus

            { NumBlocks = numBlocks; Action = action; Result = result } ) }


        

