module Alea.CUDA.Extension.MGPU.LoadBalance
// NOT IMPLEMENTED YET
open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTALoadBalance

type Plan =
    {
        NT : int
        VT : int
    }

let kernelLoadBalance (plan:Plan) =
    let NT = plan.NT
    let VT = plan.VT

    let ctaLoadBalance = ctaLoadBalance NT VT
    let deviceSharedToGlobal = deviceSharedToGlobal NT VT

    let sharedSize = NT * (VT + 1)
    <@ fun (aCount:int) (b_global:DevicePtr<int>) (bCount:int) (countingItr_global:DevicePtr<int>) (mp_global:DevicePtr<int>) (indices_global:DevicePtr<int>) ->
        let ctaLoadBalance = %ctaLoadBalance
        let deviceSharedToGlobal = %deviceSharedToGlobal

        let indices_shared = __shared__<int>(sharedSize).Ptr(0)

        let mutable aCount = aCount

        let tid = threadIdx.x
        let block = blockIdx.x
        
        let range = ctaLoadBalance aCount b_global bCount block tid countingItr_global mp_global indices_shared false
        aCount <- range.y - range.x

        deviceSharedToGlobal aCount indices_shared tid (indices_global + range.x) false
    @>


type ILoadBalanceSearch =
    {
        Action : ActionHint -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> unit
        NumPartitions : int
    }


let loadBalanceSearch() = cuda {
    let plan = { NT = 128; VT = 7 }
    let NV = plan.NT * plan.VT

    let! kernelLoadBalance = kernelLoadBalance plan |> defineKernelFuncWithName "lbs"
    let! mpp = Search.mergePathPartitions MgpuBoundsUpper (comp CompTypeLess 0)

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelLoadBalance = kernelLoadBalance.Apply m
        let mpp = mpp.Apply m

        fun (aCount:int) (bCount:int) ->
            let numBlocks = divup (aCount + bCount) NV
            let lp = LaunchParam(numBlocks, plan.NT)

            let action (hint:ActionHint) (b_global:DevicePtr<int>) (indices_global:DevicePtr<int>) (countingItr_global:DevicePtr<int>) (mp_global:DevicePtr<int>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
                    let mpp = mpp aCount bCount NV 0
                    let partitions = mpp.Action hint countingItr_global b_global mp_global
                    kernelLoadBalance.Launch lp aCount b_global bCount countingItr_global mp_global indices_global
                |> worker.Eval
            
            { Action = action; NumPartitions = numBlocks + 1 } ) }

