module Alea.cuExtension.MGPU.LoadBalance
// NOT IMPLEMENTED YET
open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension
//open Alea.cuExtension.Util
open Alea.cuExtension.MGPU
//open Alea.cuExtension.MGPU.QuotationUtil
open Alea.cuExtension.MGPU.DeviceUtil
open Alea.cuExtension.MGPU.LoadStore
open Alea.cuExtension.MGPU.CTALoadBalance

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
    <@ fun (aCount:int) (b_global:deviceptr<int>) (bCount:int) (countingItr_global:deviceptr<int>) (mp_global:deviceptr<int>) (indices_global:deviceptr<int>) ->
        let ctaLoadBalance = %ctaLoadBalance
        let deviceSharedToGlobal = %deviceSharedToGlobal

        let indices_shared = __shared__.Array<int>(sharedSize) |> __array_to_ptr

        let mutable aCount = aCount

        let tid = threadIdx.x
        let block = blockIdx.x
        
        let range = ctaLoadBalance aCount b_global bCount block tid countingItr_global mp_global indices_shared false
        aCount <- range.y - range.x

        deviceSharedToGlobal aCount indices_shared tid (indices_global + range.x) false
    @>


//type ILoadBalanceSearch =
//    {
//        Action : ActionHint -> deviceptr<int> -> deviceptr<int> -> deviceptr<int> -> deviceptr<int> -> unit
//        NumPartitions : int
//    }


let loadBalanceSearch() = cuda {
    let plan = { NT = 128; VT = 7 }
    let NV = plan.NT * plan.VT

    let! kernelLoadBalance = kernelLoadBalance plan |> Compiler.DefineKernel //"lbs"
    let! mpp = Search.mergePathPartitions MgpuBoundsUpper (comp CompTypeLess 0)

    return Entry(fun program ->
        let worker = program.Worker
        let kernelLoadBalance = program.Apply kernelLoadBalance
        //let mpp = mpp.Apply m

        fun (aCount:int) (bCount:int) ->
            let numBlocks = divup (aCount + bCount) NV
            let lp = LaunchParam(numBlocks, plan.NT)

            let run (b_global:deviceptr<int>) (indices_global:deviceptr<int>) (countingItr_global:deviceptr<int>) (mp_global:deviceptr<int>) =
                fun () ->
                    
                    let mpp = mpp aCount bCount NV 0
                    let partitions = mpp countingItr_global b_global mp_global
                    kernelLoadBalance.Launch lp aCount b_global bCount countingItr_global mp_global indices_global
                |> worker.Eval
            
            { NumPartitions = numBlocks + 1 } ) }

