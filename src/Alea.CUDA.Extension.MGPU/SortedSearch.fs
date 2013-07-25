module Alea.CUDA.Extension.MGPU.SortedSearch

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Intrinsics
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTAScan
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.CTAMerge
open Alea.CUDA.Extension.MGPU.CTASortedSearch


type Plan =
    {
        NT : int
        VT : int
    }

let deviceLoadSortedSearch (NT:int) (VT:int) (bounds:int) (indexA:bool) (matchA:bool) (indexB:bool) (matchB:bool) (compOp:IComp<int>) =
    
    let deviceLoad2ToShared = deviceLoad2ToSharedB NT VT (VT + 1) 
    let ctaSortedSearch = ctaSortedSearch NT VT bounds indexA matchA indexB matchB compOp

    <@ fun (range:int4) (a_global:DevicePtr<int>) (aCount:int) (b_global:DevicePtr<int>) (bCount:int) (tid:int) (block:int) (keys_shared:RWPtr<int>) (indices_shared:RWPtr<int>) ->
        let deviceLoad2ToShared = %deviceLoad2ToShared
        let ctaSortedSearch = %ctaSortedSearch

        let a0 = range.x
        let a1 = range.y
        let b0 = range.z
        let b1 = range.w
        let aCount2 = a1 - a0
        let bCount2 = b1 - b0

        let leftA = if (matchB && (bounds = MgpuBoundsLower)) && (a0 > 0) then 1 else 0
        let leftB = if (matchA && (bounds = MgpuBoundsUpper)) && (b0 > 0) then 1 else 0
        let rightA = if a1 < aCount then 1 else 0
        let rightB = if b1 < bCount then 1 else 0

        let aStart = leftA
        let aEnd = aStart + aCount2 + rightA
        let bStart = aEnd + leftB
        let bEnd = bStart + bCount2 + rightB

        deviceLoad2ToShared (a_global + a0 - leftA) aEnd (b_global + b0 - leftB) (bEnd - aEnd) tid keys_shared true

        let extended = 
            let mutable x = (rightA = 1) && (rightB = 1) 
            x <- x && ((not matchA) || (leftB = 1))
            x <- x && ((not matchB) || (leftA = 1))
            x
            
        let matchCount = ctaSortedSearch keys_shared aStart aCount2 aEnd a0 bStart bCount2 bEnd b0 extended tid indices_shared

        matchCount
    @>



let kernelSortedSearch (plan:Plan) (bounds:int) (indexA:int) (matchA:int) (indexB:int) (matchB:int) (compOp:IComp<int>) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = plan.NT * plan.VT

    let indexA = if indexA = 0 then false else true
    let matchA = if matchA = 0 then false else true
    let indexB = if indexB = 0 then false else true
    let matchB = if matchB = 0 then false else true
    
    let capacity, reduce = ctaReduce NT (scanOp ScanOpTypeAdd 0)
    let sharedSize = max capacity (NT * (VT + 1))

    let computeMergeRange = computeMergeRange.Device
    let deviceLoadSortedSearch = deviceLoadSortedSearch NT VT bounds indexA matchA indexB matchB compOp
    let deviceMemToMemLoop = deviceMemToMemLoop NT

    <@ fun (a_global:DevicePtr<int>) (aCount:int) (b_global:DevicePtr<int>) (bCount:int) (mp_global:DevicePtr<int>) (aIndices_global:DevicePtr<int>) (bIndices_global:DevicePtr<int>) -> //(aMatchCount:DevicePtr<int>) (bMatchCount:DevicePtr<int>) ->
        let reduce = %reduce
        let computeMergeRange = %computeMergeRange
        let deviceLoadSortedSearch = %deviceLoadSortedSearch
        let deviceMemToMemLoop = %deviceMemToMemLoop

        let shared = __shared__<int>(sharedSize).Ptr(0)
        let sharedKeys = shared
        let sharedIndices = shared
        let sharedReduce = shared

            
        let tid = threadIdx.x
        let block = blockIdx.x
        let range = computeMergeRange aCount bCount block 0 NV mp_global

            
        let matchCount = deviceLoadSortedSearch range a_global aCount b_global bCount tid block sharedKeys sharedIndices
        let aCount = range.y - range.x
        let bCount = range.w - range.z

        if (indexA || matchA) then
            deviceMemToMemLoop aCount sharedIndices tid (aIndices_global + range.x) true

        if (indexB || matchB) then
            deviceMemToMemLoop bCount (sharedIndices + aCount) tid (bIndices_global + range.z) true

//            if ((matchA || matchB) && (aMatchCount.[0] <> 0 || bMatchCount.[0] <> 0)) then
//                let x = bfi (uint32 matchCount.y) (uint32 matchCount.x) 16u 16u
//                let total = reduce tid x (sharedReduce.Reinterpret<uint32>())
//                if (tid = 0) && (aMatchCount.[0] <> 0) then atomicAdd aMatchCount (0xffff &&& total)
//                if (tid = 0) && (bMatchCount.[0] <> 0) then atomicAdd bMatchCount (total >>> 16)
        @>


type ISortedSearch =
    {                                                                                                                         // cause use until we have support for atomics  
        Action : ActionHint -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> unit //DevicePtr<int> -> DevicePtr<int> -> unit
        NumPartitions : int
    }


let sortedSearch (bounds:int) (typeA:MgpuSearchType) (typeB:MgpuSearchType) (compOp:IComp<int>) = cuda {
    let plan = {NT = 128; VT = 7}
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    let indexA = if (typeA = MgpuSearchTypeIndex) || (typeA = MgpuSearchTypeIndexMatch) then 1 else 0
    let matchA = if (typeA = MgpuSearchTypeMatch) || (typeA = MgpuSearchTypeIndexMatch) then 1 else 0
    let indexB = if (typeB = MgpuSearchTypeIndex) || (typeB = MgpuSearchTypeIndexMatch) then 1 else 0
    let matchB = if (typeB = MgpuSearchTypeMatch) || (typeB = MgpuSearchTypeIndexMatch) then 1 else 0
                

    let! kernelSortedSearch = kernelSortedSearch plan bounds indexA matchA indexB matchB compOp |> defineKernelFuncWithName "ss"
    let! mpp = Search.mergePathPartitions bounds compOp

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelSortedSearch = kernelSortedSearch.Apply m
        let mpp = mpp.Apply m

        fun (aCount:int) (bCount:int) ->
            let numBlocks = divup (aCount + bCount) NV
            let lp = LaunchParam(numBlocks, NT)


            let action (hint:ActionHint) (a_global:DevicePtr<int>) (b_global:DevicePtr<int>) (parts:DevicePtr<int>) (aIndices_global:DevicePtr<int>) (bIndices_global:DevicePtr<int>) = //(aMatchCount:DevicePtr<int>) (bMatchCount:DevicePtr<int>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
                    let mpp = mpp aCount bCount NV 0
                    let partitions = mpp.Action hint a_global b_global parts
                    kernelSortedSearch.Launch lp a_global aCount b_global bCount parts aIndices_global bIndices_global //aMatchCount bMatchCount
                |> worker.Eval
            { Action = action; NumPartitions = numBlocks + 1 } ) }
