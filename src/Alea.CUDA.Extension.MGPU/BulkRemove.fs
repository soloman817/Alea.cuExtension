module Alea.CUDA.Extension.MGPU.BulkRemove

// this maps to bulkremove.cuh

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTAScan
open Alea.CUDA.Extension.MGPU.Reduce


// this is plan
type Plan = 
    {
        NT : int
        VT : int
    }

let kernelBulkRemove (plan:Plan) (op:IScanOp<'TI, 'TV, 'TR>) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    let capacity, scan = ctaScan NT op
    let alignOfTI, sizeOfTI = TypeUtil.cudaAlignOf typeof<'TI>, sizeof<'TI>
    let alignOfTV, sizeOfTV = TypeUtil.cudaAlignOf typeof<'TV>, sizeof<'TV>
    let sharedAlign = max alignOfTI alignOfTV
    let sharedSize = max (sizeOfTI * NV) (sizeOfTV * capacity)
    let createSharedExpr = createSharedExpr sharedAlign sharedSize

    let commutative = op.Commutative
    let identity = op.Identity
    let extract = op.DExtract
    let combine = op.DCombine
    let plus = op.DPlus
    let deviceGlobalToReg = deviceGlobalToReg NT VT
    let deviceSharedToGlobal = deviceSharedToGlobal NT VT
    let deviceGather = deviceGather NT VT
    let deviceRegToGlobal = deviceRegToGlobal NT VT

    <@ fun (source_global:DevicePtr<'TI>) (sourceCount:int) (indices_global:DevicePtr<'TI>) (indicesCount:int) (p_global:RWPtr<'TV>) (dest_global:DevicePtr<'TV>) ->
        let extract = %extract
        let combine = %combine
        let plus = %plus
        let deviceGather = %deviceGather
        let deviceRegToGlobal = %deviceRegToGlobal
        let deviceGlobalToReg = %deviceGlobalToReg
        let deviceSharedToGlobal = %deviceSharedToGlobal
        let scan = %scan

        let shared = %(createSharedExpr)
        let sharedScan = shared.Reinterpret<'TV>()
        let sharedIndices = shared.Reinterpret<'TI>()
        let sharedResults = shared.Reinterpret<'TR>()

        let tid = threadIdx.x
        let block = blockIdx.x
        let gid = block * NV
        let sourceCount = min NV (sourceCount - gid)

        let p0 = p_global.[block]
        let p1 = p_global.[block + 1]

        for i = 0 to VT - 1 do
            let index = NT * i + tid
            if index < sourceCount then sharedIndices.[index] <- 1
        __syncthreads()

        let begin' = p0
        let indexCount = p1 - begin'
        let indices = __local__<'TI>(VT).Ptr(0)
        deviceGlobalToReg indexCount (indices_global + begin') tid indices 0

        for i = 0 to VT - 1 do
            if (NT * i + tid) < indexCount then
                sharedIndices.[indices.[i] - gid] <- 0
        __syncthreads()

        let mutable x = 0
        for i = 0 to VT - 1 do
            indices.[i] <- sharedIndices.[VT * tid + i]
            x <- x + indices.[i]
        __syncthreads()

        let passTotal = __local__<'TV>(1).Ptr(0)
//        let mutable s = scan tid x sharedScan passTotal ExclusiveScan
        let mutable s = scan tid x sharedScan passTotal 0
        for i = 0 to VT - 1 do
            if indices.[i] > 0 then
                s <- s + 1
                sharedIndices.[s] <- VT * tid + i
        __syncthreads()

        // should be shared to reg here
        // need to fix these
        deviceGlobalToReg NV sharedIndices tid indices 0

        let source_global = source_global + gid
        let count = sourceCount - indexCount
        let values = __local__<'TV>(VT).Ptr(0)
        deviceGather count source_global indices tid values 0

        deviceRegToGlobal count values tid (dest_global + gid - begin') @>


//let bulkRemove =
//    let plan = { NT = 128; VT = 11 }
