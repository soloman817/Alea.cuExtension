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
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.CTAScan
open Alea.CUDA.Extension.MGPU.Search


// this is plan
type Plan = 
    {
        NT : int
        VT : int
    }

let kernelBulkRemove (plan:Plan) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    // @COMMENTS@: well, 'TI means TYPE OF INPUT, 'TR means TYPE OF RESULT, 'TV means TYPE OF VALUE
    // but acturally in bulkRemove, I think 'TI should = 'TR, right? So I change it to be 'T. Tell me
    // if I was wrong that there is a conversion from 'TI to 'TR

    // and I use createSharedExpr to simulate when 'TI <> 'TR, but if you check the code, the :
    //typedef CTAScan<NT, ScanOpAdd> S;
    //union Shared {
    //	int indices[NV];
    //	typename S::Storage scan;
    //};
    //__shared__ Shared shared;
    // and ScanOpAdd is typedef ScanOp<ScanOpTypeAdd, int> ScanOpAdd
    // so it means this scan is just used to calculate index, and to make things easier, we can assume
    // that index type are always int, so here we don't need to use this way to create shared memory
    // once more, why I need use createSharedExpr to create shared memory, is because I want to simulate
    // the union, it has two cases, int for indices, or S::Storage for storage. The reason I must use
    // createSharedExpr (if you check its implementation, you can see that it is most about the alignment)
    // For example, if Storage type is float64, then its size is 8 bytes, but int is 4 bytes, So to 
    // make it work, we should define the shared memory (which will be shared as union) with alignment of 8
    // otherwise it will generate error.

    // but here it is very simple, we only use scan for index calculation, so storage type is int, so
    // no need to use createSharedExpr to do that

    let capacity, scan2 = ctaScan2 NT (scanOp ScanOpTypeAdd 0)
    let sharedSize = max NV capacity
    //let alignOfTI, sizeOfTI = TypeUtil.cudaAlignOf typeof<'TI>, sizeof<'TI>
    //let alignOfTV, sizeOfTV = TypeUtil.cudaAlignOf typeof<'TR>, sizeof<'TR>
    //let sharedAlign = max alignOfTI alignOfTV
    //let sharedSize = max (sizeOfTI * NV) (sizeOfTV * capacity)
    //let createSharedExpr = createSharedExpr sharedAlign sharedSize
    
    let deviceGlobalToReg = deviceGlobalToReg NT VT
    let deviceSharedToReg = deviceSharedToReg NT VT
    let deviceRegToGlobal = deviceRegToGlobal NT VT
    let deviceGather = deviceGather NT VT
        
    <@ fun (source_global:DevicePtr<'T>) (sourceCount:int) (indices_global:DevicePtr<int>) (indicesCount:int) (p_global:DevicePtr<int>) (dest_global:DevicePtr<'T>) ->
        
        let deviceGather = %deviceGather
        
        let deviceGlobalToReg = %deviceGlobalToReg
        let deviceSharedToReg = %deviceSharedToReg
        let deviceRegToGlobal = %deviceRegToGlobal

        let scan = %scan2

        // @COMMENTS@ so now I only need use int directly (but remember, bulkremove is the most simple one,
        // for other algorithms such as reduce, scan, we need use createSharedExpr, please check my reduce
        // example and think again why I need it (hint, for alignment)

        let shared = __shared__<int>(sharedSize).Ptr(0)
        let sharedScan = shared
        let sharedIndices = shared
        //let shared = %(createSharedExpr)
        //let sharedScan = shared.Reinterpret<int>()
        //let sharedIndices = shared.Reinterpret<int>()

        let tid = threadIdx.x
        let block = blockIdx.x
        let gid = block * NV
        
        let mutable sourceCount = sourceCount
        sourceCount <- min NV (sourceCount - gid)

        let mutable source_global = source_global

        // search for begin and end iterators of interval to load
        let p0 = p_global.[block]
        let p1 = p_global.[block + 1]

        // Set the flags to 1. The default is to copy a value
        for i = 0 to VT - 1 do
            let index = NT * i + tid
            if index < sourceCount then sharedIndices.[index] <- 1 else sharedIndices.[index] <- 0
        __syncthreads()

        // Load the indices into register
        let begin' = p0
        let indexCount = p1 - begin'
        let indices = __local__<int>(VT).Ptr(0)
        deviceGlobalToReg indexCount (indices_global + begin') tid indices false

        // Set the counter to 0 for each index we've loaded
        for i = 0 to VT - 1 do
            if (NT * i + tid) < indexCount then
                sharedIndices.[indices.[i] - gid] <- 0
        __syncthreads()

        // Run a raking scan over the flags.  We count the set flags - this is the
        // number of elements to load in per thread
        let mutable x = 0
        for i = 0 to VT - 1 do
            indices.[i] <- sharedIndices.[VT * tid + i]
            x <- x + indices.[i]
        __syncthreads()

        // Run a CTA scan and scatter the gather indices to shared memory
        let mutable s = scan tid x sharedScan
        for i = 0 to VT - 1 do
            if indices.[i] = 1 then                
                sharedIndices.[s] <- VT * tid + i
                s <- s + 1                
        __syncthreads()

        // Load the gather indices into register
        deviceSharedToReg NV sharedIndices tid indices true
        
        // Gather the data into register.  The number of values to copy
        // is sourceCount - indexCount
        source_global <- source_global + gid
        let count = sourceCount - indexCount
        let values = __local__<'T>(VT).Ptr(0)
        deviceGather count source_global indices tid values false

        // Store all the valid registers to dest_global
        deviceRegToGlobal count values tid (dest_global + gid - begin') false  @>

type IBulkRemove<'T> =
    {
        Action : ActionHint -> DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<'T> -> unit        
    }


// @COMMENTS@ : actrually, bulkRemove just need use a comp of int (for index), so you don't
// need to input a indent, right? cause for index, we assumed it is always int. for user
// of bulkremove, input type is 'TI, output type is 'TR, but we don't give it a function
// to convert 'TI to 'TR, so, here we simply use 'T for both
// and when create bsp, we just create comp CompTypeLess 0, and that gives the comp of int
// just used for int cacluation. so again, you don't even need use inline, cause no need.
let bulkRemove() = cuda {
    let plan = { NT = 128; VT = 11 }
    let NV = plan.NT * plan.VT
    
    let! kernelBulkRemove = kernelBulkRemove plan |> defineKernelFunc
    let! bsp = Search.binarySearchPartitions MgpuBoundsLower (comp CompTypeLess 0)

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelBulkRemove = kernelBulkRemove.Apply m
        let bsp = bsp.Apply m
        
        fun (sourceCount:int) (indicesCount:int) ->    
            let numBlocks = divup sourceCount NV
            let lp = LaunchParam(numBlocks, plan.NT)
            let bsp = bsp sourceCount indicesCount NV
            let parts = worker.Malloc(numBlocks + 1)
            
            let action (hint:ActionHint) (source_global:DevicePtr<'T>) (indices_global:DevicePtr<int>) (dest_global:DevicePtr<'T>) =
                let lp = lp |> hint.ModifyLaunchParam
                let partitions = (bsp.Action hint indices_global parts.Ptr)
                kernelBulkRemove.Launch lp source_global sourceCount indices_global indicesCount parts.Ptr dest_global

            { Action = action } ) }
            
