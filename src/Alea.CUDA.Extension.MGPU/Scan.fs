module Alea.CUDA.Extension.MGPU.Scan

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

// this maps to scan.cuh

type Plan =
    {
        NT : int
        VT : int
        ST : MgpuScanType
    }

let kernelParallelScan (plan:Plan) (op:IScanOp<'TI, 'TV, 'TR>) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT
    let ST = plan.ST

    let capacity, scan = ctaScan NT op
    let alignOfTI, sizeOfTI = TypeUtil.cudaAlignOf typeof<'TI>, sizeof<'TI>
    let alignOfTV, sizeOfTV = TypeUtil.cudaAlignOf typeof<'TV>, sizeof<'TV>
    let sharedAlign = max alignOfTI alignOfTV
    let sharedSize = max (sizeOfTI * NV) (sizeOfTV * capacity)
    let createSharedExpr = createSharedExpr sharedAlign sharedSize

    // get neccesary quotations for later doing slicing in quotation
    let commutative = op.Commutative
    let identity = op.Identity
    let extract = op.DExtract
    let combine = op.DCombine
    let plus = op.DPlus
    let deviceGlobalToReg = deviceGlobalToReg NT VT
    let deviceSharedToGlobal = deviceSharedToGlobal NT VT

    <@ fun (cta_global:DevicePtr<'TI>) (count:int) (total_global:DevicePtr<'TV>) (end_global:DevicePtr<'TR>) (dest_global:DevicePtr<'TI>) ->
        let extract = %extract
        let combine = %combine
        let plus = %plus
        let deviceGlobalToReg = %deviceGlobalToReg
        let deviceSharedToGlobal = %deviceSharedToGlobal
        let scan = %scan

        let shared = %(createSharedExpr)
        let sharedScan = shared.Reinterpret<'TV>()
        let sharedInputs = shared.Reinterpret<'TI>()
        let sharedResults = shared.Reinterpret<'TR>()

        let tid = threadIdx.x

        let mutable total : 'TV = extract identity -1
        let mutable totalDefined : bool = false
        let mutable start = 0
        while(start < count) do
            let count2 = min NV (count - start)
            let inputs = __local__<'TI>(VT).Ptr(0)
            let values = __local__<'TV>(VT).Ptr(0)
            deviceGlobalToReg count2 (cta_global + start) tid inputs false
            
            let mutable x = extract identity -1
            for i = 0 to VT-1 do
                let index = VT * tid + i
                if(index < count2) then
                    inputs.[i] <- inputs.[index]
                    values.[i] <- extract inputs.[i] (start + index)
                    x <- if x = i then plus x values.[i] else values.[i]
            __syncthreads()

            let passTotal:RWPtr<'TV> = RWPtr( int64(0))
            let mutable x = scan tid x sharedScan passTotal MgpuScanTypeExc
            if totalDefined then
                x <- plus total x
                total <- plus total passTotal.[0]
            else
                total <- passTotal.[0]

            let mutable x2 : 'TV = 0
            for i = 0 to VT-1 do
                let index = VT * tid + i
                if (index < count2) then
                    if (x2 = i || x2 = tid || totalDefined ) then 
                        x2 <- plus x values.[i] 
                    else 
                        x2 <- values.[i]
                        
                    // For inclusive scan, set the new value then store
                    // For exclusive scan, store the old value then set the new one
                    if(MgpuScanTypeInc = ST) then 
                        x <- x2
                        sharedResults.[index] <- combine inputs.[i] x
                    if(MgpuScanTypeExc = ST) then 
                        x <- x2
            __syncthreads()

            deviceSharedToGlobal count2 sharedResults tid (dest_global + start) true
            start <- start + NV
            totalDefined <- true

        if (total_global.[0] > 0) && (tid = 0) then
            total_global.Ref(0) := total

        if (end_global.[0] > 0) && (tid = 0) then
            end_global.Ref(0) := combine identity total @>


let kernelScanDownsweep (plan:Plan) (op:IScanOp<'TI, 'TV, 'TR>) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT
    let ST = plan.ST

    let capacity, scan = ctaScan NT op
    let alignOfTI, sizeOfTI = TypeUtil.cudaAlignOf typeof<'TI>, sizeof<'TI>
    let alignOfTV, sizeOfTV = TypeUtil.cudaAlignOf typeof<'TV>, sizeof<'TV>
    let sharedAlign = max alignOfTI alignOfTV
    let sharedSize = max (sizeOfTI * NV) (sizeOfTV * capacity)
    let createSharedExpr = createSharedExpr sharedAlign sharedSize

    // get neccesary quotations for later doing slicing in quotation
    let commutative = op.Commutative
    let identity = op.Identity
    let extract = op.DExtract
    let combine = op.DCombine
    let plus = op.DPlus
    let deviceSharedToGlobal = deviceSharedToGlobal NT VT
    let deviceGlobalToShared = deviceGlobalToShared NT VT

    <@ fun (data_global:DevicePtr<'TI>) (count:int) (task:int2) (reduction_global:RWPtr<'T>) (dest_global:DevicePtr<'TI>) (totalAtEnd:bool) ->
        let extract = %extract
        let combine = %combine
        let plus = %plus
        let deviceGlobalToShared = %deviceGlobalToShared
        let deviceSharedToGlobal = %deviceSharedToGlobal
        let scan = %scan

        let shared = %(createSharedExpr)
        let sharedScan = shared.Reinterpret<'TV>()
        let sharedInputs = shared.Reinterpret<'TI>()
        let sharedResults = shared.Reinterpret<'TR>()
    
        let tid = threadIdx.x
        let block = blockIdx.x
        let mutable range = computeTaskRangeEx block task NV count
                
        let mutable next = reduction_global.[block]
        let mutable nextDefined = false
        if block <> 0 then nextDefined <- true

        while (range.x < range.y) do
            let count2 = min NV (count - range.x)
            
            deviceGlobalToShared count2 (data_global + range.x) tid sharedInputs false

            let inputs = __local__<'TI>(VT).Ptr(0)
            let values = __local__<'TV>(VT).Ptr(0)
            let mutable x = extract identity -1

            for i = 0 to VT - 1 do
                let index = VT * tid + i
                if index < count2 then
                    inputs.[i] <- sharedInputs.[index]
                    values.[i] <- extract inputs.[i] (range.x + index)
                    if x = i then x <- plus x values.[i] else x <- values.[i]
            __syncthreads()

            let passTotal:RWPtr<'TV> = RWPtr( int64(0))
            let mutable x = scan tid x sharedScan passTotal MgpuScanTypeExc
            if nextDefined then
                x <- plus next x
                next <- plus next passTotal.[0]
            else
                next <- passTotal.[0]

            let mutable x2 : 'TV = 0
            for i = 0 to VT - 1 do
                let index = VT * tid + i
                if index < count2 then
                    if (x2 = i || x2 = tid || nextDefined) then
                        x2 <- plus x values.[i]
                    else
                        x2 <- values.[i]

                    // For inclusive scan, set the new value then store.
                    // For exclusive scan, store the old value then set the new one
                    if (MgpuScanTypeInc = ST) then
                        x <- x2
                        sharedResults.[index] <- combine inputs.[i] x
                    if (MgpuScanTypeExc = ST) then
                        x <- x2
            __syncthreads()

            deviceSharedToGlobal count2 sharedResults tid (dest_global + range.x) false
            range.x <- range.x + NV
            nextDefined <- true
               
        if ((totalAtEnd && (block <> 0)) = (((gridDim.x - 1) <> 0) && (tid <> 0))) then
            dest_global.[count] <- combine identity next @>


type IScan<'TI, 'TV> =
    {
        Action : ActionHint -> DevicePtr<'TI> -> DevicePtr<'TV> -> DevicePtr<'TV> -> DevicePtr<'TV>  -> DevicePtr<'TV> -> unit
        Result : 'TV[] -> 'TV
    }


let scan (op:IScanOp<'TI, 'TV, 'TR>) = cuda {
    let cutOff = 20000
    // count < cutOff, do parallel scan
    let psPlan = { NT = 512; VT = 3; ST = MgpuScanTypeExc} 
    
    // count >= cutOff, do parallel raking reduce as an upsweep, then
    // do parallel latency-oriented scan to reduce the spine of the 
    // raking reduction, then do a raking scan as downsweep
    let rrUpsweepPlan : Reduce.Plan = { NT = 128; VT = 7 }
    let plosPlan = { NT = 256; VT = 3; ST = MgpuScanTypeExc}
    let rsDownsweepPlan = plosPlan

    let! kernelPS = kernelParallelScan psPlan op |> defineKernelFunc
    let! kernelRRUpsweep = Reduce.kernelReduce rrUpsweepPlan op |> defineKernelFunc
    let! kernelPLOS = kernelParallelScan plosPlan op |> defineKernelFunc
    let! kernelRSDownsweep = kernelParallelScan rsDownsweepPlan op |> defineKernelFunc

    let hplus = op.HPlus

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelPS = kernelPS.Apply m
        let kernelRRUpsweep = kernelRRUpsweep.Apply m
        let kernelPLOS = kernelPLOS.Apply m
        let kernelRSDownsweep = kernelRSDownsweep.Apply m

        fun (count:int) ->
                      
            let result (reduction:'TV[]) =
                reduction |> Array.reduce hplus

            let action (hint:ActionHint) (data:DevicePtr<'TI>) (reduction:DevicePtr<'TV>) (total:DevicePtr<'TV>) (end':DevicePtr<'TR>) (dest:DevicePtr<'TI>) =
                if count < cutOff then
                    let plan = psPlan
                    let NV = plan.NT * plan.VT
                    let numTiles = divup count NV
                    let task = int2(numTiles, 1)
                    let lp = LaunchParam(1, plan.NT) |> hint.ModifyLaunchParam
                    kernelPS.Launch lp data count total end' dest
                else
                    let plan = rrUpsweepPlan
                    let NV = plan.NT * plan.VT
                    let numTiles = divup count NV
                    let numBlocks = min (worker.Device.NumSm * 25) numTiles
                    let task = divideTaskRange numTiles numBlocks
                    let lp = LaunchParam(numBlocks, plan.NT) |> hint.ModifyLaunchParam
                    kernelRRUpsweep.Launch lp data count task reduction

                    let plan = plosPlan
                    let NV = plan.NT * plan.VT
                    let numTiles = divup count NV
                    let numBlocks = min (worker.Device.NumSm * 25) numTiles
                    let task = divideTaskRange numTiles numBlocks
                    let lp = LaunchParam(numBlocks, plan.NT) |> hint.ModifyLaunchParam
                    kernelPLOS.Launch lp data count total end' dest

                    // if total ...

                    kernelRSDownsweep.Launch lp data count total end' dest
                    
            { IScan.Action = action; Result = result } ) }