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
        ST : int
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
    
    let deviceGlobalToShared = deviceGlobalToShared NT VT
    let deviceSharedToGlobal = deviceSharedToGlobal NT VT
    

    <@ fun (cta_global:DevicePtr<'TI>) (count:int) (total_global:DevicePtr<'TV>) (end_global:DevicePtr<'TR>) (dest_global:DevicePtr<'TV>) ->
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
        
        let mutable total = extract identity -1
        let mutable totalDefined = false
        let mutable start = 0G

        
        while(start < count) do
            // load data in to shared memory
            let count2 = min NV (count - start)
            deviceGlobalToShared count2 (cta_global + start) tid sharedInputs doSync
                        
            // Transpose data into register in thread order.  Reduce terms serially
            let inputs = __local__<'TI>(VT).Ptr(0)
            let values = __local__<'TV>(VT).Ptr(0)
            let mutable x = extract identity -1
            for i = 0 to VT - 1 do
                let index = VT * tid + i
                if(index < count2) then
                    inputs.[i] <- sharedInputs.[index]
                    values.[i] <- extract inputs.[i] (start + index)
                    x <- if i <> 0 then plus x values.[i] else values.[i]                    
            __syncthreads()
            
            let passTotal = __local__<'T>(1).Ptr(0)
            let mutable x = scan tid x sharedScan passTotal ExclusiveScan
            
            if totalDefined then
                x <- plus total x
                total <- plus total passTotal.[0]
            else
                total <- passTotal.[0]

            let mutable x2 = x
            for i = 0 to VT-1 do
                let index = VT * tid + i
                if (index < count2) then
                    if ((i > 0 || tid > 0 ) || totalDefined) then                     
                        x2 <- plus x values.[i] 
                    else 
                        x2 <- values.[i]                        
//                  For inclusive scan, set the new value then store
//                  For exclusive scan, store the old value then set the new one
                if(ST = InclusiveScan) then 
                        x <- x2
                sharedResults.[index] <- combine inputs.[i] x
                if(ST = ExclusiveScan) then 
                        x <- x2
            __syncthreads()
            
            deviceSharedToGlobal count2 sharedResults tid (dest_global + start) dontSync
            start <- start + NV
            totalDefined <- true

        if (total_global.Handle64 <> 0L) && (tid = 0) then
            total_global.[0] <- total
        if (end_global.Handle64 <> 0L) && (tid = 0) then
            end_global.[0] <- combine identity total 
            @>


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

    <@ fun (data_global:DevicePtr<'TI>) (count:int) (task:int2) (reduction_global:DevicePtr<'T>) (dest_global:DevicePtr<'TI>) (totalAtEnd:int) ->
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
        let mutable nextDefined = if block <> 0 then true else false
                
        while (range.x < range.y) do
            let count2 = min NV (count - range.x)
            
            // Load from global to shared memory
            deviceGlobalToShared count2 (data_global + range.x) tid sharedInputs 0

            let inputs = __local__<'TI>(VT).Ptr(0)
            let values = __local__<'TV>(VT).Ptr(0)
            let mutable x = extract identity -1

            for i = 0 to VT - 1 do
                let index = VT * tid + i
                if index < count2 then
                    inputs.[i] <- sharedInputs.[index]
                    values.[i] <- extract inputs.[i] (range.x + index)
                    if i > 0 then x <- plus x values.[i] else x <- values.[i]
            __syncthreads()

            let passTotal = __local__<'T>(1).Ptr(0)
            let mutable x = scan tid x sharedScan passTotal ExclusiveScan

            if nextDefined then
                x <- plus next x
                next <- plus next passTotal.[0]
            else
                next <- passTotal.[0]

            for i = 0 to VT - 1 do
                let index = VT * tid + i
                if index < count2 then
                    let x2 =
                        // If this is not the first element in the scan, add x values.[i] into x
                        // Otherwise initialize x to values.[i]
                        if ((i > 0 || tid > 0) || nextDefined) then
                            plus x values.[i]
                        else
                            values.[i]
                    // For inclusive scan, set the new value then store.
                    // For exclusive scan, store the old value then set the new one
                    if (ST = InclusiveScan) then
                        x <- x2
                    sharedResults.[index] <- combine inputs.[i] x
                    if (ST = ExclusiveScan) then
                        x <- x2
            __syncthreads()

            deviceSharedToGlobal count2 sharedResults tid (dest_global + range.x) 0
            range.x <- range.x + NV
            nextDefined <- true
                                
        // need totalAtEnd
        // infact, this usage of int as bool is not effiecient, you should also check xor bitwise operator
        let a = if (block <> 0) && (totalAtEnd <> 0) then 1 else 0
        let b = (gridDim.x - 1) > 0
        let c = tid = 0
        if (a = (if (b && c) then 1 else 0)) then
            dest_global.[count] <- combine identity next @>
        

type IScan<'TI, 'TV, 'TR> =
    {
        NumBlocks: int
        Action : ActionHint -> DevicePtr<'TI> -> DevicePtr<'TV> -> DevicePtr<'TR> -> unit
        //Result : 'TI -> 'TV[] -> 'TV[]
        //Total : 'TR
    }


let scan (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) = cuda {
    let cutOff = 20000
    // count < cutOff, do parallel scan
    let psPlan = { NT = 512; VT = 3; ST = mgpuScanType} 
        
    // count >= cutOff, do parallel raking reduce as an upsweep, then
    // do parallel latency-oriented scan to reduce the spine of the 
    // raking reduction, then do a raking scan as downsweep
    let rrUpsweepPlan : Reduce.Plan = { NT = 128; VT = 7 }
    let plosPlan = { NT = 256; VT = 3; ST = ExclusiveScan}
    //let plosPlan = { NT = 256; VT = 3; ST = 0}
    let rsDownsweepPlan = {NT = 256; VT = 3; ST = mgpuScanType}

    let kernelPS = kernelParallelScan psPlan op
    //printfn "1111"
    //printfn "%A" kernelPS
    let! kernelPS = kernelPS |> defineKernelFunc

    let kernelRRUpsweep = Reduce.kernelReduce rrUpsweepPlan op
    //printfn "2222"
    //printfn "%A" kernelRRUpsweep
    let! kernelRRUpsweep = kernelRRUpsweep |> defineKernelFunc

    let kernelPLOS = kernelParallelScan plosPlan op
    //printfn "3333"
    //printfn "%A" kernelPLOS
    let! kernelPLOS = kernelPLOS |> defineKernelFunc

    let kernelRSDownsweep = kernelScanDownsweep rsDownsweepPlan op
    //printfn "4444"
    //printfn "%A" kernelRSDownsweep
    let! kernelRSDownsweep = kernelRSDownsweep |> defineKernelFunc
    
    let hplus = op.HPlus
    
    
    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let numSm = worker.Device.NumSm
        let kernelPS = kernelPS.Apply m
        let kernelRRUpsweep = kernelRRUpsweep.Apply m
        let kernelPLOS = kernelPLOS.Apply m
        let kernelRSDownsweep = kernelRSDownsweep.Apply m
        fun (count:int) ->
            let numBlocks = 1
            

            let action (hint:ActionHint) (data:DevicePtr<'TI>) (total:DevicePtr<'TV>) (dest:DevicePtr<'TV>) =
                if count < cutOff then
                    //printfn "branch1, kernelParallelScan"
                    
                    let plan = psPlan
                    let NV = plan.NT * plan.VT
                    let numTiles = divup count NV
                    let task = int2(numTiles, 1)
                    let lp = LaunchParam(numBlocks, plan.NT) |> hint.ModifyLaunchParam
                    let totalDevice = worker.Malloc(1).Ptr
                    let total = if total.Handle <> 0n then totalDevice else DevicePtr<'TV>(0n)
                    //let total = if total.Handle <> 0n then totalDevice else total
                    let end' = if totalAtEnd = 1 then dest + count else DevicePtr<'TR>(0n)
                    kernelPS.Launch lp data count total end' dest                    
                else
                    //printfn "branch2"
                    let plan = rrUpsweepPlan
                    let NV = plan.NT * plan.VT
                    let numTiles = divup count NV
                    let numBlocks = min (numSm * 25) numTiles
                    let task = divideTaskRange numTiles numBlocks
                    let reductionDevice = worker.Malloc(numBlocks + 1)
                    let totalDevice = if total.Handle > 0n then reductionDevice.Ptr + numBlocks else DevicePtr<'TV>(0n)
                    let lp = LaunchParam(numBlocks, plan.NT) |> hint.ModifyLaunchParam
                    kernelRRUpsweep.Launch lp data count task reductionDevice.Ptr

                    let plan = plosPlan
                    let NV = plan.NT * plan.VT
                    let numTiles = divup count NV
                    // need numBlocks for reduction, so we dont update it here
                    let task = divideTaskRange numTiles (min (numSm * 25) numTiles)
                    let lp = LaunchParam(numBlocks, plan.NT) |> hint.ModifyLaunchParam
                    kernelPLOS.Launch lp reductionDevice.Ptr numBlocks totalDevice (worker.Malloc(1).Ptr) reductionDevice.Ptr
                                        
                    kernelRSDownsweep.Launch lp data count task reductionDevice.Ptr dest totalAtEnd
                    
            { NumBlocks = numBlocks; IScan.Action = action; (*Result = result; Total = hTotal*) } ) }