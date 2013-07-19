module Test.Alea.CUDA.Extension.MGPU.Debug

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.CTAScan
//open Alea.CUDA.Extension.MGPU.LoadStore // I put these in here
open Alea.CUDA.Extension.MGPU.CTAMerge
open Test.Alea.CUDA.Extension.MGPU.Util

open NUnit.Framework

let worker = getDefaultWorker()


module MergePartition =
    type Plan =
        {
            NT : int
            Bounds : int
        }

       
    let kernelMergePartition (plan:Plan) (mergeSearch:IMergeSearch<int>) = 
        let NT = plan.NT
        let bounds = plan.Bounds
        let mergePath = mergeSearch.DMergePath
        
        <@ fun (a_global:DevicePtr<int>) (aCount:int) (b_global:DevicePtr<int>) (bCount:int) (nv:int) (coop:int) (mp_global:DevicePtr<int>) (numSearches:int) ->
            let mergePath = %mergePath
            // @COMMENT@
            // uhmm, a stupid error here
            // C++ code: int partition = NT * blockIdx.x + threadIdx.x;
            //let partition = NT * blockIdx.x * threadIdx.x
            let partition = NT * blockIdx.x + threadIdx.x
            if partition < numSearches then                
                let a0 = 0
                let b0 = 0                
                let gid = nv * partition
                // coop always 0 for bulk insert so I deleted that part for testing
                let mp = mergePath (a_global + a0) aCount (b_global + b0) bCount (min gid (aCount + bCount))
                mp_global.[partition] <- mp @>

    type IMergePathPartitions =
        {
            Action : ActionHint -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> unit                
        }                  

    let mergePathPartitions (bounds:int) (compOp:IComp<int>) = cuda {
        let plan = { NT = 64; Bounds = bounds}
        let mergeSearch = mergeSearch bounds compOp
        let! kernelMergePartition = (kernelMergePartition plan mergeSearch) |> defineKernelFuncWithName "mpp"

        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let kernelMergePartition = kernelMergePartition.Apply m

            fun (aCount:int) (bCount:int) (nv:int) (coop:int) ->
                let numPartitions = divup (aCount + bCount) nv
                let numPartitionBlocks = divup (numPartitions + 1) plan.NT
                let lp = LaunchParam(numPartitionBlocks, plan.NT)
            
                let action (hint:ActionHint) (a_global:DevicePtr<int>) (b_global:DevicePtr<int>) (partitionsDevice:DevicePtr<int>) =
                    let lp = lp |> hint.ModifyLaunchParam
                    kernelMergePartition.Launch lp a_global aCount b_global bCount nv coop partitionsDevice (numPartitions + 1)
                
                { Action = action } ) }



let deviceRegToGlobal (NT:int) (VT:int) =
    <@ fun (count:int) (reg:RWPtr<int>) (tid:int) (dest:DevicePtr<int>) (sync:bool) ->
        for i = 0 to VT - 1 do
            let index = NT * i + tid
            if index < count then
                dest.[index] <- reg.[i]
        if sync then __syncthreads() @>


let deviceGlobalToReg (NT:int) (VT:int) =
    <@ fun (count:int) (data:DevicePtr<int>) (tid:int) (reg:RWPtr<int>) (sync:bool) ->
            if count >= NT * VT then
                for i = 0 to VT - 1 do
                    reg.[i] <- data.[NT * i + tid]
            else
                for i = 0 to VT - 1 do
                    let index = NT * i + tid
                    if index < count then reg.[i] <- data.[index]
            if sync then __syncthreads() @>    
        

let deviceTransferMergeValues (NT:int) (VT:int) =
    let deviceRegToGlobal = deviceRegToGlobal NT VT
    <@ fun (count:int) (a_global:DevicePtr<int>) (b_global:DevicePtr<int>) (bStart:int) (indices_shared:RWPtr<int>) (tid:int) (dest_global:DevicePtr<int>) (sync:bool) ->
        let deviceRegToGlobal = %deviceRegToGlobal
        
        // in C++, it is:
        // typedef typename std::iterator_traits<InputIt1>::value_type ValType;
        // ValType values[VT];
        // which means the type is ValType, a value type, so if you want to make it general, this should be generic
        let values = __local__<int>(VT).Ptr(0)
        // here is where I believe the main problem is.  I was able to get my insertions to actually show up by
        // changing the way I casted these.  The result was still incorrect, and the original indices still
        // werent being inserted/copied back to the new array
        //let bOffset = b_global.Handle64 - a_global.Handle64
        //let bOffset = int(bOffset) - bStart

        // @COMMENT@
        //let b_global = DevicePtr<int>((b_global.Handle64 - a_global.Handle64) / int64(sizeof<int>))
        // the c++ code is : 	b_global -= bStart;
        // so b_global is a pointer, while bStart is an int, why you need do that pointer handle stuff?
        let b_global = b_global - bStart
        
        if count >= NT * VT then
            for i = 0 to VT - 1 do
                //let mutable gather = indices_shared.[NT * i + tid]
                //if gather >= bStart then gather <- gather + bOffset
                //values.[i] <- a_global.[gather]
                let gather = indices_shared.[NT * i + tid]
                values.[i] <- if gather < bStart then a_global.[gather] else b_global.[gather]
        else
            for i = 0 to VT - 1 do
                let index = NT * i + tid
//                let mutable gather = indices_shared.[index]
//                if gather >= bStart then gather <- gather + bOffset
//                if index < count then values.[i] <- a_global.[gather]
                let gather = indices_shared.[index]
                if index < count then
                    values.[i] <- if gather < bStart then a_global.[gather] else b_global.[gather]

        if sync then __syncthreads()

        deviceRegToGlobal count values tid dest_global false @>


module BulkInsert =
    type Plan2 =
        {
            NT : int
            VT : int
        }
           
    // @COMMENT@ why not remove op here? because it is used for index calculation
    // and index is always int, so we can directly write it down, like bulkRemove
    let kernelBulkInsert (plan:Plan2) (op:IScanOp<int, int, int>) =
    //let kernelBulkInsert (plan:Plan2) =
        let NT = plan.NT
        let VT = plan.VT
        let NV = NT * VT
            

        let capacity, scan2 = ctaScan2 NT op
        let sharedSize = max NV capacity
        
//        let sharedSize = capacity // Alignment is another source of error
        // If I understand correctly, either the sharedSize must be constrained to NV or NV
        // needs to be the shared size.  Some of his writings on the website suggest NV isn't constant
        // throughout, but others things he says make it seem like it is.  The reason I think NV may change is
        // due to his example on the website.  The code doesn't show it changing anywhere as far as I can tell
        // so I'm not sure??

        // @COMMENT@
        // well, from the C++ code, NV is static, and the C++ code is quite similar to bulkRemove, so I
        // just copy it

        //let NV = sharedSize
        //let NV = capacity
//        let alignOfTI, sizeOfTI = TypeUtil.cudaAlignOf typeof<int>, sizeof<int>
//        let alignOfTV, sizeOfTV = TypeUtil.cudaAlignOf typeof<int>, sizeof<int>
//        let sharedAlign = max alignOfTI alignOfTV
//        let sharedSize = max (sizeOfTI * NV) (sizeOfTV * capacity)
//        let createSharedExpr = createSharedExpr sharedAlign sharedSize
        //let NV = sharedSize

        // ^^^^^^ just trying stuff


        let deviceGlobalToReg = deviceGlobalToReg NT VT
        let computeMergeRange = computeMergeRange.Device
        let deviceTransferMergeValues = deviceTransferMergeValues NT VT

        <@ fun (a_global:DevicePtr<int>) (indices_global:DevicePtr<int>) (aCount:int) (b_global:DevicePtr<int>) (bCount:int) (mp_global:DevicePtr<int>) (dest_global:DevicePtr<int>) ->
            let deviceGlobalToReg = %deviceGlobalToReg
            let computeMergeRange = %computeMergeRange
            let deviceTransferMergeValues = %deviceTransferMergeValues
            let S = %scan2
            
            let tid = threadIdx.x
            let block = blockIdx.x            
            let range = computeMergeRange aCount bCount block 0 NV mp_global

            let shared = __shared__<int>(sharedSize).Ptr(0) 
            //let shared = %(createSharedExpr)
            let sharedScan = shared//.Reinterpret<int>()
            let sharedIndices = shared //.Reinterpret<int>()
                       
            let a0 = range.x
            let a1 = range.y
            let b0 = range.z
            let b1 = range.w

            let aCount = a1 - a0
            let bCount = b1 - b0

            for i = 0 to VT - 1 do
                sharedIndices.[NT * i + tid] <- 0
            __syncthreads()

            let indices = __local__<int>(VT).Ptr(0)
            deviceGlobalToReg aCount (indices_global + a0) tid indices true
        
            for i = 0 to VT - 1 do
                let index = NT * i + tid
                if index < aCount then
                    sharedIndices.[index + indices.[i] - b0] <- 1
            __syncthreads()

            let mutable x = 0
            for i = 0 to VT - 1 do
                indices.[i] <- sharedIndices.[VT * tid + i]
                x <- x + indices.[i]
            __syncthreads()

            let mutable scan = S tid x sharedScan

            for i = 0 to VT - 1 do
                let index = VT * tid + i
                let gather = 
                    if indices.[i] > 0 then 
                        let s = scan
                        scan <- scan + 1
                        s
                    else
                        aCount + index - scan                
                sharedIndices.[index] <- gather
            __syncthreads()
            
            deviceTransferMergeValues (aCount + bCount) (a_global + a0) (b_global + b0) aCount sharedIndices tid (dest_global + a0 + b0) false 
            @>
    
    type IBulkInsert =
        {
            //Action : ActionHint -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> unit
            Action : ActionHint -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> unit
            NumPartitions : int
        }

    let bulkInsert2 (op:IScanOp<int, int, int>) (compOp:IComp<int>) = cuda {
        let plan = { NT = 128; VT = 7 }
        let NV = plan.NT * plan.VT
        
        let! kernelBulkInsert = kernelBulkInsert plan op |> defineKernelFuncWithName "bi"
        let! mpp = MergePartition.mergePathPartitions MgpuBoundsLower compOp

        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let kernelBulkInsert = kernelBulkInsert.Apply m
            let mpp = mpp.Apply m
            
            fun (aCount:int) (bCount:int) ->
                let numBlocks = divup (aCount + bCount) NV
                let lp = LaunchParam(numBlocks, plan.NT)

                let action (hint:ActionHint) (a_global:DevicePtr<int>) (indices_global:DevicePtr<int>) (*(zeroItr:DevicePtr<int>)*) (b_global:DevicePtr<int>) (parts:DevicePtr<int>) (dest_global:DevicePtr<int>) =
                    fun () ->
                        let lp = lp |> hint.ModifyLaunchParam
                        let mpp = mpp aCount bCount NV 0
                        mpp.Action hint indices_global (DevicePtr<int>(0n)) parts // @COMMENT@ try use DevicePtr<int> explictly to indicate it is int type
                        // @COMMENT@
                        // I run C++ program with same aCount=100 and bCount=400
                        // and I found the parts should be [| 0, 100 |]
                        // but on this F# program, sometime it returns correct answer, but most time it gives wrong number
                        // so first I force to set it to [| 0, 100 |], then the result be more stable
                        // so based on this sometime wrong, I guess there might be some sync problem, still checking.
                        // and I first know I should focus on mpp kernel.
                        // Uhmm, I seems fixed the problem it is a type error.
                        // so now I remove these stuff
//                        let host = Array.zeroCreate<int> 2
//                        DevicePtrUtil.Gather(worker, parts, host, 2)
//                        printfn "%A" host
//                        host.[0] <- 0
//                        host.[1] <- 100
//                        DevicePtrUtil.Scatter(worker, host, parts, 2)

                        // that (DevicePtr(0L)) is another thing causing problems.  I've tried creating it / passing it a few different ways
                        // when I had the 9999s showing up in the result like they were supposed to, I was passing it like this (above) instead
                        // of passing via the DArray.createInBlob worker 1.
                        // As far as I understand, they should both be the same thing though correct?
                        // I'm out of ideas for now.. Tomorrow I'm going to just do pointer calculations by hand and hopefully crack it by brute force.
                        // 
                        kernelBulkInsert.Launch lp a_global indices_global aCount b_global bCount parts dest_global
                    |> worker.Eval
            
                { Action = action; NumPartitions = numBlocks + 1 } ) }


    let bInsert (op:IScanOp<int, int,int>) (compOp:IComp<int>) = cuda {
        let! api = bulkInsert2 op compOp

        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (data_A:DArray<int>) (indices:DArray<int>) (data_B:DArray<int>) ->
                let aCount = data_A.Length
                let bCount = data_B.Length                
                let api = api aCount bCount
                
                pcalc {
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions
                    //printfn "api.NumPartitions=%d" api.NumPartitions
                    //let! zeroitr = DArray.createInBlob<int> worker 1
                    let! inserted = DArray.createInBlob worker (aCount + bCount)
                    do! PCalc.action (fun hint -> api.Action hint data_A.Ptr indices.Ptr (*zeroitr.Ptr*) data_B.Ptr partition.Ptr inserted.Ptr)
                    return inserted } ) }


[<Test>]
let ``bulk insert debug`` () =
    let pfunct = BulkInsert.bInsert (scanOp ScanOpTypeAdd 0) (comp CompTypeLess 0)
    let bulkin = worker.LoadPModule(pfunct).Invoke

    let aCount, bCount = 100, 400  // insert 100 elements into a 400 element array
    let hA = Array.init aCount (fun _ -> 9999) // what to insert
    let hB = Array.init bCount (fun i -> i)
    let hI = Array.init aCount (fun _ -> rng.Next(bCount)) |> Array.sort // aCount random indices ranging from 0 to bCount
    printfn "Inserting 9999 at these indicies:"
    printfn "%A" hI
            
    let dResult = pcalc {
        let! dA = DArray.scatterInBlob worker hA
        let! dB = DArray.scatterInBlob worker hB
        let! dI = DArray.scatterInBlob worker hI
        let! dR = bulkin dA dI dB
        let! results = dR.Gather()
        return results } |> PCalc.run
    

    for i = 0 to dResult.Length - 1 do
        printfn "(%d,%d)" i dResult.[i]
    //printfn "%A" dResult

//[<Test>]
//let ``bulk insert debug 2`` () =
//    let pfunct = BulkInsert.bInsert (scanOp ScanOpTypeAdd 0)
//    let bi = worker.LoadPModule(pfunct).Invoke
//    //let count = 400
//    let aCount = 100 //count / 2
//    let bCount = 400 //count - aCount
//
//    let hA = Array.init aCount (fun _ -> 999)
//    let hI = Array.init aCount (fun _ -> rng.Next(bCount)) |> Array.sort
//    let hB = Array.init bCount (fun i -> i)
//    let hZ = Array.init bCount (fun i -> i)
//    
//    let dResult = pcalc {
//        let! dA = DArray.scatterInBlob worker hA
//        let! dI = DArray.scatterInBlob worker hI
//        let! dB = DArray.scatterInBlob worker hB
//        //let! dZ = DArray.scatterInBlob worker hZ
//        let! inserted = bi dA dI dB //dZ
//        
//        let! results = inserted.Gather()
//        return results } |> PCalc.run
//    
//
//    for i = 0 to dResult.Length - 1 do
//        printfn "(%d,%d)" i dResult.[i]
//    //printfn "%A" dResult

