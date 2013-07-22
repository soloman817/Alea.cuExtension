module Test.Alea.CUDA.Extension.MGPU.Debug

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.CTAScan
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTAMerge
open Test.Alea.CUDA.Extension.MGPU.Util

open NUnit.Framework

let worker = getDefaultWorker()

let aib count =
    let aCount = count / 2
    let bCount = count - aCount
    aCount,bCount

let hostBulkInsert (dataA:int[]) (indices:int[]) (dataB:int[]) =
    let result : int[] = Array.zeroCreate (dataA.Length + dataB.Length)
    Array.blit dataB 0 result 0 indices.[0]
    Array.set result indices.[0] dataA.[0]
    for i = 1 to indices.Length - 1 do
        Array.blit dataB indices.[i - 1] result (indices.[i - 1] + i) (indices.[i] - indices.[i - 1])
        Array.set result (indices.[i] + i) dataA.[i]
    let i = indices.Length - 1
    Array.blit dataB indices.[i] result (indices.[i] + i + 1) (result.Length - (indices.[i] + i + 1))
    result

module MergePartition =
    type Plan =
        {
            NT : int
            Bounds : int
        }
       
    let kernelMergePartition() = 
        let NT = 64
        let mergePath = (mergeSearch MgpuBoundsLower (comp CompTypeLess 0)).DMergePath
        
        <@ fun (a_global:DevicePtr<int>) (aCount:int) (b_global:DevicePtr<int>) (bCount:int) (nv:int) (coop:int) (mp_global:DevicePtr<int>) (numSearches:int) ->
            let mergePath = %mergePath
            
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
            //Action : ActionHint -> DevicePtr<int> -> DevicePtr<int> -> unit
        }                  

    let mergePathPartitions = cuda {
        let bounds = MgpuBoundsLower
        let plan = { NT = 64; Bounds = bounds}
        let! kernelMergePartition = kernelMergePartition() |> defineKernelFuncWithName "mpp"

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

    let mPP() = cuda {
        let! api = mergePathPartitions

        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m

            fun (aCount:int) (bCount:int) (nv:int) (coop:int) (aGlobal:DArray<int>) ->                                                                
                pcalc {
                    let api = api aCount bCount nv coop
                    let NT = 64
                    let numPartitions = divup (aCount + bCount) nv
                    let numPartitionBlocks = divup (numPartitions + 1) NT

                    let bGlobalHost = Array.init<int> bCount (fun i -> i)
                    let! bGlobal = DArray.scatterInBlob worker bGlobalHost
                    //let zeroItr = DevicePtr<int>(0n)
                    //let zeroItr = worker.Malloc<int>([|0|])

                    let! parts = DArray.createInBlob<int> worker (numPartitions + 1)
                    do! PCalc.action (fun hint ->
                        api.Action hint aGlobal.Ptr bGlobal.Ptr parts.Ptr
                        ())

                    return parts } ) }

module BulkInsert =
    type Plan2 =
        {
            NT : int
            VT : int
        }
           
    // @COMMENT@ why not remove op here? because it is used for index calculation
    // and index is always int, so we can directly write it down, like bulkRemove
    let kernelBulkInsert (plan:Plan2) =
        let NT = plan.NT
        let VT = plan.VT
        let NV = NT * VT
            

        let capacity, scan2 = ctaScan2 NT (scanOp ScanOpTypeAdd 0)
        let sharedSize = max NV capacity
        
        let deviceGlobalToReg = deviceGlobalToReg NT VT
        let computeMergeRange = computeMergeRange.Device
        let deviceTransferMergeValues = deviceTransferMergeValuesA NT VT

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
            Action : ActionHint -> DevicePtr<int> -> DevicePtr<int> ->  DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> unit
            NumPartitions : int
        }

    let bulkInsert2() = cuda {
        let plan = { NT = 128; VT = 7 }
        let NV = plan.NT * plan.VT
        
        let! kernelBulkInsert = kernelBulkInsert plan |> defineKernelFuncWithName "bi"
        let! mpp = MergePartition.mergePathPartitions

        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let kernelBulkInsert = kernelBulkInsert.Apply m
            let mpp = mpp.Apply m
            
            fun (aCount:int) (bCount:int) ->
                let numBlocks = divup (aCount + bCount) NV
                let lp = LaunchParam(numBlocks, plan.NT)
                
                let action (hint:ActionHint) (a_global:DevicePtr<int>) (indices_global:DevicePtr<int>) (zeroItr:DevicePtr<int>) (b_global:DevicePtr<int>) (parts:DevicePtr<int>) (dest_global:DevicePtr<int>) =
                    fun () ->
                        let lp = lp |> hint.ModifyLaunchParam
                        let mpp = mpp aCount bCount NV 0
                        let partitions = mpp.Action hint indices_global zeroItr parts 
                        kernelBulkInsert.Launch lp a_global indices_global aCount b_global bCount parts dest_global
                    |> worker.Eval
            
                { Action = action; NumPartitions = numBlocks + 1 } ) }


    let bInsert() = cuda {
        let! api = bulkInsert2()

        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (aCount:int) (bCount:int) ->
                pcalc {
                    let sequence = Array.init bCount (fun i -> i)
                    
                    let insert (data_A:DArray<int>) (indices:DArray<int>) (data_B:DArray<int>) (inserted:DArray<int>) =
                        pcalc { 
                            let api = api aCount bCount
                            let! counter = DArray.scatterInBlob worker sequence
                            //let zeroItr = worker.Malloc<int>([|0|])
                            let! partition = DArray.createInBlob<int> worker api.NumPartitions                          
                            do! PCalc.action (fun hint -> api.Action hint data_A.Ptr indices.Ptr counter.Ptr data_B.Ptr partition.Ptr inserted.Ptr) }
                    return insert } ) }


//[<Test>]
//let ``bulk insert debug`` () =
//    let pfunct = BulkInsert.bInsert (scanOp ScanOpTypeAdd 0) (comp CompTypeLess 0)
//    let bulkin = worker.LoadPModule(pfunct).Invoke
//
//    let aCount, bCount = 100, 400  // insert 100 elements into a 400 element array
//    let hA = Array.init aCount (fun _ -> 9999) // what to insert
//    let hB = Array.init bCount (fun i -> i)
//    let hI = Array.init aCount (fun _ -> rng.Next(bCount)) |> Array.sort // aCount random indices ranging from 0 to bCount
//    printfn "Inserting 9999 at these indicies:"
//    printfn "%A" hI
//            
//    let dResult = pcalc {
//        let! dA = DArray.scatterInBlob worker hA
//        let! dB = DArray.scatterInBlob worker hB
//        let! dI = DArray.scatterInBlob worker hI
//        let! dR = bulkin dA dI dB
//        let! results = dR.Gather()
//        return results } |> PCalc.run
//    
//
//    for i = 0 to dResult.Length - 1 do
//        printfn "(%d,%d)" i dResult.[i]
//    

[<Test>]
let ``bulk insert debug mem debug`` () =
    let count = 1024000
    let aCount, bCount = aib count
    //let bCount = count
    //let aCount = count / 3
    printfn "count = %d" count
    printfn "aCount = %d\tbCount = %d" aCount bCount
    let hB = Array.init bCount (fun i -> i)
    let hI = Array.init aCount (fun _ -> rng.Next(bCount)) |> Array.sort
    let hA = Array.init aCount (fun _ -> 2048)

    let pfunct = BulkInsert.bInsert()
    let bulkin = worker.LoadPModule(pfunct).Invoke

    let dResults = pcalc {
        let! dA = DArray.scatterInBlob worker hA
        let! dI = DArray.scatterInBlob worker hI
        let! dB = DArray.scatterInBlob worker hB
        let! dR = DArray.createInBlob worker (aCount + bCount)
        let! insert = bulkin aCount bCount
        do! insert dA dI dB dR
        let! results = dR.Gather()
        return results } |> PCalc.run

    //printfn "%A" dResult
    let hResults = hostBulkInsert hA hI hB
    (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
    
    printfn "%A" dResults
    

[<Test>]
let ``MPP debug mem debug`` () =
    
    let aCount, bCount = 100,400
    //let bCount = count
    //let aCount = count / 3
    //printfn "count = %d" count
    printfn "aCount = %d\tbCount = %d" aCount bCount
    let hB = Array.init bCount (fun i -> i)
    let hI = Array.init aCount (fun _ -> rng.Next(bCount)) |> Array.sort
    let hA = Array.init aCount (fun _ -> 2048)

    let pfunct = MergePartition.mPP()
    let mpp = worker.LoadPModule(pfunct).Invoke

    let nv = 128 * 7
    let coop = 0

    let dParts = pcalc {
        let! dI = DArray.scatterInBlob worker hI
        //let! dZ = DArray.scatterInBlob worker [| 0 |]
        let! dR = mpp aCount bCount nv coop dI //dZ
        let! partition = dR.Gather()
        return partition} |> PCalc.run


    //printfn "%A" dResult
    printfn "%A" dParts

    let count = 512
    let aCount, bCount = aib count
    //let bCount = count
    //let aCount = count / 3
    printfn "count = %d" count
    printfn "aCount = %d\tbCount = %d" aCount bCount
    let hB = Array.init bCount (fun i -> i)
    let hI = Array.init aCount (fun _ -> rng.Next(bCount)) |> Array.sort
    let hA = Array.init aCount (fun _ -> 2048)

    let pfunct = MergePartition.mPP()
    let mpp = worker.LoadPModule(pfunct).Invoke

    let nv = 128 * 7
    let coop = 0

    let dParts = pcalc {
        let! dI = DArray.scatterInBlob worker hI
        //let! dZ = DArray.scatterInBlob worker [| 0 |]
        let! dR = mpp aCount bCount nv coop dI //dZ
        let! partition = dR.Gather()
        return partition} |> PCalc.run


    //printfn "%A" dResult
    printfn "%A" dParts

    let count = 1024
    let aCount, bCount = aib count
    //let bCount = count
    //let aCount = count / 3
    printfn "count = %d" count
    printfn "aCount = %d\tbCount = %d" aCount bCount
    let hB = Array.init bCount (fun i -> i)
    let hI = Array.init aCount (fun _ -> rng.Next(bCount)) |> Array.sort
    let hA = Array.init aCount (fun _ -> 2048)

    let pfunct = MergePartition.mPP()
    let mpp = worker.LoadPModule(pfunct).Invoke

    let nv = 128 * 7
    let coop = 0

    let dParts = pcalc {
        let! dI = DArray.scatterInBlob worker hI
        //let! dZ = DArray.scatterInBlob worker [| 0 |]
        let! dR = mpp aCount bCount nv coop dI //dZ
        let! partition = dR.Gather()
        return partition} |> PCalc.run


    //printfn "%A" dResult
    printfn "%A" dParts

    let count = 1024000
    let aCount, bCount = aib count
    //let bCount = count
    //let aCount = count / 3
    printfn "count = %d" count
    printfn "aCount = %d\tbCount = %d" aCount bCount
    let hB = Array.init bCount (fun i -> i)
    let hI = Array.init aCount (fun _ -> rng.Next(bCount)) |> Array.sort
    let hA = Array.init aCount (fun _ -> 2048)

    let pfunct = MergePartition.mPP()
    let mpp = worker.LoadPModule(pfunct).Invoke

    let nv = 128 * 7
    let coop = 0

    let dParts = pcalc {
        let! dI = DArray.scatterInBlob worker hI
        //let! dZ = DArray.scatterInBlob worker [| 0 |]
        let! dR = mpp aCount bCount nv coop dI //dZ
        let! partition = dR.Gather()
        return partition} |> PCalc.run


    //printfn "%A" dResult
    printfn "%A" dParts



// @COMMENT@ try use DevicePtr<int> explictly to indicate it is int type
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