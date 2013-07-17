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

module MergePartition =
    type Plan =
        {
            NT : int
            Bounds : int
        }

       
    let kernelMergePartition (plan:Plan) (mergeSearch:IMergeSearch<int,int>) = 
        let NT = plan.NT
        let bounds = plan.Bounds
        let mergePath = mergeSearch.DMergePath
        
        <@ fun (a_global:DevicePtr<int>) (aCount:int) (b_global:DevicePtr<int>) (bCount:int) (nv:int) (coop:int) (mp_global:DevicePtr<int>) (numSearches:int) ->
            let mergePath = %mergePath
            
                        
            let mutable aCount = aCount
            let mutable bCount = bCount
            
            
            let partition = NT * blockIdx.x * threadIdx.x
            if partition < numSearches then
                let mutable a0 = 0
                let mutable b0 = 0                
                let mutable gid = nv * partition
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
    
        
//    let mPP (bounds:int) (compOp:IComp<int>) = cuda {
//        let! api = mergePathPartitions bounds compOp
//
//        return PFunc(fun (m:Module) ->
//            let worker = m.Worker
//            let api = api.Apply m
//
//            fun (aCount:int) (bCount:int) (nv:int) (coop:int) (aGlobal:DArray<int>) (bGlobal:DArray<int>) ->
//                                                                (*aka indices_global *) (*aka 0 counting itr*)
//                pcalc {
//                    let api = api aCount bCount nv coop
//                    let NT = 64
//                    let numPartitions = divup (aCount + bCount) nv
//                    let numPartitionBlocks = divup (numPartitions + 1) NT
//
//                    let! parts = DArray.createInBlob<int> worker (numPartitions + 1)
//                    do! PCalc.action (fun hint -> api.Action hint aGlobal.Ptr bGlobal.Ptr parts.Ptr)
//
//                    return parts } ) }


module BulkInsert =
    type Plan2 =
        {
            NT : int
            VT : int
        }
           
    
    let kernelBulkInsert (plan:Plan2) =
        let NT = plan.NT
        let VT = plan.VT
        let NV = NT * VT

    
        let capacity, scan2 = ctaScan2 NT (scanOp ScanOpTypeAdd 0)
        let alignOfInt, sizeOfInt = TypeUtil.cudaAlignOf typeof<int>, sizeof<int>
        //let sharedAlign = alignOfInt
        let sharedSize = max NV capacity //max (sizeOfInt * NV) (sizeOfInt * capacity)
        //let createSharedExpr = createSharedExpr sharedAlign sharedSize
        

        let deviceGlobalToReg = deviceGlobalToReg NT VT
        let computeMergeRange = computeMergeRange.Device
        let deviceTransferMergeValues = deviceTransferMergeValuesA NT VT

        <@ fun (a_global:DevicePtr<int>) (indices_global:DevicePtr<int>) (aCount:int) (b_global:DevicePtr<int>) (bCount:int) (mp_global:DevicePtr<int>) (dest_global:DevicePtr<'T>) ->
            let deviceGlobalToReg = %deviceGlobalToReg
            let computeMergeRange = %computeMergeRange
            let deviceTransferMergeValues = %deviceTransferMergeValues
            let S = %scan2
        
//            let shared = %(createSharedExpr)
//            let sharedScan = shared.Reinterpret<int>()
//            let sharedIndices = shared.Reinterpret<int>()

            let shared = __shared__<int>(sharedSize).Ptr(0)
            let sharedScan = shared
            let sharedIndices = shared

            let tid = threadIdx.x
            let block = blockIdx.x
        
            let mutable range = computeMergeRange aCount bCount block 0 NV mp_global
            let mutable aCount = aCount
            let mutable bCount = bCount
            let gid = NV * block

//            let a0 = mp_global.[block]
//            let a1 = mp_global.[block + 1]
//            let b0 = gid - a0
//            let b1 = (min (aCount + bCount) (gid + NV)) - a1
            let a0 = range.x
            let a1 = range.y
            let b0 = range.z
            let b1 = range.w

            aCount <- a1 - a0
            bCount <- b1 - b0

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
                        scan                    
                    else 
                        aCount + index - scan
                if indices.[i] > 0 then scan <- scan + 1
                sharedIndices.[index] <- gather
            __syncthreads()

            deviceTransferMergeValues (aCount + bCount) (a_global + a0) (b_global + b0) aCount sharedIndices tid (dest_global + a0 + b0) false 
            @>
    
    type IBulkInsert =
        {
            Action : ActionHint -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> unit
            NumPartitions : int
        }

    let bulkInsert2()  = cuda {
        let plan = { NT = 128; VT = 7 }
        let NV = plan.NT * plan.VT

        let! kernelBulkInsert = kernelBulkInsert plan |> defineKernelFuncWithName "bi"
        let! mpp = MergePartition.mergePathPartitions MgpuBoundsLower (comp CompTypeLess 0)

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
            fun (data_A:DArray<int>) (indices:DArray<int>) (data_B:DArray<int>) (*zeroitr:DArray<int>*) ->
                let N = 100
                let insertCount = divup (N - 2) 5
                
                let api = api insertCount N
                pcalc {
                    let! partition = DArray.createInBlob<int> worker ((divup (insertCount + N) (128 * 7)) + 1)
                    let! zeroitr = DArray.createInBlob worker (N + insertCount)

                    let! inserted = DArray.createInBlob<int> worker (insertCount + N)
                    do! PCalc.action (fun hint -> api.Action hint data_A.Ptr indices.Ptr zeroitr.Ptr data_B.Ptr partition.Ptr inserted.Ptr)
                    return inserted } ) }


[<Test>]
let ``bulk insert debug`` () =
    let worker = getDefaultWorker()
    let pfunct = BulkInsert.bInsert()
    let bi = worker.LoadPModule(pfunct).Invoke

    let hDataSource = Array.init 100 int
    printfn "hDataSource size: (%A)" hDataSource.Length
    printfn "hDataSource: %A" hDataSource
    let hIndices = [|2..5..100|]
    printfn "hIndices size: (%A)" hIndices.Length
    printfn "hIndices: %A" hIndices
    let hDataToInsert = [|1000..10..((hIndices.Length*10+1000)-10)|]
    printfn "hDataToInsert size: (%A)" hDataToInsert.Length
    printfn "hDataToInsert: %A" hDataToInsert

    //let z = Array.zeroCreate<int> (hDataSource.Length + hDataToInsert.Length)

    let dResult = pcalc {
        let! dDataToInsert = DArray.scatterInBlob worker hDataToInsert
        let! dIndices = DArray.scatterInBlob worker hIndices
        let! dDataSource = DArray.scatterInBlob worker hDataSource
        //let! z = DArray.scatterInBlob worker z
        let! inserted = bi dDataToInsert dIndices dDataSource //z
        let! results = inserted.Gather()
        return results } |> PCalc.run
    

    for i = 0 to dResult.Length - 1 do
        printfn "(%d,%d)" i dResult.[i]
    //printfn "%A" dResult

    