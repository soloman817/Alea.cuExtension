module Alea.cuExtension.MGPU.Scan
//
//// this maps to scan.cuh
//
//open System.Runtime.InteropServices
//open Microsoft.FSharp.Collections
//open Alea.CUDA
//open Alea.CUDA.Utilities
//open Alea.cuExtension
////open Alea.cuExtension.Util
//open Alea.cuExtension.MGPU
////open Alea.cuExtension.MGPU.QuotationUtil
//open Alea.cuExtension.MGPU.DeviceUtil
//open Alea.cuExtension.MGPU.LoadStore
//open Alea.cuExtension.MGPU.CTAScan
//open Alea.cuExtension.MGPU.Reduce
//
//
//
//let kernelParallelScan (NT:int) (VT:int) (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) =
//    let NV = NT * VT
//
//    let capacity, scan = ctaScan NT op
//    let alignOfTI, sizeOfTI = TypeUtil.cudaAlignOf typeof<'TI>, sizeof<'TI>
//    let alignOfTV, sizeOfTV = TypeUtil.cudaAlignOf typeof<'TV>, sizeof<'TV>
//    let sharedAlign = max alignOfTI alignOfTV
//    let sharedSize = max (sizeOfTI * NV) (sizeOfTV * capacity)
//    //let createSharedExpr = createSharedExpr sharedAlign sharedSize
//
//    // get neccesary quotations for later doing slicing in quotation
//    let commutative = op.Commutative
//    let identity = op.Identity
//    let extract = op.DExtract
//    let combine = op.DCombine
//    let plus = op.DPlus
//    
//    let deviceGlobalToShared = deviceGlobalToShared NT VT
//    let deviceSharedToGlobal = deviceSharedToGlobal NT VT
//    
//
//    <@ fun (cta_global:deviceptr<'TI>) (count:int) (total_global:deviceptr<'TV>) (end_global:deviceptr<'TR>) (dest_global:deviceptr<'TR>) ->
//        let extract = %extract
//        let combine = %combine
//        let plus = %plus
//        
//        let deviceGlobalToShared = %deviceGlobalToShared
//        let deviceSharedToGlobal = %deviceSharedToGlobal
//        let scan = %scan
//
//        //let shared = %(createSharedExpr)
//        let sharedScan = __shared__.Array<'TV>(sharedSize)
//        let sharedInputs = __shared__.Array<'TI>(sharedSize)
//        let sharedResults = __shared__.Array<'TR>(sharedSize)
//
//        let tid = threadIdx.x
//        
//        let mutable total = extract identity -1
//        let mutable totalDefined = false
//        let mutable start = 0
//
//        
//        while(start < count) do
//            // load data in to shared memory
//            let count2 = min NV (count - start)
//            deviceGlobalToShared count2 (cta_global + start) tid (sharedInputs |> __array_to_ptr) true
//                        
//            // Transpose data into register in thread order.  Reduce terms serially
//            let inputs = __local__.Array<'TI>(VT) |> __array_to_ptr
//            let values = __local__.Array<'TV>(VT) |> __array_to_ptr
//            let mutable x = extract identity -1
//            for i = 0 to VT - 1 do
//                let index = VT * tid + i
//                if(index < count2) then
//                    inputs.[i] <- sharedInputs.[index]
//                    values.[i] <- extract inputs.[i] (start + index)
//                    x <- if i <> 0 then plus x values.[i] else values.[i]                    
//            __syncthreads()
//            
//            let passTotal = __local__.Array(1) |> __array_to_ptr
//            let mutable x = scan tid x (sharedScan |> __array_to_ptr) passTotal ExclusiveScan
//            
//            if totalDefined then
//                x <- plus total x
//                total <- plus total passTotal.[0]
//            else
//                total <- passTotal.[0]
//
//            let mutable x2 = x
//            for i = 0 to VT-1 do
//                let index = VT * tid + i
//                if (index < count2) then
//                    if ((i > 0 || tid > 0 ) || totalDefined) then                     
//                        x2 <- plus x values.[i] 
//                    else 
//                        x2 <- values.[i]                        
////                  For inclusive scan, set the new value then store
////                  For exclusive scan, store the old value then set the new one
//                if(mgpuScanType = InclusiveScan) then 
//                        x <- x2
//                sharedResults.[index] <- combine inputs.[i] x
//                if(mgpuScanType = ExclusiveScan) then 
//                        x <- x2
//            __syncthreads()
//            
//            deviceSharedToGlobal count2 (sharedResults |> __array_to_ptr) tid (dest_global + start) true
//            start <- start + NV
//            totalDefined <- true
//
//        if (total_global.Handle <> (nativeint 0)) && (tid = 0) then
//            total_global.[0] <- total
//        if (end_global.Handle <> (nativeint 0)) && (tid = 0) then
//            end_global.[0] <- combine identity total 
//            @>
//
//
//let kernelScanDownsweep (plan:Plan) (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) =
//    let NT = plan.NT
//    let VT = plan.VT
//    let NV = NT * VT
//
//    let capacity, scan = ctaScan NT op
//    let sharedSize = max (sizeof<'TI> * NV) (sizeof<'TV> * capacity)
//    
//    // get neccesary quotations for later doing slicing in quotation
//    let commutative = op.Commutative
//    let identity = op.Identity
//    let extract = op.DExtract
//    let combine = op.DCombine
//    let plus = op.DPlus
//    let deviceSharedToGlobal = deviceSharedToGlobal NT VT
//    let deviceGlobalToShared = deviceGlobalToShared NT VT
//
//    <@ fun (data_global:deviceptr<'TI>) (count:int) (task:int2) (reduction_global:deviceptr<'TV>) (dest_global:deviceptr<'TR>) (totalAtEnd:int) ->
//        let extract = %extract
//        let combine = %combine
//        let plus = %plus
//        let deviceGlobalToShared = %deviceGlobalToShared
//        let deviceSharedToGlobal = %deviceSharedToGlobal
//        let scan = %scan
//
//        
//        let sharedScan = __shared__.Array<'TV>(sharedSize)
//        let sharedInputs = __shared__.Array<'TI>(sharedSize)
//        let sharedResults = __shared__.Array<'TR>(sharedSize)
//    
//        let tid = threadIdx.x
//        let block = blockIdx.x
//        let mutable range = computeTaskRangeEx block task NV count
//                
//        let mutable next = reduction_global.[block]
//        let mutable nextDefined = if block <> 0 then true else false
//                
//        while (range.x < range.y) do
//            let count2 = min NV (count - range.x)
//            
//            // Load from global to shared memory
//            deviceGlobalToShared count2 (data_global + range.x) tid (sharedInputs |> __array_to_ptr) true
//
//            let inputs = __local__.Array<'TI>(VT) |> __array_to_ptr
//            let values = __local__.Array<'TV>(VT) |> __array_to_ptr
//            let mutable x = extract identity -1
//
//            for i = 0 to VT - 1 do
//                let index = VT * tid + i
//                if index < count2 then
//                    inputs.[i] <- sharedInputs.[index]
//                    values.[i] <- extract inputs.[i] (range.x + index)
//                    if i <> 0 then x <- plus x values.[i] else x <- values.[i]
//            __syncthreads()
//
//            let passTotal = __local__.Array(1) |> __array_to_ptr
//            let mutable x = scan tid x (sharedScan |> __array_to_ptr) passTotal ExclusiveScan
//
//            if nextDefined then
//                x <- plus next x
//                next <- plus next passTotal.[0]
//            else
//                next <- passTotal.[0]
//
//            for i = 0 to VT - 1 do
//                let index = VT * tid + i
//                if index < count2 then
//                    let x2 =
//                        // If this is not the first element in the scan, add x values.[i] into x
//                        // Otherwise initialize x to values.[i]
//                        if ((i <> 0 || tid <> 0) || nextDefined) then
//                            plus x values.[i]
//                        else
//                            values.[i]
//                    // For inclusive scan, set the new value then store.
//                    // For exclusive scan, store the old value then set the new one
//                    if (mgpuScanType = InclusiveScan) then
//                        x <- x2
//                    sharedResults.[index] <- combine inputs.[i] x
//                    if (mgpuScanType = ExclusiveScan) then
//                        x <- x2
//            __syncthreads()
//
//            deviceSharedToGlobal count2 (sharedResults |> __array_to_ptr) tid (dest_global + range.x) true
//            range.x <- range.x + NV
//            nextDefined <- true
//                                
//        // need totalAtEnd
//        // infact, this usage of int as bool is not effiecient, you should also check xor bitwise operator
//        let a = if (block <> 0) && (totalAtEnd <> 0) then 1 else 0
//        let b = if ((gridDim.x - 1) <> 0) && (tid = 0) then 1 else 0        
//        if (a = b) then
//            let dg = combine identity next
//            dest_global.[count] <- dg @>
//        
//
////type IScan<'TI, 'TV, 'TR> =
////    {
////        Action : ActionHint -> deviceptr<'TI> -> deviceptr<'TV> -> deviceptr<'TV> -> deviceptr<'TR> -> unit
////        NumBlocks: int        
////    }
//
//
//let scan (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) = cuda {
//    let cutOff = 20000
//            
//    // count >= cutOff, do parallel raking reduce as an upsweep, then
//    // do parallel latency-oriented scan to reduce the spine of the 
//    // raking reduction, then do a raking scan as downsweep
//    let tuning = {NT = 128; VT = 7}
//        
//    let NT, VT = 512, 3
//    let kernelPS = kernelParallelScan NT VT mgpuScanType op
//    let! kernelPS = kernelPS |> Compiler.DefineKernel //"ps"
//        
//    let kernelRRUpsweep = Reduce.kernelReduce tuning op
//    let! kernelRRUpsweep = kernelRRUpsweep |> Compiler.DefineKernel //"rrUpsweep"
//
//    let NT2, VT2 = 256, 3
//    let kernelPLOS = kernelParallelScan NT2 VT2 ExclusiveScan op
//    let! kernelPLOS = kernelPLOS |> Compiler.DefineKernel //"plos"
//
//    let kernelRSDownsweep = kernelScanDownsweep tuning mgpuScanType op
//    let! kernelRSDownsweep = kernelRSDownsweep |> Compiler.DefineKernel //"rsDownsweep"
//    
//    let hplus = op.HPlus
//        
//    return Entry(fun program ->
//        let worker = program.Worker
//        let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT
//        let kernelPS = program.Apply kernelPS
//        let kernelRRUpsweep = program.Apply kernelRRUpsweep
//        let kernelPLOS = program.Apply kernelPLOS
//        let kernelRSDownsweep = program.Apply kernelRSDownsweep
//        fun (count:int) ->
//            let numBlocks = ref 1            
//
//            let run (data_global:deviceptr<'TI>) (total:deviceptr<'TV>) (reductionDevice:deviceptr<'TV>) (dest_global:deviceptr<'TR>)  =
//                fun () ->
//                    let mutable total = total                 
//                    if count < cutOff then
//                        let NV = NT * VT
//                        let lp = LaunchParam(1, NT) //|> hint.ModifyLaunchParam                        
//                        kernelPS.Launch lp 
//                            data_global 
//                            count 
//                            total 
//                            (if totalAtEnd = 1 then (dest_global + count) else (deviceptr<'TR>(0n)))
//                            dest_global                    
//                    else
//                        let NV = tuning.NT * tuning.VT
//                        let numTiles = divup count NV
//                        numBlocks := min (numSm * 25) numTiles
//                        let task = divideTaskRange numTiles !numBlocks                        
//                        let totalDevice = reductionDevice + !numBlocks
//                        let lp = LaunchParam(!numBlocks, tuning.NT) //|> hint.ModifyLaunchParam
//                        
//                        kernelRRUpsweep.Launch lp 
//                            data_global 
//                            count 
//                            task 
//                            reductionDevice
//
//                        let lp = LaunchParam(1, NT2) //|> hint.ModifyLaunchParam
//                        let reductionDevice1 = reductionDevice.Reinterpret<'TI>()
//                        let reductionDevice2 = reductionDevice.Reinterpret<'TR>()
//                        kernelPLOS.Launch lp 
//                            reductionDevice1 
//                            !numBlocks 
//                            totalDevice 
//                            (deviceptr(0n))
//                            reductionDevice2
//                        
//                        total <- totalDevice
//                        let lp = LaunchParam(!numBlocks, tuning.NT) //|> hint.ModifyLaunchParam            
//                        kernelRSDownsweep.Launch lp 
//                            data_global 
//                            count 
//                            task 
//                            reductionDevice 
//                            dest_global 
//                            totalAtEnd
//
//                |> worker.Eval                    
//            { NumBlocks = !numBlocks } ) }
