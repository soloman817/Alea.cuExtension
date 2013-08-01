module Test.Alea.CUDA.Extension.MGPU.CTALoadBalance

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.CTAMerge
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.LoadStore
open NUnit.Framework

let worker = getDefaultWorker()

//[<Test>]
//let ``CTALoadBalance test`` () =
//    let hCounts = [|       4;    0;    5;    5;    0;    5;    5;    1;    3;    1;
//                           0;    3;    1;    1;    3;    5;    5;    5;    5;    5;
//                           0;    4;    5;    5;    5;    0;    2;    4;    4;    1;
//                           0;    0;    2;    0;    5;    3;    4;    5;    5;    3;
//                           3;    4;    0;    2;    5;    1;    5;    4;    4;    2 |]
//
//    let NT, VT = 128, 7
//
//      
//    let deviceSerialLoadBalanceSearch (VT:int) (rangeCheck:bool) =
//        <@ fun (b_shared:RWPtr<int>) (aBegin:int) (aEnd:int) (bFirst:int) (bBegin:int) (bEnd:int) (a_shared:RWPtr<int>) -> 
//            let mutable bKey = b_shared.[bBegin]
//            let mutable aBegin = aBegin
//            let mutable bBegin = bBegin
//
//            for i = 0 to VT - 1 do
//                let mutable p = (aBegin < aEnd) && ((bBegin >= bEnd) || (aBegin < bKey))
//                
//                if p then
//                    // Advance A (the needle).
//                    a_shared.[aBegin] <- bFirst + bBegin
//                    aBegin <- aBegin + 1
//                else
//                    // Advance B (the haystack).
//                    bBegin <- bBegin + 1
//                    bKey <- b_shared.[bBegin] @>
//
//
//
//    let ctaLoadBalance (NT:int) (VT:int) =
//        let computeMergeRange = computeMergeRange.Device
//        let mergePath = (mergePath MgpuBoundsUpper (comp CompTypeLess 0)).DMergePath
//        let deviceMemToMemLoop = deviceMemToMemLoop NT
//        let deviceSerialLoadBalanceSearchFalse = deviceSerialLoadBalanceSearch VT false
//        let deviceSerialLoadBalanceSearchTrue = deviceSerialLoadBalanceSearch VT true
//
//        let ctrSize = NT * VT + 2
//
//        <@ fun  (destCount      :int) 
//                (b_global       :RWPtr<int>)
//                (sourceCount    :int)
//                (block          :int)
//                (tid            :int)
//                (mp_global      :DevicePtr<int>)
//                (indices_shared :RWPtr<int>)
//                (loadPrecedingB :bool)
//                (mpCountingItr  :RWPtr<int>)
//                (mpOut          :DevicePtr<int>)
//                ->
//            let computeMergeRange = %computeMergeRange
//            let mergePath = %mergePath
//            let deviceMemToMemLoop = %deviceMemToMemLoop
//            let deviceSerialLoadBalanceSearchFalse = %deviceSerialLoadBalanceSearchFalse
//            let deviceSerialLoadBalanceSearchTrue = %deviceSerialLoadBalanceSearchTrue
//
//            let mutable loadPrecedingB = if loadPrecedingB then 1 else 0
//
//            let range = computeMergeRange destCount sourceCount block 0 (NT * VT) mp_global
//        
//            let a0 = range.x
//            let a1 = range.y
//            let mutable b0 = range.z
//            let b1 = range.w
//
//            let mutable loadPrecedingB = loadPrecedingB
//            if loadPrecedingB = 1 then
//                if b0 = 0 then
//                    loadPrecedingB <- 0
//                else
//                    b0 <- b0 - 1
//            
//            let mutable extended = if (a1 < destCount) && (b1 < sourceCount) then 1 else 0
//
//            let mutable aCount = a1 - a0
//            let mutable bCount = b1 - b0
//
//            //mpOut.[tid] <- indices_shared.[tid]
//            let a_shared = indices_shared
//            let b_shared = indices_shared + aCount
//            
//
//            //deviceMemToMemLoop (bCount + extended) (b_global + b0) tid b_shared true
//            deviceMemToMemLoop bCount b_global tid b_shared true
//
//            let diag = min (VT * tid) (aCount + bCount - loadPrecedingB)
//        
////            let countingItr = __local__<int>(ctrSize).Ptr(0)
////            for i = 0 to ctrSize - 1 do
////                countingItr.[i] <- i
////            __syncthreads()
//
//            //let mp = mergePath (mpCountingItr + a0) aCount (b_shared + loadPrecedingB) (bCount - loadPrecedingB) diag
//            let mp = mergePath mpCountingItr aCount b_shared bCount diag
//            mpOut.[tid] <- mp
//
//            let a0tid = a0 + mp
//            let b0tid = diag - mp + loadPrecedingB
//
//            if extended <> 0 then
//                deviceSerialLoadBalanceSearchFalse b_shared a0tid a1 (b0 - 1) b0tid bCount (a_shared - a0)
//            else
//                deviceSerialLoadBalanceSearchTrue b_shared a0tid a1 (b0 - 1) b0tid bCount (a_shared - a0)
//            __syncthreads()
//
//            int4(a0, a1, b0, b1)
//                @>
//
//
//    let pfunct = cuda {
//        let ctaLoadBalance = ctaLoadBalance NT VT
//        let deviceSharedToGlobal = deviceSharedToGlobal NT VT
//        let! kernel =
//            <@ fun  (aCount:int) 
//                    (b_global:DevicePtr<int>) 
//                    (bCount:int) 
//                    (mp_global:DevicePtr<int>) 
//                    (indices_global:DevicePtr<int>)
//                    (mpCountingItr:DevicePtr<int>)
//                    (output:DevicePtr<int>) ->
//                    let ctaLoadBalance = %ctaLoadBalance
//                    let deviceSharedToGlobal = %deviceSharedToGlobal
//
//                    let indices_shared = __shared__<int>(NT * (VT + 1)).Ptr(0)
//                    
//
//                    let tid = threadIdx.x
//                    let block = blockIdx.x
////                    let mpCountingItr = __local__<int>(NT * VT + 2).Ptr(0)
////                    for i = 0 to (NT * VT + 2) - 1 do
////                        mpCountingItr.[i] <- i
////                    __syncthreads()
//
//                    let range = ctaLoadBalance aCount b_global bCount block tid mp_global indices_shared false mpCountingItr output
//                    let aCount = range.y - range.x
////                    for i = 0 to VT - 1 do
////                        let index = NT * i + tid
////                        if index < aCount then
////                            indices_global.[index] <- indices_shared.[index]
//                    deviceSharedToGlobal aCount indices_shared tid indices_global false
//
//            @> |> defineKernelFunc
//
//        return PFunc(fun (m:Module) ->
//            let aCount = 149
//            use counts = m.Worker.Malloc(hCounts)
//            let bCount = hCounts.Length
//            use parts = m.Worker.Malloc([| 0; 149 |])
//            use indices = m.Worker.Malloc aCount
//            let sequence = Array.init bCount (fun i -> i)
//            use countingItr = m.Worker.Malloc<int> sequence
//            use output = m.Worker.Malloc(aCount)
//            let lp = LaunchParam(1, 128)
//            kernel.Launch m lp aCount counts.Ptr bCount parts.Ptr indices.Ptr countingItr.Ptr output.Ptr
//            output.ToHost()
//            (*indices.ToHost()*)
//             ) }
//
//    let pfuncm = worker.LoadPModule(pfunct)
//
//    let result = pfuncm.Invoke
//    printfn "output: %A" result