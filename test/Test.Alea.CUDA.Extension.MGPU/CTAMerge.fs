module Test.Alea.CUDA.Extension.MGPU.CTAMerge

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.CTAMerge

open NUnit.Framework

let cmr = computeMergeRange

//[<Test>]
//let ``compute merge range`` () =
//    let pfunct (plan:Plan) = cuda {
//        let NT = plan.NT
//        let VT = plan.VT
//        let NV = NT * VT
//
//        let capacity, scan2 = ctaScan2 (scanOp ScanOpTypeAdd 0)
//        let sharedSize = max NV capacity
//
//        let deviceGlobalToReg = deviceGlobalToReg NT VT
//        let computeMergeRange = computeMergeRange.Device
//        
//        let! kernel = 
//            <@ fun (aCount:int) (bCount:int) (block:int) (coop:int) (nv:int) (mp_global:DevicePtr<int>) (indices_global:DevicePtr<int>) ->
//                let deviceGlobalToReg = %deviceGlobalToReg
//                let computeMergeRange = %computeMergeRange
//                let deviceTransferMergeValues = %deviceTransferMergeValues
//                let S = %scan2
//        
//
//                let shared = __shared__<int>(sharedSize).Ptr(0)
//                let sharedScan = shared
//                let sharedIndices = shared
//
//                let tid = threadIdx.x
//                let block = blockIdx.x
//
//                let range = cmr aCount bCount block 0 nv mp_global
//                let a0 = range.x
//                let a1 = range.y
//                let b0 = range.z
//                let b1 = range.w
//
//                let mutable aCount = aCount
//                let mutable bCount = bCount
//
//                aCount <- a1 - a0
//                bCount <- b1 - b0
//
//                for i = 0 to VT - 1 do
//                    sharedIndices.[NT * i + tid] <- 0
//                __synchthreads()
//
//                deviceGlobalToReg aCount (indices_global + a0) tid indices true 
//                @> |> defineKernelFunc
//        
//
//        return PFunc(fun (m:Module) (indicesCount:int) ->
//            use indices_global = m.Worker.Malloc(indicesCount)
//            use data = m.Worker.Malloc([|2..5..100|])
//            let numBlocks = divup (aCount + bCount) NV
//            let lp = LaunchParam(numBlocks, plan.NT)
//            kernel.Launch m lp data.Length 400 0 NV 

