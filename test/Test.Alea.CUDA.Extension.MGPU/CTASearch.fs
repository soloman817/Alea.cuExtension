module Test.Alea.CUDA.Extension.MGPU.CTASearch

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.CTASearch

open NUnit.Framework

[<Test>]
let ``cta search : binary search`` () =
    let pfunct = cuda {
        let binarySearch = (binarySearchFun MgpuBoundsLower (comp CompTypeLess 0)).DBinarySearch
        let! kernel =
            <@ fun (data:DevicePtr<int>) (count:int) (key:int) (output:DevicePtr<int>) -> 
                let bs = %binarySearch
                let i = threadIdx.x
                output.[i] <- bs data count key
                 @>
            |> defineKernelFunc
        
        return PFunc(fun (m:Module) (n:int) ->
            use output = m.Worker.Malloc(n)
            use data = m.Worker.Malloc((Array.init 100 (fun i -> i)))
            let lp = LaunchParam(1,n)
            kernel.Launch m lp data.Ptr 100 8 output.Ptr
            output.ToHost() ) }
    
    let pfuncm = Engine.workers.DefaultWorker.LoadPModule(pfunct)

    let output = pfuncm.Invoke 10
    printfn "%A" output

[<Test>]
let ``cta search : merge search`` () =
    let pfunct = cuda {
        let mergePath = (mergeSearch MgpuBoundsLower (comp CompTypeLess 0)).DMergePath
        let! kernel =
            <@ fun (dataA:DevicePtr<int>) (aCount:int) (dataB:DevicePtr<int>) (bCount:int) (diag:int) (output:DevicePtr<int>) ->
                let mergePath = %mergePath
                
                let i = threadIdx.x
                let mp = mergePath dataA aCount dataB bCount diag
                output.[i] <- mp
                @>
            |> defineKernelFunc

        return PFunc(fun (m:Module) ->
            use output = m.Worker.Malloc(100)
            use dataA = m.Worker.Malloc([|2..5..100|])
            use dataB = m.Worker.Malloc([|0..400|])
            let lp = LaunchParam(7,128)
            kernel.Launch m lp dataA.Ptr 22 dataB.Ptr 78 2 output.Ptr
            output.ToHost() ) }

    let pfuncm = Engine.workers.DefaultWorker.LoadPModule(pfunct)

    let output = pfuncm.Invoke
    printfn "%A" output



