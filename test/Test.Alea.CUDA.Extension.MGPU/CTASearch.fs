module Test.Alea.CUDA.Extension.MGPU.CTASearch

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.CTASearch

open NUnit.Framework

[<Test>]
let ``cta search`` () =
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
    
    //let bsp = binarySearchFun MgpuBoundsLower CompTypeLess
    //let pfunct = pfunct bsp
    let pfuncm = Engine.workers.DefaultWorker.LoadPModule(pfunct)

    let output = pfuncm.Invoke 10
    printfn "%A" output


