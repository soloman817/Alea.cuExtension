module Test.Alea.CUDA.Extension.MGPU.CTASortedSearch

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.CTASortedSearch

open NUnit.Framework


[<Test>]
let ``CTASortedSearch : raw test`` () =
    let NT = 128
    let VT = 7
    let bounds = MgpuBoundsLower
    let compOp = (comp CompTypeLess 0)
    
    let pfunct = cuda {
        let ctaSortedSearch = ctaSortedSearch NT VT bounds true true true true compOp
        let! kernel =
            <@ fun (keys:DevicePtr<int>) (indices:DevicePtr<int>) (results:DevicePtr<int2>) -> //(resultsY:DevicePtr<int>) (resultsZ:DevicePtr<int>) ->
                let ctaSortedSearch = %ctaSortedSearch

                let i = threadIdx.x
                let matchCount = ctaSortedSearch keys 0 64 63 0 64 64 127 64 true i indices
                results.[0] <- matchCount
                @>
            |> defineKernelFunc

        return PFunc(fun (m:Module) ->
            use output = m.Worker.Malloc(1)
            use data = m.Worker.Malloc((Array.init NT (fun i -> i)))
            use inds = m.Worker.Malloc([| 3; 8; 27; 44; 73; 89 |])
            let lp = LaunchParam(1, NT)
            kernel.Launch m lp data.Ptr inds.Ptr output.Ptr
            output.ToHost() ) }

    let pfuncm = Engine.workers.DefaultWorker.LoadPModule(pfunct)

    let output = pfuncm.Invoke
    printfn "%A" output
