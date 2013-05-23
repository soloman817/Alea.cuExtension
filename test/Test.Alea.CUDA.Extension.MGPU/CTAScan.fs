module Test.Alea.CUDA.Extension.MGPU.CTAScan

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.CTAScan
open NUnit.Framework

let worker = getDefaultWorker()

[<Test>]
let ``scanop`` () =
    let pfunct (op:IScanOp<'TI, 'TV, 'TR>) = cuda {
        let identity = op.Identity

        let! kernel =
            <@ fun (output:DevicePtr<'TI>) ->
                let i = threadIdx.x
                output.[i] <- identity @>
            |> defineKernelFunc

        return PFunc(fun (m:Module) (n:int) ->
            use output = m.Worker.Malloc(n)
            let lp = LaunchParam(1, n)
            kernel.Launch m lp output.Ptr
            output.ToHost() ) }

    let scanOp = scanOp ScanOpTypeAdd 1.1
    let pfunct = pfunct scanOp

    let pfuncm = Engine.workers.DefaultWorker.LoadPModule(pfunct)

    let output = pfuncm.Invoke 100
    printfn "%A" output