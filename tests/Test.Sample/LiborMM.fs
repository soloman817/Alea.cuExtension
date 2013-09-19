module Test.Sample.LiborMM

open System
open System.IO
open System.Reflection
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.TestUtilities
open NUnit.Framework
open Sample.LiborMM
open Sample.Mrg32k3a

[<Test>]
let ``correctness on float32``() =
    let NN = 80
    let NMAT = 40
    let NOPT = 15
    let NPATH = 96000
    let maturities = [|4;4;4;8;8;8;20;20;20;28;28;28;40;40;40|]
    let delta = 0.25f
    let swaprates = [| 0.045f;0.05f;0.055f;0.045f;0.05f;0.055f;0.045f;0.05f;0.055f;0.045f;0.05f;0.055f;0.045f;0.05f;0.055f |]
    let lambda = Array.init NN (fun _ -> 0.2f)
    let L0 = Array.init NN (fun _ -> 0.051f)
    let V1 = [| 1u; 2u; 3u |]
    let V2 = [| 1u; 2u; 3u |]

    let template = cuda {
        let real = RealConverter.Real32
        
        let! rngCore = GPU.core
        let! rngNormal = GPU.normal real rngCore
        
        let param : GPU.Param = { NOPT = NOPT; NN = NN; NMAT = NMAT }
        let! simulator = GPU.template real param
        let simInit, simKernel1, simKernel2 = simulator

        return Entry(fun program ->
            let worker = program.Worker
            let rngInit = rngNormal.Init program
            let rngKernel = program.Apply(rngNormal.Kernel)
            let simInit = simInit program
            let simKernel1 = program.Apply(simKernel1)
            let simKernel2 = program.Apply(simKernel2)

            let run () =
                use start = worker.CreateEvent()
                use stop = worker.CreateEvent()
                use dz = worker.Malloc<float32>(NPATH * NMAT)
                use dv = worker.Malloc<float32>(NPATH)
                use dLb = worker.Malloc<float32>(NPATH)
                let lp = LaunchParam(NPATH / 64, 64)

                rngInit V1 V2 0
                rngKernel.Launch lp NMAT dz.Ptr

                simInit maturities delta lambda L0 swaprates

                start.Record()
                simKernel2.Launch lp dz.Ptr dv.Ptr
                stop.Record()
                stop.Synchronize()
                let msec = Event.ElapsedMilliseconds(start, stop)
                if Util.debug then printfn "GPU time (No Greeks) : %12.9f ms" msec

                let hv = dv.Gather()
                let mutable v = 0.0
                for i = 0 to NPATH - 1 do
                    v <- v + float(hv.[i]) / float(NPATH)

                Assert.That(v, Is.EqualTo(48.95406144).Within(1e-8))

                if Util.debug then printfn "average value v = %15.8f" v

                if Util.debug then printfn ""

                start.Record()
                simKernel1.Launch lp dz.Ptr dv.Ptr dLb.Ptr
                stop.Record()
                stop.Synchronize()
                let msec = Event.ElapsedMilliseconds(start, stop)
                if Util.debug then printfn "GPU time (Greeks) : %12.9f ms" msec

                let hv = dv.Gather()
                let hLb = dLb.Gather()
                let mutable v = 0.0
                let mutable Lb = 0.0
                for i = 0 to NPATH - 1 do
                    v <- v + float(hv.[i]) / float(NPATH)
                    Lb <- Lb + float(hLb.[i]) / float(NPATH)

                Assert.That(v, Is.EqualTo(48.95406141).Within(1e-8))
                Assert.That(Lb, Is.EqualTo(15.39557960).Within(1e-8))

                if Util.debug then printfn "average value v  = %15.8f" v
                if Util.debug then printfn "average value Lb = %15.8f" Lb

            run ) }

    let options = CompileOptions.Default
    //let options = { CompileOptions.Default with VarDebug=true }
    //let options = { CompileOptions.Default with FunctionParameterAsMutable=false }
    use program = template |> Util.compileWithOptions options |> Util.link |> Worker.Default.LoadProgram
    program.Run()


        