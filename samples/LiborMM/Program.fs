open System
open Alea.CUDA
open Alea.CUDA.Utilities
open Sample.Mrg32k3a
open Sample.LiborMM

let testOnSignle() =
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
                printfn "===== Test on Float32 ====="
                printfn ""

                use start = worker.CreateEvent()
                use stop = worker.CreateEvent()
                use dz = worker.Malloc<float32>(NPATH * NMAT)
                use dv = worker.Malloc<float32>(NPATH)
                use dLb = worker.Malloc<float32>(NPATH)
                let lp = LaunchParam(NPATH / 64, 64)

                rngInit V1 V2 0

                start.Record()
                rngKernel.Launch lp NMAT dz.Ptr
                stop.Record()
                stop.Synchronize()
                let msec = Event.ElapsedMilliseconds(start, stop)
                printfn "GPU random time      : %20.9f ms" msec
                printfn ""

                simInit maturities delta lambda L0 swaprates

                start.Record()
                simKernel2.Launch lp dz.Ptr dv.Ptr
                stop.Record()
                stop.Synchronize()
                let msec = Event.ElapsedMilliseconds(start, stop)

                let hv = dv.Gather()
                let mutable v = 0.0
                for i = 0 to NPATH - 1 do
                    v <- v + float(hv.[i]) / float(NPATH)

                printfn "GPU time (No Greeks) : %20.9f ms" msec
                printfn "average value v      = %19.8f" v
                printfn ""

                start.Record()
                simKernel1.Launch lp dz.Ptr dv.Ptr dLb.Ptr
                stop.Record()
                stop.Synchronize()
                let msec = Event.ElapsedMilliseconds(start, stop)

                let hv = dv.Gather()
                let hLb = dLb.Gather()
                let mutable v = 0.0
                let mutable Lb = 0.0
                for i = 0 to NPATH - 1 do
                    v <- v + float(hv.[i]) / float(NPATH)
                    Lb <- Lb + float(hLb.[i]) / float(NPATH)

                printfn "GPU time (Greeks)    : %20.9f ms" msec
                printfn "average value v      = %19.8f" v
                printfn "average value Lb     = %19.8f" Lb
                printfn ""

            run ) }

    use program = template |> Util.load Worker.Default
    program.Run()

let testOnDouble() =
    let NN = 80
    let NMAT = 40
    let NOPT = 15
    let NPATH = 96000
    let maturities = [|4;4;4;8;8;8;20;20;20;28;28;28;40;40;40|]
    let delta = 0.25
    let swaprates = [| 0.045;0.05;0.055;0.045;0.05;0.055;0.045;0.05;0.055;0.045;0.05;0.055;0.045;0.05;0.055 |]
    let lambda = Array.init NN (fun _ -> 0.2)
    let L0 = Array.init NN (fun _ -> 0.051)
    let V1 = [| 1u; 2u; 3u |]
    let V2 = [| 1u; 2u; 3u |]

    let template = cuda {
        let real = RealConverter.Real64
        
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
                printfn "===== Test on Float64 ====="
                printfn ""

                use start = worker.CreateEvent()
                use stop = worker.CreateEvent()
                use dz = worker.Malloc<float>(NPATH * NMAT)
                use dv = worker.Malloc<float>(NPATH)
                use dLb = worker.Malloc<float>(NPATH)
                let lp = LaunchParam(NPATH / 64, 64)

                rngInit V1 V2 0

                start.Record()
                rngKernel.Launch lp NMAT dz.Ptr
                stop.Record()
                stop.Synchronize()
                let msec = Event.ElapsedMilliseconds(start, stop)
                printfn "GPU random time      : %20.9f ms" msec
                printfn ""

                simInit maturities delta lambda L0 swaprates

                start.Record()
                simKernel2.Launch lp dz.Ptr dv.Ptr
                stop.Record()
                stop.Synchronize()
                let msec = Event.ElapsedMilliseconds(start, stop)

                let hv = dv.Gather()
                let mutable v = 0.0
                for i = 0 to NPATH - 1 do
                    v <- v + float(hv.[i]) / float(NPATH)

                printfn "GPU time (No Greeks) : %20.9f ms" msec
                printfn "average value v      = %19.8f" v
                printfn ""

                start.Record()
                simKernel1.Launch lp dz.Ptr dv.Ptr dLb.Ptr
                stop.Record()
                stop.Synchronize()
                let msec = Event.ElapsedMilliseconds(start, stop)

                let hv = dv.Gather()
                let hLb = dLb.Gather()
                let mutable v = 0.0
                let mutable Lb = 0.0
                for i = 0 to NPATH - 1 do
                    v <- v + float(hv.[i]) / float(NPATH)
                    Lb <- Lb + float(hLb.[i]) / float(NPATH)

                printfn "GPU time (Greeks)    : %20.9f ms" msec
                printfn "average value v      = %19.8f" v
                printfn "average value Lb     = %19.8f" Lb
                printfn ""

            run ) }

    use program = template |> Util.load Worker.Default
    program.Run()

[<EntryPoint>]
let main argv = 
    testOnSignle()
    printfn ""
    testOnDouble()
    0 // return an integer exit code
