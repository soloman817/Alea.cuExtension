module Test.Alea.CUDA.Extension.Finance.Heston

open System
open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Finance.Heston
open Alea.CUDA.Extension.Finance.Grid
open Test.Alea.CUDA.Extension

    
[<Test>]
let ``finite difference weights`` () =
      
    let worker = getDefaultWorker()

    let s = concentratedGrid 0.0 250.0 50.0 100 10.0
    let finiteDifferenceWeights = worker.LoadPModule(finiteDifferenceWeights).Invoke

    let fd = pcalc {
        let! s = DArray.scatterInBlob worker s

        // %XIANG% (1)
        
        // first, for a function wrapped with pcalc, the interface would better NOT contains
        // any raw pointers such as here, s.Ptr. Reason:
        // 1) we should use more logical high level stuff like DArray, DMatrix, DScalar
        // 2) the s.Ptr is acturally a trigger, so when you call s.Ptr for the first time
        //    it will trigger the pcalc to start build the blob, so if you use it here, we
        //    we have 2 blob acturally.
        // So, one of the main job of wrapping a raw impl to pcalc function, is to replace those
        // raw pointers with meaningful DScalar, DArray, DMatrix, etc.
        let! sDiff = finiteDifferenceWeights s.Length s

        // %XIANG% (4)
        // so inside finiteDifferenceWeights, you triggered the .Ptr, so the memory is ok
        // but the action hasn't been executed, becauuse you add action after you create the struct
        // which triggers the blob, so here for debugging, we first need let it run the pending
        // action, we could do it by force()
        //do! PCalc.force()

        // now we don't need PCalc.force(), because I use DifferenceHighLevel

        // now you gather the DArray (or DMatrix) inside the DifferenceHighLevel
        let! delta = sDiff.Delta.Gather()
        printfn "%A" delta
        // do verfiiction or you can return the host array


        // then you can gather the raw pointers.
//        do
//            let host = Array.zeroCreate<float> (s.Length - 1) 
//            DevicePtrUtil.Gather(worker, sDiff.Delta, host, s.Length - 1)
//            printfn "%A" host
            // do verification

        // to debug

        // do not return the memory which is created in blob, because after PCalc.run
        // the blob will be explicitly disposed
        
    } 

    let fd = fd |> PCalc.run

    // %XIANG% (5)
    // So when you run PCalc.run, all those blob will be disposed, if you dont want this, you need
    // 1) malloc it with worker.Malloc and return it, which means, you take responsibility for that memory
    // 2) and your pcalc function which returns some memory should aslo not be malloc in the blob

    // so that is why in my wrapping for transform, scan, sobol, xorshift, I always keep one raw impl, which
    // all use raw pointers, but with high level, I try to hide those raw stuff by DResources, which is delayed
    // so I think here in this example, you'd better use a new high level type (could be ref type), to record
    // the members with DArray, DMatrix.

    // for code orgnization, you can see what I did, for example, xorshift7, I first have XorShift7.fs, which is
    // raw implementation, it will not use any DResources, but all in raw pointers. Then I have a central public \
    // file called PRandom.fs, which is for the wrapper, and here I add high level types, and wrap the raw impl
    // with high level types and manage the memories which raw impl didn't care about

    ()


    //Assert.IsTrue(true)

[<Test>]
let ``Douglas scheme`` () =

    let worker = getDefaultWorker()
    let douglasSolver = worker.LoadPModule(douglasSolver).Invoke

    // Heston model params
    let rho = -0.5
    let sigma = 0.2
    let kappa = 0.2
    let eta = 0.2
    let rd = 0.05
    let rf = 0.01
    let heston = HestonModel(rho, sigma, rd, rf, kappa, eta)

    // contract params
    let timeToMaturity = 1.0
    let strike = 100.0
    let optionType = Call

    // PDE solver params
    let theta = 0.5
    let sMax = 1000.0
    let vMax = 16.0
    let ns = 128
    let nv = 128
    let nt = 100
    let cS = 8.0
    let cV = 15.0
    let param = Param(theta, 0.0, sMax, vMax, ns, nv, nt, cS, cV)

    let pricer = pcalc {
        let! solution = douglasSolver heston optionType strike timeToMaturity param
        //return! solution.ToArray2D()
        return! solution.Gather()
    }

    let valueMatrix = pricer |> PCalc.run
    printfn "%A" valueMatrix

    ()

[<Test>]
let ``Douglas scheme plotting`` () =

    let loop (context:Graphics.Direct3D9.Application.Context) =
        pcalc {
            // in real app, we will first compile template into irm, and load here
            // this worker from context, which is capable interop with graphics
            printf "Compiling..."
            let douglasSolver = context.Worker.LoadPModule(douglasSolver).Invoke
            printfn "[OK]"

            // Heston model params
            let rho = -0.5
            let sigma = 0.2
            let kappa = 0.2
            let eta = 0.2
            let rd = 0.05
            let rf = 0.01
            let heston = HestonModel(rho, sigma, rd, rf, kappa, eta)

            // contract params
            let timeToMaturity = 1.0
            let strike = 100.0
            let optionType = Call

            // PDE solver params
            let theta = 0.5
            let sMax = 1000.0
            let vMax = 16.0
            let ns = 256
            let nv = 128
            let nt = 100
            let cS = 8.0
            let cV = 15.0
            let param = Param(theta, 0.0, sMax, vMax, ns, nv, nt, cS, cV)

            let! solution = douglasSolver heston optionType strike timeToMaturity param

            let extend minv maxv = let maxv', _ = Graphics.Direct3D9.SurfacePlotter.defaultExtend minv maxv in maxv', 1.0
            let renderType = Graphics.Direct3D9.SurfacePlotter.RenderType.Mesh
            do! Graphics.Direct3D9.SurfacePlotter.plottingLoop context solution extend renderType }
        |> PCalc.run

    let cudaDevice = Device.AllDevices.[0]
    let param = Graphics.Direct3D9.Application.Param.Create(cudaDevice)
    let param = { param with FormTitle = "Douglas Scheme"
                             DrawingSize = System.Drawing.Size(1024, 768) }
    let application = Graphics.Direct3D9.Application.Application(param, loop)
    application.Start()
