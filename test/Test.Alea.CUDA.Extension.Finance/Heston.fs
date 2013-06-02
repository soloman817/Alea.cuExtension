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
let ``Euler scheme`` () =

    let worker = getDefaultWorker()
    let eulerSolver = worker.LoadPModule(eulerSolver).Invoke

    // Heston model params
    let rho = -0.6133
    let sigma = 0.3920
    let kappa = 1.1954
    let eta = 0.0677
    let rd = 0.02
    let rf = 0.01
    let heston = HestonModel(rho, sigma, rd, rf, kappa, eta)

    // contract params
    let timeToMaturity = 1.0
    let strike = 55.0
    let optionType = Call

    // PDE solver params
    let theta = 0.5
    let sMax = 220.0
    let vMax = 4.0
    let ns = 48
    let nv = 32
    let cS = strike/5.0
    let cV = vMax/5.0
    let param = EulerSolverParam(theta, 0.0, sMax, vMax, ns, nv, cS, cV)

    let pricer = pcalc {
        let! s, v, u = eulerSolver heston optionType strike timeToMaturity param
        return! u.Gather()
    }

    let result = pricer |> PCalc.runWithKernelTiming(10)
//    let result, loggers = pricer |> PCalc.runWithTimingLogger
//    loggers.["default"].DumpLogs()
    printfn "%A" result

[<Test>]
let ``Euler scheme plotting`` () =
    
    let verbose = true

    let loop (context:Graphics.Direct3D9.Application.Context) =
        pcalc {
            // in real app, we will first compile template into irm, and load here
            // this worker from context, which is capable interop with graphics
            printf "Compiling..."
            let eulerSolver = context.Worker.LoadPModule(eulerSolver).Invoke
            printfn "[OK]"

            // Heston model params
            let rho = -0.6133
            let sigma = 0.3920
            let kappa = 1.1954
            let eta = 0.0677
            let rd = 0.02
            let rf = 0.01
            let heston = HestonModel(rho, sigma, rd, rf, kappa, eta)

            // contract params
            let timeToMaturity = 1.0
            let strike = 55.0
            let optionType = Call

            // PDE solver params
            let theta = 0.5
            let sMax = 220.0
            let vMax = 4.0
            let ns = 48
            let nv = 32
            let cS = strike/5.0
            let cV = vMax/5.0
            let param = EulerSolverParam(theta, 0.0, sMax, vMax, ns, nv, cS, cV)

            let! s, v, u = eulerSolver heston optionType strike timeToMaturity param

            if verbose then
                let! ss = s.Gather()
                let! vv = v.Gather()
                let! uu = u.ToArray2D()
                printfn "s = %A" ss
                printfn "v = %A" vv
                printfn "u.Lx %d" (uu.GetLength(1))
                printfn "u.Ly %d" (uu.GetLength(0))
                for i = 0 to uu.GetLength(0)-1 do
                    printf "[i = %d] " i
                    for j = 0 to uu.GetLength(1)-1 do
                        printf "%.4f, " uu.[i,j]
                    printf "\n"

            let extend minv maxv = let maxv', _ = Graphics.Direct3D9.SurfacePlotter.defaultExtend minv maxv in maxv', 1.0
            let renderType = Graphics.Direct3D9.SurfacePlotter.RenderType.Mesh
            do! Graphics.Direct3D9.SurfacePlotter.plottingXYLoop context v s u extend extend extend renderType }
        |> PCalc.run

    let cudaDevice = Device.AllDevices.[0]
    let param = Graphics.Direct3D9.Application.Param.Create(cudaDevice)
    let param = { param with FormTitle = "Euler Scheme"
                             DrawingSize = System.Drawing.Size(1024, 768) }
    let application = Graphics.Direct3D9.Application.Application(param, loop)
    application.Start()

[<Test>]
let ``Douglas scheme plotting`` () =
    
    let verbose = true

    let loop (context:Graphics.Direct3D9.Application.Context) =
        pcalc {
            // in real app, we will first compile template into irm, and load here
            // this worker from context, which is capable interop with graphics
            printf "Compiling..."
            let douglasSolver = context.Worker.LoadPModule(douglasSolver).Invoke
            printfn "[OK]"

            // Heston model params
            let rho = -0.6133
            let sigma = 0.3920
            let kappa = 1.1954
            let eta = 0.0677
            let rd = 0.02
            let rf = 0.01
            let heston = HestonModel(rho, sigma, rd, rf, kappa, eta)

            // contract params
            let timeToMaturity = 1.0
            let strike = 55.0
            let optionType = Call

            // PDE solver params
            let theta = 0.5
            let sMax = 220.0
            let vMax = 4.0
            let ns = 126
            let nv = 62
            let cS = strike/5.0
            let cV = vMax/5.0
            let param = EulerSolverParam(theta, 0.0, sMax, vMax, ns, nv, cS, cV)

            let! s, v, u, y0, y1corr, y1 = douglasSolver heston optionType strike timeToMaturity param

            if verbose then
                let! ss = s.Gather()
                let! vv = v.Gather()
                let! uu = u.ToArray2D()
                printfn "s = %A" ss
                printfn "v = %A" vv
                printfn "u.Lx %d" (uu.GetLength(1))
                printfn "u.Ly %d" (uu.GetLength(0))
                for i = 0 to uu.GetLength(0)-1 do
                    printf "[i = %d] " i
                    for j = 0 to uu.GetLength(1)-1 do
                        printf "%.4f, " uu.[i,j]
                    printf "\n"
                printfn "y0"
                let! uu = y0.ToArray2D()
                for i = 0 to uu.GetLength(0)-1 do
                    printf "[i = %d] " i
                    for j = 0 to uu.GetLength(1)-1 do
                        printf "%.4f, " uu.[i,j]
                    printf "\n"
                printfn "y1corr"
                let! uu = y1corr.ToArray2D()
                for i = 0 to uu.GetLength(0)-1 do
                    printf "[i = %d] " i
                    for j = 0 to uu.GetLength(1)-1 do
                        printf "%.4f, " uu.[i,j]
                    printf "\n"
                printfn "y1"
                let! uu = y1.ToArray2D()
                for i = 0 to uu.GetLength(0)-1 do
                    printf "[i = %d] " i
                    for j = 0 to uu.GetLength(1)-1 do
                        printf "%.4f, " uu.[i,j]
                    printf "\n"

            let extend minv maxv = let maxv', _ = Graphics.Direct3D9.SurfacePlotter.defaultExtend minv maxv in maxv', 1.0
            let renderType = Graphics.Direct3D9.SurfacePlotter.RenderType.Mesh
            do! Graphics.Direct3D9.SurfacePlotter.plottingXYLoop context v s u extend extend extend renderType }
        |> PCalc.run

    let cudaDevice = Device.AllDevices.[0]
    let param = Graphics.Direct3D9.Application.Param.Create(cudaDevice)
    let param = { param with FormTitle = "Euler Scheme"
                             DrawingSize = System.Drawing.Size(1024, 768) }
    let application = Graphics.Direct3D9.Application.Application(param, loop)
    application.Start()

