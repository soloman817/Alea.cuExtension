module Test.Alea.CUDA.Extension.Finance.MatrixTiling

open System
open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Finance.MatrixTiling
open Test.Alea.CUDA.Extension

[<Test>]
let ``Matrix tiling`` () =
    
    let verbose = true

    let worker = getDefaultWorker()

    pcalc {
        printf "Compiling..."
        let matrixTiling = worker.LoadPModule(matrixTiling).Invoke
        printfn "[OK]"

        let ns = 10
        let nv = 10
        let! u = matrixTiling ns nv

        if verbose then
            let! uu = u.ToArray2D();
            printfn "u.Lx %d" (uu.GetLength(0))
            printfn "u.Ly %d" (uu.GetLength(1))
            for i = 0 to ns-1 do
                printf "[i = %d] " i
                for j = 0 to nv-1 do
                    printf "%.4f, " uu.[i,j]
                printf "\n"
    } |> PCalc.runWithKernelTiming(1)


[<Test>]
let ``Matrix tiling plotting`` () =
    
    let verbose = true

    let loop (context:Graphics.Direct3D9.Application.Context) =
        pcalc {
            // in real app, we will first compile template into irm, and load here
            // this worker from context, which is capable interop with graphics
            printf "Compiling..."
            let matrixTiling = context.Worker.LoadPModule(matrixTiling).Invoke
            printfn "[OK]"

            let ns = 10
            let nv = 10
            let! u = matrixTiling ns nv

            if verbose then
                let! uu = u.ToArray2D();
                printfn "u.Lx %d" (uu.GetLength(0))
                printfn "u.Ly %d" (uu.GetLength(1))
                for i = 0 to ns-1 do
                    printf "[i = %d] " i
                    for j = 0 to nv-1 do
                        printf "%.4f, " uu.[i,j]
                    printf "\n"

            let extend minv maxv = let maxv', _ = Graphics.Direct3D9.SurfacePlotter.defaultExtend minv maxv in maxv', 1.0
            let renderType = Graphics.Direct3D9.SurfacePlotter.RenderType.Mesh
            do! Graphics.Direct3D9.SurfacePlotter.plottingLoop context u extend renderType }
        |> PCalc.run

    let cudaDevice = Device.AllDevices.[0]
    let param = Graphics.Direct3D9.Application.Param.Create(cudaDevice)
    let param = { param with FormTitle = "Matrix Tiling Scheme"
                             DrawingSize = System.Drawing.Size(1024, 768) }
    let application = Graphics.Direct3D9.Application.Application(param, loop)
    application.Start()

