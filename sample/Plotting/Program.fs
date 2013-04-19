open System
open System.Runtime.InteropServices
open System.Threading
open SharpDX
open SharpDX.Windows
open SharpDX.Direct3D
open SharpDX.Direct3D9
open Alea.Interop.CUDA
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Finance
open Alea.CUDA.Extension.Graphics.Direct3D9

module WaveSurface =

    [<Struct;Align(16)>]
    type Param =
        val rows : int
        val cols : int
        val time : float

        new (rows, cols, time) = { rows = rows; cols = cols; time = time }

    let fillIRM =
        let transform =
            <@ fun (r:int) (c:int) (p:Param) ->
                let u = float(c) / float(p.cols)
                let v = float(r) / float(p.rows)
                let u = u * 2.0 - 1.0
                let v = v * 2.0 - 1.0

                let freq = 4.0
                sin(u * freq + p.time) * cos(v * freq + p.time) @>

        fun () ->
            printf "Compiling wave surface kernel..."
            let irm = PMatrix.fillp transform |> markStatic "Plotting.Program.WaveSurface.fillIRM" |> genirm
            printfn "[OK]"
            irm
        |> Lazy.Create

    let plottingLoop (order:Util.MatrixStorageOrder) (rows:int) (cols:int) renderType (context:Context) =
        pcalc {
            let fill = context.Worker.LoadPModule(fillIRM.Value).Invoke
            let! surface = DMatrix.createInBlob context.Worker order rows cols
            let param = Param(rows, cols, 0.0)
            do! fill param surface
            let extend minv maxv = let maxv', _ = SurfacePlotter.defaultExtend minv maxv in maxv', 0.5
            do! SurfacePlotter.plottingLoop context surface extend renderType }
        |> PCalc.run

    let animationLoop (order:Util.MatrixStorageOrder) (rows:int) (cols:int) renderType (context:Context) =
        pcalc {
            let fill = context.Worker.LoadPModule(fillIRM.Value).Invoke
            let! surface = DMatrix.createInBlob context.Worker order rows cols

            let gen (time:float) = pcalc {
                let param = Param(rows, cols, time / 400.0)
                do! fill param surface
                return surface }

            let extend minv maxv = let maxv', _ = SurfacePlotter.defaultExtend minv maxv in maxv', 0.5
            do! SurfacePlotter.animationLoop context order rows cols gen extend renderType None }
        |> PCalc.run

module HeatGauss =
    let solver init boundary source = cuda {
        let! solver = Heat2dAdi.build init boundary source

        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let solver = solver.Apply m
            fun nx ny -> solver nx ny) }

module HeatGauss1 =
    let [<ReflectedDefinition>] pi = System.Math.PI
    let [<ReflectedDefinition>] sigma1 = 0.04
    let [<ReflectedDefinition>] sigma2 = 0.04
    let [<ReflectedDefinition>] sigma3 = 0.04
    let initialCondExpr =
        <@ fun t x y -> 1.0/3.0*exp (-((x-0.2)*(x-0.2) + (y-0.2)*(y-0.2))/(2.0*sigma1*sigma1)) / (sigma1*sigma1*2.0*pi) +
                        1.0/3.0*exp (-((x-0.8)*(x-0.8) + (y-0.8)*(y-0.8))/(2.0*sigma2*sigma2)) / (sigma2*sigma2*2.0*pi) +
                        1.0/3.0*exp (-((x-0.8)*(x-0.8) + (y-0.2)*(y-0.2))/(2.0*sigma3*sigma3)) / (sigma3*sigma3*2.0*pi) @>
    let boundaryExpr = <@ fun t x y -> 0.0 @>
    let sourceFunctionExpr = <@ fun t x y -> 0.0 @>

    let solverIRM =
        fun () ->
            printf "Compiling heat gauss 1 kernel..."
            let irm = HeatGauss.solver initialCondExpr boundaryExpr sourceFunctionExpr |> markStatic "Plotting.Program.HeatGuass1.solverIRM" |> genirm
            printfn "[OK]"
            irm
        |> Lazy.Create

    let plottingLoop nx ny renderType (context:Context) =
        pcalc {
            let k = 1.0
            let tstart = 0.0
            let Lx = 1.0
            let Ly = 1.0
            let dt = 0.01

            let solver = context.Worker.LoadPModule(solverIRM.Value).Invoke
            let tstop = 0.0

            let solver = solver nx ny
            let t = Heat2dAdi.timeGrid tstart tstop dt 5
            let x, dx = solver.GenX Lx
            let y, dy = solver.GenY Ly

            let! x' = DArray.scatterInBlob context.Worker x
            let! y' = DArray.scatterInBlob context.Worker y
            let! u0 = DMatrix.createInBlob context.Worker Util.RowMajorOrder ny nx
            let! u1 = DMatrix.createInBlob context.Worker Util.RowMajorOrder ny nx
            do! PCalc.action (fun hint -> solver.Launch hint t x'.Ptr dx y'.Ptr dy u0.Ptr u1.Ptr k tstart tstop dt)

            let surface = u0
            let extend minv maxv = SurfacePlotter.defaultExtend minv maxv
            do! SurfacePlotter.plottingLoop context surface extend renderType }
        |> PCalc.run

    let animationLoop nx ny renderType (context:Context) =
        pcalc {
            let k = 1.0
            let tstart = 0.0
            let Lx = 1.0
            let Ly = 1.0
            let dt = 0.01

            let solver = context.Worker.LoadPModule(solverIRM.Value).Invoke

            let solver = solver nx ny
            let x, dx = solver.GenX Lx
            let y, dy = solver.GenY Ly

            let! x' = DArray.scatterInBlob context.Worker x
            let! y' = DArray.scatterInBlob context.Worker y
            let! u0 = DMatrix.createInBlob context.Worker Util.RowMajorOrder ny nx
            let! u1 = DMatrix.createInBlob context.Worker Util.RowMajorOrder ny nx

            let gen (tstop:float) = pcalc {
                let tstop = tstop / 1000.0 / 100.0
                let t = Heat2dAdi.timeGrid tstart tstop dt 5
                do! PCalc.action (fun hint -> solver.Launch hint t x'.Ptr dx y'.Ptr dy u0.Ptr u1.Ptr k tstart tstop dt)
                return u0 }

            let extend minv maxv = max maxv 1.0, 1.0
            do! SurfacePlotter.animationLoop context Util.RowMajorOrder ny nx gen extend renderType (Some(15.0 * 1000.0)) }
        |> PCalc.run

module HeatGauss2 =
    let [<ReflectedDefinition>] pi = System.Math.PI
    let initialCondExpr = <@ fun t x y -> exp(-t) * sin(pi*x) * cos(pi*y) @>
    let boundaryExpr = <@ fun t x y -> exp(-t) * sin(pi*x) * cos(pi*y) @>
    let sourceFunctionExpr = <@ fun t x y -> exp(-t) * sin(pi*x) * cos(pi*y) * (2.0*pi*pi - 1.0) @>

    let solverIRM =
        fun () ->
            printf "Compiling heat gauss 2 kernel..."
            let irm = HeatGauss.solver initialCondExpr boundaryExpr sourceFunctionExpr |> markStatic "Plotting.Program.HeatGuass2.solverIRM" |> genirm
            printfn "[OK]"
            irm
        |> Lazy.Create

    let plottingLoop nx ny renderType (context:Context) =
        pcalc {
            let k = 1.0
            let tstart = 0.0
            let Lx = 1.0
            let Ly = 1.0
            let dt = 0.01

            let solver = context.Worker.LoadPModule(solverIRM.Value).Invoke
            let tstop = 0.0

            let solver = solver nx ny
            let t = Heat2dAdi.timeGrid tstart tstop dt 5
            let x, dx = solver.GenX Lx
            let y, dy = solver.GenY Ly

            let! x' = DArray.scatterInBlob context.Worker x
            let! y' = DArray.scatterInBlob context.Worker y
            let! u0 = DMatrix.createInBlob context.Worker Util.RowMajorOrder ny nx
            let! u1 = DMatrix.createInBlob context.Worker Util.RowMajorOrder ny nx
            do! PCalc.action (fun hint -> solver.Launch hint t x'.Ptr dx y'.Ptr dy u0.Ptr u1.Ptr k tstart tstop dt)

            let surface = u0
            let extend minv maxv = SurfacePlotter.defaultExtend minv maxv
            do! SurfacePlotter.plottingLoop context surface extend renderType }
        |> PCalc.run

    let animationLoop nx ny renderType (context:Context) =
        pcalc {
            let k = 1.0
            let tstart = 0.0
            let Lx = 1.0
            let Ly = 1.0
            let dt = 0.01

            let solver = context.Worker.LoadPModule(solverIRM.Value).Invoke

            let solver = solver nx ny
            let x, dx = solver.GenX Lx
            let y, dy = solver.GenY Ly

            let! x' = DArray.scatterInBlob context.Worker x
            let! y' = DArray.scatterInBlob context.Worker y
            let! u0 = DMatrix.createInBlob context.Worker Util.RowMajorOrder ny nx
            let! u1 = DMatrix.createInBlob context.Worker Util.RowMajorOrder ny nx

            let gen (tstop:float) = pcalc {
                let tstop = tstop / 1000.0 / 3.0
                let t = Heat2dAdi.timeGrid tstart tstop dt 5
                do! PCalc.action (fun hint -> solver.Launch hint t x'.Ptr dx y'.Ptr dy u0.Ptr u1.Ptr k tstart tstop dt)
                return u0 }

            let extend minv maxv = max maxv 1.0, 1.0
            do! SurfacePlotter.animationLoop context Util.RowMajorOrder ny nx gen extend renderType (Some(15.0 * 1000.0)) }
        |> PCalc.run

let cudaDevice = Device.AllDevices.[0]
let param = Param.Create(cudaDevice)

let plotWaveSurface() =
    let application = Application({ param with FormTitle = "Wave Surface Plotting"
                                               DrawingSize = Drawing.Size(1024, 768) },
                                  WaveSurface.plottingLoop Util.ColMajorOrder 3000 3000 SurfacePlotter.Mesh)
    application.Start()

let animateWaveSurface() =
    let application = Application({ param with FormTitle = "Wave Surface Animation"
                                               DrawingSize = Drawing.Size(1024, 768) },
                                  WaveSurface.animationLoop Util.RowMajorOrder 1000 1000 SurfacePlotter.Point)
    application.Start()

let plotHeatGauss1() =
    let application = Application({ param with FormTitle = "Heat Gauss 1 Plotting"
                                               DrawingSize = Drawing.Size(1024, 768) },
                                  HeatGauss1.plottingLoop 512 512 SurfacePlotter.Mesh)
    application.Start()

let animateHeatGauss1() =
    let application = Application({ param with FormTitle = "Heat Gauss 1 Animation"
                                               DrawingSize = Drawing.Size(1024, 768) },
                                  HeatGauss1.animationLoop 128 128 SurfacePlotter.Mesh)
    application.Start()

let plotHeatGauss2() =
    let application = Application({ param with FormTitle = "Heat Gauss 2 Plotting"
                                               DrawingSize = Drawing.Size(1024, 768) },
                                  HeatGauss2.plottingLoop 512 512 SurfacePlotter.Mesh)
    application.Start()

let animateHeatGauss2() =
    let application = Application({ param with FormTitle = "Heat Gauss 2 Animation"
                                               DrawingSize = Drawing.Size(1024, 768) },
                                  HeatGauss2.animationLoop 128 128 SurfacePlotter.Mesh)
    application.Start()

// about code            
let genCodeRepo() =
    SurfacePlotter.genCodeRepo(@"..\..\..\..\src\Alea.CUDA.Extension.Graphics.Direct3D9\SurfacePlotter.pcr")
    Environment.clearCodeRepo()
    Environment.startRecordingCode(@"..\..\Program.pcr")
    WaveSurface.fillIRM.Force() |> ignore
    HeatGauss1.solverIRM.Force() |> ignore
    HeatGauss2.solverIRM.Force() |> ignore
    Environment.stopRecordingCode()

let tryLoadCode = Environment.loadCodeRepoFromAssemblyResource(Reflection.Assembly.GetExecutingAssembly(), "Program.pcr")
    
//genCodeRepo() 

plotWaveSurface()
animateWaveSurface()
plotHeatGauss1()
animateHeatGauss1()
plotHeatGauss2()
animateHeatGauss2()

