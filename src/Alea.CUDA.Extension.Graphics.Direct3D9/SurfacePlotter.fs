module Alea.CUDA.Extension.Graphics.Direct3D9.SurfacePlotter

#nowarn "9"
#nowarn "51"

open System
open System.Diagnostics
open System.Runtime.InteropServices
open System.Threading
open SharpDX
open SharpDX.Windows
open SharpDX.Multimedia
open SharpDX.RawInput
open SharpDX.Direct3D
open SharpDX.Direct3D9
open Alea.Interop.CUDA
open Alea.CUDA
open Alea.CUDA.Extension

[<Struct;Align(16)>]
type Vector4 =
    val x : float32
    val y : float32
    val z : float32
    val w : float32

    [<ReflectedDefinition>]
    new (x, y, z, w) = { x = x; y = y; z = z; w = w }

    override this.ToString() = sprintf "(%f,%f,%f,%f)" this.x this.y this.z this.w
    
[<Struct;Align(16)>]
type Vertex =
    val position : Vector4
    val color : Vector4

    [<ReflectedDefinition>]
    new (position, color) = { position = position; color = color }

    override this.ToString() = sprintf "[Position%A,Color%A]" this.position this.color

// minv -> maxv -> maxv', ratio
type ExtendFunc = float -> float -> float * float 

let defaultExtend (minv:float) (maxv:float) =
    let maxv' = if minv = maxv then maxv + 1.0 else maxv
    maxv', 1.0

module Kernels =
    [<Struct;Align(8)>]
    type MatrixDimension =
        val rows : int
        val cols : int

        [<ReflectedDefinition>]
        new (rows, cols) = { rows = rows; cols = cols }    

    let initMeshIBTemplate = cuda {
        let transformRawMajorOrder =
            <@ fun (i:int) (dim:MatrixDimension) ->
                if i < dim.rows * dim.cols then i
                else
                    let i = i - dim.rows * dim.cols
                    let r = i / dim.rows
                    let c = i % dim.rows
                    c * dim.cols + r @>

        let transformColMajorOrder =
            <@ fun (i:int) (dim:MatrixDimension) ->
                if i < dim.rows * dim.cols then i
                else
                    let i = i - dim.rows * dim.cols
                    let r = i / dim.cols
                    let c = i % dim.cols
                    c * dim.rows + r @>

        let! initRawMajorOrder = Transform.fillip "xxx" transformRawMajorOrder
        let! initColMajorOrder = Transform.fillip "xxx" transformColMajorOrder

        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let initRawMajorOrder = initRawMajorOrder.Apply m
            let initColMajorOrder = initColMajorOrder.Apply m

            fun (order:Util.MatrixStorageOrder) (rows:int) (cols:int) (ibRes:CUgraphicsResource) ->
                let action (hint:ActionHint) =
                    fun () ->
                        let mutable ibRes = ibRes
                        cuSafeCall(cuGraphicsMapResources(1u, &&ibRes, 0n))

                        let mutable ibPtr = 0n
                        let mutable ibSize = 0n
                        cuSafeCall(cuGraphicsResourceGetMappedPointer(&&ibPtr, &&ibSize, ibRes))

                        let ib = DevicePtr<int>(ibPtr)
                        let param = MatrixDimension(rows, cols)
                        let n = 2 * rows * cols
                        match order with
                        | Util.RowMajorOrder -> initRawMajorOrder hint n param ib
                        | Util.ColMajorOrder -> initColMajorOrder hint n param ib

                        cuSafeCall(cuGraphicsUnmapResources(1u, &&ibRes, 0n))
                    |> worker.Eval

                pcalc {
                    do! PCalc.action action
                    do! PCalc.force() } ) }

    type InitMeshIBFunc = Util.MatrixStorageOrder -> int -> int -> CUgraphicsResource -> PCalc<unit>

    let initMeshIBIRM =
        fun () ->
            printf "Compiling init mesh index buffer kernel..."
            let irm = initMeshIBTemplate |> markStatic "Alea.CUDA.Extension.Graphics.Direct3D9.SurfacePlotter.Kernels.initMeshIBIRM" |> genirm
            printfn "[OK]"
            irm
        |> Lazy.Create

    [<ReflectedDefinition>]
    let mapColor (value:float) (minv:float) (maxv:float) =
        let mapB (level:float) = max 0.0 (cos (level * MathConstant.CUDART_PI))
        let mapG (level:float) = sin (level * MathConstant.CUDART_PI)
        let mapR (level:float) = max 0.0 (-(cos (level * MathConstant.CUDART_PI)))
        let level = (value - minv) / (maxv - minv)
        Vector4(float32(mapR level), float32(mapG level), float32(mapB level), 1.0f)

    [<Struct;Align(16)>]
    type FillVBParam =
        val rows : int          // 4 byte
        val cols : int          // 4 byte
        val ratioRow : float    // 8 byte
        val ratioCol : float    // 8 byte
        val minValue : float    // 8 byte
        val maxValue : float    // 8 byte
        val ratioValue : float

        new (rows, cols, minValue, maxValue, extend:ExtendFunc) =
            let maxAxis = max rows cols |> float
            let ratioRow = float(rows) / maxAxis
            let ratioCol = float(cols) / maxAxis

            let maxValue', ratioValue = extend minValue maxValue
            let errorMessage() = sprintf "extend function is wrong: %f -> %f -> %f, %f" minValue maxValue maxValue' ratioValue
            if maxValue' <= minValue then failwith (errorMessage())
            if ratioValue <= 0.0 then failwith (errorMessage())
            if maxValue' < maxValue then failwith (errorMessage())

            { rows = rows; cols = cols; ratioRow = ratioRow; ratioCol = ratioCol; minValue = minValue; maxValue = maxValue'; ratioValue = ratioValue }

        override this.ToString() = sprintf "(%d, %d, %f, %f, %f, %f, %f)" this.rows this.cols this.ratioRow this.ratioCol this.minValue this.maxValue this.ratioValue

    let fillVBTemplate = cuda {
        let transform =
            <@ fun (r:int) (c:int) (p:FillVBParam) (v:float) ->
                let x = float32(float(c) / float(p.cols - 1) * p.ratioCol - p.ratioCol / 2.0)
                let z = float32(-(float(r) / float(p.rows - 1) * p.ratioRow) + p.ratioRow / 2.0)

                let y = float32((v - p.minValue) / (p.maxValue - p.minValue) * p.ratioValue - p.ratioValue / 2.0)

                let position = Vector4(x, y, z, 1.0f)
                let color = mapColor v p.minValue p.maxValue

                Vertex(position, color) @>

        let! map = Transform2D.transformip "xxx" transform
        let! max = PArray.reduce <@ fun () -> Double.NegativeInfinity @> <@ max @> <@ Util.identity @>
        let! min = PArray.reduce <@ fun () -> Double.PositiveInfinity @> <@ min @> <@ Util.identity @>

        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let map = map.Apply m
            let max = max.Apply m
            let min = min.Apply m

            fun (extend:ExtendFunc) (surface:DMatrix<float>) (vbRes:CUgraphicsResource) ->
                let action (minValue:float) (maxValue:float) (hint:ActionHint) =
                    fun () ->
                        let mutable vbRes = vbRes
                        cuSafeCall(cuGraphicsMapResources(1u, &&vbRes, 0n))

                        let mutable vbPtr = 0n
                        let mutable vbSize = 0n
                        cuSafeCall(cuGraphicsResourceGetMappedPointer(&&vbPtr, &&vbSize, vbRes))

                        let vb = DevicePtr<Vertex>(vbPtr)
                        let param = FillVBParam(surface.NumRows, surface.NumCols, minValue, maxValue, extend)
                        map hint surface.Order surface.NumRows surface.NumCols param surface.Ptr vb
                    
                        cuSafeCall(cuGraphicsUnmapResources(1u, &&vbRes, 0n))
                    |> worker.Eval 

                pcalc {
                    let! minValue = min surface.Storage
                    let! minValue = minValue.Gather()
                    let! maxValue = max surface.Storage
                    let! maxValue = maxValue.Gather()
                    do! action minValue maxValue |> PCalc.action
                    do! PCalc.force() } ) }

    type FillVBFunc = ExtendFunc -> DMatrix<float> -> CUgraphicsResource -> PCalc<unit>

    let fillVBIRM =
        fun () ->
            printf "Compiling vertex buffer fill kernel..."
            let irm = fillVBTemplate |> markStatic "Alea.CUDA.Extension.Graphics.Direct3D9.SurfacePlotter.Kernels.fillVBIRM" |> genirm
            printfn "[OK]"
            irm
        |> Lazy.Create

type RenderType =
    | Mesh
    | Point

let createPointIndexBuffer (context:Context) = pcalc {
    return new IndexBuffer(context.D3D9Device, sizeof<int>, Usage.WriteOnly, Pool.Managed, false) }

let createMeshIndexBuffer (context:Context) (order:Util.MatrixStorageOrder) (rows:int) (cols:int) = pcalc {
    let initIB = context.Worker.LoadPModule(Kernels.initMeshIBIRM.Value).Invoke
    let ib = new IndexBuffer(context.D3D9Device, sizeof<int> * rows * cols * 2, Usage.WriteOnly, Pool.Default, false)
    let ibRes = context.RegisterGraphicsResource(ib)
    do! initIB order rows cols ibRes
    context.UnregisterGraphicsResource(ibRes)
    return ib }

let createVertexBuffer (context:Context) (elements:int) =
    new VertexBuffer(context.D3D9Device, sizeof<Vertex> * elements, Usage.WriteOnly, VertexFormat.None, Pool.Default)

let createVertexDeclaration (context:Context) =
    let ves = [| VertexElement(0s,  0s, DeclarationType.Float4, DeclarationMethod.Default, DeclarationUsage.Position, 0uy)
                 VertexElement(0s, 16s, DeclarationType.Float4, DeclarationMethod.Default, DeclarationUsage.Color,    0uy)
                 VertexElement.VertexDeclarationEnd |]
    new VertexDeclaration(context.D3D9Device, ves)

let renderingLoop (context:Context) (vd:VertexDeclaration) (vb:VertexBuffer) (order:Util.MatrixStorageOrder) (rows:int) (cols:int) (hook:Stopwatch -> unit) (renderType:RenderType) = pcalc {
    let elements = rows * cols

    use! ib =
        match renderType with
        | RenderType.Mesh -> createMeshIndexBuffer context order rows cols
        | RenderType.Point -> createPointIndexBuffer context

    let eye = Vector3(0.0f, 2.0f, -2.0f)
    let lookat = Vector3(0.0f, 0.0f, 0.0f)
    let up = Vector3(0.0f, 1.0f, 0.0f)

    let view = Matrix.LookAtLH(eye, lookat, up)
    let proj = Matrix.PerspectiveFovLH(Math.PI * 0.25 |> float32, 1.0f, 1.0f, 100.0f)
    let world = ref (Matrix.RotationY(Math.PI * 0.25 |> float32))

    context.D3D9Device.SetTransform(TransformState.View, view)
    context.D3D9Device.SetTransform(TransformState.Projection, proj)
    context.D3D9Device.SetRenderState(RenderState.Lighting, false)

    context.D3D9Device.Indices <- ib
    context.D3D9Device.VertexDeclaration <- vd
    context.D3D9Device.SetStreamSource(0, vb, 0, sizeof<Vertex>)

    let isMouseLeftButtonDown = ref false
    RawInputDevice.RegisterDevice(UsagePage.Generic, UsageId.GenericMouse, DeviceFlags.None)
    RawInputDevice.MouseInput.Add(fun args ->
        //printfn "(x,y):(%d,%d) Buttons: %A State: %A Wheel: %A" args.X args.Y args.ButtonFlags args.Mode args.WheelDelta
        if uint32(args.ButtonFlags &&& MouseButtonFlags.LeftButtonDown) <> 0u then isMouseLeftButtonDown := true
        if uint32(args.ButtonFlags &&& MouseButtonFlags.LeftButtonUp) <> 0u then isMouseLeftButtonDown := false

        if !isMouseLeftButtonDown && args.X <> 0 then
            let r = float(-args.X) / 150.0 * Math.PI * 0.25 |> float32
            world := Matrix.Multiply(!world, Matrix.RotationY(r))

        if !isMouseLeftButtonDown && args.Y <> 0 then
            let r = float(-args.Y) / 150.0 * Math.PI * 0.25 |> float32
            world := Matrix.Multiply(!world, Matrix.RotationX(r))

        match args.WheelDelta with
        | delta when delta > 0 -> world := Matrix.Multiply(!world, Matrix.Scaling(1.01f))
        | delta when delta < 0 -> world := Matrix.Multiply(!world, Matrix.Scaling(0.99f))
        | _ -> ())

    let clock = System.Diagnostics.Stopwatch.StartNew()

    let render () = 
        hook clock

        context.D3D9Device.Clear(ClearFlags.Target ||| ClearFlags.ZBuffer, ColorBGRA(0uy, 40uy, 100uy, 0uy), 1.0f, 0)
        context.D3D9Device.BeginScene()

        context.D3D9Device.SetTransform(TransformState.World, world)

        match renderType with
        | RenderType.Point -> context.D3D9Device.DrawPrimitives(PrimitiveType.PointList, 0, elements)

        | RenderType.Mesh ->
            match order with
            | Util.RowMajorOrder ->
                for r = 0 to rows - 1 do context.D3D9Device.DrawIndexedPrimitive(PrimitiveType.LineStrip, 0, 0, cols, r * cols, cols - 1)
                for c = 0 to cols - 1 do context.D3D9Device.DrawIndexedPrimitive(PrimitiveType.LineStrip, 0, 0, rows, elements + c * rows, rows - 1)
            | Util.ColMajorOrder ->
                for c = 0 to cols - 1 do context.D3D9Device.DrawIndexedPrimitive(PrimitiveType.LineStrip, 0, 0, rows, c * rows, rows - 1)
                for r = 0 to rows - 1 do context.D3D9Device.DrawIndexedPrimitive(PrimitiveType.LineStrip, 0, 0, cols, elements + r * cols, cols - 1)

        context.D3D9Device.EndScene()
        context.D3D9Device.Present()

    RenderLoop.Run(context.Form, RenderLoop.RenderCallback(render)) }

let plottingLoop (context:Context) (surface:DMatrix<float>) (extend:ExtendFunc) (renderType:RenderType) = pcalc {
    use vb = createVertexBuffer context surface.NumElements
    use vd = createVertexDeclaration context

    let fillVB = context.Worker.LoadPModule(Kernels.fillVBIRM.Value).Invoke
    let vbRes = context.RegisterGraphicsResource(vb)
    do! fillVB extend surface vbRes
    context.UnregisterGraphicsResource(vbRes)

    do! renderingLoop context vd vb surface.Order surface.NumRows surface.NumCols (fun _ -> ()) renderType }

let animationLoop (context:Context) (order:Util.MatrixStorageOrder) (rows:int) (cols:int) (gen:float -> PCalc<DMatrix<float>>) (extend:ExtendFunc) (renderType:RenderType) (loopTime:float option) = pcalc {
    use vb = createVertexBuffer context (rows * cols)
    use vd = createVertexDeclaration context

    let vbRes = context.RegisterGraphicsResource(vb)
    let fillVB = context.Worker.LoadPModule(Kernels.fillVBIRM.Value).Invoke

    let hook (clock:Stopwatch) =
        pcalc {
            let time =
                match loopTime with
                | None -> clock.Elapsed.TotalMilliseconds
                | Some(loopTime) ->
                    if loopTime <= 0.0 then clock.Elapsed.TotalMilliseconds
                    else 
                        let time = clock.Elapsed.TotalMilliseconds
                        if time > loopTime then clock.Restart()
                        clock.Elapsed.TotalMilliseconds
            let! surface = gen time
            do! fillVB extend surface vbRes }
        |> PCalc.run

    do! renderingLoop context vd vb order rows cols hook renderType

    context.UnregisterGraphicsResource(vbRes) }


// about code            
let genCodeRepo(filename:string) =
    Environment.clearCodeRepo()
    Environment.startRecordingCode(filename)
    Kernels.initMeshIBIRM.Force() |> ignore
    Kernels.fillVBIRM.Force() |> ignore
    Environment.stopRecordingCode()

let tryLoadCode = Environment.loadCodeRepoFromAssemblyResource(Reflection.Assembly.GetExecutingAssembly(), "SurfacePlotter.pcr")

