open System
open System.Runtime.InteropServices
open Microsoft.FSharp.NativeInterop
open SharpDX
open SharpDX.Direct3D9
open SharpDX.Windows
open Alea.CUDA

// these two warning switch will turn off those warning that we are operating on unsafe 
// native stuff
#nowarn "9"
#nowarn "51"

// These two functions are used to create CUDA context with D3D9 interoperability and to register
// the vertex buffer as a CUDA graphic resource.
[<DllImport("nvcuda.dll", EntryPoint="cuD3D9CtxCreate_v2", CallingConvention=CallingConvention.StdCall)>]
extern CUresult cuD3D9CtxCreate(CUcontext* pCtx, CUdevice* pCudaDevice, uint32 Flags, nativeint pD3DDevice);

[<DllImport("nvcuda.dll", EntryPoint="cuGraphicsD3D9RegisterResource", CallingConvention=CallingConvention.StdCall)>]
extern CUresult cuGraphicsD3D9RegisterResource (CUgraphicsResource* pCudaResource, nativeint pD3DResource, uint32  Flags);

type Color = SharpDX.Color
type D3D9Device = SharpDX.Direct3D9.Device

// width and height should all be multiple of 16 according to the implementation
let width = 1024
let height = 1024
let total = width * height

// windowSize should be large, because we have 1024x1024 points, if too small, not good for display
let windowSize =
    let candidates = [| Drawing.Size(800, 600)
                        Drawing.Size(1024, 768)
                        Drawing.Size(1800, 1100) |]
    printfn "Please choose the window size:"
    candidates |> Array.iteri (fun i size -> printfn "(%d) %A" i size)
    printf "Please choose: "
    let selection = int32(Console.Read()) - 48
    candidates.[selection]

// a switcher for gpu or cpu calculation
let usegpu = true

let updateVerticesByCPU (vertices:VertexBuffer) (time:float32) =
    let genwave (time:float32) =
        Array.init total (fun i ->
            let x = i % width
            let y = i / width

            let u = float32(x) / float32(width)
            let v = float32(y) / float32(height)
            let u = u * 2.0f - 1.0f
            let v = v * 2.0f - 1.0f

            let freq = 4.0f
            let w = sin(u * freq + time) * cos(v * freq + time) * 0.5f

            Vector4(u, w, v, __nv_int_as_float(0xff00ff00)))

    vertices.Lock(0, 0, LockFlags.None).WriteRange(genwave time)
    vertices.Unlock()

// This struct has same layout as SharpDX.Vector4, but aligned with 16
[<Struct;Align(16)>]
type Vector4A16 =
    val x : float32
    val y : float32
    val z : float32
    val w : float32

    [<ReflectedDefinition>]
    new (x, y, z, w) = { x = x; y = y; z = z; w = w }

let updateVerticesByGPU = cuda {
    let! genwave =
        <@ fun (pos:deviceptr<Vector4A16>) (width:int) (height:int) (time:float32) ->
            let x = blockIdx.x * blockDim.x + threadIdx.x
            let y = blockIdx.y * blockDim.y + threadIdx.y

            let u = float32(x) / float32(width)
            let v = float32(y) / float32(height)
            let u = u * 2.0f - 1.0f
            let v = v * 2.0f - 1.0f

            let freq = 4.0f
            let w = sin(u * freq + time) * cos(v * freq + time) * 0.5f

            pos.[y * width + x] <- Vector4A16(u, w, v, __nv_int_as_float(0xff00ff00)) @>
        |> Compiler.DefineKernel

    return Entry(fun (program:Program) ->
        let worker = program.Worker
        let genwave = program.Apply(genwave)
        let blockSize = dim3(16, 16)
        let gridSize = dim3(width / blockSize.x, height / blockSize.y)
        let lp = LaunchParam(gridSize, blockSize)
        let genwave = genwave.Launch lp

        let update (vbRes:CUgraphicsResource) (time:float32) =
            fun () ->
                // 1. map resource to cuda space, means lock to cuda space
                let mutable vbRes = vbRes
                cuSafeCall(cuGraphicsMapResources(1u, &&vbRes, 0n))

                // 2. get memory pointer from mapped resource
                let mutable vbPtr = 0n
                let mutable vbSize = 0n
                cuSafeCall(cuGraphicsResourceGetMappedPointer(&&vbPtr, &&vbSize, vbRes))

                // 3. create device pointer, and run the kernel
                let pos = deviceptr<Vector4A16>(vbPtr)
                genwave pos width height time
                
                // 4. unmap resource, means unlock, so that DirectX can then use it again
                cuSafeCall(cuGraphicsUnmapResources(1u, &&vbRes, 0n))

            // make sure it is run with worker.Eval, because we are calling raw CUDA Driver API,
            // which has an implicity that it will be run under a thread which has correct CUDA
            // context created! IMPORTANT!
            |> worker.Thread.Eval

        // we return the update function as the entry of this template.
        update ) }

[<EntryPoint;STAThread>] // a Windows application must be in STAThread
let main argv =
    // create a form
    use form = new RenderForm("SimplD3D9")
    form.ClientSize <- windowSize

    // create a D3D9 device
    use device = new D3D9Device(new Direct3D(), // Direct3D interface (COM)
                                Device.Devices.[0].ID, // display adapter (device id)
                                DeviceType.Hardware, // device type
                                form.Handle, // focus window
                                CreateFlags.HardwareVertexProcessing, // behavior flags
                                PresentParameters(form.ClientSize.Width, form.ClientSize.Height))

    let ty = typeof<Vector4>

    // create vertex buffer, NOTICE, the pool type MUST be Pool.Default, which let it possible for CUDA to process.
    use vertices = new VertexBuffer(device, Utilities.SizeOf<Vector4>() * total, Usage.WriteOnly, VertexFormat.None, Pool.Default)

    // define the FVF of the vertex, first 3 float is for position, last float will be reinterpreted to 4 bytes for the color
    let vertexElems = [| VertexElement(0s, 0s, DeclarationType.Float3, DeclarationMethod.Default, DeclarationUsage.Position, 0uy)
                         VertexElement(0s, 12s, DeclarationType.Ubyte4, DeclarationMethod.Default, DeclarationUsage.Color, 0uy)
                         VertexElement.VertexDeclarationEnd |]
    use vertexDecl = new VertexDeclaration(device, vertexElems)

    let view = Matrix.LookAtLH(Vector3(0.0f, 3.0f, -2.0f),          // the camera position
                               Vector3(0.0f, 0.0f, 0.0f),           // the look-at position
                               Vector3(0.0f, 1.0f, 0.0f))           // the up direction
    let proj = Matrix.PerspectiveFovLH(float32(Math.PI / 4.0),      // the horizontal field of view
                                       1.0f,
                                       1.0f,
                                       100.0f)

    // IMPORTANT: to interop with CUDA, CUDA context must use special API to create, so we use a customized device worker
    // constructor, which takes a context generation function: unit -> nativeint * Engine.Device. the device worker will
    // call it in its working thread later, which thus bind the cuda context to the working thread.
    use worker =
        let generate() =
            let mutable ctx = 0n
            let mutable dev = -1
            cuSafeCall(cuD3D9CtxCreate(&&ctx, &&dev, 0u, device.NativePointer))
            let dev = Device.DeviceDict.[dev]
            dev, ctx
        Worker.Create(generate)

    // compile the gpu module if needed
    let updateVerticesByGPU =
        if usegpu then
            printf "Compiling..."
            let updateVerticesByGPU = worker.LoadProgram(updateVerticesByGPU).Run
            printfn "[OK]"
            updateVerticesByGPU
        else
            fun _ _ -> ()
   
    // register and unregister the vertex buffer as CUDA graphic resource.
    // NOTICE, they are all called by worker.Eval. so again, they are raw CUDA Driver API
    // and need to be called within the worker's thread.
    let registerVerticesResource() =
        fun () ->
            let mutable res = 0n
            cuSafeCall(cuGraphicsD3D9RegisterResource(&&res, vertices.NativePointer, 0u))
            res
        |> worker.Thread.Eval

    let unregisterVerticesResource res =
        fun () -> cuSafeCall(cuGraphicsUnregisterResource(res))
        |> worker.Thread.Eval

    let vbres = if usegpu then registerVerticesResource() else 0n

    device.SetTransform(TransformState.View, view)
    device.SetTransform(TransformState.Projection, proj)
    device.SetRenderState(RenderState.Lighting, false)

    let clock = System.Diagnostics.Stopwatch.StartNew()

    let render () = 
        // on each render, first calculate the vertex buffer.
        let time = float32(clock.Elapsed.TotalMilliseconds / 300.0 )
        if usegpu then updateVerticesByGPU vbres time
        else updateVerticesByCPU vertices time
        
        // Now normal D3D9 render procedure.
        device.Clear(ClearFlags.Target ||| ClearFlags.ZBuffer, ColorBGRA(0uy, 40uy, 100uy, 0uy), 1.0f, 0)
        device.BeginScene()

        device.VertexDeclaration <- vertexDecl
        device.SetStreamSource(0, vertices, 0, Utilities.SizeOf<Vector4>())
        // we use PointList as the graphics primitives
        device.DrawPrimitives(PrimitiveType.PointList, 0, total)

        device.EndScene()
        device.Present()

    RenderLoop.Run(form, RenderLoop.RenderCallback(render))

    // unregister the vertex buffer
    if usegpu then unregisterVerticesResource vbres

    0 // return an integer exit code
