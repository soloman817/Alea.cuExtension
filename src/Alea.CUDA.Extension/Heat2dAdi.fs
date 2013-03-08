module Alea.CUDA.Extension.Heat2dAdi

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Alea.CUDA.Extension.TriDiag

open Util 

let [<ReflectedDefinition>] pi = System.Math.PI

let [<ReflectedDefinition>] xSweep (boundary:float -> float -> float -> float) (sourceFunction:float -> float -> float -> float)
                                   nx ny (x:DevicePtr<float>) (y:DevicePtr<float>) Cx Cy dt t0 t1 (u0:DevicePtr<float>) (u1:DevicePtr<float>) =
    let shared = __extern_shared__()
    let h = shared.Reinterpret<'T>()
    let d = h + (nx+1)
    let l = d + (nx+1)
    let u = l + (nx+1)

    let mutable xj = 0.0
    let mutable yi = 0.0

    let mstride = ny+1

    let mutable i = blockIdx.x
    while i <= ny do  
        yi <- y.[i]

        if i = 0 || i = ny then

            let mutable j = threadIdx.x
            while j <= nx do  
                xj <- x.[j]
                u1.[i*mstride+j] <- boundary t1 xj yi 
                j <- j + blockDim.x

            __syncthreads()

        else

            let mutable j = threadIdx.x
            while j <= nx do
                xj <- x.[j]

                if j = 0 then
                    d.[j] <- 1.0
                    u.[j] <- 0.0
                    h.[j] <- boundary t1 xj yi
                else if j = nx then
                    l.[j] <- 0.0
                    d.[j] <- 1.0
                    h.[j] <- boundary t1 xj yi
                else
                    l.[j] <- -Cx
                    d.[j] <- 2.0 + 2.0*Cx
                    u.[j] <- -Cx
                    h.[j] <- 2.0*u0.[i*mstride+j] +
                             Cy*(u0.[(i-1)*mstride+j] - 2.0*u0.[i*mstride+j] + u0.[(i+1)*mstride+j]) +
                             dt*(sourceFunction t1 xj yi)

                j <- j + blockDim.x

            __syncthreads()

            triDiagPcrSingleBlock (nx+1) l d u h
            //linalg::cuda::triDiagonalSystemSolveOpt<RealType>(threadIdx.x, blockDim.x, nx+1, l, d, u, H)

            j <- threadIdx.x
            while j <= nx do  
                u1.[i*mstride+j] <- h.[j]
                j <- j + blockDim.x

            __syncthreads()

        i <- i + gridDim.x

let [<ReflectedDefinition>] ySweep (boundary:float -> float -> float -> float) (sourceFunction:float -> float -> float -> float)
                                   nx ny (x:DevicePtr<float>) (y:DevicePtr<float>) Cx Cy dt t0 t1 (u0:DevicePtr<float>) (u1:DevicePtr<float>) =
    let shared = __extern_shared__()
    let h = shared.Reinterpret<'T>()
    let d = h + (nx+1)
    let l = d + (nx+1)
    let u = l + (nx+1)

    let mutable xj = 0.0
    let mutable yi = 0.0

    let mstride = ny+1

    let mutable j = blockIdx.x
    while j <= nx do

        xj <- x.[j]

        if j = 0 || j = nx then

            let mutable i = threadIdx.x
            while i <= ny do
                yi <- y.[i]
                u1.[i*mstride+j] <- boundary t1 xj yi
                i <- i + blockDim.x

            __syncthreads()
        
        else

            let mutable i = threadIdx.x
            while i <= ny do  
                yi <- y.[i]

                if i = 0 then
                    d.[i] <- 1.0
                    u.[i] <- 0.0
                    h.[i] <- boundary t1 xj yi
                else if i = ny then
                    l.[i] <- 0.0
                    d.[i] <- 1.0
                    h.[i] <- boundary t1 xj yi
                else
                    l.[i] <- -Cy
                    d.[i] <- 2.0 + 2.0*Cy
                    u.[i] <- -Cy
                    h.[i] <- 2.0*u0.[i*mstride+j] +
                             Cx*(u0.[i*mstride+j-1] - 2.0*u0.[i*mstride+j] + u0.[i*mstride+j+1]) +
                             dt*(sourceFunction t1 xj yi)

                i <- i + blockDim.x

            __syncthreads()

            triDiagPcrSingleBlock (ny+1) l d u h

            i <- threadIdx.x
            while i <= ny do 
                u1.[i*mstride+j] <- h.[i]
                i <- i + blockDim.x

            __syncthreads()

    j <- j + gridDim.x

/// Exact solution of heat equation 
///
///     u_t = u_{xx} + u_{yy} + f(t, x, y)
///
/// with boundary condition b(t, x, y) and source function f(t, x, y)
///
let inline adiSolver (initCondExpr:Expr<float -> float -> float -> float>) 
                     (boundaryExpr:Expr<float -> float -> float -> float>) 
                     (sourceExpr:Expr<float -> float -> float -> float>) = cuda {

    let! initCondKernel =     
        <@ fun nx ny t (x:DevicePtr<float>) (y:DevicePtr<float>) (u:DevicePtr<float>) ->
            let initCond = %initCondExpr
            let i = blockIdx.x*blockDim.x + threadIdx.x
            let j = blockIdx.y*blockDim.y + threadIdx.y
            let mstride = ny+1
            if i <= nx && j <= ny then u.[i*mstride+j] <- initCond t x.[i] y.[j] @> |> defineKernelFunc

    let! xSweepKernel =     
        <@ fun nx ny (x:DevicePtr<float>) (y:DevicePtr<float>) Cx Cy dt t0 t1 (u0:DevicePtr<float>) (u1:DevicePtr<float>) ->     
            let boundary = %boundaryExpr
            let source = % sourceExpr     
            xSweep boundary source nx ny x y Cx Cy dt t0 t1 u0 u1 @> |> defineKernelFunc

    let! ySweepKernel =     
        <@ fun nx ny (x:DevicePtr<float>) (y:DevicePtr<float>) Cx Cy dt t0 t1 (u0:DevicePtr<float>) (u1:DevicePtr<float>) ->          
            let boundary = %boundaryExpr
            let source = % sourceExpr     
            ySweep boundary source nx ny x y Cx Cy dt t0 t1 u0 u1 @> |> defineKernelFunc

    return PFunc(fun (m:Module) k tstart tstop Lx Ly nx ny dt  ->
        let maxThreads = m.Worker.Device.Attribute DeviceAttribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        let dx = Lx / float(nx)
        let dy = Ly / float(ny)
        let Cx = k * dt / (dx*dx)
        let Cy = k * dt / (dy*dy)
        let x = Array.init (nx+1) (fun i -> float(i)*dx)
        let y = Array.init (ny+1) (fun i -> float(i)*dy)
        use dx = m.Worker.Malloc(x)
        use dy = m.Worker.Malloc(y)

        let usize = (nx+1)*(ny+1)
        use du0 = m.Worker.Malloc<float>(usize)
        use du1 = m.Worker.Malloc<float>(usize)

        let lp = LaunchParam(dim3(divup (nx+1) 256, divup (ny+1) 256), dim3(256, 256))
        let lpx = LaunchParam(ny+1, nx+1, 4*(nx+1)*sizeof<float>)
        let lpy = LaunchParam(nx+1, ny+1, 4*(ny+1)*sizeof<float>)

        let initCondKernelFunc = initCondKernel.Launch m lp 
        let xSweepKernelFunc = xSweepKernel.Launch m lpx
        let ySweepKernelFunc = ySweepKernel.Launch m lpx

        initCondKernelFunc nx ny tstart dx.Ptr dy.Ptr du0.Ptr

        let mutable t0 = tstart
        let mutable t1 = tstart
        while t1 <= tstop do
            t0 <- t1
            t1 <- t1 + 0.5 * dt

            printfn "xSweep t = %f (dt = %f)" t0 dt

            // x-direction sweep
            xSweepKernelFunc nx ny dx.Ptr dy.Ptr Cx Cy dt t0 t1 du0.Ptr du1.Ptr

            printfn "done xSweep t = %f" t0

            t0 <- t1
            t1 <- t1 + 0.5 * dt

            printfn "ySweep t = %f" t0

            // y-direction sweep
            ySweepKernelFunc nx ny dx.Ptr dy.Ptr Cx Cy dt t0 t1 du1.Ptr du0.Ptr

            printfn "done ySweep t = %f" t0

        x, y, du0.ToHost()) }


