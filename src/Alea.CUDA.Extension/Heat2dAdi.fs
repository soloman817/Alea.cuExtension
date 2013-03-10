module Alea.CUDA.Extension.Heat2dAdi

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Alea.CUDA.Extension.TriDiag

open Util 

let [<ReflectedDefinition>] pi = System.Math.PI

/// Solves ny-1 systems of dimension nx+1 in the x-coordinate direction 
[<ReflectedDefinition>]
let xSweep (boundary:float -> float -> float -> float) (sourceFunction:float -> float -> float -> float)
           nx ny (x:DevicePtr<float>) (y:DevicePtr<float>) (Cx:float) (Cy:float) (dt:float) (t0:float) (t1:float) (u0:DevicePtr<float>) (u1:DevicePtr<float>) =
    let shared = __extern_shared__()
    let h = shared.Reinterpret<float>()
    let d = h + (nx+1)
    let l = d + (nx+1)
    let u = l + (nx+1)

    let mutable xi = 0.0
    let mutable yj = 0.0

    let mstride = ny+1

    let mutable j = blockIdx.x
    while j <= ny do  
        yj <- y.[j]

        if j = 0 || j = ny then

            let mutable i = threadIdx.x
            while i <= nx do  
                xi <- x.[i]
                u1.[i*mstride+j] <- boundary t1 xi yj 
                i <- i + blockDim.x

            __syncthreads()

        else

            let mutable i = threadIdx.x
            while i <= nx do
                xi <- x.[i]

                if i = 0 then
                    d.[i] <- 1.0
                    u.[i] <- 0.0
                    h.[i] <- boundary t1 xi yj
                else if i = nx then
                    l.[i] <- 0.0
                    d.[i] <- 1.0
                    h.[i] <- boundary t1 xi yj
                else
                    l.[i] <- -Cx
                    d.[i] <- 2.0 + 2.0*Cx
                    u.[i] <- -Cx
                    h.[i] <- 2.0*u0.[i*mstride+j] +
                             Cy*(u0.[i*mstride+(j-1)] - 2.0*u0.[i*mstride+j] + u0.[i*mstride+(j+1)]) +
                             dt*(sourceFunction t1 xi yj)

                i <- i + blockDim.x

            __syncthreads()

            triDiagPcrSingleBlock (nx+1) l d u h

            i <- threadIdx.x
            while i <= nx do  
                u1.[i*mstride+j] <- h.[i]
                i <- i + blockDim.x

            __syncthreads()

        j <- j + gridDim.x

/// Solves nx-1 systems of dimension ny+1 in the y-coordinate direction 
[<ReflectedDefinition>]
let ySweep (boundary:float -> float -> float -> float) (sourceFunction:float -> float -> float -> float)
           nx ny (x:DevicePtr<float>) (y:DevicePtr<float>) (Cx:float) (Cy:float) (dt:float) (t0:float) (t1:float) (u0:DevicePtr<float>) (u1:DevicePtr<float>) =
    let shared = __extern_shared__()
    let h = shared.Reinterpret<float>()
    let d = h + (nx+1)
    let l = d + (nx+1)
    let u = l + (nx+1)

    let mutable xi = 0.0
    let mutable yj = 0.0

    let mstride = ny+1

    let mutable i = blockIdx.x
    while i <= nx do

        xi <- x.[i]

        if i = 0 || i = nx then

            let mutable j = threadIdx.x
            while j <= ny do
                yj <- y.[j]
                u1.[i*mstride+j] <- boundary t1 xi yj
                j <- j + blockDim.x

            __syncthreads()
        
        else

            let mutable j = threadIdx.x
            while j <= ny do  
                yj <- y.[j]

                if j = 0 then
                    d.[j] <- 1.0
                    u.[j] <- 0.0
                    h.[j] <- boundary t1 xi yj
                else if j = ny then
                    l.[j] <- 0.0
                    d.[j] <- 1.0
                    h.[j] <- boundary t1 xi yj
                else
                    l.[j] <- -Cy
                    d.[j] <- 2.0 + 2.0*Cy
                    u.[j] <- -Cy
                    h.[j] <- 2.0*u0.[i*mstride+j] +
                             Cx*(u0.[(i-1)*mstride+j] - 2.0*u0.[i*mstride+j] + u0.[(i+1)*mstride+j]) +
                             dt*(sourceFunction t1 xi yj)

                j <- j + blockDim.x

            __syncthreads()

            triDiagPcrSingleBlock (ny+1) l d u h

            j <- threadIdx.x
            while j <= ny do 
                u1.[i*mstride+j] <- h.[j]
                j <- j + blockDim.x

            __syncthreads()

        i <- i + gridDim.x

/// Exact solution of heat equation 
///
///     u_t = u_{xx} + u_{yy} + f(t, x, y)
///
/// with boundary condition b(t, x, y) and source function f(t, x, y)
///
/// The x coordinate is mapped to the rows and the y coordinate to the columns
/// of the solution matrix. The solution matrix is stored in row major format.
let adiSolver (initCondExpr:Expr<float -> float -> float -> float>) 
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
            let source = %sourceExpr     
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

        let lp = LaunchParam(dim3(divup (nx+1) 16, divup (ny+1) 16), dim3(16, 16))
        let lpx = LaunchParam(ny+1, nx+1, 4*(nx+1)*sizeof<float>)
        let lpy = LaunchParam(nx+1, ny+1, 4*(ny+1)*sizeof<float>)

        let initCondKernelFunc = initCondKernel.Launch m lp 
        let xSweepKernelFunc = xSweepKernel.Launch m lpx
        let ySweepKernelFunc = ySweepKernel.Launch m lpx

        initCondKernelFunc nx ny tstart dx.Ptr dy.Ptr du0.Ptr

        let mutable t0 = tstart
        let mutable t1 = tstart      
        while t1 < tstop do
            t0 <- t1
            t1 <- t1 + 0.5 * dt

            xSweepKernelFunc nx ny dx.Ptr dy.Ptr Cx Cy dt t0 t1 du0.Ptr du1.Ptr

            t0 <- t1
            t1 <- t1 + 0.5 * dt

            ySweepKernelFunc nx ny dx.Ptr dy.Ptr Cx Cy dt t0 t1 du1.Ptr du0.Ptr

        x, y, du0.ToHost()) }


