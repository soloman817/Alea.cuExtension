module Alea.CUDA.Extension.Finance.Heat2dAdi

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Alea.CUDA.Extension
open Alea.CUDA.Extension.TriDiag
open Alea.CUDA.Extension.Finance.Grid

open Util 

/// Simple homogeneous state grid startig at zero, covering interval [0, L]
let stateGrid = homogeneousGrid

/// Create a time grid up to tstop of step size not larger than dt, with nc condensing points in the first interval
let timeGrid = exponentiallyCondensedGrid 

/// Solves ny-2 systems of dimension nx in the x-coordinate direction 
[<ReflectedDefinition>]
let xSweep (boundary:float -> float -> float -> float) (sourceFunction:float -> float -> float -> float)
           nx ny (x:DevicePtr<float>) (y:DevicePtr<float>) (Cx:float) (Cy:float) (dt:float) (t0:float) (t1:float) (u0:DevicePtr<float>) (u1:DevicePtr<float>) =
    let shared = __extern_shared__<float>()
    let h = shared
    let d = h + nx
    let l = d + nx
    let u = l + nx

    let mutable xi = 0.0
    let mutable yj = 0.0

    let mstride = ny

    let mutable j = blockIdx.x
    while j < ny do  
        yj <- y.[j]

        if j = 0 || j = ny-1 then

            let mutable i = threadIdx.x
            while i < nx do  
                xi <- x.[i]
                u1.[i*mstride+j] <- boundary t1 xi yj 
                i <- i + blockDim.x

            __syncthreads()

        else

            let mutable i = threadIdx.x
            while i < nx do
                xi <- x.[i]

                if i = 0 then
                    d.[i] <- 1.0
                    u.[i] <- 0.0
                    h.[i] <- boundary t1 xi yj
                else if i = nx-1 then
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

            triDiagPcrSingleBlock nx l d u h

            i <- threadIdx.x
            while i < nx do  
                u1.[i*mstride+j] <- h.[i]
                i <- i + blockDim.x

            __syncthreads()

        j <- j + gridDim.x

/// Solves nx-2 systems of dimension ny in the y-coordinate direction 
[<ReflectedDefinition>]
let ySweep (boundary:float -> float -> float -> float) (sourceFunction:float -> float -> float -> float)
           nx ny (x:DevicePtr<float>) (y:DevicePtr<float>) (Cx:float) (Cy:float) (dt:float) (t0:float) (t1:float) (u0:DevicePtr<float>) (u1:DevicePtr<float>) =
    let shared = __extern_shared__<float>()
    let h = shared
    let d = h + ny
    let l = d + ny
    let u = l + ny

    let mutable xi = 0.0
    let mutable yj = 0.0

    let mstride = ny

    let mutable i = blockIdx.x
    while i < nx do

        xi <- x.[i]

        if i = 0 || i = nx-1 then

            let mutable j = threadIdx.x
            while j < ny do
                yj <- y.[j]
                u1.[i*mstride+j] <- boundary t1 xi yj
                j <- j + blockDim.x

            __syncthreads()
        
        else

            let mutable j = threadIdx.x
            while j < ny do  
                yj <- y.[j]

                if j = 0 then
                    d.[j] <- 1.0
                    u.[j] <- 0.0
                    h.[j] <- boundary t1 xi yj
                else if j = ny-1 then
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

            triDiagPcrSingleBlock ny l d u h

            j <- threadIdx.x
            while j < ny do 
                u1.[i*mstride+j] <- h.[j]
                j <- j + blockDim.x

            __syncthreads()

        i <- i + gridDim.x

// Launch hint x dx y dy u0 u1 k tstart tstop dt
type ISolver =
    abstract GenT : float -> float -> float -> float[] // tstart -> tstop -> dt -> timeGrid
    abstract GenX : float -> float[] * float // Lx -> x, dx
    abstract GenY : float -> float[] * float // Ly -> y, dy
    abstract NumU : int
    abstract Launch : ActionHint -> float[] -> DevicePtr<float> -> float -> DevicePtr<float> -> float -> DevicePtr<float> -> DevicePtr<float> -> float -> float -> float -> float -> unit

/// Solve heat equation 
///
///     u_t = u_{xx} + u_{yy} + f(t, x, y)
///
/// with boundary condition b(t, x, y) and source function f(t, x, y)
///
/// The x coordinate is mapped to the rows and the y coordinate to the columns
/// of the solution matrix. The solution matrix is stored in row major format.
let build (initCondExpr:Expr<float -> float -> float -> float>) 
          (boundaryExpr:Expr<float -> float -> float -> float>) 
          (sourceExpr:Expr<float -> float -> float -> float>) = cuda {

    let! initCondKernel =     
        <@ fun nx ny t (x:DevicePtr<float>) (y:DevicePtr<float>) (u:DevicePtr<float>) ->
            let initCond = %initCondExpr
            let i = blockIdx.x*blockDim.x + threadIdx.x
            let j = blockIdx.y*blockDim.y + threadIdx.y
            let mstride = ny
            if i < nx && j < ny then u.[i*mstride+j] <- initCond t x.[i] y.[j] @> |> defineKernelFunc

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

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let initCondKernel = initCondKernel.Apply m
        let xSweepKernel = xSweepKernel.Apply m
        let ySweepKernel = ySweepKernel.Apply m

        fun (nx:int) (ny:int) ->
            let nu = nx * ny
            let lp0 = LaunchParam(dim3(divup nx 16, divup ny 16), dim3(16, 16))
            let lpx = LaunchParam(ny, nx, 4*nx*sizeof<float>)
            let lpy = LaunchParam(nx, ny, 4*ny*sizeof<float>)

            let launch (hint:ActionHint) (t:float[]) (x:DevicePtr<float>) dx (y:DevicePtr<float>) dy (u0:DevicePtr<float>) (u1:DevicePtr<float>) k tstart tstop dt =
                let lp0 = lp0 |> hint.ModifyLaunchParam
                let lpx = lpx |> hint.ModifyLaunchParam
                let lpy = lpy |> hint.ModifyLaunchParam

                let initCondKernelFunc = initCondKernel.Launch lp0 
                let xSweepKernelFunc = xSweepKernel.Launch lpx
                let ySweepKernelFunc = ySweepKernel.Launch lpy

                initCondKernelFunc nx ny tstart x y u0

                if t.Length > 1 then
                    let step (t0, t1) =
                        let dt = t1 - t0
                        let Cx = k * dt / (dx * dx)
                        let Cy = k * dt / (dy * dy)
                        xSweepKernelFunc nx ny x y Cx Cy dt t0 (t0 + 0.5 * dt) u0 u1
                        ySweepKernelFunc nx ny x y Cx Cy dt (t0 + 0.5 * dt) t1 u1 u0

                    let timeIntervals = t |> Seq.pairwise |> Seq.toArray
                    timeIntervals |> Array.iter step

            { new ISolver with
                member this.GenT tstart tstop dt = timeGrid tstart tstop dt 5
                member this.GenX Lx = stateGrid nx Lx
                member this.GenY Ly = stateGrid ny Ly
                member this.NumU = nu
                member this.Launch hint t x dx y dy u0 u1 k tstart tstop dt = launch hint t x dx y dy u0 u1 k tstart tstop dt
            } ) }

let solve init boundary source = cuda {
    let! solver = build init boundary source
    
    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let solver = solver.Apply m
        fun k tstart tstop Lx Ly nx ny dt ->
            let solver = solver nx ny
            let t = timeGrid tstart tstop dt 5
            let x, dx = solver.GenX Lx
            let y, dy = solver.GenY Ly
            let nu = solver.NumU
            pcalc {
                let! x' = DArray.scatterInBlob worker x
                let! y' = DArray.scatterInBlob worker y
                let! u0 = DArray.createInBlob worker nu
                let! u1 = DArray.createInBlob worker nu
                do! PCalc.action (fun hint -> solver.Launch hint t x'.Ptr dx y'.Ptr dy u0.Ptr u1.Ptr k tstart tstop dt)
                return x, y, u0 } ) }

// Launch hint nt t x dx y dy u0 u1 k tstart tstop dt
type ISolverFused =
    abstract GenT : float -> float -> float -> float[] // tstart -> tstop -> dt -> timeGrid
    abstract GenX : float -> float[] * float // Lx -> x, dx
    abstract GenY : float -> float[] * float // Ly -> y, dy
    abstract NumU : int
    abstract Launch : ActionHint -> int -> DevicePtr<float> -> DevicePtr<float> -> float -> DevicePtr<float> -> float -> DevicePtr<float> -> DevicePtr<float> -> float -> float -> float -> float -> unit

let buildFused (initCondExpr:Expr<float -> float -> float -> float>) 
               (boundaryExpr:Expr<float -> float -> float -> float>) 
               (sourceExpr:Expr<float -> float -> float -> float>) = cuda {

    let! initCondKernel =     
        <@ fun nx ny t (x:DevicePtr<float>) (y:DevicePtr<float>) (u:DevicePtr<float>) ->
            let initCond = %initCondExpr
            let i = blockIdx.x*blockDim.x + threadIdx.x
            let j = blockIdx.y*blockDim.y + threadIdx.y
            let mstride = ny
            if i < nx && j < ny then u.[i*mstride+j] <- initCond t x.[i] y.[j] @> |> defineKernelFunc

    let! stepKernel =     
        <@ fun k nt nx ny (t:DevicePtr<float>) (x:DevicePtr<float>) dx (y:DevicePtr<float>) dy (u0:DevicePtr<float>) (u1:DevicePtr<float>) ->     
            let boundary = %boundaryExpr
            let source = %sourceExpr  
            
            let mutable i = 0
            while i < nt - 1 do
                let t0 = t.[i]
                let t1 = t.[i + 1]
                let dt = t1 - t0
                let Cx = k * dt / (dx * dx)
                let Cy = k * dt / (dy * dy)
              
                xSweep boundary source nx ny x y Cx Cy dt t0 (t0 + 0.5 * dt) u0 u1 

                __syncthreads()
                
                ySweep boundary source nx ny x y Cx Cy dt (t0 + 0.5 * dt) t1 u0 u1

                __syncthreads()

                i <- i + 1

        @> |> defineKernelFunc

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let initCondKernel = initCondKernel.Apply m
        let stepKernel = stepKernel.Apply m

        fun (nx:int) (ny:int) ->
            let nu = nx * ny
            let lp0 = LaunchParam(dim3(divup nx 16, divup ny 16), dim3(16, 16))
            let n = max nx ny
            let lp1 = LaunchParam(n, n, 4*n*sizeof<float>)

            let launch (hint:ActionHint) nt (t:DevicePtr<float>) (x:DevicePtr<float>) dx (y:DevicePtr<float>) dy (u0:DevicePtr<float>) (u1:DevicePtr<float>) k tstart tstop dt =
                let lp0 = lp0 |> hint.ModifyLaunchParam
                let initCondKernelFunc = initCondKernel.Launch lp0 
                initCondKernelFunc nx ny tstart x y u0

                if nt > 1 then
                    let lp1 = lp1 |> hint.ModifyLaunchParam
                    let stepKernelFunc = stepKernel.Launch lp1
                    stepKernelFunc k nt nx ny t x dx y dy u0 u1

            { new ISolverFused with
                member this.GenT tstart tstop dt = timeGrid tstart tstop dt 5
                member this.GenX Lx = stateGrid nx Lx
                member this.GenY Ly = stateGrid ny Ly
                member this.NumU = nu
                member this.Launch hint nt t x dx y dy u0 u1 k tstart tstop dt = launch hint nt t x dx y dy u0 u1 k tstart tstop dt
            } ) }

let solveFused init boundary source = cuda {
    let! solver = buildFused init boundary source
    
    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let solver = solver.Apply m
        fun k tstart tstop Lx Ly nx ny dt ->
            let solver = solver nx ny
            let t = solver.GenT tstart tstop dt
            let x, dx = solver.GenX Lx
            let y, dy = solver.GenY Ly
            let nu = solver.NumU
            pcalc {
                let! t' = DArray.scatterInBlob worker t
                let! x' = DArray.scatterInBlob worker x
                let! y' = DArray.scatterInBlob worker y
                let! u0 = DArray.createInBlob worker nu
                let! u1 = DArray.createInBlob worker nu
                do! PCalc.action (fun hint -> solver.Launch hint t.Length t'.Ptr x'.Ptr dx y'.Ptr dy u0.Ptr u1.Ptr k tstart tstop dt)
                return x, y, u0 } ) }