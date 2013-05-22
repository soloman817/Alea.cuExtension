module Alea.CUDA.Extension.Finance.Heston

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Alea.CUDA.Extension
open Alea.CUDA.Extension.TriDiag
open Alea.CUDA.Extension.Finance.Grid

open Util 

// shorthands 
// TODO refactor / move
let [<ReflectedDefinition>] get (u:RMatrixRowMajor ref) (si:int) (vi:int) =
    RMatrixRowMajor.Get(u, si, vi)

let [<ReflectedDefinition>] elem (u:RMatrixRowMajor ref) (si:int) (vi:int) =
    RMatrixRowMajor.Get(u, si, vi)

let [<ReflectedDefinition>] set (u:RMatrixRowMajor ref) (si:int) (vi:int) (value:float) =
    RMatrixRowMajor.Set(u, si, vi, value)

/// Concentrate at critical point and map critical point to grid point
let stateGrid = concentratedGridAt

/// Heston model in terms of expressions to be compiled into various kernels
type HestonModelExpr =
    val mutable rho : Expr<float -> float>
    val mutable sigma : Expr<float -> float>
    val mutable rd : Expr<float -> float>
    val mutable rf : Expr<float -> float>
    val mutable kappa : Expr<float -> float>
    val mutable eta : Expr<float -> float>

    new(rho:float, sigma:float, rd:float, rf:float, kappa:float, eta:float) = 
        { rho = <@ fun _ -> rho @>; sigma = <@ fun _ -> sigma @>; rd = <@ fun _ -> rd @>; 
          rf = <@ fun _ -> rf @>;  kappa = <@ fun _ -> kappa @>; eta = <@ fun _ -> eta @> }

[<Struct>]
type HestonModel =
    val rho : float
    val sigma : float
    val rd : float
    val rf : float
    val kappa : float
    val eta : float

    [<ReflectedDefinition>]
    new (rho:float, sigma:float, rd:float, rf:float, kappa:float, eta:float) =
        { rho = rho; sigma = sigma; rd = rd; rf = rf; kappa = kappa; eta = eta } 

[<Struct>]
type FdWeights =
    // first derivative
    val v1 : float
    val v2 : float
    val v3 : float
    // second derivative
    val w1 : float
    val w2 : float
    val w3 : float

    [<ReflectedDefinition>]
    new (v1:float, v2:float, v3:float, w1:float, w2:float, w3:float) =
        { v1 = v1; v2 = v2; v3 = v3; w1 = w1; w2 = w2; w3 = w3 } 

let [<ReflectedDefinition>] centralWeights (dx:float) (dxp:float) = 
    let v1 = -dxp / (dx*(dx + dxp))
    let v2 = (dxp - dx) / (dx*dxp)
    let v3 = dx / (dxp*(dx + dxp))
    let w1 = 2.0 / (dx*(dx + dxp))
    let w2 = -2.0 / (dx*dxp)
    let w3 = 2.0 / (dxp*(dx + dxp))
    FdWeights(v1, v2, v3, w1, w2, w3)

let [<ReflectedDefinition>] forwardWeights (dxp:float) (dxpp:float) = 
    let v1 = (-2.0*dxp - dxpp) / (dxp*(dxp + dxpp))
    let v2 = (dxp + dxpp) / (dxp*dxpp);
    let v3 = -dxp / (dxpp*(dxp + dxpp))
    let w1 = 2.0 / (dxp*(dxp + dxpp))
    let w2 = -2.0 / (dxp*dxpp)
    let w3 = 2.0 / (dxpp*(dxp + dxpp))
    FdWeights(v1, v2, v3, w1, w2, w3)

let [<ReflectedDefinition>] forwardWeightsSimple (dx:float)  = 
    FdWeights(-1.0/dx, 1.0/dx, 0.0, 0.0, 0.0, 0.0)

let [<ReflectedDefinition>] backwardWeights (dxm:float) (dx:float) = 
    let v1 = dx / (dxm*(dxm + dx))
    let v2 = (-dxm - dx) / (dxm*dx)
    let v3 = (dxm + 2.0*dx) / (dx*(dxm + dx))
    let w1 = 2.0 / (dxm*(dxm + dx))
    let w2 = -2.0 / (dxm*dx)
    let w3 = 2.0 / (dx*(dxm + dx))
    FdWeights(v1, v2, v3, w1, w2, w3)

/// Explicit apply operator for Euler scheme.
let [<ReflectedDefinition>] applyF (heston:HestonModel) t (dt:float) (u:RMatrixRowMajor ref) (s:DevicePtr<float>) (v:DevicePtr<float>) ns nv =
    let start = blockIdx.x * blockDim.x + threadIdx.x
    let stride = gridDim.x * blockDim.x

    let mutable a0 = 0.0
    let mutable a1 = 0.0
    let mutable a2 = 0.0
    let mutable b1 = 0.0
    let mutable b2 = 0.0
    let mutable a1l = 0.0
    let mutable a1d = 0.0
    let mutable a1u = 0.0
    let mutable a2l = 0.0
    let mutable a2d = 0.0
    let mutable a2u = 0.0

    let mutable j = blockIdx.y * blockDim.y + threadIdx.y

    // Dirichlet boundary at v = max, so we do not need to process j = nv-1
    while j < nv-1 do 

        // Dirichlet boundary at s = 0, so we do not need to process i = 0, we start i at 1
        let mutable i = blockIdx.x * blockDim.x + threadIdx.x + 1

        let vj = v.[j]

        // j = 0, special boundary at v = 0, one sided forward difference quotient
        let fdv =
            if j = 0 then                
                let dv = v.[1] - vj
                let fdv = forwardWeightsSimple dv         

                a2d <- heston.kappa*heston.eta*fdv.v1 - 0.5*heston.rd
                a2u <- heston.kappa*heston.eta*fdv.v2   
                
                fdv
            else
                // here we have 0 < j < nv-1 so we can always build central weights for v               
                let dv = vj - v.[j-1]
                let dvp = v.[j+1] - vj
                let fdv = centralWeights dv dvp

                // operator A2, j-th row, is independent of i so we can construct it here
                if j < nv-2 then
                    a2l <- 0.5*heston.sigma*vj*fdv.w1 + heston.kappa*(heston.eta - vj)*fdv.v1
                    a2d <- 0.5*heston.sigma*vj*fdv.w2 + heston.kappa*(heston.eta - vj)*fdv.v2 - 0.5*heston.rd
                    a2u <- 0.5*heston.sigma*vj*fdv.w3 + heston.kappa*(heston.eta - vj)*fdv.v3
                else
                    // Dirichlet boundary at v = max, the constant term 
                    //   (0.5*sigma*v(j)*wp2 + kappa*(eta - v(j))*wp1)*s(i)*exp(-t*rf)
                    // is absorbed into b2
                    a2l <- 0.5*heston.sigma*vj*fdv.w1 + heston.kappa*(heston.eta - vj)*fdv.v1
                    a2d <- 0.5*heston.sigma*vj*fdv.w2 + heston.kappa*(heston.eta - vj)*fdv.v2 - 0.5*heston.rd
                    a2u <- 0.0
                
                fdv
                             
        // we always have i > 0 
        while i < ns do

            let u00 = get u i j

            if j = 0 then
                            
                // j = 0, special boundary at v = 0, one sided forward difference quotient
                let um0 = get u (i-1) 0 
                let u0p = get u i     1

                let si = s.[i]
                let ds = si - s.[i-1]

                // inside points: 0 < i < ns-1
                if i < ns-1 then
                    let up0 = get u (i+1) 0

                    let dsp = s.[i+1] - si
                    let fds = centralWeights ds dsp

                    // operator A1, i-th row
                    a1l <- (heston.rd-heston.rf)*si*fds.v1          
                    a1d <- (heston.rd-heston.rf)*si*fds.v2 - 0.5*heston.rd
                    a1u <- (heston.rd-heston.rf)*si*fds.v3 

                    a0 <- 0.0
                    a1 <- a1l*um0 + a1d*u00 + a1u*up0 // A1*u                    
                    a2 <- a2d*u00 + a2u*u0p // A2*u             
                    b1 <- 0.0
                    b2 <- 0.0

                // boundary s = max: i = ns-1 
                else
                    let fds = centralWeights ds ds

                    // Neumann boundary in s direction with additonal ghost point and extrapolation
                    a1l <- 0.5*si*si*vj*fds.w1 + (heston.rd-heston.rf)*si*fds.v1
                    a1d <- 0.5*si*si*vj*(fds.w2+fds.w3) + (heston.rd-heston.rf)*si*(fds.v2+fds.v3) - 0.5*heston.rd
                    a1u <- 0.0

                    a0 <- 0.0
                    a1 <- a1l*um0 + a1d*u00 // A1*u
                    a2 <- a2d*u00 + a2u*u0p // A2*u
                    b1 <- (heston.rd-heston.rf)*si*0.5; // wp = ds / (2.0*ds*ds) => duds = wp*ds = 0.5
                    b2 <- 0.0
               
            else

                // we add a zero boundary around u to have no memory access issues
                let umm = get u (i-1) (j-1)
                let ump = get u (i-1) (j+1)
                let um0 = get u (i-1) j 
                let u0m = get u i     (j-1)
                let u0p = get u i     (j+1)
                let upm = get u (i+1) (j-1)
                let up0 = get u (i+1) j
                let upp = get u (i+1) (j+1)

                let si = s.[i]
                let ds = si - s.[i-1]
           
                if j = nv-2 then
                    b2 <- (0.5*heston.sigma*vj*fdv.w3 + heston.kappa*(heston.eta - vj)*fdv.v3)*si*exp(-t*heston.rf)
                else
                    b2 <- 0.0

                // inside points: 0 < i < ns-1
                if i < ns-1 then
                    let dsp = s.[i+1] - si
                    let fds = centralWeights ds dsp

                    // a0 <> 0 only on 0 < j < nv-1 && 0 < i < ns-1 
                    let mixed = fds.v1*fdv.v1*umm + fds.v2*fdv.v1*u0m + fds.v3*fdv.v1*upm +
                                fds.v1*fdv.v2*um0 + fds.v2*fdv.v2*u00 + fds.v3*fdv.v2*up0 + 
                                fds.v1*fdv.v3*ump + fds.v2*fdv.v3*u0p + fds.v3*fdv.v3*upp
                    a0 <- heston.rho*heston.sigma*si*vj*mixed  

                    // operator A1, i-th row
                    // Dirichlet boundary at s = 0 for i = 1
                    a1l <- if i = 1 then 0.0 else 0.5*si*si*vj*fds.w1 + (heston.rd-heston.rf)*si*fds.v1          
                    a1d <- 0.5*si*si*vj*fds.w2 + (heston.rd-heston.rf)*si*fds.v2 - 0.5*heston.rd
                    a1u <- 0.5*si*si*vj*fds.w3 + (heston.rd-heston.rf)*si*fds.v3 

                    a1 <- a1l*um0 + a1d*u00 + a1u*up0 // A1*u                    
                    a2 <- a2l*u0m + a2d*u00 + a2u*u0p // A2*u    
                    b1 <- 0.0                             
            
                // boundary s = max: i = ns-1 
                else
                    a0 <- 0.0
                    let fds = centralWeights ds ds

                    // Neumann boundary with additonal ghost point and extrapolation
                    a1l <- 0.5*si*si*vj*fds.w1 + (heston.rd-heston.rf)*si*fds.v1
                    a1d <- 0.5*si*si*vj*(fds.w2+fds.w3) + (heston.rd-heston.rf)*si*(fds.v2+fds.v3) - 0.5*heston.rd
                    a1u <- 0.0

                    a1 <- a1l*um0 + a1d*u00 
                    a2 <- a2l*u0m + a2d*u00 + a2u*u0p
                    b1 <- 0.5*si*si*vj/ds + (heston.rd-heston.rf)*si*0.5 // wp = 1.0/(ds*ds), d2ud2s = wp*ds = 1.0/ds, wp = ds / (2.0*ds*ds) duds = wp*ds = 0.5

            // set u for i = 1,..., ns-1, j=1,..., nv-2
            set u i j (u00 + dt*(a0+a1+a2+b1+b2))

            //set u i j a2d

            i <- i + blockDim.x * gridDim.x
               
        j <- j + blockDim.y * gridDim.y   

type OptionType =
| Call
| Put
    member this.sign =
        match this with
        | Call -> 1.0
        | Put -> -1.0

/// Initial condition for vanilla call put option.
/// We add artifical zeros to avoid access violation in the kernel.
let [<ReflectedDefinition>] initConditionVanilla ns nv (s:DevicePtr<float>) (u:RMatrixRowMajor) optionType strike =    
    let i = blockIdx.x*blockDim.x + threadIdx.x
    let j = blockIdx.y*blockDim.y + threadIdx.y
    if i < ns + 1 && j < nv + 1 then
        let payoff = 
            if i = ns || j = nv then 
                0.0 
            else
                match optionType with
                |  1.0 -> max (s.[i] - strike) 0.0
                | -1.0 -> max (strike - s.[i]) 0.0 
                | _ -> 0.0
        set (ref u) i j payoff 

/// Boundary condition for vanilla call put option.
let [<ReflectedDefinition>] boundaryConditionVanilla (rf:float) t ns nv (s:DevicePtr<float>) (u:RMatrixRowMajor) =
    let i = blockIdx.x*blockDim.x + threadIdx.x
    if i < ns then
        set (ref u) i (nv-1) (exp(-rf*t)*s.[i])
    if i < nv then
        set (ref u) 0 i 0.0

/// Copy kernel to copy data on device to reduce to valid range
let [<ReflectedDefinition>] copyValues ns nv (u0:RMatrixRowMajor) (u1:RMatrixRowMajor) =    
    let i = blockIdx.x*blockDim.x + threadIdx.x
    let j = blockIdx.y*blockDim.y + threadIdx.y
    if i < ns && j < nv then
        set (ref u1) i j (get (ref u0) i j) 

/// Copy kernel to copy data on device to reduce to valid range
let [<ReflectedDefinition>] copyGrids ns nv (s0:DevicePtr<float>) (v0:DevicePtr<float>) (s1:DevicePtr<float>) (v1:DevicePtr<float>) =    
    let i = blockIdx.x*blockDim.x + threadIdx.x
    if i < ns then
        s1.[i] <- s0.[i]        
    if i < nv then
        v1.[i] <- v0.[i]

type EulerSolverParam =
    val theta:float
    val sMin:float
    val sMax:float
    val vMax:float
    val ns:int
    val nv:int
    val sC:float
    val vC:float

    new(theta:float, sMin:float, sMax:float, vMax:float, ns:int, nv:int, sC:float, vC:float) =
        {theta = theta; sMin = sMin; sMax = sMax; vMax = vMax; ns = ns; nv = nv; sC = sC; vC = vC}

/// Solve Hesten with explicit Euler scheme.
/// Because the time stepping has to be selected according to the state discretization
/// we may need to have small time steps to maintain stability.
let eulerSolver = cuda {

    // we add artifical zeros to avoid access violation in the kernel
    let! initCondKernel =     
        <@ fun ns nv (s:DevicePtr<float>) (u:RMatrixRowMajor) optionType strike ->
            initConditionVanilla ns nv s u optionType strike @> |> defineKernelFuncWithName "initCondition"

    let! boundaryCondKernel =     
        <@ fun (rf:float) t ns nv (s:DevicePtr<float>) (u:RMatrixRowMajor) ->
            boundaryConditionVanilla rf t ns nv s u @> |> defineKernelFuncWithName "boundaryCondition"
   
    let! appFKernel =
        <@ fun (heston:HestonModel) t (dt:float) (u:RMatrixRowMajor) (s:DevicePtr<float>) (v:DevicePtr<float>) ns nv ->
            applyF heston t dt (ref u) s v ns nv @> |> defineKernelFuncWithName "applyF"
    
    let! copyValuesKernel = 
        <@ fun ns nv (u0:RMatrixRowMajor) (u1:RMatrixRowMajor) ->
            copyValues ns nv u0 u1 @> |> defineKernelFuncWithName "copyValues"    

    let! copyGridsKernel = 
        <@ fun ns nv (s0:DevicePtr<float>) (v0:DevicePtr<float>) (s1:DevicePtr<float>) (v1:DevicePtr<float>) ->
            copyGrids ns nv s0 v0 s1 v1 @> |> defineKernelFuncWithName "copyGrids"    
                         
    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let initCondKernel = initCondKernel.Apply m
        let boundaryCondKernel = boundaryCondKernel.Apply m
        let appFKernel = appFKernel.Apply m
        let copyValuesKernel = copyValuesKernel.Apply m
        let copyGridsKernel = copyGridsKernel.Apply m

        fun (heston:HestonModel) (optionType:OptionType) strike timeToMaturity (param:EulerSolverParam) ->

            // we add one more point to the state grids because the value surface has a ghost aerea as well
            let s, ds = concentratedGridBetween param.sMin param.sMax strike param.ns param.sC
            let v, dv = concentratedGridBetween 0.0 param.vMax 0.0 param.nv param.vC

            // calculate a time step which is compatible with the space discretization
            let dt = ds*ds/100.0
            let nt = int(timeToMaturity/dt) + 1
            let t, dt = homogeneousGrid nt 0.0 timeToMaturity

            pcalc {
                let! s = DArray.scatterInBlob worker s
                let! v = DArray.scatterInBlob worker v

                // add a ghost point to the value surface to allow simpler access in the kernel, these
                // ghost points will have value zero 
                let! u = DMatrix.createInBlob<float> worker RowMajorOrder (param.ns+1) (param.nv+1)

                // storage for reduced values
                let! ured = DMatrix.createInBlob<float> worker RowMajorOrder param.ns param.nv
                
                do! PCalc.action (fun hint ->
                    let lpm = LaunchParam(dim3(divup param.ns 16, divup param.ns 16), dim3(16, 16)) |> hint.ModifyLaunchParam
                    let lpb = LaunchParam(divup (max param.ns param.nv) 256, 256) |> hint.ModifyLaunchParam

                    let u = RMatrixRowMajor(u.NumRows, u.NumCols, u.Storage.Ptr)
                    let ured = RMatrixRowMajor(ured.NumRows, ured.NumCols, ured.Storage.Ptr)

                    initCondKernel.Launch lpm param.ns param.nv s.Ptr u optionType.sign strike

                    for ti = 0 to nt-2 do

                        let t0 = t.[ti]
                        let t1 = t.[ti + 1]
                        let dt = t1 - t0
                        let thetaDt = dt * param.theta

                        boundaryCondKernel.Launch lpb heston.rf t0 param.ns param.nv s.Ptr u
                        appFKernel.Launch lpm heston t0 dt u s.Ptr v.Ptr param.ns param.nv

                    // this is a temporary solution, later we should use a view on it
                    copyValuesKernel.Launch lpm param.ns param.nv u ured
                )
                
                return s, v, ured } ) }


//****** Refactor below, some problem with boundary conditons

[<Struct>]
type RFiniteDifferenceWeights =
    val n : int
    [<PointerField(MemorySpace.Global)>] val mutable x       : int64
    [<PointerField(MemorySpace.Global)>] val mutable delta   : int64
    [<PointerField(MemorySpace.Global)>] val mutable alpha0  : int64
    [<PointerField(MemorySpace.Global)>] val mutable alphaM1 : int64
    [<PointerField(MemorySpace.Global)>] val mutable alphaM2 : int64
    [<PointerField(MemorySpace.Global)>] val mutable beta0   : int64
    [<PointerField(MemorySpace.Global)>] val mutable betaP   : int64
    [<PointerField(MemorySpace.Global)>] val mutable betaM   : int64
    [<PointerField(MemorySpace.Global)>] val mutable gamma0  : int64
    [<PointerField(MemorySpace.Global)>] val mutable gammaP1 : int64
    [<PointerField(MemorySpace.Global)>] val mutable gammaP2 : int64
    [<PointerField(MemorySpace.Global)>] val mutable delta0  : int64
    [<PointerField(MemorySpace.Global)>] val mutable deltaP  : int64
    [<PointerField(MemorySpace.Global)>] val mutable deltaM  : int64

    [<PointerProperty("x")>]       member this.X       with get () = DevicePtr<float>(this.x) and set (ptr:DevicePtr<float>) = this.x <- ptr.Handle64
    [<PointerProperty("delta")>]   member this.Delta   with get () = DevicePtr<float>(this.delta) and set (ptr:DevicePtr<float>) = this.delta <- ptr.Handle64
    [<PointerProperty("alpha0")>]  member this.Alpha0  with get () = DevicePtr<float>(this.alpha0) and set (ptr:DevicePtr<float>) = this.alpha0 <- ptr.Handle64
    [<PointerProperty("alphaM1")>] member this.AlphaM1 with get () = DevicePtr<float>(this.alphaM1) and set (ptr:DevicePtr<float>) = this.alphaM1 <- ptr.Handle64
    [<PointerProperty("alphaM2")>] member this.AlphaM2 with get () = DevicePtr<float>(this.alphaM2) and set (ptr:DevicePtr<float>) = this.alphaM2 <- ptr.Handle64
    [<PointerProperty("beta0")>]   member this.Beta0   with get () = DevicePtr<float>(this.beta0) and set (ptr:DevicePtr<float>) = this.beta0 <- ptr.Handle64
    [<PointerProperty("betaP")>]   member this.BetaP   with get () = DevicePtr<float>(this.betaP) and set (ptr:DevicePtr<float>) = this.betaP <- ptr.Handle64
    [<PointerProperty("betaM")>]   member this.BetaM   with get () = DevicePtr<float>(this.betaM) and set (ptr:DevicePtr<float>) = this.betaM <- ptr.Handle64
    [<PointerProperty("gamma0")>]  member this.Gamma0  with get () = DevicePtr<float>(this.gamma0) and set (ptr:DevicePtr<float>) = this.gamma0 <- ptr.Handle64
    [<PointerProperty("gammaP1")>] member this.GammaP1 with get () = DevicePtr<float>(this.gammaP1) and set (ptr:DevicePtr<float>) = this.gammaP1 <- ptr.Handle64
    [<PointerProperty("gammaP2")>] member this.GammaP2 with get () = DevicePtr<float>(this.gammaP2) and set (ptr:DevicePtr<float>) = this.gammaP2 <- ptr.Handle64
    [<PointerProperty("delta0")>]  member this.Delta0  with get () = DevicePtr<float>(this.delta0) and set (ptr:DevicePtr<float>) = this.delta0 <- ptr.Handle64
    [<PointerProperty("deltaP")>]  member this.DeltaP  with get () = DevicePtr<float>(this.deltaP) and set (ptr:DevicePtr<float>) = this.deltaP <- ptr.Handle64
    [<PointerProperty("deltaM")>]  member this.DeltaM  with get () = DevicePtr<float>(this.deltaM) and set (ptr:DevicePtr<float>) = this.deltaM <- ptr.Handle64

    [<ReflectedDefinition>]
    new (n:int, x:DevicePtr<float>, delta:DevicePtr<float>, 
         alpha0:DevicePtr<float>, alphaM1:DevicePtr<float>, alphaM2:DevicePtr<float>, 
         beta0:DevicePtr<float>, betaP:DevicePtr<float>, betaM:DevicePtr<float>, 
         gamma0:DevicePtr<float>, gammaP1:DevicePtr<float>, gammaP2:DevicePtr<float>, 
         delta0:DevicePtr<float>, deltaP:DevicePtr<float>, deltaM:DevicePtr<float>) =
        { n = n; x = x.Handle64; delta = delta.Handle64
          alpha0 = alpha0.Handle64; alphaM1 = alphaM1.Handle64; alphaM2 = alphaM2.Handle64
          beta0 = beta0.Handle64; betaP = betaP.Handle64; betaM = betaM.Handle64
          gamma0 = gamma0.Handle64; gammaP1 = gammaP1.Handle64; gammaP2 = gammaP2.Handle64
          delta0 = delta0.Handle64; deltaP = deltaP.Handle64; deltaM = deltaM.Handle64 }

// helper functions to shift index properly
let [<ReflectedDefinition>] delta(diff:RFiniteDifferenceWeights) i = diff.Delta.[i-1]
let [<ReflectedDefinition>] alphaM2(diff:RFiniteDifferenceWeights) i = diff.AlphaM2.[i-2]
let [<ReflectedDefinition>] alphaM1(diff:RFiniteDifferenceWeights) i = diff.AlphaM1.[i-2]
let [<ReflectedDefinition>] alpha0(diff:RFiniteDifferenceWeights) i = diff.Alpha0.[i-2]
let [<ReflectedDefinition>] betaM(diff:RFiniteDifferenceWeights) i = diff.BetaM.[i-1]
let [<ReflectedDefinition>] beta0(diff:RFiniteDifferenceWeights) i = diff.Beta0.[i-1]
let [<ReflectedDefinition>] betaP(diff:RFiniteDifferenceWeights) i = diff.BetaP.[i-1]
let [<ReflectedDefinition>] gamma0(diff:RFiniteDifferenceWeights) i = diff.Gamma0.[i]
let [<ReflectedDefinition>] gammaP1(diff:RFiniteDifferenceWeights) i = diff.GammaP1.[i]
let [<ReflectedDefinition>] gammaP2(diff:RFiniteDifferenceWeights) i = diff.GammaP2.[i]
let [<ReflectedDefinition>] deltaM(diff:RFiniteDifferenceWeights) i = diff.DeltaM.[i-1]
let [<ReflectedDefinition>] delta0(diff:RFiniteDifferenceWeights) i = diff.Delta0.[i-1]
let [<ReflectedDefinition>] deltaP(diff:RFiniteDifferenceWeights) i = diff.DeltaP.[i-1]

type HFiniteDifferenceWeights = {
    X       : float[]
    Delta   : float[]
    Alpha0  : float[]
    AlphaM1 : float[]
    AlphaM2 : float[]
    Beta0   : float[]
    BetaP   : float[]
    BetaM   : float[]
    Gamma0  : float[]
    GammaP1 : float[]
    GammaP2 : float[]
    Delta0  : float[]
    DeltaP  : float[]
    DeltaM  : float[]
}

type DFiniteDifferenceWeights = 
    val X       : DArray<float>
    val Delta   : DArray<float>
    val Alpha0  : DArray<float>
    val AlphaM1 : DArray<float>
    val AlphaM2 : DArray<float>
    val Beta0   : DArray<float>
    val BetaP   : DArray<float>
    val BetaM   : DArray<float>
    val Gamma0  : DArray<float>
    val GammaP1 : DArray<float>
    val GammaP2 : DArray<float>
    val Delta0  : DArray<float>
    val DeltaP  : DArray<float>
    val DeltaM  : DArray<float>
    
    new(x:DArray<float>, delta:DArray<float>, 
        alpha0:DArray<float>, alphaM1:DArray<float>, alphaM2:DArray<float>, 
        beta0:DArray<float>, betaP:DArray<float>, betaM:DArray<float>, 
        gamma0:DArray<float>, gammaP1:DArray<float>, gammaP2:DArray<float>, 
        delta0:DArray<float>, deltaP:DArray<float>, deltaM:DArray<float>) = 
       { X = x; Delta = delta; Alpha0 = alpha0; AlphaM1 = alphaM1; AlphaM2 = alphaM2; 
         Beta0 = beta0; BetaP = betaP; BetaM = betaM; Gamma0 = gamma0; GammaP1 = gammaP1; 
         GammaP2 = gammaP2; Delta0 = delta0; DeltaP = deltaP; DeltaM = deltaM }

    member this.Raw() = RFiniteDifferenceWeights(this.X.Length, this.X.Ptr, this.Delta.Ptr, 
                                                 this.Alpha0.Ptr, this.AlphaM1.Ptr, this.AlphaM2.Ptr, 
                                                 this.Beta0.Ptr, this.BetaP.Ptr, this.BetaM.Ptr, 
                                                 this.Gamma0.Ptr, this.GammaP1.Ptr, this.GammaP2.Ptr, 
                                                 this.Delta0.Ptr, this.DeltaP.Ptr, this.DeltaM.Ptr)

    member this.Gather() = pcalc {
            let! x = this.X.Gather()
            let! delta = this.Delta.Gather()
            let! alpha0 = this.Alpha0.Gather()
            let! alphaM1 = this.AlphaM1.Gather()
            let! alphaM2 = this.AlphaM2.Gather()
            let! beta0 = this.Beta0.Gather()
            let! betaP = this.BetaP.Gather()
            let! betaM = this.BetaM.Gather()
            let! gamma0 = this.Gamma0.Gather()
            let! gammaP1 = this.GammaP1.Gather()
            let! gammaP2 = this.GammaP2.Gather()
            let! delta0 = this.Delta0.Gather()
            let! deltaP = this.DeltaP.Gather()
            let! deltaM = this.DeltaM.Gather()
            let diff : HFiniteDifferenceWeights = { 
                X = x; Delta = delta; Alpha0 = alpha0; AlphaM1 = alphaM1; AlphaM2 = alphaM2; 
                Beta0 = beta0; BetaP = betaP; BetaM = betaM; Gamma0 = gamma0; GammaP1 = gammaP1; 
                GammaP2 = gammaP2; Delta0 = delta0; DeltaP = deltaP; DeltaM = deltaM } 
            return diff
        }

let finiteDifferenceWeights = cuda {

    let! diffKernel = 
        <@  fun (n:int) (x:DevicePtr<float>) (dx:DevicePtr<float>) -> 
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < n - 1 do
                dx.[i] <- x.[i + 1] - x.[i]
                i <- i + stride @> |> defineKernelFuncWithName "diff"

    let! finiteDifferenceKernel = 
        <@ fun (diff:RFiniteDifferenceWeights) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < diff.n - 2 do
                let dx0 = diff.Delta.[i]
                let dx1 = diff.Delta.[i+1]

                diff.AlphaM2.[i] <- dx1 / (dx0 * (dx0 + dx1))
                diff.AlphaM1.[i] <- -(dx0 + dx1) / (dx0 * dx1)
                diff.Alpha0.[i]  <- (dx0 + 2.0 * dx1) / (dx1 * (dx0 + dx1))
        
                diff.BetaM.[i]   <- -dx1 / (dx0 * (dx0 + dx1))
                diff.Beta0.[i]   <- (dx1 - dx0) / (dx0 * dx1)
                diff.BetaP.[i]   <- dx0 / (dx1 * (dx0 + dx1))
       
                diff.Gamma0.[i]  <- (-2.0 * dx0 - dx1) / (dx0 * (dx0 + dx1))
                diff.GammaP1.[i] <- (dx0 + dx1) / (dx0 * dx1)
                diff.GammaP2.[i] <- -dx0 / (dx1 * (dx0 + dx1))
        
                diff.DeltaM.[i]  <- 2.0 / (dx0 * (dx0 + dx1))
                diff.Delta0.[i]  <- -2.0 / (dx0 * dx1)
                diff.DeltaP.[i]  <- 2.0 / (dx1 * (dx0 + dx1))

                i <- i + stride @> |> defineKernelFuncWithName "finiteDifference"

    return PFunc(fun (m:Module) n (x:DArray<float>) ->
        let worker = m.Worker
        pcalc {
            let! delta = DArray.createInBlob<float> worker (n-1)
            let! alpha0 = DArray.createInBlob<float> worker (n-2)
            let! alphaM1 = DArray.createInBlob<float> worker (n-2)
            let! alphaM2 = DArray.createInBlob<float> worker (n-2)
            let! beta0 = DArray.createInBlob<float> worker (n-2)
            let! betaP = DArray.createInBlob<float> worker (n-2)
            let! betaM = DArray.createInBlob<float> worker (n-2)
            let! gamma0 = DArray.createInBlob<float> worker (n-2)
            let! gammaP1 = DArray.createInBlob<float> worker (n-2)
            let! gammaP2 = DArray.createInBlob<float> worker (n-2)
            let! delta0 = DArray.createInBlob<float> worker (n-2)
            let! deltaP = DArray.createInBlob<float> worker (n-2)
            let! deltaM = DArray.createInBlob<float> worker (n-2)

            let diff = DFiniteDifferenceWeights(x, delta, alpha0, alphaM1, alphaM2, beta0, betaP, betaM, 
                                                gamma0, gammaP1, gammaP2, delta0, deltaP, deltaM)

            do! PCalc.action (fun hint ->
                let blockSize = 256
                let gridSize = Util.divup n blockSize           
                let lp = LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam
                diffKernel.Launch m lp n x.Ptr delta.Ptr
                finiteDifferenceKernel.Launch m lp (diff.Raw()))

            return diff } ) }


[<Struct>]
type Stencil = 
    val mutable si:int
    val mutable vi:int
        
    val mutable vs:float
    val mutable vv:float
    val mutable vu:float
      
    val mutable ds0:float
    val mutable ds1:float
    val mutable ds2:float
    val mutable dv0:float
    val mutable dv1:float
    val mutable dv2:float
       
    val mutable u00:float
    val mutable u01:float
    val mutable u02:float
    val mutable u10:float
    val mutable u11:float
    val mutable u12:float
    val mutable u20:float
    val mutable u21:float
    val mutable u22:float
         
    val mutable dds0:float
    val mutable dds1:float
    val mutable dds2:float
    val mutable ddv0:float
    val mutable ddv1:float
    val mutable ddv2:float
        
    val mutable us0:float
    val mutable us1:float
    val mutable us2:float
    val mutable uv0:float
    val mutable uv1:float
    val mutable uv2:float

    [<ReflectedDefinition>]
    new (si:int, vi:int, vs:float, vv:float, vu:float,
         ds0:float, ds1:float, ds2:float, dv0:float, dv1:float, dv2:float, 
         u00:float, u01:float, u02:float, u10:float, u11:float, u12:float, u20:float, u21:float, u22:float,
         dds0:float, dds1:float, dds2:float, ddv0:float, ddv1:float, ddv2:float,
         us0:float, us1:float, us2:float, uv0:float, uv1:float, uv2:float) = 
        {si = si; vi = vi; 
         vs = vs; vv = vv; vu = vu;
         ds0 = ds0; ds1 = ds1; ds2 = ds2; dv0 = dv0; dv1 = dv1; dv2 = dv2;
         u00 = u00; u01 = u01; u02 = u02; u10 = u10; u11 = u11; u12 = u12; u20 = u20; u21 = u21; u22 = u22;
         dds0 = dds0; dds1 = dds1; dds2 = dds2; ddv0 = ddv0; ddv1 = ddv1; ddv2 = ddv2;
         us0 = us0; us1 = us1; us2 = us2; uv0 = uv0; uv1 = uv1; uv2 = uv2}

let [<ReflectedDefinition>] stencil si vi (ds:RFiniteDifferenceWeights) (dv:RFiniteDifferenceWeights) (u:RMatrixRowMajor ref)  =
    let mutable stencil = Stencil()
    stencil.si <- si
    stencil.vi <- vi

    if si = 0 && vi = 0 then  // 1 corner

        let u00 = elem u (si  ) (vi  )
        let u01 = elem u (si  ) (vi+1)
        let u02 = elem u (si  ) (vi+2)
        let u10 = elem u (si+1) (vi  )
        let u11 = elem u (si+1) (vi+1)
        let u12 = elem u (si+1) (vi+2)
        let u20 = elem u (si+2) (vi  )
        let u21 = elem u (si+2) (vi+1)
        let u22 = elem u (si+2) (vi+2)
        
        stencil.ds0 <- gamma0 ds si; stencil.ds1 <- gammaP1 ds si; stencil.ds2 <- gammaP2 ds si
        stencil.dv0 <- gamma0 dv vi; stencil.dv1 <- gammaP1 dv vi; stencil.dv2 <- gammaP2 dv vi

        stencil.u00 <- u00; stencil.u01 <- u01; stencil.u02 <- u02
        stencil.u10 <- u10; stencil.u11 <- u11; stencil.u12 <- u12
        stencil.u20 <- u20; stencil.u21 <- u21; stencil.u22 <- u22
 
        stencil.vs <- 0.0; stencil.vv <- 0.0; stencil.vu <- u00

        stencil.us0 <- u00; stencil.us1 <- u10; stencil.us2 <- u20
        stencil.uv0 <- u00; stencil.uv1 <- u01; stencil.uv2 <- u02

        stencil.dds0 <- 0.0; stencil.dds1 <- 0.0; stencil.dds2 <- 0.0
        stencil.ddv0 <- 0.0; stencil.ddv1 <- 0.0; stencil.ddv2 <- 0.0

    else if si = ds.n - 1 && vi = 0 then // 2 corner

        let u00 = elem u (si-2) (vi  )
        let u01 = elem u (si-2) (vi+1)
        let u02 = elem u (si-2) (vi+2)
        let u10 = elem u (si-1) (vi  )
        let u11 = elem u (si-1) (vi+1)
        let u12 = elem u (si-1) (vi+2)
        let u20 = elem u (si  ) (vi  )
        let u21 = elem u (si  ) (vi+1)
        let u22 = elem u (si  ) (vi+2)

        stencil.ds0 <- alphaM2 ds si; stencil.ds1 <- alphaM1 ds si; stencil.ds2 <- alpha0 ds si
        stencil.dv0 <- gamma0 dv vi; stencil.dv1 <- gammaP1 dv vi; stencil.dv2 <- gammaP2 dv vi

        stencil.u00 <- u00; stencil.u01 <- u01; stencil.u02 <- u02
        stencil.u10 <- u10; stencil.u11 <- u11; stencil.u12 <- u12
        stencil.u20 <- u20; stencil.u21 <- u21; stencil.u22 <- u22

        stencil.vs <- 0.0; stencil.vv <- 0.0; stencil.vu <- u20

        stencil.us0 <- u00; stencil.us1 <- u10; stencil.us2 <- u20
        stencil.uv0 <- u20; stencil.uv1 <- u21; stencil.uv2 <- u22

        stencil.dds0 <- 0.0; stencil.dds1 <- 0.0; stencil.dds2 <- 0.0
        stencil.ddv0 <- 0.0; stencil.ddv1 <- 0.0; stencil.ddv2 <- 0.0

    else if si = 0 && vi = dv.n - 1 then // 3 corner

        let u00 = elem u (si  ) (vi-2)
        let u01 = elem u (si  ) (vi-1)
        let u02 = elem u (si  ) (vi  )
        let u10 = elem u (si+1) (vi-2)
        let u11 = elem u (si+1) (vi-1)
        let u12 = elem u (si+1) (vi  )
        let u20 = elem u (si+2) (vi-2)
        let u21 = elem u (si+2) (vi-1)
        let u22 = elem u (si+2) (vi  )

        stencil.ds0 <- gamma0 ds si; stencil.ds1 <- gammaP1 ds si; stencil.ds2 <- gammaP2 ds si
        stencil.dv0 <- alphaM2 dv vi; stencil.dv1 <- alphaM1 dv vi; stencil.dv2 <- alpha0 dv vi

        stencil.u00 <- u00; stencil.u01 <- u01; stencil.u02 <- u02
        stencil.u10 <- u10; stencil.u11 <- u11; stencil.u12 <- u12
        stencil.u20 <- u20; stencil.u21 <- u21; stencil.u22 <- u22

        stencil.vs <- 0.0; stencil.vv <- 0.0; stencil.vu <- u02

        stencil.us0 <- u02; stencil.us1 <- u12; stencil.us2 <- u22
        stencil.uv0 <- u00; stencil.uv1 <- u01; stencil.uv2 <- u02 

        stencil.dds0 <- 0.0; stencil.dds1 <- 0.0; stencil.dds2 <- 0.0
        stencil.ddv0 <- 0.0; stencil.ddv1 <- 0.0; stencil.ddv2 <- 0.0
    
    else if si = ds.n - 1 && vi = dv.n - 1 then // 4 corner

        let u00 = elem u (si-2) (vi-2)
        let u01 = elem u (si-2) (vi-1)
        let u02 = elem u (si-2) (vi  )
        let u10 = elem u (si-1) (vi-2)
        let u11 = elem u (si-1) (vi-1)
        let u12 = elem u (si-1) (vi  )
        let u20 = elem u (si  ) (vi-2)
        let u21 = elem u (si  ) (vi-1)
        let u22 = elem u (si  ) (vi  )

        stencil.ds0 <- alphaM2 ds si; stencil.ds1 <- alphaM1 ds si; stencil.ds2 <- alpha0 ds si
        stencil.dv0 <- alphaM2 dv vi; stencil.dv1 <- alphaM1 dv vi; stencil.dv2 <- alpha0 dv vi

        stencil.u00 <- u00; stencil.u01 <- u01; stencil.u02 <- u02 
        stencil.u10 <- u10; stencil.u11 <- u11; stencil.u12 <- u12
        stencil.u20 <- u20; stencil.u21 <- u21; stencil.u22 <- u22

        stencil.vs <- 0.0; stencil.vv <- 0.0; stencil.vu <- u22

        stencil.us0 <- u02; stencil.us1 <- u12; stencil.us2 <- u22
        stencil.uv0 <- u20; stencil.uv1 <- u21; stencil.uv2 <- u22

        stencil.dds0 <- 0.0; stencil.dds1 <- 0.0; stencil.dds2 <- 0.0
        stencil.ddv0 <- 0.0; stencil.ddv1 <- 0.0; stencil.ddv2 <- 0.0
    
    else if si = 0 then // 5 face
    
        let u00 = elem u (si  ) (vi-1)
        let u01 = elem u (si  ) (vi  )
        let u02 = elem u (si  ) (vi+1)
        let u10 = elem u (si+1) (vi-1)
        let u11 = elem u (si+1) (vi  )
        let u12 = elem u (si+1) (vi+1)
        let u20 = elem u (si+2) (vi-1)
        let u21 = elem u (si+2) (vi  )
        let u22 = elem u (si+2) (vi+1)

        stencil.ds0 <- gamma0  ds si; stencil.ds1 <- gammaP1 ds si; stencil.ds2 <- gammaP2 ds si;
        stencil.dv0 <- betaM dv vi; stencil.dv1 <- beta0 dv vi; stencil.dv2 <- betaP dv vi;

        stencil.u00 <- u00; stencil.u01 <- u01; stencil.u02 <- u02 
        stencil.u10 <- u10; stencil.u11 <- u11; stencil.u12 <- u12
        stencil.u20 <- u20; stencil.u21 <- u21; stencil.u22 <- u22

        stencil.dds0 <- 0.0; stencil.dds1 <- 0.0; stencil.dds2 <- 0.0
        stencil.ddv0 <- deltaM dv vi; stencil.ddv1 <- delta0 dv vi; stencil.ddv2 <- deltaP dv vi

        stencil.vs <- 0.0; stencil.vv <- 0.0; stencil.vu <- u01

        stencil.us0 <- u01; stencil.us1 <- u11; stencil.us2 <- u21
        stencil.uv0 <- u00; stencil.uv1 <- u01; stencil.uv2 <- u02

    else if si = ds.n - 1 then // 6 face

        let u00 = elem u (si-2) (vi-1)
        let u01 = elem u (si-2) (vi  )
        let u02 = elem u (si-2) (vi+1)
        let u10 = elem u (si-1) (vi-1)
        let u11 = elem u (si-1) (vi  )
        let u12 = elem u (si-1) (vi+1)
        let u20 = elem u (si  ) (vi-1)
        let u21 = elem u (si  ) (vi  )
        let u22 = elem u (si  ) (vi+1)

        stencil.ds0 <- alphaM2 ds si; stencil.ds1 <- alphaM1 ds si; stencil.ds2 <- alpha0 ds si
        stencil.dv0 <- betaM dv vi; stencil.dv1 <- beta0 dv vi; stencil.dv2 <- betaP dv vi

        stencil.u00 <- u00; stencil.u01 <- u01; stencil.u02 <- u02
        stencil.u10 <- u10; stencil.u11 <- u11; stencil.u12 <- u12
        stencil.u20 <- u20; stencil.u21 <- u21; stencil.u22 <- u22
 
        stencil.dds0 <- 0.0; stencil.dds1 <- 0.0; stencil.dds2 <- 0.0
        stencil.ddv0 <- deltaM dv vi; stencil.ddv1 <- delta0 dv vi; stencil.ddv2 <- deltaP dv vi

        stencil.vs <- 0.0; stencil.vv <- 0.0; stencil.vu <- u21

        stencil.us0 <- u01; stencil.us1 <- u11; stencil.us2 <- u21
        stencil.uv0 <- u20; stencil.uv1 <- u21; stencil.uv2 <- u22
    
    else if vi = dv.n - 1 then // 7 face

        let u00 = elem u (si-1) (vi-2)
        let u01 = elem u (si-1) (vi-1)
        let u02 = elem u (si-1) (vi  )
        let u10 = elem u (si  ) (vi-2)
        let u11 = elem u (si  ) (vi-1)
        let u12 = elem u (si  ) (vi  )
        let u20 = elem u (si+1) (vi-2)
        let u21 = elem u (si+1) (vi-1)
        let u22 = elem u (si+1) (vi  )

        stencil.ds0 <- betaM ds si; stencil.ds1 <- beta0 ds si; stencil.ds2 <- betaP ds si
        stencil.dv0 <- alphaM2 dv vi; stencil.dv1 <- alphaM1 dv vi; stencil.dv2 <- alpha0  dv vi

        stencil.u00 <- u00; stencil.u01 <- u01; stencil.u02 <- u02
        stencil.u10 <- u10; stencil.u11 <- u11; stencil.u12 <- u12
        stencil.u20 <- u20; stencil.u21 <- u21; stencil.u22 <- u22

        stencil.dds0 <- deltaM ds si; stencil.dds1 <- delta0 ds si; stencil.dds2 <- deltaP ds si
        stencil.ddv0 <- 0.0; stencil.ddv1 <- 0.0; stencil.ddv2 <- 0.0

        stencil.vs <- 0.0; stencil.vv <- 0.0; stencil.vu <- u12

        stencil.us0 <- u02; stencil.us1 <- u12; stencil.us2 <- u22
        stencil.uv0 <- u10; stencil.uv1 <- u11; stencil.uv2 <- u12
    
    else if vi = 0 then // 8 face

        let u00 = elem u (si-1) (vi  )
        let u01 = elem u (si-1) (vi+1)
        let u02 = elem u (si-1) (vi+2)
        let u10 = elem u (si  ) (vi  )
        let u11 = elem u (si  ) (vi+1)
        let u12 = elem u (si  ) (vi+2)
        let u20 = elem u (si+1) (vi  )
        let u21 = elem u (si+1) (vi+1)
        let u22 = elem u (si+1) (vi+2)

        stencil.ds0 <- betaM ds si; stencil.ds1 <- beta0 ds si; stencil.ds2 <- betaP ds si
        stencil.dv0 <- gamma0  dv vi; stencil.dv1 <- gammaP1 dv vi; stencil.dv2 <- gammaP2 dv vi

        stencil.u00 <- u00; stencil.u01 <- u01; stencil.u02 <- u02
        stencil.u10 <- u10; stencil.u11 <- u11; stencil.u12 <- u12
        stencil.u20 <- u20; stencil.u21 <- u21; stencil.u22 <- u22

        stencil.dds0 <- deltaM ds si; stencil.dds1 <- delta0 ds si; stencil.dds2 <- deltaP ds si
        stencil.ddv0 <- 0.0; stencil.ddv1 <- 0.0; stencil.ddv2 <- 0.0

        stencil.vs <- 0.0; stencil.vv <- 0.0; stencil.vu <- u10

        stencil.us0 <- u00; stencil.us1 <- u10; stencil.us2 <- u20
        stencil.uv0 <- u10; stencil.uv1 <- u11; stencil.uv2 <- u12
    
    else // 9 inner

        let u00 = elem u (si-1) (vi-1)
        let u01 = elem u (si-1) (vi  )
        let u02 = elem u (si-1) (vi+1)
        let u10 = elem u (si  ) (vi-1)
        let u11 = elem u (si  ) (vi  )
        let u12 = elem u (si  ) (vi+1)
        let u20 = elem u (si+1) (vi-1)
        let u21 = elem u (si+1) (vi  )
        let u22 = elem u (si+1) (vi+1)

        stencil.ds0 <- betaM ds si; stencil.ds1 <- beta0 ds si; stencil.ds2 <- betaP ds si
        stencil.dv0 <- betaM dv vi; stencil.dv1 <- beta0 dv vi; stencil.dv2 <- betaP dv vi

        stencil.u00 <- u00; stencil.u01 <- u01; stencil.u02 <- u02
        stencil.u10 <- u10; stencil.u11 <- u11; stencil.u12 <- u12 
        stencil.u20 <- u20; stencil.u21 <- u21; stencil.u22 <- u22

        stencil.dds0 <- deltaM ds si; stencil.dds1 <- delta0 ds si; stencil.dds2 <- deltaP ds si
        stencil.ddv0 <- deltaM dv vi; stencil.ddv1 <- delta0 dv vi; stencil.ddv2 <- deltaP dv vi

        stencil.vs <- 0.0; stencil.vv <- 0.0; stencil.vu <- u11

        stencil.us0 <- u01; stencil.us1 <- u11; stencil.us2 <- u21
        stencil.uv0 <- u10; stencil.uv1 <- u11; stencil.uv2 <- u12

    stencil

let [<ReflectedDefinition>] appF0 (heston:HestonModel) t (stencil:Stencil) =  
    let u0 =
        if stencil.vi = 0 then
            0.0
        else
            let u0 = stencil.ds0 * stencil.dv0 * stencil.u00
                    + stencil.ds0 * stencil.dv1 * stencil.u01
                    + stencil.ds0 * stencil.dv2 * stencil.u02
                    + stencil.ds1 * stencil.dv0 * stencil.u10
                    + stencil.ds1 * stencil.dv1 * stencil.u11
                    + stencil.ds1 * stencil.dv2 * stencil.u12
                    + stencil.ds2 * stencil.dv0 * stencil.u20
                    + stencil.ds2 * stencil.dv1 * stencil.u21
                    + stencil.ds2 * stencil.dv2 * stencil.u22
            u0 * heston.rho * heston.sigma * stencil.vs * stencil.vv
    u0

let [<ReflectedDefinition>] appF1 (heston:HestonModel) t (ds:RFiniteDifferenceWeights) (stencil:Stencil) =
    let u1 = 
        let rdt = heston.rd
        let rft = heston.rf
        if stencil.vi = 0 then
            let v2 = (rdt - rft) * stencil.vs

            if stencil.si = 0 then
                (v2 * stencil.ds0 - 0.5 * rdt) * stencil.us0
                + (v2 * stencil.ds1) * stencil.us1
                + (v2 * stencil.ds2) * stencil.us2
                    
            else if stencil.si = ds.n - 1 then
                (v2 * stencil.ds0) * stencil.us0
                + (v2 * stencil.ds1) * stencil.us1
                + (v2 * stencil.ds2 - 0.5 * rdt) * stencil.us2
                    
            else
                (v2 * stencil.ds0) * stencil.us0
                + (v2 * stencil.ds1 - 0.5 * rdt) * stencil.us1
                + (v2 * stencil.ds2) * stencil.us2
                    
        else
            let v1 = 0.5 * stencil.vv * stencil.vs * stencil.vs
            let v2 = (rdt - rft) * stencil.vs

            if stencil.si = 0 then
                let ds0 = delta ds 1 
                let ds1 = delta ds 2

                (v1 * 2.0 / (ds0 * (ds0 + ds1)) + v2 * stencil.ds0 - 0.5 * rdt) * stencil.us0
                + (-v1 * 2.0 / (ds0 * ds1) + v2 * stencil.ds1) * stencil.us1
                + (v1 * 2.0 / (ds1 * (ds0 + ds1)) + v2 * stencil.ds2) * stencil.us2
                    
            else if stencil.si = ds.n - 1 then
                let ds1 = delta ds (ds.n - 2)
                let ds0 = delta ds (ds.n - 1)

                (v1 * 2.0 / (ds1 * (ds0 + ds1)) + v2 * stencil.ds0) * stencil.us0
                + (-v1 * 2.0 / (ds0 * ds1) + v2 * stencil.ds1) * stencil.us1
                + (v1 * 2.0 / (ds0 * (ds0 + ds1)) + v2 * stencil.ds2 - 0.5 * rdt) * stencil.us2

            else
                (v1 * stencil.dds0 + v2 * stencil.ds0) * stencil.us0
                + (v1 * stencil.dds1 + v2 * stencil.ds1 - 0.5 * rdt) * stencil.us1
                + (v1 * stencil.dds2 + v2 * stencil.ds2) * stencil.us2
    u1                    

let [<ReflectedDefinition>] appF2 (heston:HestonModel) t (dv:RFiniteDifferenceWeights) (stencil:Stencil) =
    let u2 = 
        let rdt = heston.rd 
        let sigmat = heston.sigma 
        let kappat = heston.kappa 
        let etat = heston.eta 
        let v1 = 0.5 * sigmat * sigmat * stencil.vv
        let v2 = kappat * (etat - stencil.vv)

        if stencil.vi = 0 then
            let dv0 = delta dv 1
            let dv1 = delta dv 2

            (v1 * 2.0 / (dv0 * (dv0 + dv1)) + v2 * stencil.dv0 - 0.5 * rdt) * stencil.uv0
            + (-v1 * 2.0 / (dv0 * dv1) + v2 * stencil.dv1) * stencil.uv1
            + (v1 * 2.0 / (dv1 * (dv0 + dv1)) + v2 * stencil.dv2) * stencil.uv2
        
        else if stencil.vi = dv.n - 1 then
            let dv1 = delta dv (dv.n - 2)
            let dv0 = delta dv (dv.n - 1)

            (v1 * 2.0 / (dv1 * (dv0 + dv1)) + v2 * stencil.dv0) * stencil.uv0
            + (-v1 * 2.0 / (dv0 * dv1) + v2 * stencil.dv1) * stencil.uv1
            + (v1 * 2.0 / (dv0 * (dv0 + dv1)) + v2 * stencil.dv2 - 0.5 * rdt) * stencil.uv2
              
        else
            (v1 * stencil.ddv0 + v2 * stencil.dv0) * stencil.uv0
            + (v1 * stencil.ddv1 + v2 * stencil.dv1 - 0.5 * rdt) * stencil.uv1
            + (v1 * stencil.ddv2 + v2 * stencil.dv2) * stencil.uv2
    u2

[<ReflectedDefinition>]
let appF (heston:HestonModel) (t:float) (ds:RFiniteDifferenceWeights) (dv:RFiniteDifferenceWeights) (u:RMatrixRowMajor ref) (func:int -> int -> float -> float -> float -> float -> unit) =
    let start = blockIdx.x * blockDim.x + threadIdx.x
    let stride = gridDim.x * blockDim.x
    let mutable si = blockIdx.x * blockDim.x + threadIdx.x
    let ns = ds.n
    let nv = dv.n

    while si < ns do

        let mutable vi = blockIdx.y * blockDim.y + threadIdx.y

        while vi < nv do
            let s = stencil si vi ds dv u
            let u0 = appF0 heston t s 
            let u1 = appF1 heston t ds s 
            let u2 = appF2 heston t dv s 
            func si vi s.vu u0 u1 u2

            vi <- vi + blockDim.y * gridDim.y

        si <- si + blockDim.x * gridDim.x

[<ReflectedDefinition>]
let solveF1 (heston:HestonModel) (t:float) (t1:float) (thetaDt:float) (ds:RFiniteDifferenceWeights) (dv:RFiniteDifferenceWeights) (b:RMatrixRowMajor ref) (func:int -> int -> float -> unit) =
    let ns = ds.n
    let nv = dv.n
    let rdt = heston.rd 
    let rft = heston.rf 

    let shared = __extern_shared__<float>()
    let h = shared
    let d = h + ns
    let l = d + ns
    let u = l + ns

    let mutable vi = blockIdx.x
    while vi < nv do
        
        let vv = dv.X.[vi]

        let mutable si = threadIdx.x
        while si < ns do
       
            let vs = ds.X.[si]

            if si = 0 then
                l.[si] <- 0.0
                d.[si] <- 1.0
                u.[si] <- 0.0
                h.[si] <- 0.0                           
            else if si = ns - 1 then
                l.[si] <- -1.0 / (delta ds si)
                d.[si] <- 1.0 / (delta ds si)
                u.[si] <- 0.0
                h.[si] <- exp(-t1 * rft)
            else
                if vv > 0.0 then
                    let v1 = 0.5 * vv * vs * vs
                    let v2 = (rdt - rft) * vs

                    let deltaSM = deltaM ds si
                    let deltaS0 = delta0 ds si
                    let deltaSP = deltaP ds si

                    let betaSM = betaM ds si
                    let betaS0 = beta0 ds si
                    let betaSP = betaP ds si

                    l.[si] <- -(v1 * deltaSM + v2 * betaSM) * thetaDt
                    d.[si] <- 1.0 - (v1 * deltaS0 + v2 * betaS0 - 0.5 * rdt) * thetaDt
                    u.[si] <- -(v1 * deltaSP + v2 * betaSP) * thetaDt
                else
                    let v = (rdt - rft) * vs / (delta ds si)
                    l.[si] <- -v * thetaDt
                    d.[si] <- 1.0 - (-v - 0.5 * rdt) * thetaDt
                    u.[si] <- 0.0

                h.[si] <- elem b si vi

            si <- si + blockDim.x

        __syncthreads()

        triDiagPcrSingleBlock ns l d u h

        si <- threadIdx.x
        while si < ns do       
            func si vi h.[si] 
            si <- si + blockDim.x

        __syncthreads()

        vi <- vi + gridDim.x
   
[<ReflectedDefinition>]
let solveF2 (heston:HestonModel) (t:float) (t1:float) (thetaDt:float) (ds:RFiniteDifferenceWeights) (dv:RFiniteDifferenceWeights) (b:RMatrixRowMajor ref) (func:int -> int -> float -> unit)  =
    let ns = ds.n
    let nv = dv.n
    let rdt = heston.rd 
    let rft = heston.rf 
    let sigmat = heston.sigma 
    let kappat = heston.kappa 
    let etat = heston.eta 

    let shared = __extern_shared__<float>()
    let h = shared
    let d = h + nv
    let l = d + nv
    let u = l + nv

    let mutable si = blockIdx.x
    while si < ns - 1 do

        let vs = ds.X.[si]

        let mutable vi = threadIdx.x
        while vi < dv.n do

            let vv = dv.X.[vi]

            if si = 0 then
                h.[vi] <- 0.0
            else
                if vi = 0 then
                    l.[vi] <- 0.0
                    d.[vi] <- 1.0
                    u.[vi] <- 0.0
                    h.[vi] <- elem b si vi
                else if vi = nv - 1 then
                    l.[vi] <- 0.0
                    d.[vi] <- 1.0
                    u.[vi] <- 0.0
                    h.[vi] <- vs * exp(-t1 * rft)
                else
                    let deltaVM = deltaM dv vi
                    let deltaV0 = delta0 dv vi
                    let deltaVP = deltaP dv vi

                    let betaVM = betaM dv vi
                    let betaV0 = beta0 dv vi
                    let betaVP = betaP dv vi

                    let v1 = 0.5 * sigmat * sigmat * vv
                    let v2 = kappat * (etat - vv)

                    l.[vi] <- -thetaDt * v1 * deltaVM - thetaDt * v2 * betaVM
                    d.[vi] <- 1.0 - thetaDt * (v1 * deltaV0 + v2 * betaV0 - 0.5 * rdt)
                    u.[vi] <- -thetaDt * (v1 * deltaVP + v2 * betaVP)
                    h.[vi] <- elem b si vi

            vi <- vi + blockDim.x

        __syncthreads()

        if si <> 0 then
            triDiagPcrSingleBlock nv l d u h
        
        vi <- threadIdx.x    
        while vi < nv do  
            func si vi h.[vi]
            vi <- vi + blockDim.x

        si <- si + gridDim.x

module DouglasScheme =

    [<ReflectedDefinition>]
    let appF dt thetaDt (u:RMatrixRowMajor ref) (b:RMatrixRowMajor ref) si vi vu u0 u1 u2 =
        set b si vi (vu + dt * (u0 + u1 + u2) - thetaDt * u1)
        set u si vi u2

    [<ReflectedDefinition>]
    let solveF1 thetaDt (u:RMatrixRowMajor ref) (b:RMatrixRowMajor ref) si vi x =
        set b si vi (x - thetaDt * elem u si vi)

    [<ReflectedDefinition>]
    let solveF2 (heston:HestonModel) (ds:RFiniteDifferenceWeights) (u:RMatrixRowMajor ref) si vi x =
        set u si vi x
        if si = ds.n - 2 then 
            set u (si+1) vi ((delta ds (ds.n-1)) * exp(-heston.rf) + x)

module HVScheme =

    [<ReflectedDefinition>]
    let appF1 dt thetaDt (u:RMatrixRowMajor ref) (b:RMatrixRowMajor ref) (y:RMatrixRowMajor ref) si vi vu u0 u1 u2 =
        set b si vi (vu + dt * (u0 + u1 + u2) - thetaDt * u1)
        set u si vi u2
        set y si vi (vu + dt * (u0 + u1 + u2) - 0.5 * dt * (u0 + u1 + u2))

    [<ReflectedDefinition>]
    let appF2 dt thetaDt (u:RMatrixRowMajor ref) (b:RMatrixRowMajor ref) (y:RMatrixRowMajor ref) si vi vu u0 u1 u2 =
        set b si vi ((elem y si vi) + dt * (u0 + u1 + u2) - thetaDt * u1)
        set u si vi u2

    [<ReflectedDefinition>]
    let solveF1 thetaDt (u:RMatrixRowMajor ref) (b:RMatrixRowMajor ref) si vi x =
        set b si vi (x - thetaDt * elem u si vi)

    [<ReflectedDefinition>]
    let solveF2 (heston:HestonModel) t thetaDt t1 (ds:RFiniteDifferenceWeights) (u:RMatrixRowMajor ref) si vi x =
        set u si vi x
        if si = ds.n - 2 then 
            set u (si+1) vi ((delta ds (ds.n-1)) * exp(-heston.rf) + x) 

type DouglasSolverParam =
    val theta:float
    val sMin:float
    val sMax:float
    val vMax:float
    val ns:int
    val nv:int
    val nt:int
    val sC:float
    val vC:float

    new(theta:float, sMin:float, sMax:float, vMax:float, ns:int, nv:int, nt:int, sC:float, vC:float) =
        {theta = theta; sMin = sMin; sMax = sMax; vMax = vMax; ns = ns; nv = nv; nt = nt; sC = sC; vC = vC}


/// Solve Hesten pde.
let douglasSolver = cuda {

    let! initCondKernel =     
        <@ fun ns nv (s:DevicePtr<float>) (u:RMatrixRowMajor) optionType strike ->
            let i = blockIdx.x*blockDim.x + threadIdx.x
            let j = blockIdx.y*blockDim.y + threadIdx.y
            if i < ns && j < nv then 
                let payoff = match optionType with
                             |  1.0 -> max (s.[i] - strike) 0.0
                             | -1.0 -> max (strike - s.[i]) 0.0 
                             | _ -> 0.0
                set (ref u) i j payoff @> |> defineKernelFuncWithName "initCondition"

    let! appFKernel =
        <@ fun heston t dt thetaDt ds dv u u2 b ->
            appF heston t ds dv (ref u) (DouglasScheme.appF dt thetaDt (ref u2) (ref b)) @> |> defineKernelFuncWithName "applyF"
                
    let! solveF1Kernel =
        <@ fun heston t t1 thetaDt ds dv u2 b ->
            solveF1 heston t t1 thetaDt ds dv (ref b) (DouglasScheme.solveF1 thetaDt (ref u2) (ref b)) @> |> defineKernelFuncWithName "solveF1"

    let! solveF2Kernel =
        <@ fun heston t t1 thetaDt ds dv u b ->
            solveF2 heston t t1 thetaDt ds dv (ref b) (DouglasScheme.solveF2 heston ds (ref u)) @> |> defineKernelFuncWithName "solveF2"

    let! finiteDifferenceWeights = finiteDifferenceWeights

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let initCondKernel = initCondKernel.Apply m
        let appFKernel = appFKernel.Apply m
        let solveF1Kernel = solveF1Kernel.Apply m
        let solveF2Kernel = solveF2Kernel.Apply m
        let finiteDifferenceWeights = finiteDifferenceWeights.Apply m
         
        fun (heston:HestonModel) (optionType:OptionType) strike timeToMaturity (param:DouglasSolverParam) ->

            let s, ds = concentratedGridBetween param.sMin param.sMax strike param.ns param.sC
            let v, dt = concentratedGridBetween 0.0 param.vMax 0.0 param.nv param.vC
            let t, dt = homogeneousGrid param.nt 0.0 timeToMaturity

            pcalc {
                let! s = DArray.scatterInBlob worker s
                let! v = DArray.scatterInBlob worker v
                
                let! b = DMatrix.createInBlob<float> worker RowMajorOrder param.ns param.nv
                let! u = DMatrix.createInBlob<float> worker RowMajorOrder param.ns param.nv
                let! u2 = DMatrix.createInBlob<float> worker RowMajorOrder param.ns param.nv
                
                let! sDiff = finiteDifferenceWeights s.Length s   
                let! vDiff = finiteDifferenceWeights v.Length v               

                do! PCalc.action (fun hint ->
                    let lpm = LaunchParam(dim3(divup param.ns 16, divup param.ns 16), dim3(16, 16)) |> hint.ModifyLaunchParam
                    let lps = LaunchParam(param.nv, param.ns, 4*param.ns*sizeof<float>) |> hint.ModifyLaunchParam
                    let lpv = LaunchParam(param.ns, param.nv, 4*param.nv*sizeof<float>) |> hint.ModifyLaunchParam

                    let sDiff = sDiff.Raw()
                    let vDiff = vDiff.Raw()

                    let b = RMatrixRowMajor(b.NumRows, b.NumCols, b.Storage.Ptr)
                    let u = RMatrixRowMajor(u.NumRows, u.NumCols, u.Storage.Ptr)
                    let u2 = RMatrixRowMajor(u2.NumRows, u2.NumCols, u2.Storage.Ptr)

                    initCondKernel.Launch lpm param.ns param.nv sDiff.X u optionType.sign strike

                    for ti = 0 to param.nt - 2 do
                    
                        let t0 = t.[ti]
                        let t1 = t.[ti + 1]
                        let dt = t1 - t0
                        let thetaDt = dt * param.theta

                        appFKernel.Launch lpm heston t0 dt thetaDt sDiff vDiff u u2 b

                        solveF1Kernel.Launch lps heston t0 t1 thetaDt sDiff vDiff u2 b

                        solveF2Kernel.Launch lpv heston t0 t1 thetaDt sDiff vDiff u b
                )
                    
                return s, v, u } ) }


