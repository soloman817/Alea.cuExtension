module Alea.CUDA.Extension.Finance.Heston

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Alea.CUDA.Extension
open Alea.CUDA.Extension.TriDiag
open Alea.CUDA.Extension.Finance.Grid

open Util 

/// Concentrate at critical point and map critical point to grid point
let stateGrid = concentratedGrid

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
type HestonModel=
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
type Differences =
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
let [<ReflectedDefinition>] delta(diff:Differences) i = diff.Delta.[i-1]
let [<ReflectedDefinition>] alphaM2(diff:Differences) i = diff.AlphaM2.[i-2]
let [<ReflectedDefinition>] alphaM1(diff:Differences) i = diff.AlphaM1.[i-2]
let [<ReflectedDefinition>] alpha0(diff:Differences) i = diff.Alpha0.[i-2]
let [<ReflectedDefinition>] betaM(diff:Differences) i = diff.BetaM.[i-1]
let [<ReflectedDefinition>] beta0(diff:Differences) i = diff.Beta0.[i-1]
let [<ReflectedDefinition>] betaP(diff:Differences) i = diff.BetaP.[i-1]
let [<ReflectedDefinition>] gamma0(diff:Differences) i = diff.Gamma0.[i]
let [<ReflectedDefinition>] gammaP1(diff:Differences) i = diff.GammaP1.[i]
let [<ReflectedDefinition>] gammaP2(diff:Differences) i = diff.GammaP2.[i]
let [<ReflectedDefinition>] deltaM(diff:Differences) i = diff.DeltaM.[i-1]
let [<ReflectedDefinition>] delta0(diff:Differences) i = diff.Delta0.[i-1]
let [<ReflectedDefinition>] deltaP(diff:Differences) i = diff.DeltaP.[i-1]

type DifferenceHighLevel =
    {
        Delta : DArray<float>
    }

let finiteDifferenceWeights = cuda {

    let! diffKernel = 
        <@  fun (n:int) (x:DevicePtr<float>) (dx:DevicePtr<float>) -> 
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < n - 1 do
                dx.[i] <- x.[i + 1] - x.[i]
                i <- i + stride @> |> defineKernelFunc

    let! finiteDifferenceKernel = 
        <@ fun (diff:Differences) ->
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

                i <- i + stride @> |> defineKernelFunc

    return PFunc(fun (m:Module) n (x:DevicePtr<float>) ->
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

            // %XIANG% (2)

            // here again, you should better move this statement inside the action,
            // because x.Ptr will trigger the blob malloc.
            // So the difference between a raw pointer and a DScalar, DArray, DMatrix is
            // DXXXX is DELAYED, first it is just a blob slot, then if you call its .Ptr
            // that means you really do want that memory, then that will trigger the blob.
            // So here acturally you need have a higher level of the struct, please reference
            // the xorshift implementation.
//            let diff = Differences(n, x, delta.Ptr, alpha0.Ptr, alphaM1.Ptr, alphaM2.Ptr, 
//                                   beta0.Ptr, betaP.Ptr, betaM.Ptr, gamma0.Ptr, gammaP1.Ptr, gammaP2.Ptr, 
//                                   delta0.Ptr, deltaP.Ptr, deltaM.Ptr)  
            
            do! PCalc.action (fun hint ->
                // now I move it here, to trigger the memories malloc inside the action
                // because action is delayed
                let diff = Differences(n, x, delta.Ptr, alpha0.Ptr, alphaM1.Ptr, alphaM2.Ptr, 
                                       beta0.Ptr, betaP.Ptr, betaM.Ptr, gamma0.Ptr, gammaP1.Ptr, gammaP2.Ptr, 
                                       delta0.Ptr, deltaP.Ptr, deltaM.Ptr)  
                let blockSize = 256
                let gridSize = Util.divup n blockSize           
                let lp = LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam
                diffKernel.Launch m lp n x delta.Ptr
                finiteDifferenceKernel.Launch m lp diff)

            // and I return the high level struct which is delayed (with DArray, or you might need DMatrix)
            let diff : DifferenceHighLevel =
                {
                    Delta = delta
                }
                                  
            return diff      
        } ) }


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

[<Struct>]
type StateVolMatrix =
    val ns : int // number of state grid points in rows
    val nv : int // number of volatiltiy grid points in columns
    [<PointerField(MemorySpace.Global)>] val mutable u : int64

    [<PointerProperty("u")>] member this.U with get () = DevicePtr<float>(this.u) and set (ptr:DevicePtr<float>) = this.u <- ptr.Handle64

    [<ReflectedDefinition>]
    new (ns:int, nv:int, u:DevicePtr<float>) = { ns = ns; nv = nv; u = u.Handle64 }
        
    [<ReflectedDefinition>]
    static member Elem(s:StateVolMatrix ref, si:int, vi:int) =
        s.contents.U.[vi + si*s.contents.nv]  

    [<ReflectedDefinition>]
    static member Elem(s:StateVolMatrix ref, si:int, vi:int, value:float) =
        s.contents.U.[vi + si*s.contents.nv] <- value

let [<ReflectedDefinition>] elem (u:DevicePtr<StateVolMatrix>) (si:int) (vi:int) =
    StateVolMatrix.Elem(u.Ref(0), si, vi)

let [<ReflectedDefinition>] set (u:DevicePtr<StateVolMatrix>) (si:int) (vi:int) (value:float) =
    StateVolMatrix.Elem(u.Ref(0), si, vi, value)

let [<ReflectedDefinition>] stencil si vi (ds:Differences) (dv:Differences) (u:DevicePtr<StateVolMatrix>)  =
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

let [<ReflectedDefinition>] applyF0 (heston:HestonModel) t (stencil:Stencil) =  
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

let [<ReflectedDefinition>] applyF1 (heston:HestonModel) t (ds:Differences) (stencil:Stencil) =
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

let [<ReflectedDefinition>] applyF2 (heston:HestonModel) t (dv:Differences) (stencil:Stencil) =
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
let applyF (heston:HestonModel) ns nv t (ds:Differences) (dv:Differences) (u:DevicePtr<StateVolMatrix>) (func:int -> int -> float -> float -> float -> float -> unit) =
    let start = blockIdx.x * blockDim.x + threadIdx.x
    let stride = gridDim.x * blockDim.x
    let mutable si = blockIdx.x * blockDim.x + threadIdx.x

    while si < ns do

        let mutable vi = blockIdx.y * blockDim.y + threadIdx.y

        while vi < nv do
            let s = stencil si vi ds dv u
            let u0 = applyF0 heston t s 
            let u1 = applyF1 heston t ds s 
            let u2 = applyF2 heston t dv s 
            func si vi s.vu u0 u1 u2

            vi <- vi + blockDim.y * gridDim.y

        si <- si + blockDim.x * gridDim.x

[<ReflectedDefinition>]
let solveF1 (heston:HestonModel) t (ds:Differences) (dv:Differences) (b:DevicePtr<StateVolMatrix>) (func:int -> int -> float -> unit) tk1 thetaDt =
    let rdt = heston.rd 
    let rft = heston.rf 

    let shared = __extern_shared__<float>()
    let h = shared
    let d = h + ds.n
    let l = d + ds.n
    let u = l + ds.n

    let mutable vi = blockIdx.x
    while vi < dv.n do
        
        let vv = dv.X.[vi]

        let mutable si = threadIdx.x
        while si < ds.n do
       
            let vs = ds.X.[si]

            if si = 0 then
                l.[si] <- 0.0
                d.[si] <- 1.0
                u.[si] <- 0.0
                h.[si] <- 0.0                           
            else if si = ds.n - 1 then
                l.[si] <- -1.0 / (delta ds si)
                d.[si] <- 1.0 / (delta ds si)
                u.[si] <- 0.0
                h.[si] <- exp(-tk1 * rft)
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

        triDiagPcrSingleBlock ds.n l d u h

        si <- threadIdx.x
        while si < ds.n do       
            func si vi h.[si] 
            si <- si + blockDim.x

        __syncthreads()

        vi <- vi + gridDim.x
   
[<ReflectedDefinition>]
let solveF2 (heston:HestonModel) t (ds:Differences) (dv:Differences) (b:DevicePtr<StateVolMatrix>) (func:int -> int -> float -> unit) tk1 thetaDt =
    let rdt = heston.rd 
    let rft = heston.rf 
    let sigmat = heston.sigma 
    let kappat = heston.kappa 
    let etat = heston.eta 

    let shared = __extern_shared__<float>()
    let h = shared
    let d = h + ds.n
    let l = d + ds.n
    let u = l + ds.n

    let mutable si = blockIdx.x
    while si < ds.n - 1 do

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
                else if vi = dv.n - 1 then
                    l.[vi] <- 0.0
                    d.[vi] <- 1.0
                    u.[vi] <- 0.0
                    h.[vi] <- vs * exp(-tk1 * rft)
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
            triDiagPcrSingleBlock ds.n l d u h
        
        vi <- threadIdx.x    
        while vi < dv.n do  
            func si vi h.[vi]
            vi <- vi + blockDim.x

        si <- si + gridDim.x

module DouglasScheme =

    [<ReflectedDefinition>]
    let applyF dt thetaDt (b:DevicePtr<StateVolMatrix>) (u:DevicePtr<StateVolMatrix>) si vi vu u0 u1 u2 =
        set b si vi (vu + dt * (u0 + u1 + u2) - thetaDt * u1)
        set u si vi u2

    [<ReflectedDefinition>]
    let solveF1 thetaDt (b:DevicePtr<StateVolMatrix>) (u:DevicePtr<StateVolMatrix>) si vi x =
        set b si vi (x - thetaDt * elem u si vi)

    [<ReflectedDefinition>]
    let solveF2 (heston:HestonModel) t (ds:Differences) (u:DevicePtr<StateVolMatrix>) si vi x =
        set u si vi x
        if si = ds.n - 2 then 
            set u (si+1) vi ((delta ds (ds.n-1)) * exp(-heston.rf) + x)

module HVScheme =

    [<ReflectedDefinition>]
    let applyF1 dt thetaDt (b:DevicePtr<StateVolMatrix>) (u:DevicePtr<StateVolMatrix>) (y:DevicePtr<StateVolMatrix>) si vi vu u0 u1 u2 =
        set b si vi (vu + dt * (u0 + u1 + u2) - thetaDt * u1)
        set u si vi u2
        set y si vi (vu + dt * (u0 + u1 + u2) - 0.5 * dt * (u0 + u1 + u2))

    [<ReflectedDefinition>]
    let applyF2 dt thetaDt (b:DevicePtr<StateVolMatrix>) (u:DevicePtr<StateVolMatrix>) (y:DevicePtr<StateVolMatrix>) si vi vu u0 u1 u2 =
        set b si vi ((elem y si vi) + dt * (u0 + u1 + u2) - thetaDt * u1)
        set u si vi u2

    [<ReflectedDefinition>]
    let solveF1 thetaDt (b:DevicePtr<StateVolMatrix>) (u:DevicePtr<StateVolMatrix>) si vi x =
        set b si vi (x - thetaDt * elem u si vi)

    [<ReflectedDefinition>]
    let solveF2 (heston:HestonModel) t thetaDt tk1 (ds:Differences) (u:DevicePtr<StateVolMatrix>) si vi x =
        set u si vi x
        if si = ds.n - 2 then 
            set u (si+1) vi ((delta ds (ds.n-1)) * exp(-heston.rf) + x) 

type OptionType =
| Call
| Put

type Param =
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
let buildDouglas = cuda {

    let! initCondKernel =     
        <@ fun ns nv t (s:DevicePtr<float>) (y:DevicePtr<float>) (u:DevicePtr<StateVolMatrix>) optionType K ->
            let i = blockIdx.x*blockDim.x + threadIdx.x
            let j = blockIdx.y*blockDim.y + threadIdx.y
            if i < ns && j < nv then 
                let payoff = match optionType with
                             | Call -> max (s.[i] - K) 0.0
                             | Put  -> max (K - s.[i]) 0.0 
                set u i j payoff @> |> defineKernelFunc

    let! applyFKernel =
        <@ fun dt thetaDt b ns nv t ds dv u heston ->
            applyF heston ns nv t ds dv u (DouglasScheme.applyF dt thetaDt b u) @> |> defineKernelFunc
                
    let! solveF1Kernel =
        <@ fun dt thetaDt tk1 b ns nv t ds dv u heston ->
            solveF1 heston t ds dv b (DouglasScheme.solveF1 thetaDt b u) tk1 thetaDt @> |> defineKernelFunc

    let! solveF2Kernel =
        <@ fun dt thetaDt tk1 b ns nv t ds dv u heston ->
            solveF2 heston t ds dv b (DouglasScheme.solveF2 heston t ds u) tk1 thetaDt @> |> defineKernelFunc

    let! finiteDifferenceWeights = finiteDifferenceWeights

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let initCondKernel = initCondKernel.Apply m
        let applyFKernel = applyFKernel.Apply m
        let solveF1Kernel = solveF1Kernel.Apply m
        let solveF2Kernel = solveF2Kernel.Apply m
        let finiteDifferenceWeights = finiteDifferenceWeights.Apply m
         
        fun (heston:HestonModel) strike timeToMaturity (param:Param) ->

            let s = concentratedGrid param.sMin param.sMax strike param.ns param.sC
            let v = concentratedGrid 0.0 param.vMax 0.0 param.nv param.vC
            let t = homogeneousGrid param.nt 0.0 timeToMaturity

            pcalc {
                let! s = DArray.scatterInBlob worker s
                let! v = DArray.scatterInBlob worker v
                let! sDiff = finiteDifferenceWeights s.Length s.Ptr  
                let! vDiff = finiteDifferenceWeights v.Length v.Ptr  
                return sDiff, vDiff
            } ) }

//            let nu = nx * ny
//            let lp0 = LaunchParam(dim3(divup nx 16, divup ny 16), dim3(16, 16))
//
//
//            let lpx = LaunchParam(ny, nx, 4*nx*sizeof<float>)
//            let lpy = LaunchParam(nx, ny, 4*ny*sizeof<float>)
//
//            let launch (hint:ActionHint) (t:float[]) (x:DevicePtr<float>) dx (y:DevicePtr<float>) dy (u0:DevicePtr<float>) (u1:DevicePtr<float>) k tstart tstop dt =
//                let lp0 = lp0 |> hint.ModifyLaunchParam
//                let lpx = lpx |> hint.ModifyLaunchParam
//                let lpy = lpy |> hint.ModifyLaunchParam
//
//                let initCondKernelFunc = initCondKernel.Launch lp0 
//                let xSweepKernelFunc = xSweepKernel.Launch lpx
//                let ySweepKernelFunc = ySweepKernel.Launch lpy
//
//                initCondKernelFunc nx ny tstart x y u0
//
//                if t.Length > 1 then
//                    let step (t0, t1) =
//                        let dt = t1 - t0
//                        let Cx = k * dt / (dx * dx)
//                        let Cy = k * dt / (dy * dy)
//                        xSweepKernelFunc nx ny x y Cx Cy dt t0 (t0 + 0.5 * dt) u0 u1
//                        ySweepKernelFunc nx ny x y Cx Cy dt (t0 + 0.5 * dt) t1 u1 u0
//
//                    let timeIntervals = t |> Seq.pairwise |> Seq.toArray
//                    timeIntervals |> Array.iter step
//
//            { new ISolver with
//                member this.GenT tstart tstop dt = timeGrid tstart tstop dt 5
//                member this.GenX Lx = stateGrid nx Lx
//                member this.GenY Ly = stateGrid ny Ly
//                member this.NumU = nu
//                member this.Launch hint t x dx y dy u0 u1 k tstart tstop dt = launch hint t x dx y dy u0 u1 k tstart tstop dt
//            } ) }
