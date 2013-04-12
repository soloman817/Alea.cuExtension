module Alea.CUDA.Extension.Finance.Heston

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Alea.CUDA.Extension.TriDiag

open Util 

// can struct of functions be used?
type HestonModel = {
    rho : float -> float
    sigma : float -> float
    rd : float -> float
    rf : float -> float
    kappa : float -> float
    eta : float -> float
}

type Differences = {
    n:int; x:DevicePtr<float>; delta:DevicePtr<float>
    alpha0:DevicePtr<float>; alphaM1:DevicePtr<float>; alphaM2:DevicePtr<float> 
    beta0:DevicePtr<float>;  betaP:DevicePtr<float>;   betaM:DevicePtr<float>   
    gamma0:DevicePtr<float>; gammaP1:DevicePtr<float>; gammaP2:DevicePtr<float> 
    delta0:DevicePtr<float>; deltaP:DevicePtr<float>;  deltaM:DevicePtr<float> 
}

type Stencil = {
    si:int; vi:int

    vs:float; vv:float; vu:float
    u0:float; u1:float; u2:float
    
    ds0:float; ds1:float; ds2:float
    dv0:float; dv1:float; dv2:float

    u00:float; u01:float; u02:float
    u10:float; u11:float; u12:float
    u20:float; u21:float; u22:float
    
    dds0:float; dds1:float; dds2:float
    ddv0:float; ddv1:float; ddv2:float
    
    us0:float; us1:float; us2:float
    uv0:float; uv1:float; uv2:float
}

[<ReflectedDefinition>]
let deltaX (n:int) (x:DevicePtr<float>) (dx:DevicePtr<float>) =
    let start = blockIdx.x * blockDim.x + threadIdx.x
    let stride = gridDim.x * blockDim.x
    let mutable i = start
    while i < n - 1 do
        dx.[i] <- x.[i + 1] - x.[i]
        i <- i + stride 

[<ReflectedDefinition>]
let finiteDifferenceWeights (diff:Differences) =
    let start = blockIdx.x * blockDim.x + threadIdx.x
    let stride = gridDim.x * blockDim.x
    let mutable i = start
    while i < diff.n - 2 do
        let dx0 = diff.delta.[i]
        let dx1 = diff.delta.[i+1]

        diff.alphaM2.[i] <- dx1 / (dx0 * (dx0 + dx1))
        diff.alphaM1.[i] <- -(dx0 + dx1) / (dx0 * dx1)
        diff.alpha0.[i]  <- (dx0 + 2.0 * dx1) / (dx1 * (dx0 + dx1))
        
        diff.betaM.[i]   <- -dx1 / (dx0 * (dx0 + dx1))
        diff.beta0.[i]   <- (dx1 - dx0) / (dx0 * dx1)
        diff.betaP.[i]   <- dx0 / (dx1 * (dx0 + dx1))
       
        diff.gamma0.[i]  <- (-2.0 * dx0 - dx1) / (dx0 * (dx0 + dx1))
        diff.gammaP1.[i] <- (dx0 + dx1) / (dx0 * dx1)
        diff.gammaP2.[i] <- -dx0 / (dx1 * (dx0 + dx1))
        
        diff.deltaM.[i]  <- 2.0 / (dx0 * (dx0 + dx1))
        diff.delta0.[i]  <- -2.0 / (dx0 * dx1)
        diff.deltaP.[i]  <- 2.0 / (dx1 * (dx0 + dx1))

        i <- i + stride 

let [<ReflectedDefinition>] delta (diff:Differences) i = diff.delta.[i-1]
 
let [<ReflectedDefinition>] alphaM2 (diff:Differences) i = diff.alphaM2.[i-2]
let [<ReflectedDefinition>] alphaM1(diff:Differences) i = diff.alphaM1.[i-2]
let [<ReflectedDefinition>] alpha0(diff:Differences) i = diff.alpha0.[i-2]

let [<ReflectedDefinition>] betaM(diff:Differences) i = diff.betaM.[i-1]
let [<ReflectedDefinition>] beta0(diff:Differences) i = diff.beta0.[i-1]
let [<ReflectedDefinition>] betaP(diff:Differences) i = diff.betaP.[i-1]

let [<ReflectedDefinition>] gamma0(diff:Differences) i = diff.gamma0.[i]
let [<ReflectedDefinition>] gammaP1(diff:Differences) i = diff.gammaP1.[i]
let [<ReflectedDefinition>] gammaP2(diff:Differences) i = diff.gammaP2.[i]

let [<ReflectedDefinition>] deltaM(diff:Differences) i = diff.deltaM.[i-1]
let [<ReflectedDefinition>] delta0(diff:Differences) i = diff.delta0.[i-1]
let [<ReflectedDefinition>] deltaP(diff:Differences) i = diff.deltaP.[i-1]

let [<ReflectedDefinition>] stencil si vi (ds:Differences) (dv:Differences) (u:float[,]) (* u should be a DeviceMatrix *) =
    if si = 0 && vi = 0 then  // 1 corner

        let u00 = u.[si  , vi  ]
        let u01 = u.[si  , vi+1]
        let u02 = u.[si  , vi+2]
        let u10 = u.[si+1, vi  ]
        let u11 = u.[si+1, vi+1]
        let u12 = u.[si+1, vi+2]
        let u20 = u.[si+2, vi  ]
        let u21 = u.[si+2, vi+1]
        let u22 = u.[si+2, vi+2]

        { si = si; vi = vi
        
          ds0 = gamma0 ds si; ds1 = gammaP1 ds si; ds2 = gammaP2 ds si
          dv0 = gamma0 dv vi; dv1 = gammaP1 dv vi; dv2 = gammaP2 dv vi

          u00 = u00; u01 = u01; u02 = u02; u10 = u10; u11 = u11; u12 = u12; u20 = u20; u21 = u21; u22 = u22
 
          vs = 0.0; vv = 0.0; vu = u00
          u0 = 0.0; u1 = 0.0; u2 = 0.0

          us0 = u00; us1 = u10; us2 = u20
          uv0 = u00; uv1 = u01; uv2 = u02

          dds0 = 0.0; dds1 = 0.0; dds2 = 0.0
          ddv0 = 0.0; ddv1 = 0.0; ddv2 = 0.0
        }

    else if si = ds.n - 1 && vi = 0 then // 2 corner

        let u00 = u.[si-2, vi  ]
        let u01 = u.[si-2, vi+1]
        let u02 = u.[si-2, vi+2]
        let u10 = u.[si-1, vi  ]
        let u11 = u.[si-1, vi+1]
        let u12 = u.[si-1, vi+2]
        let u20 = u.[si  , vi  ]
        let u21 = u.[si  , vi+1]
        let u22 = u.[si  , vi+2]

        { si = si; vi = vi
          
          ds0 = alphaM2 ds si; ds1 = alphaM1 ds si; ds2 = alpha0 ds si
          dv0 = gamma0 dv vi; dv1 = gammaP1 dv vi; dv2 = gammaP2 dv vi

          u00 = u00; u01 = u01; u02 = u02; u10 = u10; u11 = u11; u12 = u12; u20 = u20; u21 = u21; u22 = u22

          vs = 0.0; vv = 0.0; vu = u20
          u0 = 0.0; u1 = 0.0; u2 = 0.0

          us0 = u00; us1 = u10; us2 = u20
          uv0 = u20; uv1 = u21; uv2 = u22

          dds0 = 0.0; dds1 = 0.0; dds2 = 0.0
          ddv0 = 0.0; ddv1 = 0.0; ddv2 = 0.0
        }

    else if si = 0 && vi = dv.n - 1 then // 3 corner

        let u00 = u.[si  , vi-2]
        let u01 = u.[si  , vi-1]
        let u02 = u.[si  , vi  ]
        let u10 = u.[si+1, vi-2]
        let u11 = u.[si+1, vi-1]
        let u12 = u.[si+1, vi  ]
        let u20 = u.[si+2, vi-2]
        let u21 = u.[si+2, vi-1]
        let u22 = u.[si+2, vi  ]

        { si = si; vi = vi
          
          ds0 = gamma0 ds si; ds1 = gammaP1 ds si; ds2 = gammaP2 ds si
          dv0 = alphaM2 dv vi; dv1 = alphaM1 dv vi; dv2 = alpha0 dv vi

          u00 = u00; u01 = u01; u02 = u02; u10 = u10; u11 = u11; u12 = u12; u20 = u20; u21 = u21; u22 = u22

          vs = 0.0; vv = 0.0; vu = u02
          u0 = 0.0; u1 = 0.0; u2 = 0.0

          us0 = u02; us1 = u12; us2 = u22
          uv0 = u00; uv1 = u01; uv2 = u02 

          dds0 = 0.0; dds1 = 0.0; dds2 = 0.0
          ddv0 = 0.0; ddv1 = 0.0; ddv2 = 0.0
        }
    
    else if si = ds.n - 1 && vi = dv.n - 1 then // 4 corner

        let u00 = u.[si-2, vi-2]
        let u01 = u.[si-2, vi-1]
        let u02 = u.[si-2, vi  ]
        let u10 = u.[si-1, vi-2]
        let u11 = u.[si-1, vi-1]
        let u12 = u.[si-1, vi  ]
        let u20 = u.[si  , vi-2]
        let u21 = u.[si  , vi-1]
        let u22 = u.[si  , vi  ]

        { si = si; vi = vi
          
          ds0 = alphaM2 ds si; ds1 = alphaM1 ds si; ds2 = alpha0 ds si
          dv0 = alphaM2 dv vi; dv1 = alphaM1 dv vi; dv2 = alpha0 dv vi

          u00 = u00; u01 = u01; u02 = u02; u10 = u10; u11 = u11; u12 = u12; u20 = u20; u21 = u21; u22 = u22

          vs = 0.0; vv = 0.0; vu = u22
          u0 = 0.0; u1 = 0.0; u2 = 0.0

          us0 = u02; us1 = u12; us2 = u22
          uv0 = u20; uv1 = u21; uv2 = u22

          dds0 = 0.0; dds1 = 0.0; dds2 = 0.0
          ddv0 = 0.0; ddv1 = 0.0; ddv2 = 0.0
        }
    
    else if si = 0 then // 5 face
    
        let u00 = u.[si  , vi-1]
        let u01 = u.[si  , vi  ]
        let u02 = u.[si  , vi+1]
        let u10 = u.[si+1, vi-1]
        let u11 = u.[si+1, vi  ]
        let u12 = u.[si+1, vi+1]
        let u20 = u.[si+2, vi-1]
        let u21 = u.[si+2, vi  ]
        let u22 = u.[si+2, vi+1]

        { si = si; vi = vi
        
          ds0 = gamma0  ds si; ds1 = gammaP1 ds si; ds2 = gammaP2 ds si;
          dv0 = betaM dv vi; dv1 = beta0 dv vi; dv2 = betaP dv vi;

          u00 = u00; u01 = u01; u02 = u02; u10 = u10; u11 = u11; u12 = u12; u20 = u20; u21 = u21; u22 = u22

          dds0 = 0.0; dds1 = 0.0; dds2 = 0.0
          ddv0 = deltaM dv vi; ddv1 = delta0 dv vi; ddv2 = deltaP dv vi

          vs = 0.0; vv = 0.0; vu = u01
          u0 = 0.0; u1 = 0.0; u2 = 0.0

          us0 = u01; us1 = u11; us2 = u21
          uv0 = u00; uv1 = u01; uv2 = u02
        }

    else if si = ds.n - 1 then // 6 face

        let u00 = u.[si-2, vi-1]
        let u01 = u.[si-2, vi  ]
        let u02 = u.[si-2, vi+1]
        let u10 = u.[si-1, vi-1]
        let u11 = u.[si-1, vi  ]
        let u12 = u.[si-1, vi+1]
        let u20 = u.[si  , vi-1]
        let u21 = u.[si  , vi  ]
        let u22 = u.[si  , vi+1]

        { si = si; vi = vi

          ds0 = alphaM2 ds si; ds1 = alphaM1 ds si; ds2 = alpha0 ds si
          dv0 = betaM dv vi; dv1 = beta0 dv vi; dv2 = betaP dv vi

          u00 = u00; u01 = u01; u02 = u02; u10 = u10; u11 = u11; u12 = u12; u20 = u20; u21 = u21; u22 = u22
 
          dds0 = 0.0; dds1 = 0.0; dds2 = 0.0
          ddv0 = deltaM dv vi; ddv1 = delta0 dv vi; ddv2 = deltaP dv vi

          vs = 0.0; vv = 0.0; vu = u21
          u0 = 0.0; u1 = 0.0; u2 = 0.0

          us0 = u01; us1 = u11; us2 = u21
          uv0 = u20; uv1 = u21; uv2 = u22
        }
    
    else if vi = dv.n - 1 then // 7 face

        let u00 = u.[si-1, vi-2]
        let u01 = u.[si-1, vi-1]
        let u02 = u.[si-1, vi  ]
        let u10 = u.[si  , vi-2]
        let u11 = u.[si  , vi-1]
        let u12 = u.[si  , vi  ]
        let u20 = u.[si+1, vi-2]
        let u21 = u.[si+1, vi-1]
        let u22 = u.[si+1, vi  ]

        { si = si; vi = vi

          ds0 = betaM ds si; ds1 = beta0 ds si; ds2 = betaP ds si
          dv0 = alphaM2 dv vi; dv1 = alphaM1 dv vi; dv2 = alpha0  dv vi

          u00 = u00; u01 = u01; u02 = u02; u10 = u10; u11 = u11; u12 = u12; u20 = u20; u21 = u21; u22 = u22

          dds0 = deltaM ds si; dds1 = delta0 ds si; dds2 = deltaP ds si
          ddv0 = 0.0; ddv1 = 0.0; ddv2 = 0.0

          vs = 0.0; vv = 0.0; vu = u12
          u0 = 0.0; u1 = 0.0; u2 = 0.0

          us0 = u02; us1 = u12; us2 = u22
          uv0 = u10; uv1 = u11; uv2 = u12
        }
    
    else if vi = 0 then // 8 face

        let u00 = u.[si-1, vi  ]
        let u01 = u.[si-1, vi+1]
        let u02 = u.[si-1, vi+2]
        let u10 = u.[si  , vi  ]
        let u11 = u.[si  , vi+1]
        let u12 = u.[si  , vi+2]
        let u20 = u.[si+1, vi  ]
        let u21 = u.[si+1, vi+1]
        let u22 = u.[si+1, vi+2]

        { si = si; vi = vi

          ds0 = betaM ds si; ds1 = beta0 ds si; ds2 = betaP ds si
          dv0 = gamma0  dv vi; dv1 = gammaP1 dv vi; dv2 = gammaP2 dv vi

          u00 = u00; u01 = u01; u02 = u02; u10 = u10; u11 = u11; u12 = u12; u20 = u20; u21 = u21; u22 = u22

          dds0 = deltaM ds si; dds1 = delta0 ds si; dds2 = deltaP ds si
          ddv0 = 0.0; ddv1 = 0.0; ddv2 = 0.0

          vs = 0.0; vv = 0.0; vu = u10
          u0 = 0.0; u1 = 0.0; u2 = 0.0

          us0 = u00; us1 = u10; us2 = u20
          uv0 = u10; uv1 = u11; uv2 = u12
        }
    
    else // 9 inner

        let u00 = u.[si-1, vi-1]
        let u01 = u.[si-1, vi  ]
        let u02 = u.[si-1, vi+1]
        let u10 = u.[si  , vi-1]
        let u11 = u.[si  , vi  ]
        let u12 = u.[si  , vi+1]
        let u20 = u.[si+1, vi-1]
        let u21 = u.[si+1, vi  ]
        let u22 = u.[si+1, vi+1]

        { si = si; vi = vi

          ds0 = betaM ds si; ds1 = beta0 ds si; ds2 = betaP ds si
          dv0 = betaM dv vi; dv1 = beta0 dv vi; dv2 = betaP dv vi

          u00 = u00; u01 = u01; u02 = u02; u10 = u10; u11 = u11; u12 = u12; u20 = u20; u21 = u21; u22 = u22


          dds0 = deltaM ds si; dds1 = delta0 ds si; dds2 = deltaP ds si
          ddv0 = deltaM dv vi; ddv1 = delta0 dv vi; ddv2 = deltaP dv vi

          vs = 0.0; vv = 0.0; vu = u11
          u0 = 0.0; u1 = 0.0; u2 = 0.0

          us0 = u01; us1 = u11; us2 = u21
          uv0 = u10; uv1 = u11; uv2 = u12
        }

// TODO C++ version sets stencil.u0
let [<ReflectedDefinition>] applyF0 t (stencil:Stencil) (heston:HestonModel) =    
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
            u0 * (heston.rho t) * (heston.sigma t) * stencil.vs * stencil.vv
    u0

// TODO C++ version sets u1 in stencil
let [<ReflectedDefinition>] applyF1 t (ds:Differences) (stencil:Stencil) (heston:HestonModel) = 
    let u1 = 
        if stencil.vi = 0 then
            let v2 = ((heston.rd t) - (heston.rf t)) * stencil.vs

            if stencil.si = 0 then
                (v2 * stencil.ds0 - 0.5 * heston.rd t) * stencil.us0
                + (v2 * stencil.ds1) * stencil.us1
                + (v2 * stencil.ds2) * stencil.us2
                    
            else if stencil.si = ds.n - 1 then
                (v2 * stencil.ds0) * stencil.us0
                + (v2 * stencil.ds1) * stencil.us1
                + (v2 * stencil.ds2 - 0.5 * heston.rd t) * stencil.us2
                    
            else
                (v2 * stencil.ds0) * stencil.us0
                + (v2 * stencil.ds1 - 0.5 * heston.rd t) * stencil.us1
                + (v2 * stencil.ds2) * stencil.us2
                    
        else
            let v1 = 0.5 * stencil.vv * stencil.vs * stencil.vs
            let v2 = ((heston.rd t) - (heston.rf t)) * stencil.vs

            if stencil.si = 0 then
                let ds0 = delta ds 1 
                let ds1 = delta ds 2

                (v1 * 2.0 / (ds0 * (ds0 + ds1)) + v2 * stencil.ds0 - 0.5 * heston.rd t) * stencil.us0
                + (-v1 * 2.0 / (ds0 * ds1) + v2 * stencil.ds1) * stencil.us1
                + (v1 * 2.0 / (ds1 * (ds0 + ds1)) + v2 * stencil.ds2) * stencil.us2
                    
            else if stencil.si = ds.n - 1 then
                let ds1 = delta ds (ds.n - 2)
                let ds0 = delta ds (ds.n - 1)

                (v1 * 2.0 / (ds1 * (ds0 + ds1)) + v2 * stencil.ds0) * stencil.us0
                + (-v1 * 2.0 / (ds0 * ds1) + v2 * stencil.ds1) * stencil.us1
                + (v1 * 2.0 / (ds0 * (ds0 + ds1)) + v2 * stencil.ds2 - 0.5 * heston.rd t) * stencil.us2

            else
                (v1 * stencil.dds0 + v2 * stencil.ds0) * stencil.us0
                + (v1 * stencil.dds1 + v2 * stencil.ds1 - 0.5 * heston.rd t) * stencil.us1
                + (v1 * stencil.dds2 + v2 * stencil.ds2) * stencil.us2
    u1                    

// TODO C++ version sets u2 in stencil
let [<ReflectedDefinition>] applyF2 t (dv:Differences) (stencil:Stencil) (heston:HestonModel) = 
    let u2 = 
        let v1 = 0.5 * (heston.sigma t)* (heston.sigma t) * stencil.vv
        let v2 = (heston.kappa t) * ((heston.eta t)- stencil.vv)

        if stencil.vi = 0 then
            let dv0 = delta dv 1
            let dv1 = delta dv 2

            (v1 * 2.0 / (dv0 * (dv0 + dv1)) + v2 * stencil.dv0 - 0.5 * heston.rd t) * stencil.uv0
            + (-v1 * 2.0 / (dv0 * dv1) + v2 * stencil.dv1) * stencil.uv1
            + (v1 * 2.0 / (dv1 * (dv0 + dv1)) + v2 * stencil.dv2) * stencil.uv2
        
        else if stencil.vi = dv.n - 1 then
            let dv1 = delta dv (dv.n - 2)
            let dv0 = delta dv (dv.n - 1)

            (v1 * 2.0 / (dv1 * (dv0 + dv1)) + v2 * stencil.dv0) * stencil.uv0
            + (-v1 * 2.0 / (dv0 * dv1) + v2 * stencil.dv1) * stencil.uv1
            + (v1 * 2.0 / (dv0 * (dv0 + dv1)) + v2 * stencil.dv2 - 0.5 * heston.rd t) * stencil.uv2
              
        else
            (v1 * stencil.ddv0 + v2 * stencil.dv0) * stencil.uv0
            + (v1 * stencil.ddv1 + v2 * stencil.dv1 - 0.5 * heston.rd t) * stencil.uv1
            + (v1 * stencil.ddv2 + v2 * stencil.dv2) * stencil.uv2
    u2

[<ReflectedDefinition>]
let applyF ns nv t (ds:Differences) (dv:Differences) (u:float[,]) (func:int -> int -> float -> float -> float -> float -> unit) (heston:HestonModel) =
    let start = blockIdx.x * blockDim.x + threadIdx.x
    let stride = gridDim.x * blockDim.x
    let mutable si = blockIdx.x * blockDim.x + threadIdx.x

    while si < ns do

        let mutable vi = blockIdx.y * blockDim.y + threadIdx.y

        while vi < nv do
            let s = stencil si vi ds dv u
            let u0 = applyF0 t s heston
            let u1 = applyF1 t ds s heston
            let u2 = applyF2 t dv s heston
            func si vi s.vu u0 u1 u2

            vi <- vi + blockDim.y * gridDim.y

        si <- si + blockDim.x * gridDim.x


[<ReflectedDefinition>]
let solveF1 t (ds:Differences) (dv:Differences) (b:float[,]) (func:int -> int -> float -> float -> unit) (tk1:float) (thetaDt:float) (heston:HestonModel) =

    let shared = __extern_shared__<float>()
    let h = shared
    let d = h + ds.n
    let l = d + ds.n
    let u = l + ds.n

    let mutable vi = blockIdx.x
    while vi < dv.n do
        
        let vv = dv.x.[vi]

        let mutable si = threadIdx.x
        while si < ds.n do
       
            let vs = ds.x.[si]

            if si = 0 then
                l.[si] <- 0.0
                d.[si] <- 1.0
                u.[si] <- 0.0
                h.[si] <- 0.0                           
            else if si = ds.n - 1 then
                l.[si] <- -1.0 / (delta ds si)
                d.[si] <- 1.0 / (delta ds si)
                u.[si] <- 0.0
                h.[si] <- exp(-tk1 * (heston.rf t))
            else
                if vv > 0.0 then
                    let v1 = 0.5 * vv * vs * vs
                    let v2 = (heston.rd t - heston.rf t) * vs

                    let deltaSM = deltaM ds si
                    let deltaS0 = delta0 ds si
                    let deltaSP = deltaP ds si

                    let betaSM = betaM ds si
                    let betaS0 = beta0 ds si
                    let betaSP = betaP ds si

                    l.[si] <- -(v1 * deltaSM + v2 * betaSM) * thetaDt
                    d.[si] <- 1.0 - (v1 * deltaS0 + v2 * betaS0 - 0.5 * heston.rd t) * thetaDt
                    u.[si] <- -(v1 * deltaSP + v2 * betaSP) * thetaDt
                else
                    let v = (heston.rd t - heston.rf t) * vs / (delta ds si)
                    l.[si] <- -v * thetaDt
                    d.[si] <- 1.0 - (-v - 0.5 * heston.rd t) * thetaDt
                    u.[si] <- 0.0

                h.[si] <- b.[si, vi]

            si <- si + blockDim.x

        __syncthreads()

        triDiagPcrSingleBlock ds.n l d u h

        si <- threadIdx.x
        while si < ds.n do       
            func si vi h.[si] thetaDt
            si <- si + blockDim.x

        __syncthreads()

        vi <- vi + gridDim.x
   
[<ReflectedDefinition>]
let solveF2 t (ds:Differences) (dv:Differences) (b:float[,]) (func:int -> int -> float -> unit) (tk1:float) (thetaDt:float) (heston:HestonModel) =

    let shared = __extern_shared__<float>()
    let h = shared
    let d = h + ds.n
    let l = d + ds.n
    let u = l + ds.n

    let mutable si = blockIdx.x
    while si < ds.n - 1 do

        let vs = ds.x.[si]

        let mutable vi = threadIdx.x
        while vi < dv.n do

            let vv = dv.x.[vi]

            if si = 0 then
                h.[vi] <- 0.0
            else
                if vi = 0 then
                    l.[vi] <- 0.0
                    d.[vi] <- 1.0
                    u.[vi] <- 0.0
                    h.[vi] <- b.[si, vi]
                else if vi = dv.n - 1 then
                    l.[vi] <- 0.0
                    d.[vi] <- 1.0
                    u.[vi] <- 0.0
                    h.[vi] <- vs * exp(-tk1 * heston.rf t)
                else
                    let deltaVM = deltaM dv vi
                    let deltaV0 = delta0 dv vi
                    let deltaVP = deltaP dv vi

                    let betaVM = betaM dv vi
                    let betaV0 = beta0 dv vi
                    let betaVP = betaP dv vi

                    let v1 = 0.5 * (heston.sigma t)* (heston.sigma t)* vv
                    let v2 = (heston.kappa t) * ((heston.eta t)- vv)

                    l.[vi] <- -thetaDt * v1 * deltaVM - thetaDt * v2 * betaVM
                    d.[vi] <- 1.0 - thetaDt * (v1 * deltaV0 + v2 * betaV0 - 0.5 * heston.rd t)
                    u.[vi] <- -thetaDt * (v1 * deltaVP + v2 * betaVP)
                    h.[vi] <- b.[si, vi]

            vi <- vi + blockDim.x

        __syncthreads()

        if si <> 0 then
            triDiagPcrSingleBlock ds.n l d u h
        
        vi <- threadIdx.x    
        while vi < dv.n do  
            func si vi h.[vi]
            vi <- vi + blockDim.x

        si <- si + gridDim.x

