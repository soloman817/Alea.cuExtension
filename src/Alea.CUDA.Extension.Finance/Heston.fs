﻿module Alea.CUDA.Extension.Finance.Heston

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Alea.CUDA.Extension
open Alea.CUDA.Extension.TriDiag
open Alea.CUDA.Extension.Finance.Grid

open Util 

// shorthands 
let [<ReflectedDefinition>] get (u:RMatrixRowMajor ref) (si:int) (vi:int) =
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

[<Struct>]
type A =
    // row of A2
    val al : float
    val ad : float
    val au : float
    // first derivative
    val v1 : float
    val v2 : float
    val v3 : float
    // second derivative
    val w1 : float
    val w2 : float
    val w3 : float

    [<ReflectedDefinition>]
    new (al:float, ad:float, au:float, v1:float, v2:float, v3:float, w1:float, w2:float, w3:float) =
        { al = al; ad = ad; au = au; v1 = v1; v2 = v2; v3 = v3; w1 = w1; w2 = w2; w3 = w3 } 

    [<ReflectedDefinition>]
    static member apply (a:A) um u0 up = a.al*um + a.ad*u0 + a.au*up

/// Operator A1 i-th row  
let [<ReflectedDefinition>] a1Operator (j:int) (ns:int) (heston:HestonModel) sj vi ds dsp =   
    let fd = centralWeights ds dsp  
                  
    if j < ns-1 then
        
        // Dirichlet boundary at s = 0 for i = 1
        let al = if j = 1 then 0.0 else 0.5*sj*sj*vi*fd.w1 + (heston.rd-heston.rf)*sj*fd.v1          
        let ad = 0.5*sj*sj*vi*fd.w2 + (heston.rd-heston.rf)*sj*fd.v2 - 0.5*heston.rd
        let au = 0.5*sj*sj*vi*fd.w3 + (heston.rd-heston.rf)*sj*fd.v3 

        A(al, ad, au, fd.v1, fd.v2, fd.v3, fd.w1, fd.w2, fd.w3)
    else
        // Neumann boundary with additonal ghost point and extrapolation
        let al = 0.5*sj*sj*vi*fd.w1 + (heston.rd-heston.rf)*sj*fd.v1
        let ad = 0.5*sj*sj*vi*(fd.w2+fd.w3) + (heston.rd-heston.rf)*sj*(fd.v2+fd.v3) - 0.5*heston.rd
        let au = 0.0
    
        A(al, ad, au, fd.v1, fd.v2, fd.v3, fd.w1, fd.w2, fd.w3)

/// Operator A2, j-th row for 0 < j < nv-1 so we can always build central weights for v  
/// Note that by assumption iMin = 1 
let [<ReflectedDefinition>] a2Operator (i:int) (nv:int) (heston:HestonModel) vi dv dvp =   
    // i = iMin, special boundary at v = 0, one sided forward difference quotient 
    if i = 1 then
        let fd = forwardWeightsSimple dvp         

        let al = 0.0
        let ad = heston.kappa*heston.eta*fd.v1 - 0.5*heston.rd
        let au = heston.kappa*heston.eta*fd.v2   
                
        A(al, ad, au, fd.v1, fd.v2, fd.v3, fd.w1, fd.w2, fd.w3)
    else               
        let fd = centralWeights dv dvp

        if i < nv-1 then
            let al = 0.5*heston.sigma*vi*fd.w1 + heston.kappa*(heston.eta - vi)*fd.v1
            let ad = 0.5*heston.sigma*vi*fd.w2 + heston.kappa*(heston.eta - vi)*fd.v2 - 0.5*heston.rd
            let au = 0.5*heston.sigma*vi*fd.w3 + heston.kappa*(heston.eta - vi)*fd.v3

            A(al, ad, au, fd.v1, fd.v2, fd.v3, fd.w1, fd.w2, fd.w3)
        else
            // Dirichlet boundary at v = max, the constant term 
            //   (0.5*sigma*v(j)*wp2 + kappa*(eta - v(j))*wp1)*s(i)*exp(-t*rf)
            // is absorbed into b2
            let al = 0.5*heston.sigma*vi*fd.w1 + heston.kappa*(heston.eta - vi)*fd.v1
            let ad = 0.5*heston.sigma*vi*fd.w2 + heston.kappa*(heston.eta - vi)*fd.v2 - 0.5*heston.rd
            let au = 0.0

            A(al, ad, au, fd.v1, fd.v2, fd.v3, fd.w1, fd.w2, fd.w3)               

let [<ReflectedDefinition>] b1Value j jMax (heston:HestonModel) sj vi ds =   
    if j = jMax then 
        // wp = 1.0/(ds*ds), d2ud2s = wp*ds = 1.0/ds, wp = ds / (2.0*ds*ds) duds = wp*ds = 0.5
        0.5*sj*sj*vi/ds + (heston.rd-heston.rf)*sj*0.5 
    else 
        0.0

let [<ReflectedDefinition>] b2Value i iMax (heston:HestonModel) t sj vi w3 v3 =               
    if i = iMax then
        (0.5*heston.sigma*vi*w3 + heston.kappa*(heston.eta - vi)*v3)*sj*exp(-t*heston.rf)
    else
        0.0

/// Explicit apply operator for Euler scheme using optimized loads through shared memory.
/// We introduce ghost points at v = 0 (i = 0) and s = smax (j = ns) in order to simplify
/// the memory access pattern so that no additional checks at the boundary are required.
/// The grids have to be extended properly as well.
///                                 
///      jMin = 1                   jMax = ns - 1
///      |                          |
///      v                          v
///                               
///     0                            ns
///     ---------------------------------->   S (j, x, columns)
///  0  |OOOOOOOOOOOOOOOOOOOOOOOOOOOOO
///     |*...........................O  <--- iMin = 1 
///     |*...........................O    
///     |*...........................O    
///     |*...........................O    
///     |*...........................O    
///     |*...........................O    
///     |*...........................O  <--- iMax = nv - 1
///  nv |****************************O 
///     |
///     v
///
///     V (i, y, rows)                            
///
/// Total dimension of solution matrix (nv+1) x (ns+1)  
/// Meaning of points:
///     - O     ghost points added to unify memory reading without bound checks, must remain zero 
///             will be removed by copying to a reduced matrix
///     - *     Dirichelt boundary points, which are not going into equation and are fixed
///     - .     Innter points, which are effectively updated by the solver
///
let [<ReflectedDefinition>] applyF (heston:HestonModel) t (dt:float) (u:RMatrixRowMajor ref) (s:DevicePtr<float>) (v:DevicePtr<float>) ns nv =
    let tx = threadIdx.x
    let ty = threadIdx.y
    let bx = blockDim.x
    let by = blockDim.y
    let gx = gridDim.x
    let gy = gridDim.y
 
    let uPtr = u.contents.Storage
    let uShared = __extern_shared__<float>()
    
    // for coalesicing access we need to map x to columns and y to rows of the matrix because threads are aligned that way
    // i, k / y -> row -> v
    // j, l / x -> col -> s

    let jMin = 1
    let jMax = ns - 1 
    let iMin = 1
    let iMax = nv - 1

    let nRows = nv + 1
    let nCols = ns + 1

    let mutable k = blockIdx.y * by + ty 
                                                
    while k <= gy*by do

        let mutable l = blockIdx.x * bx + tx

        while l <= gx*bx do 

            // find out the tile in which we are working because the grid may not cover all of the matrix u
            let ktile = k / by    
            let ltile = l / bx    
            let i0 = ktile * by
            let j0 = ltile * bx
            let I0 = i0 * nCols + j0
 
            // use all threads of block to load bx*by elements of u 
            let offset = ty*bx + tx
            let I = offset / (bx + 2)  // row
            let J = offset % (bx + 2)  // col

            if i0 + I < nRows && j0 + J < nCols then
                uShared.[I*(bx+2) + J] <- uPtr.[I0 + I*nCols + J]

            // second round to load remaining (bx+2)*(by+2) - bx*by elements of u
            // note that some threads do not need to load data anymore
            let offset = bx*by + ty*bx + tx
            let I = offset / (bx + 2)
            let J = offset % (bx + 2)

            if offset < (bx + 2)*(by + 2) && i0 + I < nRows && j0 + J < nCols then
                uShared.[I*(bx+2) + J] <- uPtr.[I0 + I*nCols + J]

            __syncthreads()

            // we added a ghost points at j = 0, so we can start at j = 1 and read from u the same way for each thread
            let i = k + 1
            let j = l + 1

            // Dirichlet boundary at s = 0, so we do not need to process 0, we start 1
            // Dirichlet boundary at v = max, so we do not need to process nv
            if i <= iMax && j <= jMax then

                let vi = v.[i]       
                let a2op = a2Operator i nv heston vi (vi - v.[i-1]) (v.[i+1] - vi)

                // relative addressing in the shared memory tile
                let ir = i - i0  
                let jr = j - j0 

                // we add a ghost points of zero value around u to have no memory access issues
                let umm = uShared.[(ir-1)*(bx+2) + (jr-1)]
                let u0m = uShared.[ ir   *(bx+2) + (jr-1)]
                let upm = uShared.[(ir+1)*(bx+2) + (jr-1)]
                let um0 = uShared.[(ir-1)*(bx+2) +  jr   ]
                let u00 = uShared.[ ir   *(bx+2) +  jr   ]
                let up0 = uShared.[(ir+1)*(bx+2) +  jr   ]
                let ump = uShared.[(ir-1)*(bx+2) + (jr+1)]
                let u0p = uShared.[ ir   *(bx+2) + (jr+1)]
                let upp = uShared.[(ir+1)*(bx+2) + (jr+1)]

                let sj = s.[j]
                let ds = sj - s.[j-1]

                let b1 = b1Value j jMax heston sj vi ds
                let b2 = b2Value i iMax heston t sj vi a2op.w3 a2op.v3
                
                let a1op = a1Operator j ns heston sj vi ds (s.[j+1] - sj)
                    
                // a0 <> 0 only on iMin < i <= iMax && jMin < j < jMax
                let mixed = 
                    // we do not need to test for i > iMin because at iMin vi = 0 hend a0 becomes zero there too
                    if j < jMax then                     
                        a1op.v1*a2op.v1*umm + a1op.v1*a2op.v2*u0m + a1op.v1*a2op.v3*upm +
                        a1op.v2*a2op.v1*um0 + a1op.v2*a2op.v2*u00 + a1op.v2*a2op.v3*up0 + 
                        a1op.v3*a2op.v1*ump + a1op.v3*a2op.v2*u0p + a1op.v3*a2op.v3*upp
                    else
                        0.0   
            
                let a0 = heston.rho*heston.sigma*sj*vi*mixed                    
                let a1 = A.apply a1op u0m u00 u0p                    
                let a2 = A.apply a2op um0 u00 up0         
            
                // set u for i = 1,...,iMax, j = 1,...,jMax
                set u i j (u00 + dt*(a0+a1+a2+b1+b2))

                __syncthreads()
               
            l <- l + bx * gridDim.x   

        k <- k + by * gridDim.y

/// Explicit apply operator for Euler scheme with all data loaded directly from global device memory.
let [<ReflectedDefinition>] applyFDevMemory (heston:HestonModel) t (dt:float) (u:RMatrixRowMajor ref) (s:DevicePtr<float>) (v:DevicePtr<float>) ns nv =
    let tx = threadIdx.x
    let ty = threadIdx.y
    let bx = blockDim.x
    let by = blockDim.y
    let gx = gridDim.x
    let gy = gridDim.y
     
    // for coalesicing access we need to map x to columns and y to rows of the matrix because threads are aligned that way
    // i, k / y -> row -> v
    // j, l / x -> col -> s

    let jMin = 1
    let jMax = ns - 1 
    let iMin = 1
    let iMax = nv - 1

    let nRows = nv + 1
    let nCols = ns + 1

    let mutable k = blockIdx.y * by + ty 
                                                
    while k <= gy*by do

        let mutable l = blockIdx.x * bx + tx

        while l <= gx*bx do 

            // we added a ghost points at j = 0, so we can start at j = 1 and read from u the same way for each thread
            let i = k + 1
            let j = l + 1

            // Dirichlet boundary at s = 0, so we do not need to process 0, we start 1
            // Dirichlet boundary at v = max, so we do not need to process nv
            if i <= iMax && j <= jMax then

                let vi = v.[i]       
                let a2op = a2Operator i nv heston vi (vi - v.[i-1]) (v.[i+1] - vi)

                // we add a ghost points of zero value around u to have no memory access issues
                let umm = get u (i-1) (j-1)
                let u0m = get u  i    (j-1)
                let upm = get u (i+1) (j-1)
                let um0 = get u (i-1)  j 
                let u00 = get u  i     j
                let up0 = get u (i+1)  j
                let ump = get u (i-1) (j+1)
                let u0p = get u  i    (j+1)
                let upp = get u (i+1) (j+1)

                let sj = s.[j]
                let ds = sj - s.[j-1]

                let b1 = b1Value j jMax heston sj vi ds
                let b2 = b2Value i iMax heston t sj vi a2op.w3 a2op.v3
                
                let a1op = a1Operator j ns heston sj vi ds (s.[j+1] - sj)
                    
                // a0 <> 0 only on iMin < i <= iMax && jMin < j < jMax
                let mixed = 
                    // we do not need to test for i > iMin because at iMin vi = 0 hend a0 becomes zero there too
                    if j < jMax then                     
                        a1op.v1*a2op.v1*umm + a1op.v1*a2op.v2*u0m + a1op.v1*a2op.v3*upm +
                        a1op.v2*a2op.v1*um0 + a1op.v2*a2op.v2*u00 + a1op.v2*a2op.v3*up0 + 
                        a1op.v3*a2op.v1*ump + a1op.v3*a2op.v2*u0p + a1op.v3*a2op.v3*upp
                    else
                        0.0   
            
                let a0 = heston.rho*heston.sigma*sj*vi*mixed                    
                let a1 = A.apply a1op u0m u00 u0p                    
                let a2 = A.apply a2op um0 u00 up0         
            
                // set u for i = 1,...,iMax, j = 1,...,jMax
                set u i j (u00 + dt*(a0+a1+a2+b1+b2))
               
            l <- l + bx * gridDim.x   

        k <- k + by * gridDim.y

/// Explicit part for Douglas and other schemes which require intermediate results.
let [<ReflectedDefinition>] applyFDouglas (heston:HestonModel) t tnext (dt:float) (theta:float) (u:RMatrixRowMajor ref) (s:DevicePtr<float>) (v:DevicePtr<float>) ns nv (y0:RMatrixRowMajor ref) (y1corr:RMatrixRowMajor ref) =
    let tx = threadIdx.x
    let ty = threadIdx.y
    let bx = blockDim.x
    let by = blockDim.y
    let gx = gridDim.x
    let gy = gridDim.y
 
    let uPtr = u.contents.Storage
    let uShared = __extern_shared__<float>()
    
    // for coalesicing access we need to map x to columns and y to rows of the matrix because threads are aligned that way
    // i, k / y -> row -> v
    // j, l / x -> col -> s

    let jMin = 1
    let jMax = ns - 1 
    let iMin = 1
    let iMax = nv - 1

    let nRows = nv + 1
    let nCols = ns + 1

    let mutable k = blockIdx.y * by + ty 
                                                
    while k <= gy*by do

        let mutable l = blockIdx.x * bx + tx

        while l <= gx*bx do 

            // find out the tile in which we are working because the grid may not cover all of the matrix u
            let ktile = k / by    
            let ltile = l / bx    
            let i0 = ktile * by
            let j0 = ltile * bx
            let I0 = i0 * nCols + j0
 
            // use all threads of block to load bx*by elements of u 
            let offset = ty*bx + tx
            let I = offset / (bx + 2)  // row
            let J = offset % (bx + 2)  // col

            if i0 + I < nRows && j0 + J < nCols then
                uShared.[I*(bx+2) + J] <- uPtr.[I0 + I*nCols + J]

            // second round to load remaining (bx+2)*(by+2) - bx*by elements of u
            // note that some threads do not need to load data anymore
            let offset = bx*by + ty*bx + tx
            let I = offset / (bx + 2)
            let J = offset % (bx + 2)

            if offset < (bx + 2)*(by + 2) && i0 + I < nRows && j0 + J < nCols then
                uShared.[I*(bx+2) + J] <- uPtr.[I0 + I*nCols + J]

            __syncthreads()

            // we added a ghost points at j = 0, so we can start at j = 1 and read from u the same way for each thread
            let i = k + 1
            let j = l + 1

            // Dirichlet boundary at s = 0, so we do not need to process 0, we start 1
            // Dirichlet boundary at v = max, so we do not need to process nv
            if i <= iMax && j <= jMax then

                let vi = v.[i]       
                let a2op = a2Operator i nv heston vi (vi - v.[i-1]) (v.[i+1] - vi)

                // relative addressing in the shared memory tile
                let ir = i - i0  
                let jr = j - j0 

                // we add a ghost points of zero value around u to have no memory access issues
                let umm = uShared.[(ir-1)*(bx+2) + (jr-1)]
                let u0m = uShared.[ ir   *(bx+2) + (jr-1)]
                let upm = uShared.[(ir+1)*(bx+2) + (jr-1)]
                let um0 = uShared.[(ir-1)*(bx+2) +  jr   ]
                let u00 = uShared.[ ir   *(bx+2) +  jr   ]
                let up0 = uShared.[(ir+1)*(bx+2) +  jr   ]
                let ump = uShared.[(ir-1)*(bx+2) + (jr+1)]
                let u0p = uShared.[ ir   *(bx+2) + (jr+1)]
                let upp = uShared.[(ir+1)*(bx+2) + (jr+1)]

                let sj = s.[j]
                let ds = sj - s.[j-1]

                let b1 = b1Value j jMax heston sj vi ds
                let b2 = b2Value i iMax heston t sj vi a2op.w3 a2op.v3
                let b2next = b2Value i iMax heston tnext sj vi a2op.w3 a2op.v3
                
                let a1op = a1Operator j ns heston sj vi ds (s.[j+1] - sj)
                    
                // a0 <> 0 only on iMin < i <= iMax && jMin < j < jMax
                let mixed = 
                    // we do not need to test for i > iMin because at iMin vi = 0 hend a0 becomes zero there too
                    if j < jMax then                     
                        a1op.v1*a2op.v1*umm + a1op.v1*a2op.v2*u0m + a1op.v1*a2op.v3*upm +
                        a1op.v2*a2op.v1*um0 + a1op.v2*a2op.v2*u00 + a1op.v2*a2op.v3*up0 + 
                        a1op.v3*a2op.v1*ump + a1op.v3*a2op.v2*u0p + a1op.v3*a2op.v3*upp
                    else
                        0.0   
            
                let a0 = heston.rho*heston.sigma*sj*vi*mixed                    
                let a1 = A.apply a1op u0m u00 u0p                    
                let a2 = A.apply a2op um0 u00 up0         
            
                // set u for i = 1,...,iMax, j = 1,...,jMax
                set y0 i j (u00 + dt*(a0+a2+b1+b2) + (1.0 - theta)*dt*a1)
                set y1corr i j (theta*dt*(a2 + b2 - b2next))                   
                 
            __syncthreads()
               
            l <- l + bx * gridDim.x   

        k <- k + by * gridDim.y

/// Sweep over the volatility grid and solve for each volatility value a system of ns-1 variables for the inner spot points.
/// The kernel requires 4*(ns-1)*sizeof<float> shared memory. 
[<ReflectedDefinition>]
let solveF1 (heston:HestonModel) (t:float) (thetaDt:float) (rhs:int -> int -> float) (s:DevicePtr<float>) (v:DevicePtr<float>) ns nv (result:RMatrixRowMajor ref) =
    let rdt = heston.rd 
    let rft = heston.rf 

    let shared = __extern_shared__<float>()
    let h = shared
    let d = h + (ns-1)
    let l = d + (ns-1)
    let u = l + (ns-1)

    let mutable i = blockIdx.x + 1
    while i < nv do
        
        let vi = v.[i]

        let mutable j = threadIdx.x + 1
        while j < ns do
       
            let sj = s.[j]
            let a1op = a1Operator j ns heston sj vi (sj - s.[j-1]) (s.[j+1] - sj)

            l.[j-1] <- -thetaDt*a1op.al
            d.[j-1] <- 1.0 - thetaDt*a1op.ad
            u.[j-1] <- -thetaDt*a1op.au
            h.[j-1] <- rhs i j

            j <- j + blockDim.x

        __syncthreads()

        triDiagPcrSingleBlock (ns-1) l d u h

        j <- threadIdx.x + 1
        while j < ns do       
            set result i j h.[j-1] 
            j <- j + blockDim.x

        __syncthreads()

        i <- i + gridDim.x

[<ReflectedDefinition>]
let rhsSolveF1 (y0:RMatrixRowMajor ref) i j =
    get y0 i j  

/// Sweep over the spot grid and solve for each spot value a system of nv-1 variables for the inner volatility points.
/// The kernel requires 4*(nv-1)*sizeof<float> shared memory. 
[<ReflectedDefinition>]
let solveF2 (heston:HestonModel) (t:float) (thetaDt:float) (rhs:int -> int -> float) (s:DevicePtr<float>) (v:DevicePtr<float>) ns nv (result:RMatrixRowMajor ref) =
    let rdt = heston.rd 
    let rft = heston.rf 
    let sigmat = heston.sigma 
    let kappat = heston.kappa 
    let etat = heston.eta 

    let shared = __extern_shared__<float>()
    let h = shared
    let d = h + (nv-1)
    let l = d + (nv-1)
    let u = l + (nv-1)

    let mutable j = blockIdx.x + 1
    while j < ns do

        let sj = s.[j]

        let mutable i = threadIdx.x + 1
        while i < nv do

            let vi = v.[i]
            let a2op = a2Operator i nv heston vi (vi - v.[i-1]) (v.[i+1] - vi)

            l.[i-1] <- -thetaDt*a2op.al
            d.[i-1] <- 1.0 - thetaDt*a2op.ad
            u.[i-1] <- -thetaDt*a2op.au
            h.[i-1] <- rhs i j
                        
            i <- i + blockDim.x

        __syncthreads()

        triDiagPcrSingleBlock (nv-1) l d u h

        i <- threadIdx.x + 1   
        while i < nv do  
            set result i j h.[i-1]
            i <- i + blockDim.x

        j <- j + gridDim.x

[<ReflectedDefinition>]
let rhsSolveF2 (y1:RMatrixRowMajor ref) (y:RMatrixRowMajor ref) i j =
    (get y1 i j) - (get y i j)  

type OptionType =
| Call
| Put
    member this.sign =
        match this with
        | Call -> 1.0
        | Put -> -1.0

/// Initial condition for vanilla call put option.
/// We add artifical zeros to avoid access violation in the kernel. See the documentation above.
let [<ReflectedDefinition>] initConditionVanilla ns nv (s:DevicePtr<float>) (u:RMatrixRowMajor) optionType strike =    
    let i = blockIdx.y*blockDim.y + threadIdx.y
    let j = blockIdx.x*blockDim.x + threadIdx.x
    if i < nv + 1 && j < ns + 1 then
        let payoff = 
            // these are ghost points where we set the value to 0 and which the kernels must not change
            if j = ns || i = 0 then 
                0.0
            else
                match optionType with
                |  1.0 -> max (s.[j] - strike) 0.0
                | -1.0 -> max (strike - s.[j]) 0.0 
                | _ -> 0.0
        set (ref u) i j payoff 

/// Boundary condition for vanilla call put option.
let [<ReflectedDefinition>] boundaryConditionVanilla (rf:float) t ns nv (s:DevicePtr<float>) (u:RMatrixRowMajor) =
    let i = blockIdx.x*blockDim.x + threadIdx.x
    if i < ns then
        set (ref u) nv i (exp(-rf*t)*s.[i])
    if i < nv then
        set (ref u) i 0 0.0

/// Set all values to zero.
let [<ReflectedDefinition>] zeros ns nv (u:RMatrixRowMajor) =    
    let i = blockIdx.y*blockDim.y + threadIdx.y
    let j = blockIdx.x*blockDim.x + threadIdx.x
    if i < nv + 1 && j < ns + 1 then
        set (ref u) i j 0.0 

/// Copy kernel to copy data on device to reduce to valid range
let [<ReflectedDefinition>] copyValues ns nv (u0:RMatrixRowMajor) (u1:RMatrixRowMajor) =    
    let j = blockIdx.x*blockDim.x + threadIdx.x
    let i = blockIdx.y*blockDim.y + threadIdx.y
    if i < nv && j < ns then
        set (ref u1) i j (get (ref u0) (i+1) j) 

/// Copy kernel to copy data on device to reduce to valid range
let [<ReflectedDefinition>] copyGrids ns nv (s0:DevicePtr<float>) (v0:DevicePtr<float>) (s1:DevicePtr<float>) (v1:DevicePtr<float>) =    
    let i = blockIdx.x*blockDim.x + threadIdx.x
    if i < ns then
        s1.[i] <- s0.[i]        
    if i < nv+2 then
        v1.[i] <- v0.[i+1]

type HestonEulerSolverParam =
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
   
    let! applyFKernel =
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
        let applyFKernel = applyFKernel.Apply m
        let copyValuesKernel = copyValuesKernel.Apply m
        let copyGridsKernel = copyGridsKernel.Apply m

        fun (heston:HestonModel) (optionType:OptionType) strike timeToMaturity (param:HestonEulerSolverParam) ->

            // we add one more point to the state grids because the value surface has a ghost aerea as well
            let s, ds = concentratedGridBetween param.sMin param.sMax strike param.ns param.sC
            let v, dv = concentratedGridBetween 0.0 param.vMax 0.0 param.nv param.vC

            // extend grids for ghost points
            let sGhost = 2.0*s.[s.Length - 1] - s.[s.Length - 2]
            let s = Array.append s [|sGhost|]
            let vLowerGhost = 2.0*v.[0] - v.[1]
            let v = Array.append [|vLowerGhost|] v

            // calculate a time step which is compatible with the space discretization
            let dt = ds*ds/100.0
            let nt = int(timeToMaturity/dt) + 1
            let t, dt = homogeneousGrid nt 0.0 timeToMaturity

            pcalc {
                let! s = DArray.scatterInBlob worker s
                let! v = DArray.scatterInBlob worker v

                // add zero value ghost points to the value surface to allow simpler access in the kernel
                // one ghost point at s = smax and v = 0
                // no ghost point needed at v = vmax and s = 0 because there we have Dirichlet boundary 
                let ns1 = param.ns+1
                let nv1 = param.nv+1
                let! u = DMatrix.createInBlob<float> worker RowMajorOrder nv1 ns1 

                // storage for reduced values without ghost points
                let! sred = DArray.createInBlob<float> worker param.ns
                let! vred = DArray.createInBlob<float> worker param.nv
                let! ured = DMatrix.createInBlob<float> worker RowMajorOrder param.nv param.ns 
                
                do! PCalc.action (fun hint ->
                    let sharedSize = 10 * 10 * sizeof<float>
                    let lpms = LaunchParam(dim3(divup ns1 8, divup nv1 8), dim3(8, 8), sharedSize) |> hint.ModifyLaunchParam
                    let lpm = LaunchParam(dim3(divup ns1 16, divup nv1 16), dim3(16, 16)) |> hint.ModifyLaunchParam
                    let lpb = LaunchParam(divup (max ns1 nv1) 256, 256) |> hint.ModifyLaunchParam

                    let u = RMatrixRowMajor(u.NumRows, u.NumCols, u.Storage.Ptr)
                    let ured = RMatrixRowMajor(ured.NumRows, ured.NumCols, ured.Storage.Ptr)

                    initCondKernel.Launch lpm param.ns param.nv s.Ptr u optionType.sign strike

                    for ti = 0 to nt-2 do

                        let t0 = t.[ti]
                        let t1 = t.[ti + 1]
                        let dt = t1 - t0

                        boundaryCondKernel.Launch lpb heston.rf t0 param.ns param.nv s.Ptr u
                        applyFKernel.Launch lpms heston t0 dt u s.Ptr v.Ptr param.ns param.nv

                    // copy solution, later we could use a view on it
                    copyValuesKernel.Launch lpm param.ns param.nv u ured
                    copyGridsKernel.Launch lpb param.ns param.nv s.Ptr v.Ptr sred.Ptr vred.Ptr
                )
                
                return sred, vred, ured } ) }

type HestonDouglasSolverParam =
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
  
/// Solve Hesten with Douglas scheme.
let douglasSolver = cuda {

    // we add artifical zeros to avoid access violation in the kernel
    let! initCondKernel =     
        <@ fun ns nv (s:DevicePtr<float>) (u:RMatrixRowMajor) optionType strike ->
            initConditionVanilla ns nv s u optionType strike @> |> defineKernelFuncWithName "initCondition"

    let! boundaryCondKernel =     
        <@ fun (rf:float) t ns nv (s:DevicePtr<float>) (u:RMatrixRowMajor) ->
            boundaryConditionVanilla rf t ns nv s u @> |> defineKernelFuncWithName "boundaryCondition"
   
    let! applyFKernel =
        <@ fun (heston:HestonModel) t tnext (dt:float) (theta:float) (u:RMatrixRowMajor) (s:DevicePtr<float>) (v:DevicePtr<float>) ns nv (y0:RMatrixRowMajor) (y1corr:RMatrixRowMajor) ->
            applyFDouglas heston t tnext dt theta (ref u) s v ns nv (ref y0) (ref y1corr) @> |> defineKernelFuncWithName "applyF"

    let! solveF1Kernel =
        <@ fun (heston:HestonModel) (t:float) (thetaDt:float) (y0:RMatrixRowMajor) (s:DevicePtr<float>) (v:DevicePtr<float>) ns nv (result:RMatrixRowMajor) ->
            solveF1 heston t thetaDt (rhsSolveF1 (ref y0)) s v ns nv (ref result) @> |> defineKernelFuncWithName "solveF1"

    let! solveF2Kernel =
        <@ fun (heston:HestonModel) (t:float) (thetaDt:float) (y1:RMatrixRowMajor) (y1corr:RMatrixRowMajor) (s:DevicePtr<float>) (v:DevicePtr<float>) ns nv (result:RMatrixRowMajor) ->
            solveF2 heston t thetaDt (rhsSolveF2 (ref y1) (ref y1corr)) s v ns nv (ref result) @> |> defineKernelFuncWithName "solveF2"

    let! zerosKernel = 
        <@ fun ns nv (u:RMatrixRowMajor) ->
            zeros ns nv u @> |> defineKernelFuncWithName "zeros"    
    
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
        let appFKernel = applyFKernel.Apply m
        let zerosKernel = zerosKernel.Apply m
        let copyValuesKernel = copyValuesKernel.Apply m
        let copyGridsKernel = copyGridsKernel.Apply m
        let solveF1Kernel = solveF1Kernel.Apply m
        let solveF2Kernel = solveF2Kernel.Apply m

        fun (heston:HestonModel) (optionType:OptionType) strike timeToMaturity (param:HestonDouglasSolverParam) ->

            // we add one more point to the state grids because the value surface has a ghost aerea as well
            let s, ds = concentratedGridBetween param.sMin param.sMax strike param.ns param.sC
            let v, dv = concentratedGridBetween 0.0 param.vMax 0.0 param.nv param.vC

            // extend grids for ghost points
            let sGhost = 2.0*s.[s.Length - 1] - s.[s.Length - 2]
            let s = Array.append s [|sGhost|]
            let vLowerGhost = 2.0*v.[0] - v.[1]
            let v = Array.append [|vLowerGhost|] v

            let t, dt = homogeneousGrid param.nt 0.0 timeToMaturity

            pcalc {
                let! s = DArray.scatterInBlob worker s
                let! v = DArray.scatterInBlob worker v

                // add zero value ghost points to the value surface to allow simpler access in the kernel
                // one ghost point at s = smax and v = 0
                // no ghost point needed at v = vmax and s = 0 because there we have Dirichlet boundary 
                let ns1 = param.ns+1
                let nv1 = param.nv+1
                let! u = DMatrix.createInBlob<float> worker RowMajorOrder nv1 ns1 
                let! y0 = DMatrix.createInBlob<float> worker RowMajorOrder nv1 ns1 
                let! y1 = DMatrix.createInBlob<float> worker RowMajorOrder nv1 ns1 
                let! y1corr = DMatrix.createInBlob<float> worker RowMajorOrder nv1 ns1 

                // storage for reduced values without ghost points
                let! sred = DArray.createInBlob<float> worker param.ns
                let! vred = DArray.createInBlob<float> worker param.nv
                let! ured = DMatrix.createInBlob<float> worker RowMajorOrder param.nv param.ns 
                
                do! PCalc.action (fun hint ->
                    let blockSize = 16
                    let sharedSize = (blockSize + 2) * (blockSize + 2) * sizeof<float>
                    let lpms = LaunchParam(dim3(divup ns1 blockSize, divup nv1 blockSize), dim3(blockSize, blockSize), sharedSize) |> hint.ModifyLaunchParam
                    let lpm = LaunchParam(dim3(divup ns1 blockSize, divup nv1 blockSize), dim3(blockSize, blockSize)) |> hint.ModifyLaunchParam
                    let lpb = LaunchParam(divup (max ns1 nv1) 256, 256) |> hint.ModifyLaunchParam
                    let lps = LaunchParam(param.nv, param.ns-1, 4*(param.ns-1)*sizeof<float>) |> hint.ModifyLaunchParam
                    let lpv = LaunchParam(param.ns, param.nv-1, 4*(param.nv-1)*sizeof<float>) |> hint.ModifyLaunchParam

                    let u = RMatrixRowMajor(u.NumRows, u.NumCols, u.Storage.Ptr)
                    let y0 = RMatrixRowMajor(y0.NumRows, y0.NumCols, y0.Storage.Ptr)
                    let y1 = RMatrixRowMajor(y1.NumRows, y1.NumCols, y1.Storage.Ptr)
                    let y1corr = RMatrixRowMajor(y1corr.NumRows, y1corr.NumCols, y1corr.Storage.Ptr)
                    let ured = RMatrixRowMajor(ured.NumRows, ured.NumCols, ured.Storage.Ptr)

                    initCondKernel.Launch lpm param.ns param.nv s.Ptr u optionType.sign strike

                    zerosKernel.Launch lpm param.ns param.nv y0
                    zerosKernel.Launch lpm param.ns param.nv y1
                    zerosKernel.Launch lpm param.ns param.nv y1corr

                    for ti = 0 to param.nt-2 do

                        let t0 = t.[ti]
                        let t1 = t.[ti + 1]
                        let dt = t1 - t0

                        //let dt = 3.3984e-4

                        let thetaDt = param.theta*dt

                        boundaryCondKernel.Launch lpb heston.rf t0 param.ns param.nv s.Ptr u
                        appFKernel.Launch lpms heston t0 t1 dt param.theta u s.Ptr v.Ptr param.ns param.nv y0 y1corr

                        solveF1Kernel.Launch lps heston t0 thetaDt y0 s.Ptr v.Ptr param.ns param.nv y1
                        solveF2Kernel.Launch lps heston t0 thetaDt y1 y1corr s.Ptr v.Ptr param.ns param.nv u

                    // copy solution, later we could use a view on it
                    copyValuesKernel.Launch lpm param.ns param.nv u ured
                    copyGridsKernel.Launch lpb param.ns param.nv s.Ptr v.Ptr sred.Ptr vred.Ptr
                )
                
                return sred, vred, ured } ) }
                                