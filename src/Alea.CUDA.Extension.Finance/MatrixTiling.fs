module Alea.CUDA.Extension.Finance.MatrixTiling

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

/// Explicit apply operator for Euler scheme.
/// We require ghost points at j = 0 and s = smax and require that the grids are properly extended as well.
///
///     V                             
///                                    
///     ^                             
///  nv |****************************O 
///     |*...........................O  <--- jMax = nv - 1
///     |*...........................O    
///     |*...........................O    
///     |*...........................O    
///     |*...........................O    
///     |*...........................O    
///     |*...........................O  <--- jMin = 1 
///  0  |OOOOOOOOOOOOOOOOOOOOOOOOOOOOO
///     ---------------------------------->   S
///     0                            ns
///      ^                          ^
///      |                          |
///      iMin = 1                   iMax = ns - 1
///
/// Total dimension of solution matrix (ns+1) x (nv+1) 
/// Meaning of points:
///     - O     ghost points added to unify memory reading without bound checks, must remain zero 
///             will be removed by copying to a reduced matrix
///     - *     Dirichelt boundary points, which are not going into equation and are fixed
///     - .     Innter points, which are effectively updated by the solver
///
let [<ReflectedDefinition>] tiling (u:RMatrixRowMajor ref) ns nv =
    let tx = threadIdx.x
    let ty = threadIdx.y
    let bx = blockDim.x
    let by = blockDim.y
    
    let jMin = 1
    let jMax = nv - 1
    let iMin = 1
    let iMax = ns - 1

    let uPtr = u.contents.Storage
    let uShared = __extern_shared__<float>()

    let mutable j = blockIdx.y * by + ty

    while j < nv - 2 do 

        let mutable i = blockIdx.x * bx + tx 
                                            
        while i < ns - 2 do

            // find out the tile in which we are working because the grid may not cover all of the matrix u
            let itile = i / bx
            let jtile = j / by
            let i0 = itile * bx
            let j0 = jtile * by
 
            // use all threads of block to load bx*by elements of u 
            let l = ty*bx + tx
            let I = l % (bx + 2)
            let J = l / (bx + 2)
 
            let mutable test = float(l)
 
            if i0 + I < ns + 1 && j0 + J < nv + 1 then
                uShared.[J*(bx+2) + I] <- get u (i0 + I) (j0 + J)

            // second round to load remaining (bx+2)*(by+2) - bx*by elements of u, some threads do not need to load
            let l = bx*by + ty*bx + tx
            let I = l % (bx + 2)
            let J = l / (bx + 2)

            //test <- float(J)

            if l < (bx + 2)*(by + 2) && i0 + I < ns && j0 + J < nv then
                uShared.[J*(bx+2) + I] <- get u (i0 + I) (j0 + J)

            __syncthreads()

            // relative index into submatrix copied into shared memory 
            let k = i - i0
            let l = j - j0

            // we add a ghost points of zero value around u to have no memory access issues
//            let umm = uShared.[(l-1)*(bx+2) + (k-1)]
//            let ump = uShared.[(l+1)*(bx+2) + (k-1)]
//            let um0 = uShared.[ l   *(bx+2) + (k-1)]
//            let u0m = uShared.[(l-1)*(bx+2) +  k   ]
//            let u00 = uShared.[ l   *(bx+2) +  k   ]
//            let u0p = uShared.[(l+1)*(bx+2) +  k   ]
//            let upm = uShared.[(l-1)*(bx+2) + (k+1)]
//            let up0 = uShared.[ l   *(bx+2) + (k+1)]
//            let upp = uShared.[(l+1)*(bx+2) + (k+1)]

            let u00 = uShared.[ j   *(bx+2) +  i   ]
            //test <- get u (i) (j)
            test <- u00 + 100.0

            set u i j test

            __syncthreads()

            i <- i + bx * gridDim.x
               
        j <- j + by * gridDim.y   


/// Initial condition for vanilla call put option.
/// We add artifical zeros to avoid access violation in the kernel. See the documentation above.
let [<ReflectedDefinition>] init ns nv (u:RMatrixRowMajor) =    
    let i = blockIdx.x*blockDim.x + threadIdx.x
    let j = blockIdx.y*blockDim.y + threadIdx.y
    if i < ns && j < nv then
        set (ref u) i j (float(j*nv+i)) 

/// Solve Hesten with explicit Euler scheme.
/// Because the time stepping has to be selected according to the state discretization
/// we may need to have small time steps to maintain stability.
let matrixTiling = cuda {

    // we add artifical zeros to avoid access violation in the kernel
    let! initKernel =     
        <@ fun ns nv (u:RMatrixRowMajor) ->
            init ns nv u @> |> defineKernelFuncWithName "init"
   
    let! tilingKernel =
        <@ fun (u:RMatrixRowMajor) ns nv ->
            tiling (ref u) ns nv @> |> defineKernelFuncWithName "tiling"
                                     
    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let initKernel = initKernel.Apply m
        let tilingKernel = tilingKernel.Apply m

        fun (ns:int) (nv:int) ->

            pcalc {
                let! u = DMatrix.createInBlob<float> worker RowMajorOrder ns nv
                
                // block / tile size
                let bx = 8
                let by = 8

                // we have halo, one to each side
                let ns2 = ns - 2
                let nv2 = nv - 2

                do! PCalc.action (fun hint ->
                    let sharedSize = (bx+2) * (by+2) * sizeof<float>
                    let lpm = LaunchParam(dim3(divup ns bx, divup nv by), dim3(bx, by)) |> hint.ModifyLaunchParam
                    let lpms = LaunchParam(dim3(divup ns2 bx, divup nv2 by), dim3(bx, by), sharedSize) |> hint.ModifyLaunchParam

                    let u = RMatrixRowMajor(u.NumRows, u.NumCols, u.Storage.Ptr)

                    initKernel.Launch lpm ns nv u
                    tilingKernel.Launch lpms u ns nv
                )
                
                return u } ) }
                
