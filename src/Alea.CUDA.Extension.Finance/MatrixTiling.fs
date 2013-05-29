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

/// Tiling of a matrix
/// The rows of the matrix u have to be padded so that ld-2 is a multiple of blockDim.x 
let [<ReflectedDefinition>] tiling nRows nCols ld (u:RMatrixRowMajor ref) (o1:RMatrixRowMajor ref) (o2:RMatrixRowMajor ref) =
    let tx = threadIdx.x
    let ty = threadIdx.y
    let bx = blockDim.x
    let by = blockDim.y

    let uPtr = u.contents.Storage
    let uShared = __extern_shared__<float>()

    // need to map x to columns and y to rows of the matrix
    // i / y -> row
    // j / x -> col

    // we need to have enough threads to load the elements so we need to loop 
    // possibly over more than nRows - 2 
    let nby = (nRows - 2) / by
    let n = nby * by

    let mutable i = blockIdx.y * by + ty                                           
    while i < n do

        let mutable j = blockIdx.x * bx + tx 
        while j < ld - 2 do 

            // find out the tile in which we are working because the grid may not cover all of the matrix u
            let itile = i / by    
            let jtile = j / bx    
            let i0 = itile * by
            let j0 = jtile * bx
            let I0 = i0 * ld + j0
 
            // use all threads of block to load bx*by elements of u 
            let l = ty*bx + tx
            let I = l / (bx + 2)  // row
            let J = l % (bx + 2)  // col
 
            let mutable test = -1.0

            if i0 + I < nRows + 1 && j0 + J < nCols + 1 then
                uShared.[I*(bx+2) + J] <- uPtr.[I0 + I*ld + J]
                set o1 i j (float(I0 + I*ld + J))
            else 
                set o1 i j -1.0

            // second round to load remaining (bx+2)*(by+2) - bx*by elements of u, some threads do not need to load
            let l = bx*by + ty*bx + tx
            let I = l / (bx + 2)
            let J = l % (bx + 2)

            if l < (bx + 2)*(by + 2) && i0 + I < nRows && j0 + J < nCols then
                uShared.[I*(bx+2) + J] <- uPtr.[I0 + I*ld + J]
                set o2 i j (float(I0 + I*ld + J))
            else
                set o2 i j -1.0

            __syncthreads()

            if i < nRows - 2 && j < nCols - 2 then 

                // relative index into submatrix copied into shared memory
                // shift by one to account for translation into interior domain
                let k = i - i0 + 1
                let l = j - j0 + 1

                let umm = uShared.[(k-1)*(bx+2) + (l-1)]
                let ump = uShared.[(k+1)*(bx+2) + (l-1)]
                let um0 = uShared.[ k   *(bx+2) + (l-1)]
                let u0m = uShared.[(k-1)*(bx+2) +  l   ]
                let u00 = uShared.[ k   *(bx+2) +  l   ]
                let u0p = uShared.[(k+1)*(bx+2) +  l   ]
                let upm = uShared.[(k-1)*(bx+2) + (l+1)]
                let up0 = uShared.[ k   *(bx+2) + (l+1)]
                let upp = uShared.[(k+1)*(bx+2) + (l+1)]

                set u (i+1) (j+1) u00

                __syncthreads()
              
            j <- j + bx * gridDim.x   

        i <- i + by * gridDim.y

/// Initial condition for vanilla call put option.
/// We add artifical zeros to avoid access violation in the kernel.  
let [<ReflectedDefinition>] init nRows nCols ld (u:RMatrixRowMajor) (o1:RMatrixRowMajor) (o2:RMatrixRowMajor)=    
    let i = blockIdx.y*blockDim.y + threadIdx.y
    let j = blockIdx.x*blockDim.x + threadIdx.x
    if i < nRows then
        if j < ld then
            set (ref o1) i j -1.0
            set (ref o2) i j -1.0
        if j < nCols then
            set (ref u) i j (float(i*nCols+j))
        else if j < ld then
            set (ref u) i j -1.0

/// Solve Hesten with explicit Euler scheme.
/// Because the time stepping has to be selected according to the state discretization
/// we may need to have small time steps to maintain stability.
let matrixTiling = cuda {

    // we add artifical zeros to avoid access violation in the kernel
    let! initKernel =     
        <@ fun nRows nCols ld (u:RMatrixRowMajor) (o1:RMatrixRowMajor) (o2:RMatrixRowMajor) ->
            init nRows nCols ld u o1 o2 @> |> defineKernelFuncWithName "init"
   
    let! tilingKernel =
        <@ fun nRows nCols ld (u:RMatrixRowMajor) (o1:RMatrixRowMajor) (o2:RMatrixRowMajor) ->
            tiling nRows nCols ld (ref u) (ref o1) (ref o2) @> |> defineKernelFuncWithName "tiling"
                                     
    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let initKernel = initKernel.Apply m
        let tilingKernel = tilingKernel.Apply m

        fun (nRows:int) (nCols:int) ->

            pcalc {
                // we have halo, one to each side
                let nRows2 = nRows - 2
                let nCols2 = nCols - 2

                // block / tile size
                let bx = 8
                let by = 8

                // create row padding 
                let nbx = divup nCols2 bx
                let ld = nbx * bx + 2

                let! u = DMatrix.createInBlob<float> worker RowMajorOrder nRows ld  
                let! o1 = DMatrix.createInBlob<float> worker RowMajorOrder nRows ld               
                let! o2 = DMatrix.createInBlob<float> worker RowMajorOrder nRows ld               

                do! PCalc.action (fun hint ->
                    let sharedSize = (bx+2) * (by+2) * sizeof<float>
                    let lpm = LaunchParam(dim3(nbx + 1, divup nRows by), dim3(bx, by)) |> hint.ModifyLaunchParam
                    let lpms = LaunchParam(dim3(nbx, divup nRows2 by), dim3(bx, by), sharedSize) |> hint.ModifyLaunchParam

                    let u = RMatrixRowMajor(u.NumRows, u.NumCols, u.Storage.Ptr)
                    let o1 = RMatrixRowMajor(o1.NumRows, o1.NumCols, o1.Storage.Ptr)
                    let o2 = RMatrixRowMajor(o2.NumRows, o2.NumCols, o2.Storage.Ptr)

                    initKernel.Launch lpm nRows nCols ld u o1 o2
                    tilingKernel.Launch lpms nRows nCols ld u o1 o2
                )
                
                return u, o1, o2 } ) }
                
