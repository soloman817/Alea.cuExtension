module Alea.CUDA.Extension.PMatrix

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Util

/// <summary>PMatrix.init</summary>
/// <remarks></remarks>
let init (f:Expr<int -> int -> 'T>) = cuda {
    let! pfunc = Transform2D.filli "init" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (order:MatrixStorageOrder) (rows:int) (cols:int) ->
            pcalc {
                let! matrix = DMatrix.createInBlob worker order rows cols
                do! PCalc.action (fun hint -> pfunc hint matrix.Order matrix.NumRows matrix.NumCols matrix.Ptr)
                return matrix } ) }

/// <summary>PMatrix.initp</summary>
/// <remarks></remarks>
let initp (f:Expr<int -> int -> 'P -> 'T>) = cuda {
    let! pfunc = Transform2D.fillip "initp" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (param:'P) (order:MatrixStorageOrder) (rows:int) (cols:int) ->
            pcalc {
                let! matrix = DMatrix.createInBlob worker order rows cols
                do! PCalc.action (fun hint -> pfunc hint matrix.Order matrix.NumRows matrix.NumCols param matrix.Ptr)
                return matrix } ) }

/// <summary>PMatrix.fillip</summary>
/// <remarks></remarks>
let fillip (f:Expr<int -> int -> 'P -> 'T>) = cuda {
    let! pfunc = Transform2D.fillip "fillip" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (param:'P) (matrix:DMatrix<'T>) ->
            pcalc { do! PCalc.action (fun hint -> pfunc hint matrix.Order matrix.NumRows matrix.NumCols param matrix.Ptr) } ) }

/// <summary>PMatrix.transformip</summary>
/// <remarks></remarks>
let transformip (f:Expr<int -> int -> 'P -> 'T -> 'U>) = cuda {
    let! pfunc = Transform2D.transformip "transformip" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (param:'P) (input:DMatrix<'T>) (output:DMatrix<'U>) ->
            if input.Order <> output.Order then failwith "matrix should be same order"
            if input.NumRows <> output.NumRows then failwith "matrix rows not equal"
            if input.NumCols <> output.NumCols then failwith "matrix cols not equal"
            let order = input.Order
            let rows = input.NumRows
            let cols = input.NumCols
            pcalc { do! PCalc.action (fun hint -> pfunc hint order rows cols param input.Ptr output.Ptr) } ) }

/// <summary>Transpose with diagonal block reordering for row major storage format</summary>
/// <remarks></remarks>
let [<ReflectedDefinition>] transposeRowMajor tileDim blockRows sizeY sizeX (dInput:DevicePtr<'T>) (dOutput:DevicePtr<'T>) =
    let tile = __shared__<'T>(tileDim*(tileDim+1))
                
    // diagonal reordering of blocks
    let mutable blockIdxX = 0
    let mutable blockIdxY = 0
    if sizeX = sizeY then
        blockIdxX <- (blockIdx.x + blockIdx.y) % gridDim.x
        blockIdxY <- blockIdx.x
    else
        let bid = blockIdx.x + gridDim.x*blockIdx.y
        blockIdxY <- bid % gridDim.y
        blockIdxX <- ((bid/gridDim.y) + blockIdxY) % gridDim.x           
                 
    // from here on same code as before
    let xIndex = blockIdx.x * tileDim + threadIdx.x
    let yIndex = blockIdx.y * tileDim + threadIdx.y
    let indexIn = xIndex + yIndex*sizeX

    // Besides i < tileDim we must have yIndex + i < sizeY to not overrun rows.
    let mutable i = 0                   
    while i < tileDim do
        if xIndex < sizeX && yIndex + i < sizeY then    
            tile.[(threadIdx.y + i)*(tileDim + 1) + threadIdx.x] <- dInput.[indexIn + i*sizeX]
        i <- i + blockRows

    // (xIndex, yIndex) index into transposed matrix     
    let xIndex = blockIdx.y * tileDim + threadIdx.x
    let yIndex = blockIdx.x * tileDim + threadIdx.y
    let indexOut = xIndex + yIndex*sizeY

    __syncthreads()

    // Besides i < tileDim we must have yIndex + i < sizeX to not overrun rows.
    i <- 0
    while i < tileDim do
        if xIndex < sizeY && yIndex + i < sizeX then    
            dOutput.[indexOut + i*sizeY] <- tile.[threadIdx.x*(tileDim + 1) + threadIdx.y+i]
        i <- i + blockRows

/// <summary>PMatrix.transpose</summary>
/// <remarks></remarks>
let transposeOpBuilder tileDim blockRows = cuda {
    let! transposeRowMajor = <@ fun sizeY sizeX (dInput:DevicePtr<'T>) (dOutput:DevicePtr<'T>) ->
        transposeRowMajor tileDim blockRows sizeY sizeX dInput dOutput @> |> defineKernelFunc

    return PFunc(fun (m:Module) ->
        let transposeRowMajor = transposeRowMajor.Apply m

        fun (input:DMatrix<'T>) (output:DMatrix<'T>) ->
            if input.NumCols = output.NumRows && input.NumRows = output.NumCols then 
                failwith "matrix dimensions not compatible for inplace tranform"
            let sizeX = input.NumCols
            let sizeY = input.NumRows
            let dimBlock = dim3(tileDim, blockRows)
            let dimGrid = dim3(divup sizeX tileDim, divup sizeY tileDim)
            pcalc {
                do! PCalc.action (fun hint ->
                    let lp = LaunchParam(dimGrid, dimBlock) |> hint.ModifyLaunchParam
                    match input.Order with
                    | RowMajorOrder -> transposeRowMajor.Launch lp sizeY sizeX input.Ptr output.Ptr
                    | ColMajorOrder -> failwith "not supported yet" ) } ) } 

/// <summary>PMatrix.transpose</summary>
/// <remarks></remarks>
let transposeOp () = transposeOpBuilder 32 8 

/// <summary>PMatrix.transpose</summary>
/// <remarks>Transpose a matrix and store it in new matrix</remarks>
let transposeBuilder tileDim blockRows = cuda {
    let! transposeRowMajor = <@ fun sizeY sizeX (dInput:DevicePtr<'T>) (dOutput:DevicePtr<'T>) ->
        transposeRowMajor tileDim blockRows sizeY sizeX dInput dOutput @> |> defineKernelFunc

    return PFunc(fun (m:Module) ->
        let transposeRowMajor = transposeRowMajor.Apply m

        fun (input:DMatrix<'T>) ->
            let sizeX = input.NumCols
            let sizeY = input.NumRows
            let dimBlock = dim3(tileDim, blockRows)
            let dimGrid = dim3(divup sizeX tileDim, divup sizeY tileDim)
            pcalc {
                let! output = DMatrix.createInBlob<'T> m.Worker input.Order input.NumCols input.NumRows 

                do! PCalc.action (fun hint ->
                    let lp = LaunchParam(dimGrid, dimBlock) |> hint.ModifyLaunchParam
                    match input.Order with
                    | RowMajorOrder -> transposeRowMajor.Launch lp sizeY sizeX input.Ptr output.Ptr
                    | ColMajorOrder -> failwith "not supported yet" ) } ) } 

/// <summary>PMatrix.transpose</summary>
/// <remarks>Transpose a matrix and store it in new matrix</remarks>
let transpose () = transposeBuilder 32 8 