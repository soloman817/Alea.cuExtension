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

