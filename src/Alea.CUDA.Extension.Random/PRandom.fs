﻿module Alea.CUDA.Extension.Random.PRandom

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension

let sobol converter = cuda {
    let! generator = Sobol.generator converter

    return PFunc(fun (m:Module) ->
        let generator = generator.Apply m
        let worker = m.Worker
        fun dimensions vectors offset ->
            let generator = generator dimensions
            pcalc {
                let! directions = DArray.scatterInBlob worker generator.Directions
                let! output = DArray.createInBlob worker (dimensions * vectors)
                do! PCalc.action (fun lphint -> generator.Generate lphint vectors offset directions.Ptr output.Ptr)
                return output }) }

let sobolRng converter = cuda {
    let! generator = Sobol.generator converter

    return PFunc(fun (m:Module) ->
        let generator = generator.Apply m
        let worker = m.Worker
        fun dimensions ->
            let generator = generator dimensions
            pcalc {
                let! directions = DArray.scatterInBlob worker generator.Directions
                return fun vectors offset (output:DArray<'T>) ->
                    pcalc { do! PCalc.action (fun lphint -> generator.Generate lphint vectors offset directions.Ptr output.Ptr) }}) }
