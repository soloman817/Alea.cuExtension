module Alea.CUDA.Extension.PRandom

open Microsoft.FSharp.Quotations
open Alea.CUDA

let sobol converter = cuda {
    let! generator = Sobol.generator converter

    return PFunc(fun (m:Module) ->
        let generator = generator.Apply m
        let worker = m.Worker
        fun dimensions vectors offset ->
            let generator = generator dimensions
            pcalc {
                let! lpmod = PCalc.lpmod()
                let! directions = DArray.scatterInBlob worker generator.Directions
                let! output = DArray.createInBlob worker (dimensions * vectors)
                do! PCalc.action (fun () -> generator.Generate lpmod vectors offset directions.Ptr output.Ptr)
                return output }) }

let sobolRng converter = cuda {
    let! generator = Sobol.generator converter

    return PFunc(fun (m:Module) ->
        let generator = generator.Apply m
        let worker = m.Worker
        fun dimensions ->
            let generator = generator dimensions
            pcalc {
                let! lpmod = PCalc.lpmod()
                let! directions = DArray.scatterInBlob worker generator.Directions
                return fun vectors offset (output:DArray<'T>) ->
                    pcalc {
                        do! PCalc.action (fun () -> generator.Generate lpmod vectors offset directions.Ptr output.Ptr) }}) }
