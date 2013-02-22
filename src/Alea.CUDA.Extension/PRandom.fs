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
                let! directions = DArray.scatterInBlob worker generator.Directions
                let! output = DArray.createInBlob worker (dimensions * vectors)
                let! lpmod = PCalc.lpmod()
                do! PCalc.action (lazy (generator.Generate lpmod vectors offset directions.Ptr output.Ptr))
                return output }) }

let sobolIter converter = cuda {
    let! generator = Sobol.generator converter

    return PFunc(fun (m:Module) ->
        let generator = generator.Apply m
        let worker = m.Worker
        fun dimensions ->
            let generator = generator dimensions
            pcalc {
                let! directions = DArray.scatterInBlob worker generator.Directions
                return fun vectors offset (output:DArray<'T>) ->
                    pcalc {
                        let! lpmod = PCalc.lpmod()
                        do! PCalc.action (lazy (generator.Generate lpmod vectors offset directions.Ptr output.Ptr)) }}) }
