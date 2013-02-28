module Alea.CUDA.Extension.PArray

open Microsoft.FSharp.Quotations
open Alea.CUDA

let transform (f:Expr<'T -> 'U>) = cuda {
    let! pfunc = Transform.transform "transform" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (input:DArray<'T>) (output:DArray<'U>) ->
            let n = input.Length
            pcalc { do! PCalc.action (fun lphint -> pfunc lphint n input.Ptr output.Ptr) } ) }

let transform2 (f:Expr<'T1 -> 'T2 -> 'U>) = cuda {
    let! pfunc = Transform.transform2 "transform2" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (input1:DArray<'T1>) (input2:DArray<'T2>) (output:DArray<'U>) ->
            let n = input1.Length
            pcalc { do! PCalc.action (fun lphint -> pfunc lphint n input1.Ptr input2.Ptr output.Ptr) } ) }

let transformi (f:Expr<int -> 'T -> 'U>) = cuda {
    let! pfunc = Transform.transformi "transformi" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (input:DArray<'T>) (output:DArray<'U>) ->
            let n = input.Length
            pcalc { do! PCalc.action (fun lphint -> pfunc lphint n input.Ptr output.Ptr) } ) }

let transformi2 (f:Expr<int -> 'T1 -> 'T2 -> 'U>) = cuda {
    let! pfunc = Transform.transformi2 "transformi2" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (input1:DArray<'T1>) (input2:DArray<'T2>) (output:DArray<'U>) ->
            let n = input1.Length
            pcalc { do! PCalc.action (fun lphint -> pfunc lphint n input1.Ptr input2.Ptr output.Ptr) } ) }

let map (f:Expr<'T -> 'U>) = cuda {
    let! pfunc = Transform.transform "map" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (input:DArray<'T>) ->
            let n = input.Length
            pcalc {
                let! output = DArray.createInBlob worker n
                do! PCalc.action (fun lphint -> pfunc lphint n input.Ptr output.Ptr)
                return output } ) }

let map2 (f:Expr<'T1 -> 'T2 -> 'U>) = cuda {
    let! pfunc = Transform.transform2 "map2" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (input1:DArray<'T1>) (input2:DArray<'T2>) ->
            let n = input1.Length
            pcalc {
                let! output = DArray.createInBlob worker n
                do! PCalc.action (fun lphint -> pfunc lphint n input1.Ptr input2.Ptr output.Ptr)
                return output } ) }

let mapi (f:Expr<int -> 'T -> 'U>) = cuda {
    let! pfunc = Transform.transformi "mapi" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (input:DArray<'T>) ->
            let n = input.Length
            pcalc {
                let! output = DArray.createInBlob worker n
                do! PCalc.action (fun lphint -> pfunc lphint n input.Ptr output.Ptr)
                return output } ) }

let mapi2 (f:Expr<int -> 'T1 -> 'T2 -> 'U>) = cuda {
    let! pfunc = Transform.transformi2 "mapi2" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (input1:DArray<'T1>) (input2:DArray<'T2>) ->
            let n = input1.Length
            pcalc {
                let! output = DArray.createInBlob worker n
                do! PCalc.action (fun lphint -> pfunc lphint n input1.Ptr input2.Ptr output.Ptr)
                return output } ) }

let reduce (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (transf:Expr<'T -> 'T>) = cuda {
    let! reducer = Reduce.reduce init op transf

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let reducer = reducer.Apply m
        fun (values:DArray<'T>) ->
            let n = values.Length
            let reducer = reducer n 
            pcalc {
                let! ranges = DArray.scatterInBlob worker reducer.Ranges
                let! rangeTotals = DArray.createInBlob worker reducer.NumRanges
                do! PCalc.action (fun lphint -> reducer.Reduce lphint ranges.Ptr rangeTotals.Ptr values.Ptr)
                return DScalar.ofArray rangeTotals 0 } ) }

let reducer (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (transf:Expr<'T -> 'T>) = cuda {
    let! reducer = Reduce.reduce init op transf

    return PFunc(fun (m:Module) ->
        let reducer = reducer.Apply m
        let worker = m.Worker
        fun (n:int) ->
            let reducer = reducer n
            pcalc {
                let! ranges = DArray.scatterInBlob worker reducer.Ranges
                let! rangeTotals = DArray.createInBlob worker reducer.NumRanges
                let result = DScalar.ofArray rangeTotals 0
                return fun (values:DArray<'T>) ->
                    if values.Length <> n then failwith "Reducer n not match the input values.Length!"
                    pcalc {
                        do! PCalc.action (fun lphint -> reducer.Reduce lphint ranges.Ptr rangeTotals.Ptr values.Ptr)
                        return result } } ) }

let inline sum () = cuda {
    let! reducer = Reduce.sum()

    return PFunc(fun (m:Module) ->
        let reducer = reducer.Apply m
        let worker = m.Worker
        fun (values:DArray<'T>) ->
            let n = values.Length
            let reducer = reducer n 
            pcalc {
                let! ranges = DArray.scatterInBlob worker reducer.Ranges
                let! rangeTotals = DArray.createInBlob worker reducer.NumRanges
                do! PCalc.action (fun lphint -> reducer.Reduce lphint ranges.Ptr rangeTotals.Ptr values.Ptr)
                return DScalar.ofArray rangeTotals 0 } ) }

let inline sumer () = cuda {
    let! reducer = Reduce.sum()

    return PFunc(fun (m:Module) ->
        let reducer = reducer.Apply m
        let worker = m.Worker
        fun (n:int) ->
            let reducer = reducer n
            pcalc {
                let! ranges = DArray.scatterInBlob worker reducer.Ranges
                let! rangeTotals = DArray.createInBlob worker reducer.NumRanges
                let result = DScalar.ofArray rangeTotals 0
                return fun (values:DArray<'T>) ->
                    if values.Length <> n then failwith "Reducer n not match the input values.Length!"
                    pcalc {
                        do! PCalc.action (fun lphint -> reducer.Reduce lphint ranges.Ptr rangeTotals.Ptr values.Ptr)
                        return result } } ) }


