module Alea.CUDA.Extension.PArray

open Microsoft.FSharp.Quotations
open Alea.CUDA

let transform (f:Expr<'T -> 'U>) = cuda {
    let! pfunc = Transform.transform "transform" f

    return PFunc(fun (m:Module) ->
        let pfunc = pfunc.Apply m
        let worker = m.Worker
        fun (input:DArray<'T>) (output:DArray<'U>) ->
            pcalc {
                let! lpmod = PCalc.lpmod()
                let n = input.Length
                do! PCalc.action (lazy (pfunc lpmod n input.Ptr output.Ptr)) } ) }

let transform2 (f:Expr<'T1 -> 'T2 -> 'U>) = cuda {
    let! pfunc = Transform.transform2 "transform2" f

    return PFunc(fun (m:Module) ->
        let pfunc = pfunc.Apply m
        let worker = m.Worker
        fun (input1:DArray<'T1>) (input2:DArray<'T2>) (output:DArray<'U>) ->
            pcalc {
                let! lpmod = PCalc.lpmod()
                let n = input1.Length
                do! PCalc.action (lazy (pfunc lpmod n input1.Ptr input2.Ptr output.Ptr)) } ) }

let transformi (f:Expr<int -> 'T -> 'U>) = cuda {
    let! pfunc = Transform.transformi "transformi" f

    return PFunc(fun (m:Module) ->
        let pfunc = pfunc.Apply m
        let worker = m.Worker
        fun (input:DArray<'T>) (output:DArray<'U>) ->
            pcalc {
                let! lpmod = PCalc.lpmod()
                let n = input.Length
                do! PCalc.action (lazy (pfunc lpmod n input.Ptr output.Ptr)) } ) }

let transformi2 (f:Expr<int -> 'T1 -> 'T2 -> 'U>) = cuda {
    let! pfunc = Transform.transformi2 "transformi2" f

    return PFunc(fun (m:Module) ->
        let pfunc = pfunc.Apply m
        let worker = m.Worker
        fun (input1:DArray<'T1>) (input2:DArray<'T2>) (output:DArray<'U>) ->
            pcalc {
                let! lpmod = PCalc.lpmod()
                let n = input1.Length
                do! PCalc.action (lazy (pfunc lpmod n input1.Ptr input2.Ptr output.Ptr)) } ) }

let map (f:Expr<'T -> 'U>) = cuda {
    let! pfunc = Transform.transform "map" f

    return PFunc(fun (m:Module) ->
        let pfunc = pfunc.Apply m
        let worker = m.Worker
        fun (input:DArray<'T>) ->
            pcalc {
                let! lpmod = PCalc.lpmod()
                let n = input.Length
                let! output = DArray.createInBlob worker n
                do! PCalc.action (lazy (pfunc lpmod n input.Ptr output.Ptr))
                return output } ) }

let map2 (f:Expr<'T1 -> 'T2 -> 'U>) = cuda {
    let! pfunc = Transform.transform2 "map2" f

    return PFunc(fun (m:Module) ->
        let pfunc = pfunc.Apply m
        let worker = m.Worker
        fun (input1:DArray<'T1>) (input2:DArray<'T2>) ->
            pcalc {
                let! lpmod = PCalc.lpmod()
                let n = input1.Length
                let! output = DArray.createInBlob worker n
                do! PCalc.action (lazy (pfunc lpmod n input1.Ptr input2.Ptr output.Ptr))
                return output } ) }

let mapi (f:Expr<int -> 'T -> 'U>) = cuda {
    let! pfunc = Transform.transformi "mapi" f

    return PFunc(fun (m:Module) ->
        let pfunc = pfunc.Apply m
        let worker = m.Worker
        fun (input:DArray<'T>) ->
            pcalc {
                let! lpmod = PCalc.lpmod()
                let n = input.Length
                let! output = DArray.createInBlob worker n
                do! PCalc.action (lazy (pfunc lpmod n input.Ptr output.Ptr))
                return output } ) }

let mapi2 (f:Expr<int -> 'T1 -> 'T2 -> 'U>) = cuda {
    let! pfunc = Transform.transformi2 "mapi2" f

    return PFunc(fun (m:Module) ->
        let pfunc = pfunc.Apply m
        let worker = m.Worker
        fun (input1:DArray<'T1>) (input2:DArray<'T2>) ->
            pcalc {
                let! lpmod = PCalc.lpmod()
                let n = input1.Length
                let! output = DArray.createInBlob worker n
                do! PCalc.action (lazy (pfunc lpmod n input1.Ptr input2.Ptr output.Ptr))
                return output } ) }

let reduce (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (transf:Expr<'T -> 'T>) = cuda {
    let! reducer = Reduce.reduceBuilder (Reduce.Generic.reduceUpSweepKernel init op transf)
                                        (Reduce.Generic.reduceRangeTotalsKernel init op)

    return PFunc(fun (m:Module) ->
        let reducer = reducer.Apply m
        let worker = m.Worker
        fun (values:DArray<'T>) ->
            let n = values.Length
            let reducer = reducer n 
            pcalc {
                let! lpmod = PCalc.lpmod()
                let! ranges = DArray.scatterInBlob worker reducer.Ranges
                let! rangeTotals = DArray.createInBlob worker reducer.NumRangeTotals
                do! PCalc.action (lazy (reducer.Reduce lpmod ranges.Ptr rangeTotals.Ptr values.Ptr))
                return DScalar.ofArray rangeTotals 0 } ) }

let inline sum () = cuda {
    let! reducer = Reduce.reduceBuilder Reduce.Sum.reduceUpSweepKernel Reduce.Sum.reduceRangeTotalsKernel

    return PFunc(fun (m:Module) ->
        let reducer = reducer.Apply m
        let worker = m.Worker
        fun (values:DArray<'T>) ->
            let n = values.Length
            let reducer = reducer n 
            pcalc {
                let! lpmod = PCalc.lpmod()
                let! ranges = DArray.scatterInBlob worker reducer.Ranges
                let! rangeTotals = DArray.createInBlob worker reducer.NumRangeTotals
                do! PCalc.action (lazy (reducer.Reduce lpmod ranges.Ptr rangeTotals.Ptr values.Ptr))
                return DScalar.ofArray rangeTotals 0 } ) }
