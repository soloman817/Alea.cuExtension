module Alea.CUDA.Extension.PArray

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Util

let fill (f:Expr<'T>) = cuda {
    let! pfunc = Transform.fill "fill" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (data:DArray<'T>) ->
            let n = data.Length
            pcalc { do! PCalc.action (fun hint -> pfunc hint n data.Ptr) } ) }

let filli (f:Expr<int -> 'T>) = cuda {
    let! pfunc = Transform.filli "filli" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (data:DArray<'T>) ->
            let n = data.Length
            pcalc { do! PCalc.action (fun hint -> pfunc hint n data.Ptr) } ) }

let init (f:Expr<int -> 'T>) = cuda {
    let! pfunc = Transform.filli "init" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (n:int) ->
            pcalc {
                let! data = DArray.createInBlob worker n
                do! PCalc.action (fun hint -> pfunc hint n data.Ptr)
                return data } ) }

let create (f:Expr<'T>) = cuda {
    let! pfunc = Transform.fill "create" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (n:int) ->
            pcalc {
                let! data = DArray.createInBlob worker n
                do! PCalc.action (fun hint -> pfunc hint n data.Ptr)
                return data } ) }

let inline zeroCreate() = cuda {
    let! pfunc = Transform.fill "zeroCreate" <@ 0G @>

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (n:int) ->
            pcalc {
                let! data = DArray.createInBlob worker n
                do! PCalc.action (fun hint -> pfunc hint n data.Ptr)
                return data } ) }

let transform (f:Expr<'T -> 'U>) = cuda {
    let! pfunc = Transform.transform "transform" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (input:DArray<'T>) (output:DArray<'U>) ->
            let n = input.Length
            pcalc { do! PCalc.action (fun hint -> pfunc hint n input.Ptr output.Ptr) } ) }

let transform2 (f:Expr<'T1 -> 'T2 -> 'U>) = cuda {
    let! pfunc = Transform.transform2 "transform2" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (input1:DArray<'T1>) (input2:DArray<'T2>) (output:DArray<'U>) ->
            let n = input1.Length
            pcalc { do! PCalc.action (fun hint -> pfunc hint n input1.Ptr input2.Ptr output.Ptr) } ) }

let transformi (f:Expr<int -> 'T -> 'U>) = cuda {
    let! pfunc = Transform.transformi "transformi" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (input:DArray<'T>) (output:DArray<'U>) ->
            let n = input.Length
            pcalc { do! PCalc.action (fun hint -> pfunc hint n input.Ptr output.Ptr) } ) }

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
    let! reducer = Reduce.generic init op transf

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let reducer = reducer.Apply m
        fun (values:DArray<'T>) ->
            let n = values.Length
            let reducer = reducer n 
            pcalc {
                let! ranges = DArray.scatterInBlob worker reducer.Ranges
                let! rangeTotals = DArray.createInBlob worker reducer.NumRangeTotals
                do! PCalc.action (fun lphint -> reducer.Reduce lphint ranges.Ptr rangeTotals.Ptr values.Ptr)
                return DScalar.ofArray rangeTotals 0 } ) }

let reducer (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (transf:Expr<'T -> 'T>) = cuda {
    let! reducer = Reduce.generic init op transf

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let reducer = reducer.Apply m
        fun (n:int) ->
            let reducer = reducer n
            pcalc {
                let! ranges = DArray.scatterInBlob worker reducer.Ranges
                let! rangeTotals = DArray.createInBlob worker reducer.NumRangeTotals
                return fun (values:DArray<'T>) (result:DScalar<'T>) ->
                    if values.Length <> n then failwith "Reducer n not match the input values.Length!"
                    pcalc {
                        let action hint =
                            fun () ->
                                reducer.Reduce hint ranges.Ptr rangeTotals.Ptr values.Ptr
                                cuSafeCall(cuMemcpyDtoDAsync(result.Ptr.Handle, rangeTotals.Ptr.Handle, nativeint(sizeof<'T>), hint.Stream.Handle))
                            |> worker.Eval
                        do! PCalc.action action } } ) }

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
                let! rangeTotals = DArray.createInBlob worker reducer.NumRangeTotals
                do! PCalc.action (fun lphint -> reducer.Reduce lphint ranges.Ptr rangeTotals.Ptr values.Ptr)
                return DScalar.ofArray rangeTotals 0 } ) }

let inline sumer () = cuda {
    let! reducer = Reduce.sum()

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let reducer = reducer.Apply m
        fun (n:int) ->
            let reducer = reducer n
            pcalc {
                let! ranges = DArray.scatterInBlob worker reducer.Ranges
                let! rangeTotals = DArray.createInBlob worker reducer.NumRangeTotals
                return fun (values:DArray<'T>) (result:DScalar<'T>) ->
                    if values.Length <> n then failwith "Reducer n not match the input values.Length!"
                    pcalc {
                        let action hint =
                            fun () ->
                                reducer.Reduce hint ranges.Ptr rangeTotals.Ptr values.Ptr
                                cuSafeCall(cuMemcpyDtoDAsync(result.Ptr.Handle, rangeTotals.Ptr.Handle, nativeint(sizeof<'T>), hint.Stream.Handle))
                            |> worker.Eval
                        do! PCalc.action action } } ) }

let scan (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (transf:Expr<'T -> 'T>) = cuda {
    let! scanner = Scan.generic init op transf

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let scanner = scanner.Apply m
        fun (inclusive:bool) (values:DArray<'T>) ->
            let n = values.Length
            let scanner = scanner n 
            pcalc {
                let! ranges = DArray.scatterInBlob worker scanner.Ranges
                let! rangeTotals = DArray.createInBlob worker scanner.NumRangeTotals
                let! results = DArray.createInBlob worker n
                do! PCalc.action (fun hint -> scanner.Scan hint ranges.Ptr rangeTotals.Ptr values.Ptr results.Ptr inclusive)
                return results } ) }

let scanner (init:Expr<unit -> 'T>) (op:Expr<'T -> 'T -> 'T>) (transf:Expr<'T -> 'T>) = cuda {
    let! scanner = Scan.generic init op transf

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let scanner = scanner.Apply m
        fun (n:int) ->
            let scanner = scanner n
            pcalc {
                let! ranges = DArray.scatterInBlob worker scanner.Ranges
                let! rangeTotals = DArray.createInBlob worker scanner.NumRangeTotals
                return fun (inclusive:bool) (values:DArray<'T>) (results:DArray<'T>) ->
                    if values.Length <> n then failwith "Scanner input and output should all equals to n!"
                    if results.Length <> n then failwith "Scanner input and output should all equals to n!"
                    pcalc { do! PCalc.action (fun hint -> scanner.Scan hint ranges.Ptr rangeTotals.Ptr values.Ptr results.Ptr inclusive) } } ) }

let inline sumscan() = cuda {
    let! scanner = Scan.sum()

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let scanner = scanner.Apply m
        fun (inclusive:bool) (values:DArray<'T>) ->
            let n = values.Length
            let scanner = scanner n 
            pcalc {
                let! ranges = DArray.scatterInBlob worker scanner.Ranges
                let! rangeTotals = DArray.createInBlob worker scanner.NumRangeTotals
                let! results = DArray.createInBlob worker n
                do! PCalc.action (fun hint -> scanner.Scan hint ranges.Ptr rangeTotals.Ptr values.Ptr results.Ptr inclusive)
                return results } ) }

let inline sumscanner() = cuda {
    let! scanner = Scan.sum()

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let scanner = scanner.Apply m
        fun (n:int) ->
            let scanner = scanner n
            pcalc {
                let! ranges = DArray.scatterInBlob worker scanner.Ranges
                let! rangeTotals = DArray.createInBlob worker scanner.NumRangeTotals
                return fun (inclusive:bool) (values:DArray<'T>) (results:DArray<'T>) ->
                    if values.Length <> n then failwith "Scanner input and output should all equals to n!"
                    if results.Length <> n then failwith "Scanner input and output should all equals to n!"
                    pcalc { do! PCalc.action (fun hint -> scanner.Scan hint ranges.Ptr rangeTotals.Ptr values.Ptr results.Ptr inclusive) } } ) }

let inline sumsegscan() = cuda {
    let! scanner = SegmentedScan.sum()

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let scanner = scanner.Apply m
        fun (inclusive:bool) (values:DArray<'T>) (flags:DArray<int>) ->
            let n = values.Length
            let scanner = scanner n 
            pcalc {
                let! ranges = DArray.scatterInBlob worker scanner.Ranges
                let! rangeTotals = DArray.createInBlob worker scanner.NumRangeTotals
                let! headFlags = DArray.createInBlob worker scanner.NumHeadFlags
                let! results = DArray.createInBlob worker n
                do! PCalc.action (fun hint -> scanner.Scan hint ranges.Ptr rangeTotals.Ptr headFlags.Ptr values.Ptr flags.Ptr results.Ptr inclusive)
                return results } ) }


