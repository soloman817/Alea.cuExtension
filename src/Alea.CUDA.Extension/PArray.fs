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

let reduce' (reducer:PTemplate<PFunc<int -> Reduce.IReduce<'T>>>) = cuda {
    let! reducer = reducer

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

let reducer' (reducer:PTemplate<PFunc<int -> Reduce.IReduce<'T>>>) = cuda {
    let! reducer = reducer

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

let reduce init op transf = Reduce.generic Reduce.Planner.Default init op transf |> reduce'
let reducer init op transf = Reduce.generic Reduce.Planner.Default init op transf |> reducer'
let inline sum() = Reduce.sum Reduce.Planner.Default |> reduce'
let inline sumer() = Reduce.sum Reduce.Planner.Default |> reducer'

let scan' (scanner:PTemplate<PFunc<int -> Scan.IScan<'T>>>) = cuda {
    let! scanner = scanner

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

let scanner' (scanner:PTemplate<PFunc<int -> Scan.IScan<'T>>>) = cuda {
    let! scanner = scanner

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

let scan init op transf = Scan.generic Scan.Planner.Default init op transf |> scan'
let scanner init op transf = Scan.generic Scan.Planner.Default init op transf |> scanner'
let inline sumscan() = Scan.sum Scan.Planner.Default |> scan'
let inline sumscanner() = Scan.sum Scan.Planner.Default |> scanner'

let segscan' (scanner:PTemplate<PFunc<int -> SegmentedScan.ISegmentedScan<'T>>>) = cuda {
    let! scanner = scanner

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let scanner = scanner.Apply m
        fun (inclusive:bool) (marks:DArray<int>) (values:DArray<'T>) ->
            let n = values.Length
            let scanner = scanner n 
            pcalc {
                let! ranges = DArray.scatterInBlob worker scanner.Ranges
                let! rangeTotals = DArray.createInBlob worker scanner.NumRangeTotals
                let! headFlags = DArray.createInBlob worker scanner.NumHeadFlags
                let! results = DArray.createInBlob worker n
                do! PCalc.action (fun hint -> scanner.Scan hint ranges.Ptr rangeTotals.Ptr headFlags.Ptr marks.Ptr values.Ptr results.Ptr inclusive)
                return results } ) }

let segscanner' (scanner:PTemplate<PFunc<int -> SegmentedScan.ISegmentedScan<'T>>>) = cuda {
    let! scanner = scanner

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let scanner = scanner.Apply m
        fun (n:int) ->
            let scanner = scanner n
            pcalc {
                let! ranges = DArray.scatterInBlob worker scanner.Ranges
                let! rangeTotals = DArray.createInBlob worker scanner.NumRangeTotals
                let! headFlags = DArray.createInBlob worker scanner.NumHeadFlags
                return fun (inclusive:bool) (marks:DArray<int>) (values:DArray<'T>) (results:DArray<'T>) ->
                    if values.Length <> n then failwith "Scanner input and output should all equals to n!"
                    if marks.Length <> n then failwith "Scanner marks should be equal to n!"
                    if results.Length <> n then failwith "Scanner input and output should all equals to n!"
                    pcalc { do! PCalc.action (fun hint -> scanner.Scan hint ranges.Ptr rangeTotals.Ptr headFlags.Ptr marks.Ptr values.Ptr results.Ptr inclusive) } } ) }
                    
let fsegscan init op transf = SegmentedScan.genericf SegmentedScan.Planner.Default init op transf |> segscan'
let fsegscanner init op transf = SegmentedScan.genericf SegmentedScan.Planner.Default init op transf |> segscanner'
let inline sumfsegscan() = SegmentedScan.sumf SegmentedScan.Planner.Default |> segscan'
let inline sumfsegscanner() = SegmentedScan.sumf SegmentedScan.Planner.Default |> segscanner'

let ksegscan init op transf = SegmentedScan.generick SegmentedScan.Planner.Default init op transf |> segscan'
let ksegscanner init op transf = SegmentedScan.generick SegmentedScan.Planner.Default init op transf |> segscanner'
let inline sumksegscan() = SegmentedScan.sumk SegmentedScan.Planner.Default |> segscan'
let inline sumksegscanner() = SegmentedScan.sumk SegmentedScan.Planner.Default |> segscanner'

