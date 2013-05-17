module Alea.CUDA.Extension.PArray

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Util

/// <summary>PArray.init</summary>
/// <remarks></remarks>
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

/// <summary>PArray.initp</summary>
/// <remarks></remarks>
let initp (f:Expr<int -> 'P -> 'T>) = cuda {
    let! pfunc = Transform.fillip "initp" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (param:'P) (n:int) ->
            pcalc {
                let! data = DArray.createInBlob worker n
                do! PCalc.action (fun hint -> pfunc hint n param data.Ptr)
                return data } ) }

/// <summary>PArray.fill</summary>
/// <remarks></remarks>
let fill (f:Expr<'T>) = cuda {
    let! pfunc = Transform.fill "fill" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (data:DArray<'T>) ->
            let n = data.Length
            pcalc { do! PCalc.action (fun hint -> pfunc hint n data.Ptr) } ) }

/// <summary>PArray.fillp</summary>
/// <remarks></remarks>
let fillp (f:Expr<'P -> 'T>) = cuda {
    let! pfunc = Transform.fillp "fillp" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (param:'P) (data:DArray<'T>) ->
            let n = data.Length
            pcalc { do! PCalc.action (fun hint -> pfunc hint n param data.Ptr) } ) }

/// <summary>PArray.filli</summary>
/// <remarks></remarks>
let filli (f:Expr<int -> 'T>) = cuda {
    let! pfunc = Transform.filli "filli" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (data:DArray<'T>) ->
            let n = data.Length
            pcalc { do! PCalc.action (fun hint -> pfunc hint n data.Ptr) } ) }

/// <summary>PArray.fillip</summary>
/// <remarks></remarks>
let fillip (f:Expr<int -> 'P -> 'T>) = cuda {
    let! pfunc = Transform.fillip "fillip" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (param:'P) (data:DArray<'T>) ->
            let n = data.Length
            pcalc { do! PCalc.action (fun hint -> pfunc hint n param data.Ptr) } ) }

/// <summary>PArray.create</summary>
/// <remarks></remarks>
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

/// <summary>PArray.zeroCreate</summary>
/// <remarks></remarks>
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

/// <summary>PArray.transform</summary>
/// <remarks></remarks>
let transform (f:Expr<'T -> 'U>) = cuda {
    let! pfunc = Transform.transform "transform" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (input:DArray<'T>) (output:DArray<'U>) ->
            let n = input.Length
            pcalc { do! PCalc.action (fun hint -> pfunc hint n input.Ptr output.Ptr) } ) }

/// <summary>PArray.transformp</summary>
/// <remarks></remarks>
let transformp (f:Expr<'P -> 'T -> 'U>) = cuda {
    let! pfunc = Transform.transformp "transformp" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (param:'P) (input:DArray<'T>) (output:DArray<'U>) ->
            let n = input.Length
            pcalc { do! PCalc.action (fun hint -> pfunc hint n param input.Ptr output.Ptr) } ) }

/// <summary>PArray.transformi</summary>
/// <remarks></remarks>
let transformi (f:Expr<int -> 'T -> 'U>) = cuda {
    let! pfunc = Transform.transformi "transformi" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (input:DArray<'T>) (output:DArray<'U>) ->
            let n = input.Length
            pcalc { do! PCalc.action (fun hint -> pfunc hint n input.Ptr output.Ptr) } ) }

/// <summary>PArray.transformip</summary>
/// <remarks></remarks>
let transformip (f:Expr<int -> 'P -> 'T -> 'U>) = cuda {
    let! pfunc = Transform.transformip "transformip" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (param:'P) (input:DArray<'T>) (output:DArray<'U>) ->
            let n = input.Length
            pcalc { do! PCalc.action (fun hint -> pfunc hint n param input.Ptr output.Ptr) } ) }

/// <summary>PArray.transform2</summary>
/// <remarks></remarks>
let transform2 (f:Expr<'T1 -> 'T2 -> 'U>) = cuda {
    let! pfunc = Transform.transform2 "transform2" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (input1:DArray<'T1>) (input2:DArray<'T2>) (output:DArray<'U>) ->
            let n = input1.Length
            pcalc { do! PCalc.action (fun hint -> pfunc hint n input1.Ptr input2.Ptr output.Ptr) } ) }

/// <summary>PArray.transformp2</summary>
/// <remarks></remarks>
let transformp2 (f:Expr<'P -> 'T1 -> 'T2 -> 'U>) = cuda {
    let! pfunc = Transform.transformp2 "transformp2" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (param:'P) (input1:DArray<'T1>) (input2:DArray<'T2>) (output:DArray<'U>) ->
            let n = input1.Length
            pcalc { do! PCalc.action (fun hint -> pfunc hint n param input1.Ptr input2.Ptr output.Ptr) } ) }

/// <summary>PArray.transformi2</summary>
/// <remarks></remarks>
let transformi2 (f:Expr<int -> 'T1 -> 'T2 -> 'U>) = cuda {
    let! pfunc = Transform.transformi2 "transformi2" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (input1:DArray<'T1>) (input2:DArray<'T2>) (output:DArray<'U>) ->
            let n = input1.Length
            pcalc { do! PCalc.action (fun lphint -> pfunc lphint n input1.Ptr input2.Ptr output.Ptr) } ) }

/// <summary>PArray.transformip2</summary>
/// <remarks></remarks>
let transformip2 (f:Expr<int -> 'P -> 'T1 -> 'T2 -> 'U>) = cuda {
    let! pfunc = Transform.transformip2 "transformip2" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (param:'P) (input1:DArray<'T1>) (input2:DArray<'T2>) (output:DArray<'U>) ->
            let n = input1.Length
            pcalc { do! PCalc.action (fun lphint -> pfunc lphint n param input1.Ptr input2.Ptr output.Ptr) } ) }

/// <summary>PArray.map</summary>
/// <remarks></remarks>
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

/// <summary>PArray.mapp</summary>
/// <remarks></remarks>
let mapp (f:Expr<'P -> 'T -> 'U>) = cuda {
    let! pfunc = Transform.transformp "mapp" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (param:'P) (input:DArray<'T>) ->
            let n = input.Length
            pcalc {
                let! output = DArray.createInBlob worker n
                do! PCalc.action (fun lphint -> pfunc lphint n param input.Ptr output.Ptr)
                return output } ) }

/// <summary>PArray.mapi</summary>
/// <remarks></remarks>
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

/// <summary>PArray.mapip</summary>
/// <remarks></remarks>
let mapip (f:Expr<int -> 'P -> 'T -> 'U>) = cuda {
    let! pfunc = Transform.transformip "mapip" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (param:'P) (input:DArray<'T>) ->
            let n = input.Length
            pcalc {
                let! output = DArray.createInBlob worker n
                do! PCalc.action (fun lphint -> pfunc lphint n param input.Ptr output.Ptr)
                return output } ) }

/// <summary>PArray.map2</summary>
/// <remarks></remarks>
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

/// <summary>PArray.mapp2</summary>
/// <remarks></remarks>
let mapp2 (f:Expr<'P -> 'T1 -> 'T2 -> 'U>) = cuda {
    let! pfunc = Transform.transformp2 "mapp2" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (param:'P) (input1:DArray<'T1>) (input2:DArray<'T2>) ->
            let n = input1.Length
            pcalc {
                let! output = DArray.createInBlob worker n
                do! PCalc.action (fun lphint -> pfunc lphint n param input1.Ptr input2.Ptr output.Ptr)
                return output } ) }

/// <summary>PArray.mapi2</summary>
/// <remarks></remarks>
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

/// <summary>PArray.mapip2</summary>
/// <remarks></remarks>
let mapip2 (f:Expr<int -> 'P -> 'T1 -> 'T2 -> 'U>) = cuda {
    let! pfunc = Transform.transformip2 "mapip2" f

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pfunc = pfunc.Apply m
        fun (param:'P) (input1:DArray<'T1>) (input2:DArray<'T2>) ->
            let n = input1.Length
            pcalc {
                let! output = DArray.createInBlob worker n
                do! PCalc.action (fun lphint -> pfunc lphint n param input1.Ptr input2.Ptr output.Ptr)
                return output } ) }

/// <summary>PArray.reduce'</summary>
/// <remarks></remarks>
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

/// <summary>PArray.reducer'</summary>
/// <remarks></remarks>
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

/// <summary>PArray.scan'</summary>
/// <remarks></remarks>
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

/// <summary>PArray.scanner'</summary>
/// <remarks></remarks>
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

/// <summary>PArray.segscan'</summary>
/// <remarks></remarks>
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

/// <summary>PArray.segscanner'</summary>
/// <remarks></remarks>
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

