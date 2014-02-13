module Lab.Lab

open System
open Microsoft.FSharp.Quotations

open Alea.CUDA

#nowarn "9"
#nowarn "51"


//
//let mdpt = true
//type IO<'T> =
//    | STMD of deviceptr<'T> * deviceptr<'T> 
//    | STSD of 'T * Ref<'T>

//type STMD<'T, 'BlockPrefixCallbackOp> =
//    abstract ExclusiveSum : Expr<deviceptr<'T> -> deviceptr<'T> -> unit>
//    abstract ExclusiveSum : Expr<deviceptr<'T> -> deviceptr<'T> -> Ref<'T> -> unit>
//    abstract ExclusiveSum : Expr<deviceptr<'T> -> deviceptr<'T> -> Ref<'BlockPrefixCallbackOp> -> unit>
//    abstract ExclusiveSum : Expr<deviceptr<'T> -> deviceptr<'T> -> Ref<'T> -> Ref<'BlockPrefixCallbackOp> -> unit>


//type STMD<'T, 'BlockPrefixCallbackOp> =
//    abstract ExclusiveSum : deviceptr<'T> -> deviceptr<'T> -> Ref<'T> option -> Ref<'BlockPrefixCallbackOp> option -> unit
//    abstract ExclusiveScan : deviceptr<'T> -> deviceptr<'T> -> Ref<'T> -> Expr<'T -> 'T -> 'T> -> Ref<'T> option -> Ref<'BlockPrefixCallbackOp> option -> unit
//    abstract ExclusiveScan : deviceptr<'T> -> deviceptr<'T> -> Expr<'T -> 'T -> 'T> -> Ref<'T> option -> Ref<'BlockPrefixCallbackOp> option -> unit

//type STMD<'T, 'BlockPrefixCallbackOp> =
//    abstract Sum : (deviceptr<'T> * deviceptr<'T>) -> unit
//    abstract Sum : (deviceptr<'T> * deviceptr<'T> * Ref<'T>) -> unit
//    abstract Sum : (deviceptr<'T> * deviceptr<'T> * Ref<'T> * Ref<'BlockPrefixCallbackOp>) -> unit

//type STMDsum<'T, 'BlockPrefixCallbackOp> =
//    | Sum                           of (deviceptr<'T> * deviceptr<'T>)
//    | SumWithAggregate              of (deviceptr<'T> * deviceptr<'T> * Ref<'T>)
//    | SumWithAggregateAndCallback   of (deviceptr<'T> * deviceptr<'T> * Ref<'T> * Ref<'BlockPrefixCallbackOp>)
//
//type STSDsum<'T, 'BlockPrefixCallbackOp> =
//    | Sum                           of ('T * Ref<'T>)
//    | SumWithAggregate              of ('T * Ref<'T> * Ref<'T>)
//    | SumWithAggregateAndCallback   of ('T * Ref<'T> * Ref<'T> * Ref<'BlockPrefixCallbackOp>)

//type ISTMD<'T, 'BlockPrefixCallbackOp> =
//    abstract ExclusiveSum : deviceptr<'T> -> deviceptr<'T> -> Ref<'T> option -> Ref<'BlockPrefixCallbackOp> option -> unit
//    abstract ExclusiveScan : deviceptr<'T> -> deviceptr<'T> -> Ref<'T> option -> Expr<'T -> 'T -> 'T> -> Ref<'T> option -> Ref<'BlockPrefixCallbackOp> option -> unit
//
//type ISTSD<'T, 'BlockPrefixCallbackOp> =
//    abstract ExclusiveSum : 'T -> Ref<'T> -> Ref<'T> option -> Ref<'BlockPrefixCallbackOp> option -> unit
//    abstract ExclusiveScan : 'T -> Ref<'T> -> Ref<'T> option -> Expr<'T -> 'T -> 'T> -> Ref<'T> option -> Ref<'BlockPrefixCallbackOp> option -> unit
//
//
//
//let (|STMD|STSD|) (items_per_thread:int option) =
//    if items_per_thread.IsSome then
//        STMD
//    else
//        STSD
//
//let testItems x =
//    match x with
//    | STMD -> (fun (input:deviceptr<'T>) (output:deviceptr<'T>) -> Expr )
//    | STSD -> (fun (input:'T) (output:Ref<'T>) -> Expr)

//let __T<'T>() = null
//let __R<'T>() = ref null
//
//type Exclusive =
//    | Sum
//    | Scan
//
//type Inclusive =
//    | Sum
//    | Scan
//
//
//type Scan =
//    | Exclusive of Exclusive
//    | Inclusive of Inclusive
////
////let (|STMD|STSD|) f =
////    if typeof<f> = deviceptr then
////        STMD
////    else
////        STSD
////
//
//type ParamSet<'T> =
//    | STMD of (deviceptr<'T> -> deviceptr<'T> -> Expr)
//    | STSD of ('T -> Ref<'T> -> Expr)
//
//type ISTMD<'T> =
//    abstract Sum : deviceptr<'T> -> deviceptr<'T> -> unit
//
//type ExclusiveSumApi<'T> =
//    {
//        Sum : ParamSet<'T>        
//    }
//
//
//let (|STMD|STSD|) (p:ParamSet<'T>) =
//    if typeof<ParamSet<'T>> = ParamSet<'T>.STMD then
//        STMD
//    else
//        STSD 
type BlockScanAlgorithm =
    | BLOCK_SCAN_RAKING
    | BLOCK_SCAN_RAKING_MEMOIZE
    | BLOCK_SCAN_WARP_SCANS

//type IExclusiveSum<'T> =
//    abstract Sum : 'T -> Ref<'T> -> unit
//    abstract SumWithAggregate : 'T -> Ref<'T> -> Ref<'T> -> unit
//    abstract SumWithAggregateAndCallback : 'T -> Ref<'T> -> Ref<'T> -> unit

type IExclusiveSum<'T, 'BlockPrefixCallbackOp> =
    abstract Sum : ('T * Ref<'T>) -> unit
    abstract Sum : ('T * Ref<'T> * Ref<'T>) -> unit
    abstract Sum : ('T * Ref<'T> * Ref<'T> * Ref<'BlockPrefixCallbackOp>) -> unit

//type STMD = HasIdentity | Identityless
//and STSD = HasIdentity | Identityless
//and ExclusiveScan = STMD | STSD
//and InclusiveScan = STMD | STSD
//and Scan = ExclusiveScan | InclusiveScan 
//
//type Scan = ExclusiveScan | InclusiveScan
//and STMD = HasIdentity | Identityless
//and STSD = HasIdentity | Identityless
//and ExclusiveScan = STMD | STSD
//and InclusiveScan = STMD | STSD
//
//
//let inline x() = Scan.ExclusiveScan.STMD

let inline exclusiveSum() = ()

let template (block_threads:int) (algorithm:BlockScanAlgorithm option) = cuda {
    
    return Entry(fun program -> ())

    }