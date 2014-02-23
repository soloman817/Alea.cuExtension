module Lab.Lab

open System
open Microsoft.FSharp.Quotations

open Alea.CUDA

#nowarn "9"
#nowarn "51"


//
//let mdpt = true
//type IO<int> =
//    | STMD of deviceptr<int> * deviceptr<int> 
//    | STSD of 'T * Ref<int>

//type STMD<'T, 'BlockPrefixCallbackOp> =
//    abstract ExclusiveSum : Expr<deviceptr<int> -> deviceptr<int> -> unit>
//    abstract ExclusiveSum : Expr<deviceptr<int> -> deviceptr<int> -> Ref<int> -> unit>
//    abstract ExclusiveSum : Expr<deviceptr<int> -> deviceptr<int> -> Ref<'BlockPrefixCallbackOp> -> unit>
//    abstract ExclusiveSum : Expr<deviceptr<int> -> deviceptr<int> -> Ref<int> -> Ref<'BlockPrefixCallbackOp> -> unit>


//type STMD<'T, 'BlockPrefixCallbackOp> =
//    abstract ExclusiveSum : deviceptr<int> -> deviceptr<int> -> Ref<int> option -> Ref<'BlockPrefixCallbackOp> option -> unit
//    abstract ExclusiveScan : deviceptr<int> -> deviceptr<int> -> Ref<int> -> Expr<'T -> 'T -> 'T> -> Ref<int> option -> Ref<'BlockPrefixCallbackOp> option -> unit
//    abstract ExclusiveScan : deviceptr<int> -> deviceptr<int> -> Expr<'T -> 'T -> 'T> -> Ref<int> option -> Ref<'BlockPrefixCallbackOp> option -> unit

//type STMD<'T, 'BlockPrefixCallbackOp> =
//    abstract Sum : (deviceptr<int> * deviceptr<int>) -> unit
//    abstract Sum : (deviceptr<int> * deviceptr<int> * Ref<int>) -> unit
//    abstract Sum : (deviceptr<int> * deviceptr<int> * Ref<int> * Ref<'BlockPrefixCallbackOp>) -> unit

//type STMDsum<'T, 'BlockPrefixCallbackOp> =
//    | Sum                           of (deviceptr<int> * deviceptr<int>)
//    | SumWithAggregate              of (deviceptr<int> * deviceptr<int> * Ref<int>)
//    | SumWithAggregateAndCallback   of (deviceptr<int> * deviceptr<int> * Ref<int> * Ref<'BlockPrefixCallbackOp>)
//
//type STSDsum<'T, 'BlockPrefixCallbackOp> =
//    | Sum                           of ('T * Ref<int>)
//    | SumWithAggregate              of ('T * Ref<int> * Ref<int>)
//    | SumWithAggregateAndCallback   of ('T * Ref<int> * Ref<int> * Ref<'BlockPrefixCallbackOp>)

//type ISTMD<'T, 'BlockPrefixCallbackOp> =
//    abstract ExclusiveSum : deviceptr<int> -> deviceptr<int> -> Ref<int> option -> Ref<'BlockPrefixCallbackOp> option -> unit
//    abstract ExclusiveScan : deviceptr<int> -> deviceptr<int> -> Ref<int> option -> Expr<'T -> 'T -> 'T> -> Ref<int> option -> Ref<'BlockPrefixCallbackOp> option -> unit
//
//type ISTSD<'T, 'BlockPrefixCallbackOp> =
//    abstract ExclusiveSum : 'T -> Ref<int> -> Ref<int> option -> Ref<'BlockPrefixCallbackOp> option -> unit
//    abstract ExclusiveScan : 'T -> Ref<int> -> Ref<int> option -> Expr<'T -> 'T -> 'T> -> Ref<int> option -> Ref<'BlockPrefixCallbackOp> option -> unit
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
//    | STMD -> (fun (input:deviceptr<int>) (output:deviceptr<int>) -> Expr )
//    | STSD -> (fun (input:int) (output:Ref<int>) -> Expr)

//let __T<int>() = null
//let __R<int>() = ref null
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
//type ParamSet<int> =
//    | STMD of (deviceptr<int> -> deviceptr<int> -> Expr)
//    | STSD of ('T -> Ref<int> -> Expr)
//
//type ISTMD<int> =
//    abstract Sum : deviceptr<int> -> deviceptr<int> -> unit
//
//type ExclusiveSumApi<int> =
//    {
//        Sum : ParamSet<int>        
//    }
//
//
//let (|STMD|STSD|) (p:ParamSet<int>) =
//    if typeof<ParamSet<int>> = ParamSet<int>.STMD then
//        STMD
//    else
//        STSD 
type BlockScanAlgorithm =
    | BLOCK_SCAN_RAKING
    | BLOCK_SCAN_RAKING_MEMOIZE
    | BLOCK_SCAN_WARP_SCANS

//type IExclusiveSum<int> =
//    abstract Sum : 'T -> Ref<int> -> unit
//    abstract SumWithAggregate : 'T -> Ref<int> -> Ref<int> -> unit
//    abstract SumWithAggregateAndCallback : 'T -> Ref<int> -> Ref<int> -> unit

type IExclusiveSum<'T, 'BlockPrefixCallbackOp> =
    abstract Sum : ('T * Ref<int>) -> unit
    abstract Sum : ('T * Ref<int> * Ref<int>) -> unit
    abstract Sum : ('T * Ref<int> * Ref<int> * Ref<'BlockPrefixCallbackOp>) -> unit

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