module Alea.cuExtension.CUB.Block
    
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common

module Discontinuity =
    let f() = "discontinuity"
module Exchange =
    let f() = "exchange"
module Histogram =
    let f() = "histogram"

module Load =
    type BlockLoadAlgorithm =
        | BLOCK_LOAD_DIRECT
        | BLOCK_LOAD_VECTORIZE
        | BLOCK_LOAD_TRANSPOSE
        | BLOCK_LOAD_WARP_TRANSPOSE

    type BlockLoad<'T> =
        abstract BLOCK_THREADS      : int
        abstract ITEMS_PER_THREAD   : int
        abstract ALGORITHM          : BlockLoadAlgorithm
        abstract WARP_TIME_SLICING  : bool

    type LoadInternal<'T> =
        abstract Load : unit -> (Template<Function<deviceptr<'T> -> deviceptr<'T> -> unit>>)
        abstract Load : int -> (Template<Function<deviceptr<'T> -> deviceptr<'T> -> unit>>)
        abstract Load : int * 'T -> (Template<Function<deviceptr<'T> -> deviceptr<'T> -> unit>>)


//    let loadInternal (real:RealTraits<'T>) (linear_tid:int) =
//        { new LoadInternal<'T> with
//            member this.Load () = cuda { return!
//                <@ fun (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//                    for ITEM = 0 to 10 do items.[ITEM] <- block_itr.[(linear_tid * 10) + ITEM]
//                @> |> Compiler.DefineFunction }
//                
//            member this.Load (valid_items:int) = cuda { return! 
//                <@ fun (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//                    let bounds = valid_items - (linear_tid * 10)
//                    for ITEM = 0 to 10 do if (ITEM < bounds) then items.[ITEM] <- block_itr.[(linear_tid * 10) + ITEM]
//                @> |> Compiler.DefineFunction }
//            
//            member this.Load (valid_items:int, oob_default:int) = 
//                for ITEM = 0 to (10 - 1) do items.[ITEM] <- oob_default
//                this.Load(valid_items)
//        }


//    type LoadDirectBlocked<'T>(linear_tid:int, block_itr:InputIterator<'T>, items:deviceptr<'T>, ?valid_items:int, ?oob_default:'T) =
//        member this.Invoke =
//            match valid_items with
//            | None -> 
//                cuda { return! 
//                    <@ fun _ITEMS_PER_THREAD_ -> 
//                        for ITEM = 0 to (_ITEMS_PER_THREAD_ - 1) do items.[ITEM] <- block_itr.[(linear_tid * _ITEMS_PER_THREAD_) + ITEM] 
//                    @> |> Compiler.DefineFunction }
//            | Some valid_items ->
//                match oob_default with
//                | None ->
//                    cuda { return!
//                        <@ fun _ITEMS_PER_THREAD_ ->
//                            let bounds = valid_items - (linear_tid * _ITEMS_PER_THREAD_)
//                            for ITEM = 0 to (_ITEMS_PER_THREAD_ - 1) do if (ITEM < bounds) then items.[ITEM] <- block_itr.[(linear_tid * _ITEMS_PER_THREAD_) + ITEM]
//                        @> |> Compiler.DefineFunction }
//                | Some oob_default ->
//                    cuda { return!                    
//                        <@ fun _ITEMS_PER_THREAD_ ->
//                            for ITEM = 0 to (_ITEMS_PER_THREAD_ - 1) do items.[ITEM] <- oob_default
//                            let bounds = valid_items - (linear_tid * _ITEMS_PER_THREAD_)
//                            for ITEM = 0 to (_ITEMS_PER_THREAD_ - 1) do if (ITEM < bounds) then items.[ITEM] <- block_itr.[(linear_tid * _ITEMS_PER_THREAD_) + ITEM]
//                        @> |> Compiler.DefineFunction }


//    type LoadDirectBlocked<'T>(_ITEMS_PER_THREAD_:int) =
//        member this.Invoke() = 
//                cuda { return! 
//                    <@ fun (linear_tid:int) (block_itr:InputIterator<'T>) (items:deviceptr<'T>) -> 
//                        for ITEM = 0 to (_ITEMS_PER_THREAD_ - 1) do items.[ITEM] <- block_itr.[(linear_tid * _ITEMS_PER_THREAD_) + ITEM] 
//                    @> |> Compiler.DefineFunction }
//            
//        member this.Invoke(valid_items:int) =
//                cuda { return!
//                    <@ fun (linear_tid:int) (block_itr:InputIterator<'T>) (items:deviceptr<'T>) ->
//                        let bounds = valid_items - (linear_tid * _ITEMS_PER_THREAD_)
//                        for ITEM = 0 to (_ITEMS_PER_THREAD_ - 1) do if (ITEM < bounds) then items.[ITEM] <- block_itr.[(linear_tid * _ITEMS_PER_THREAD_) + ITEM]
//                    @> |> Compiler.DefineFunction }
//                
//        member this.Invoke(valid_items:int, oob_default:'T) =
//                cuda { return!                    
//                    <@ fun (linear_tid:int) (block_itr:InputIterator<'T>) (items:deviceptr<'T>) ->
//                        for ITEM = 0 to (_ITEMS_PER_THREAD_ - 1) do items.[ITEM] <- oob_default
//                        let bounds = valid_items - (linear_tid * _ITEMS_PER_THREAD_)
//                        for ITEM = 0 to (_ITEMS_PER_THREAD_ - 1) do if (ITEM < bounds) then items.[ITEM] <- block_itr.[(linear_tid * _ITEMS_PER_THREAD_) + ITEM]
//                    @> |> Compiler.DefineFunction }
//
//    type LoadDirectBlockedVectorized<'T>(linear_tid:int, block_ptr:deviceptr<'T>, items:deviceptr<'T>) =
//        member this.Invoke = cuda { return! <@ fun _ITEMS_PER_THREAD_ -> () @> |> Compiler.DefineFunction }

//    type BlockLoad<'T>(?_ITEMS_PER_THREAD_:int,?BLOCK_THREADS:int) =
//        abstract LoadDirectBlocked : (int * deviceptr<'T> * deviceptr<'T>) -> unit
//        abstract LoadDirectBlocked : (int * deviceptr<'T> * deviceptr<'T> * int) -> unit
//        abstract LoadDirectBlocked : (int * deviceptr<'T> * deviceptr<'T> * int * 'T) -> unit
//        abstract LoadDirectBlockedVectorized : (int * deviceptr<'T> * deviceptr<'T>) -> unit
//        abstract LoadDirectStriped : (int * deviceptr<'T> * deviceptr<'T>) -> unit
//        abstract LoadDirectStriped : (int * deviceptr<'T> * deviceptr<'T> * int) -> unit
//        abstract LoadDirectStriped : (int * deviceptr<'T> * deviceptr<'T> * int * 'T) -> unit
//        abstract LoadDirectWarpStriped : (int * deviceptr<'T> * deviceptr<'T>) -> unit
//        abstract LoadDirectWarpStriped : (int * deviceptr<'T> * deviceptr<'T> * int) -> unit
//        abstract LoadDirectWarpStriped : (int * deviceptr<'T> * deviceptr<'T> * int * 'T) -> unit

module RadixRank =
    let f() = "radix rank"
module RadixSort =
    let f() = "radix sort"
module RakingLayout =
    let f() = "raking layout"
module Reduce =
    let f() = "reduce"
module Scan =
    let f() = "scan"
module Shift =
    let f() = "shift"
module Store =
    let f() = "store"


module Specializations =

    module HistogramAtomic =
        let f() = "histogram atomic"

    module HistogramSort =
        let f() = "histogram sort"

    module ReduceRanking =
        let f() = "reduce ranking"

    module ReduceWarpReduction =
        let f() = "reduce warp reduction"

    module ScanRanking =
        let f() = "scan ranking"

    module ScanWarpScans =
        let f() = "scan warp scans"