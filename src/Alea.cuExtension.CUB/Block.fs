module Alea.cuExtension.CUB.Block
    
open Microsoft.FSharp.Quotations

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

    [<Record>]
    type LoadDirectBlocked<'T> =
        {
            mutable ITEMS_PER_THREAD    : int
            [<RecordExcludedField>] real : RealTraits<'T>
        }

        [<ReflectedDefinition>]
        member inline this.LoadDirectBlocked(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>) =
            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * this.ITEMS_PER_THREAD) + ITEM]

        [<ReflectedDefinition>]
        member inline this.LoadDirectBlocked(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
            let bounds = valid_items - (linear_tid * this.ITEMS_PER_THREAD)
            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * this.ITEMS_PER_THREAD) + ITEM]

        [<ReflectedDefinition>]
        member inline this.LoadDirectBlocked(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:'T) =
            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
            this.LoadDirectBlocked(linear_tid, block_itr, items, valid_items)

        [<ReflectedDefinition>]
        static member Create(real:RealTraits<'T>, _ITEMS_PER_THREAD:int) =
            {   ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
                real = real }

        [<ReflectedDefinition>]
        static member Default(real:RealTraits<'T>) =
            {   ITEMS_PER_THREAD = 128;
                real = real }


    // LoadDirectBlockedVectorized

    [<Record>]
    type LoadDirectStriped<'T> =
        {
            mutable BLOCK_THREADS : int
            mutable ITEMS_PER_THREAD : int
            [<RecordExcludedField>] real : RealTraits<'T>
        }

        [<ReflectedDefinition>]
        member inline this.LoadDirectStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>) =
            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(ITEM * this.BLOCK_THREADS) + linear_tid]

        [<ReflectedDefinition>]
        member inline this.LoadDirectStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
            let bounds = valid_items - linear_tid
            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do 
                if (ITEM * this.BLOCK_THREADS < bounds) then items.[ITEM] <- block_itr.[(ITEM * this.BLOCK_THREADS) + linear_tid]

        [<ReflectedDefinition>]
        member inline this.LoadDirectStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:'T) =
            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
            this.LoadDirectStriped(linear_tid, block_itr, items, valid_items)

        [<ReflectedDefinition>]
        static member Create(real:RealTraits<'T>, _BLOCK_THREADS:int, _ITEMS_PER_THREAD:int) =
            {   BLOCK_THREADS = _BLOCK_THREADS
                ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
                real = real }

        [<ReflectedDefinition>]
        static member Default(real:RealTraits<'T>) =
            {   BLOCK_THREADS = 128;
                ITEMS_PER_THREAD = 128;
                real = real }


    [<Record>]
    type LoadDirectWarpStriped<'T> =
        {
            mutable ITEMS_PER_THREAD : int
            [<RecordExcludedField>] real : RealTraits<'T>
        }

        [<ReflectedDefinition>]
        member inline this.LoadDirectWarpStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>) =
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * this.ITEMS_PER_THREAD

            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]

        [<ReflectedDefinition>]
        member inline this.LoadDirectWarpStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * this.ITEMS_PER_THREAD
            let bounds = valid_items - warp_offset - tid

            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do 
                if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]

        [<ReflectedDefinition>]
        member inline this.LoadDirectWarpStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:'T) =
            for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
            this.LoadDirectWarpStriped(linear_tid, block_itr, items, valid_items)

        [<ReflectedDefinition>]
        static member Create(real:RealTraits<'T>, _ITEMS_PER_THREAD:int) =
            {   ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
                real = real }

        [<ReflectedDefinition>]
        static member Default(real:RealTraits<'T>) =
            {   ITEMS_PER_THREAD = 128;
                real = real }


    type BlockLoadAlgorithm =
        | BLOCK_LOAD_DIRECT
        | BLOCK_LOAD_VECTORIZE
        | BLOCK_LOAD_TRANSPOSE
        | BLOCK_LOAD_WARP_TRANSPOSE


    [<Record>]
    type LoadInternal<'T> =
        {
            mutable real : RealTraits<'T>
            mutable ITEMS_PER_THREAD : int option
            mutable BLOCK_THREADS : int option
            mutable ALGORITHM : BlockLoadAlgorithm
            mutable temp_storage : deviceptr<'T> option
            mutable linear_tid : int option
            mutable LoadDirectBlocked : LoadDirectBlocked<'T> option
            mutable LoadDirectStriped : LoadDirectStriped<'T> option
        }


        [<ReflectedDefinition>]
        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>) =
            match this.ALGORITHM with
            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->
                this.LoadDirectBlocked <- LoadDirectBlocked.Default(this.real) |> Some
            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE -> ()
            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE -> ()
            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE -> ()

        [<ReflectedDefinition>]
        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
            match this.ALGORITHM with
            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT -> ()
            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE -> ()
            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE -> ()
            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE -> ()

        [<ReflectedDefinition>]
        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:int) =
            match this.ALGORITHM with
            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT -> ()
            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE -> ()
            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE -> ()
            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE -> ()

        [<ReflectedDefinition>]
        static member inline Create(real:RealTraits<'T>, _ALGORITHM:BlockLoadAlgorithm, linear_tid:int) =
            {   real = real;
                ITEMS_PER_THREAD = None;
                BLOCK_THREADS = None;
                ALGORITHM = _ALGORITHM;
                temp_storage = None;
                linear_tid = linear_tid |> Some;
                LoadDirectBlocked = None;
                LoadDirectStriped = None}

        [<ReflectedDefinition>]
        static member inline Create(real:RealTraits<'T>, _ALGORITHM:BlockLoadAlgorithm) =
            {   real = real;
                ITEMS_PER_THREAD = None;
                BLOCK_THREADS = None;
                ALGORITHM = _ALGORITHM;
                temp_storage = None;
                linear_tid = None;
                LoadDirectBlocked = None;
                LoadDirectStriped = None}


    [<Record>]
    type BlockLoad<'T> =
        {
            real : RealTraits<'T>
            mutable BLOCK_THREADS      : int
            mutable ITEMS_PER_THREAD   : int
            mutable ALGORITHM          : BlockLoadAlgorithm
            mutable WARP_TIME_SLICING  : bool
            TempStorage : Expr<unit -> deviceptr<'T>> option
            LoadInternal : LoadInternal<'T> option
        }

        [<ReflectedDefinition>]
        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>) = 
            if this.LoadInternal.IsSome then this.LoadInternal.Value.Load(block_itr, items) else failwith "need to initialize LoadInternal"

        [<ReflectedDefinition>]
        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) = 
            if this.LoadInternal.IsSome then this.LoadInternal.Value.Load(block_itr, items, valid_items) else failwith "need to initialize LoadInternal"

        [<ReflectedDefinition>]
        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:int) = 
            if this.LoadInternal.IsSome then this.LoadInternal.Value.Load(block_itr, items, valid_items) else failwith "need to initialize LoadInternal"

        [<ReflectedDefinition>]
        static member Create(real:RealTraits<'T>, _BLOCK_THREADS:int, _ITEMS_PER_THREAD:int, _ALGORITHM:BlockLoadAlgorithm, _WARP_TIME_SLICING:bool) =
            {   real = real;
                BLOCK_THREADS = _BLOCK_THREADS;
                ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
                ALGORITHM = _ALGORITHM;
                WARP_TIME_SLICING = _WARP_TIME_SLICING;
                TempStorage = None;
                LoadInternal = LoadInternal<'T>.Create(real, _ALGORITHM) |> Some}


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