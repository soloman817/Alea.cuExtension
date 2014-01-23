module Alea.cuExtension.CUB.Block.Load
    
open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

open Macro
open Vector


[<Record>]
type LoadDirectBlocked<'T> =
    {
        ITEMS_PER_THREAD    : int
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


[<Record>]
type LoadDirectBlockedVectorized<'T> =
    {
        ITEMS_PER_THREAD    : int
        [<RecordExcludedField>] real : RealTraits<'T>
    }

        
    [<ReflectedDefinition>]
    member inline this.LoadDirectBlockedVectorized(linear_tid:int, block_ptr:deviceptr<'T>, items:deviceptr<'T>) =
        let MAX_VEC_SIZE = CUB_MIN 4 this.ITEMS_PER_THREAD
        let VEC_SIZE = if (((MAX_VEC_SIZE - 1) &&& MAX_VEC_SIZE) = 0) && ((this.ITEMS_PER_THREAD % MAX_VEC_SIZE) = 0) then MAX_VEC_SIZE else 1
        let VECTORS_PER_THREAD = this.ITEMS_PER_THREAD / VEC_SIZE
        let ptr = (block_ptr + (linear_tid * VEC_SIZE * VECTORS_PER_THREAD)) |> __ptr_reinterpret

        let vec_items = __local__.Array<CubVector<'T>>(VECTORS_PER_THREAD) |> __array_to_ptr

        for ITEM = 0 to (VECTORS_PER_THREAD - 1) do vec_items.[ITEM] <- ptr.[ITEM]
        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- vec_items.[ITEM].Ptr |> __ptr_to_obj

    [<ReflectedDefinition>]
    static member Create(real:RealTraits<'T>, _ITEMS_PER_THREAD:int) =
        {   ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
            real = real }

    [<ReflectedDefinition>]
    static member Default(real:RealTraits<'T>) =
        {   ITEMS_PER_THREAD = 128;
            real = real }

[<Record>]
type LoadDirectStriped<'T> =
    {
        BLOCK_THREADS : int
        ITEMS_PER_THREAD : int
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
    //| BLOCK_LOAD_TRANSPOSE
    //| BLOCK_LOAD_WARP_TRANSPOSE

type LoadSignatures<'T> =
    abstract Load : int * deviceptr<'T> * deviceptr<'T> -> unit
    abstract Load : int * deviceptr<'T> * deviceptr<'T> * int -> unit
    abstract Load : int * deviceptr<'T> * deviceptr<'T> * int * 'T -> unit

    
let loadInternal (_ALGORITHM:BlockLoadAlgorithm) =
    fun (real:RealTraits<'T>) (items_per_thread:int) (block_threads:int option) ->
        match _ALGORITHM with
        | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->
            {   new LoadSignatures<'T> with
                    member this.Load(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>) =
                        (LoadDirectBlocked.Create(real, items_per_thread)).LoadDirectBlocked(linear_tid, block_itr, items)

                    member this.Load(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
                        (LoadDirectBlocked.Create(real, items_per_thread)).LoadDirectBlocked(linear_tid, block_itr, items, valid_items)

                    member this.Load(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:'T) =
                        (LoadDirectBlocked.Create(real, items_per_thread)).LoadDirectBlocked(linear_tid, block_itr, items, valid_items, oob_default)  }
            
        | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->
            {   new LoadSignatures<'T> with
                    member this.Load(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>) =
                        (LoadDirectBlockedVectorized.Create(real, items_per_thread)).LoadDirectBlockedVectorized(linear_tid, block_itr, items)

                    member this.Load(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
                        //(LoadDirectBlockedVectorized.Create(real, items_per_thread)).LoadDirectBlockedVectorized(linear_tid, block_itr, items)
                        this.Load(linear_tid, block_itr, items)

                    member this.Load(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:'T) =
                        //(LoadDirectBlockedVectorized.Create(real, items_per_thread)).LoadDirectBlockedVectorized(linear_tid, block_itr, items)
                        this.Load(linear_tid, block_itr, items) }

        //| BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE
        //| BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE
[<Record>]
type LoadInternalVars<'T> =
    {
        temp_storage : deviceptr<'T>
        linear_tid : int
    }

[<Record>]
type LoadInternal<'T> = 
    {
        x : int
    }
    
[<Record>]
type BlockLoadVars<'T> =
    {
        InternalLoad : LoadInternal<'T>
        temp_storage : deviceptr<'T>
    }

[<Record>]
type BlockLoad<'T> =
    {
        BLOCK_THREADS : int
        ITEMS_PER_THREAD : int
        ALGORITHM : BlockLoadAlgorithm
        WARP_TIME_SLICING : bool
    }
//    [<Record>]
//    type LoadInternal<'T> =
//        {
//            mutable real : RealTraits<'T>
//            mutable ITEMS_PER_THREAD : int option
//            mutable BLOCK_THREADS : int option
//            mutable ALGORITHM : BlockLoadAlgorithm
//            mutable temp_storage : deviceptr<'T> option
//            mutable linear_tid : int option
//            mutable LoadDirectBlocked : LoadDirectBlocked<'T> option
//            mutable LoadDirectBlockedVectorized : LoadDirectBlockedVectorized<'T> option
//            mutable LoadDirectStriped : LoadDirectStriped<'T> option
//        }
//
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>) =
//            match this.ALGORITHM with
//            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->
//                this.LoadDirectBlocked <- LoadDirectBlocked.Create(this.real, this.ITEMS_PER_THREAD.Value) |> Some
//            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE -> 
//                this.LoadDirectBlockedVectorized <- LoadDirectBlockedVectorized.Create(this.real, this.ITEMS_PER_THREAD.Value) |> Some
//            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE -> ()
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
//            match this.ALGORITHM with
//            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE -> ()
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:int) =
//            match this.ALGORITHM with
//            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE -> ()
//
//        [<ReflectedDefinition>]
//        static member inline Create(real:RealTraits<'T>, _ALGORITHM:BlockLoadAlgorithm, linear_tid:int) =
//            {   real = real;
//                ITEMS_PER_THREAD = None;
//                BLOCK_THREADS = None;
//                ALGORITHM = _ALGORITHM;
//                temp_storage = None;
//                linear_tid = linear_tid |> Some;
//                LoadDirectBlocked = None;
//                LoadDirectStriped = None}
//
//        [<ReflectedDefinition>]
//        static member inline Create(real:RealTraits<'T>, _ALGORITHM:BlockLoadAlgorithm) =
//            {   real = real;
//                ITEMS_PER_THREAD = None;
//                BLOCK_THREADS = None;
//                ALGORITHM = _ALGORITHM;
//                temp_storage = None;
//                linear_tid = None;
//                LoadDirectBlocked = None;
//                LoadDirectStriped = None}

//
//    [<Record>]
//    type BlockLoad<'T> =
//        {
//            real : RealTraits<'T>
//            mutable BLOCK_THREADS      : int
//            mutable ITEMS_PER_THREAD   : int
//            mutable ALGORITHM          : BlockLoadAlgorithm
//            mutable WARP_TIME_SLICING  : bool
//            TempStorage : Expr<unit -> deviceptr<'T>> option
//            LoadInternal : LoadInternal<'T> option
//        }
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>) = 
//            if this.LoadInternal.IsSome then this.LoadInternal.Value.Load(block_itr, items) else failwith "need to initialize LoadInternal"
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) = 
//            if this.LoadInternal.IsSome then this.LoadInternal.Value.Load(block_itr, items, valid_items) else failwith "need to initialize LoadInternal"
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:int) = 
//            if this.LoadInternal.IsSome then this.LoadInternal.Value.Load(block_itr, items, valid_items) else failwith "need to initialize LoadInternal"
//
//        [<ReflectedDefinition>]
//        static member Create(real:RealTraits<'T>, _BLOCK_THREADS:int, _ITEMS_PER_THREAD:int, _ALGORITHM:BlockLoadAlgorithm, _WARP_TIME_SLICING:bool) =
//            {   real = real;
//                BLOCK_THREADS = _BLOCK_THREADS;
//                ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//                ALGORITHM = _ALGORITHM;
//                WARP_TIME_SLICING = _WARP_TIME_SLICING;
//                TempStorage = None;
//                LoadInternal = LoadInternal<'T>.Create(real, _ALGORITHM) |> Some}


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