[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Load
    
open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

open Macro
open Vector

//
//type BlockLoadAlgorithm =
//    | BLOCK_LOAD_DIRECT
//    | BLOCK_LOAD_VECTORIZE
//    | BLOCK_LOAD_TRANSPOSE
//    | BLOCK_LOAD_WARP_TRANSPOSE
//
//
//let loadDirectBlocked (items_per_thread:int) (block_threads:int) = 
//    fun (valid_items:int option) (oob_default:'T option) ->
//        match valid_items, oob_default with
//        | None, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
//
//        | Some valid_items, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//                let bounds = valid_items - (linear_tid * items_per_thread)
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
//                
//        | Some valid_items, Some oob_default ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- oob_default
//                let bounds = valid_items - (linear_tid * items_per_thread)
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
//
//        | _, _ ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) -> ()
//
//
//let loadDirectBlockedVectorized (items_per_thread:int) (block_threads:int) =
//    fun _ _ ->
//        fun (linear_tid:int) (block_ptr:deviceptr<'Vector>) (items:deviceptr<'Vector>) ->
//            let MAX_VEC_SIZE = CUB_MIN 4 items_per_thread
//            let VEC_SIZE = if (((MAX_VEC_SIZE - 1) &&& MAX_VEC_SIZE) = 0) && ((items_per_thread % MAX_VEC_SIZE) = 0) then MAX_VEC_SIZE else 1
//            let VECTORS_PER_THREAD = items_per_thread / VEC_SIZE
//            let ptr = (block_ptr + (linear_tid * VEC_SIZE * VECTORS_PER_THREAD)) |> __ptr_reinterpret
//
//            let vec_items = __local__.Array<'Vector>(VECTORS_PER_THREAD) |> __array_to_ptr
//
//            for ITEM = 0 to (VECTORS_PER_THREAD - 1) do vec_items.[ITEM] <- ptr.[ITEM]
//            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- vec_items.[ITEM] //|> __ptr_to_obj
//
//
//let loadDirectStriped (items_per_thread:int) (block_threads:int) = 
//    fun (valid_items:int option) (oob_default:'T option) ->
//        match valid_items, oob_default with
//        | None, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//               for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(ITEM * block_threads) + linear_tid]
//
//        | Some valid_items, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//                let bounds = valid_items - linear_tid
//                for ITEM = 0 to (items_per_thread - 1) do 
//                    if (ITEM * block_threads < bounds) then items.[ITEM] <- block_itr.[(ITEM * block_threads) + linear_tid]
//                
//        | Some valid_items, Some oob_default ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- oob_default
//                let bounds = valid_items - linear_tid
//                for ITEM = 0 to (items_per_thread - 1) do 
//                    if (ITEM * block_threads < bounds) then items.[ITEM] <- block_itr.[(ITEM * block_threads) + linear_tid]
//
//        | _, _ ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) -> ()
//
//
//let loadDirectWarpStriped (items_per_thread:int) (block_threads:int) = 
//    fun (valid_items:int option) (oob_default:'T option) ->
//        match valid_items, oob_default with
//        | None, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//                let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//                let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
//                let warp_offset = wid * CUB_PTX_WARP_THREADS * items_per_thread
//
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
//
//        | Some valid_items, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//                let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//                let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
//                let warp_offset = wid * CUB_PTX_WARP_THREADS * items_per_thread
//                let bounds = valid_items - warp_offset - tid
//
//                for ITEM = 0 to (items_per_thread - 1) do 
//                    if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
//                
//        | Some valid_items, Some oob_default ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- oob_default
//                let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//                let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
//                let warp_offset = wid * CUB_PTX_WARP_THREADS * items_per_thread
//                let bounds = valid_items - warp_offset - tid
//
//                for ITEM = 0 to (items_per_thread - 1) do 
//                    if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
//
//        | _, _ ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) -> ()
//
//
//let loadInternal (_ALGORITHM:BlockLoadAlgorithm) =
//    fun (items_per_thread:int) (block_threads:int) ->
//        match _ALGORITHM with
//        | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->           (items_per_thread, block_threads) ||> loadDirectBlocked
//        | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->        (items_per_thread, block_threads) ||> loadDirectBlockedVectorized
//        | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->        (items_per_thread, block_threads) ||> loadDirectStriped
//        | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->   (items_per_thread, block_threads) ||> loadDirectWarpStriped
//
//
//let blockLoad (block_threads:int) (items_per_thread:int) (algorithm:BlockLoadAlgorithm) (warp_time_slicing:bool) =
//    match algorithm with
//    | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->
//        fun _ linear_tid ->
//            fun (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int option) (oob_default:'T option) ->
//                algorithm |> loadInternal 
//                <||     (items_per_thread, block_threads)
//                <||     (valid_items, oob_default)
//                <|||    (linear_tid, block_itr, items)
//
//    | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->
//        fun _ linear_tid ->    
//            fun (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int option) (oob_default:'T option) ->
//                algorithm |> loadInternal 
//                <||     (items_per_thread, block_threads)
//                <||     (valid_items, oob_default)
//                <|||    (linear_tid, block_itr, items)
//
//    | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->
//        let stripedToBlocked = (block_threads, items_per_thread, warp_time_slicing) |||> Exchange.stripedToBlocked
//        
//        fun temp_storage linear_tid ->
//            let stripedToBlocked = (temp_storage, linear_tid) ||> stripedToBlocked
//            fun (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int option) (oob_default:'T option) ->
//                algorithm |> loadInternal 
//                <||     (items_per_thread, block_threads)
//                <||     (valid_items, oob_default)
//                <|||    (linear_tid, block_itr, items)
//                items |> stripedToBlocked
//
//    | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->
//        let warpStripedToBlocked = (block_threads, items_per_thread, warp_time_slicing) |||> Exchange.warpStripedToBlocked
//        
//        fun temp_storage linear_tid ->
//            let warpStripedToBlocked = (temp_storage, linear_tid) ||> warpStripedToBlocked
//            fun (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int option) (oob_default:'T option) ->
//                algorithm |> loadInternal 
//                <||     (items_per_thread, block_threads)
//                <||     (valid_items, oob_default)
//                <|||    (linear_tid, block_itr, items)
//                items |> warpStripedToBlocked
//
//
//    
//[<Record>]
//type ThreadFields<'T> =
//    {
//        temp_storage : deviceptr<'T>
//        linear_tid : int
//    }
//
//    member this.Get() = (this.temp_storage, this.linear_tid)
//
//    static member Init() =        
//        {
//            temp_storage = privateStorage()
//            linear_tid = threadIdx.x
//        }
//
//    static member inline Init(temp_storage:deviceptr<_>) =
//        {
//            temp_storage = temp_storage
//            linear_tid = threadIdx.x
//        }
//
//    static member Init(linear_tid:int) =
//        {
//            temp_storage = privateStorage()
//            linear_tid = linear_tid
//        }
//
//    static member inline Init(temp_storage:deviceptr<_>, linear_tid:int) =
//        {
//            temp_storage = temp_storage
//            linear_tid = linear_tid
//        }
//
//
//[<Record>]
//type BlockLoad<'T> =
//    {
//        ThreadFields : ThreadFields<'T>
//        BLOCK_THREADS : int
//        ITEMS_PER_THREAD : int
//        ALGORITHM : BlockLoadAlgorithm
//        WARP_TIME_SLICING : bool
//    }
//
//
//    member this.Load(block_itr:deviceptr<_>, items:deviceptr<_>) = 
//        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items) 
//            <|| (None, None)
//
//    member this.Load(block_itr:deviceptr<_>, items:deviceptr<_>, valid_items:int) =
//        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items) 
//            <|| (valid_items |> Some, None)
//
//    member inline this.Load(block_itr:deviceptr<_>, items:deviceptr<_>, valid_items:int, oob_default:'T) =
//        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items)
//            <|| (valid_items |> Some, oob_default |> Some)
//
//    static member Init(threadFields, block_threads, items_per_thread, algorithm, warp_time_slicing) =
//        {
//            ThreadFields        = threadFields
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = algorithm
//            WARP_TIME_SLICING   = warp_time_slicing
//        }
//
//    static member Init(threadFields, block_threads, items_per_thread) = 
//        {
//            ThreadFields        = threadFields
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = BlockLoadAlgorithm.BLOCK_LOAD_DIRECT
//            WARP_TIME_SLICING   = false
//        }



//let vars (temp_storage:deviceptr<'T> option) (linear_tid:int option) =
//    match temp_storage, linear_tid with
//    | Some temp_storage, Some linear_tid -> temp_storage,       linear_tid
//    | None,              Some linear_tid -> privateStorage(),   linear_tid
//    | Some temp_storage, None ->            temp_storage,       threadIdx.x
//    | None,              None ->            privateStorage(),   threadIdx.x



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
//
//[<Record>]
//type LoadDirectBlocked<'T> =
//    {
//        ITEMS_PER_THREAD    : int
//        [<RecordExcludedField>] real : RealTraits<'T>
//    }
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectBlocked(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>) =
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * this.ITEMS_PER_THREAD) + ITEM]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectBlocked(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
//        let bounds = valid_items - (linear_tid * this.ITEMS_PER_THREAD)
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * this.ITEMS_PER_THREAD) + ITEM]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectBlocked(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:'T) =
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
//        this.LoadDirectBlocked(linear_tid, block_itr, items, valid_items)
//
//    [<ReflectedDefinition>]
//    static member Create(real:RealTraits<'T>, _ITEMS_PER_THREAD:int) =
//        {   ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//            real = real }
//
//    [<ReflectedDefinition>]
//    static member Default(real:RealTraits<'T>) =
//        {   ITEMS_PER_THREAD = 128;
//            real = real }
//
//
//[<Record>]
//type LoadDirectBlockedVectorized<'T> =
//    {
//        ITEMS_PER_THREAD    : int
//        [<RecordExcludedField>] real : RealTraits<'T>
//    }
//
//        
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectBlockedVectorized(linear_tid:int, block_ptr:deviceptr<'T>, items:deviceptr<'T>) =
//        let MAX_VEC_SIZE = CUB_MIN 4 this.ITEMS_PER_THREAD
//        let VEC_SIZE = if (((MAX_VEC_SIZE - 1) &&& MAX_VEC_SIZE) = 0) && ((this.ITEMS_PER_THREAD % MAX_VEC_SIZE) = 0) then MAX_VEC_SIZE else 1
//        let VECTORS_PER_THREAD = this.ITEMS_PER_THREAD / VEC_SIZE
//        let ptr = (block_ptr + (linear_tid * VEC_SIZE * VECTORS_PER_THREAD)) |> __ptr_reinterpret
//
//        let vec_items = __local__.Array<CubVector<'T>>(VECTORS_PER_THREAD) |> __array_to_ptr
//
//        for ITEM = 0 to (VECTORS_PER_THREAD - 1) do vec_items.[ITEM] <- ptr.[ITEM]
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- vec_items.[ITEM].Ptr |> __ptr_to_obj
//
//    [<ReflectedDefinition>]
//    static member Create(real:RealTraits<'T>, _ITEMS_PER_THREAD:int) =
//        {   ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//            real = real }
//
//    [<ReflectedDefinition>]
//    static member Default(real:RealTraits<'T>) =
//        {   ITEMS_PER_THREAD = 128;
//            real = real }
//
//[<Record>]
//type LoadDirectStriped<'T> =
//    {
//        BLOCK_THREADS : int
//        ITEMS_PER_THREAD : int
//        [<RecordExcludedField>] real : RealTraits<'T>
//    }
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>) =
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(ITEM * this.BLOCK_THREADS) + linear_tid]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
//        let bounds = valid_items - linear_tid
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do 
//            if (ITEM * this.BLOCK_THREADS < bounds) then items.[ITEM] <- block_itr.[(ITEM * this.BLOCK_THREADS) + linear_tid]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:'T) =
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
//        this.LoadDirectStriped(linear_tid, block_itr, items, valid_items)
//
//    [<ReflectedDefinition>]
//    static member Create(real:RealTraits<'T>, _BLOCK_THREADS:int, _ITEMS_PER_THREAD:int) =
//        {   BLOCK_THREADS = _BLOCK_THREADS
//            ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//            real = real }
//
//    [<ReflectedDefinition>]
//    static member Default(real:RealTraits<'T>) =
//        {   BLOCK_THREADS = 128;
//            ITEMS_PER_THREAD = 128;
//            real = real }
//
//
//[<Record>]
//type LoadDirectWarpStriped<'T> =
//    {
//        mutable ITEMS_PER_THREAD : int
//        [<RecordExcludedField>] real : RealTraits<'T>
//    }
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectWarpStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>) =
//        let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//        let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
//        let warp_offset = wid * CUB_PTX_WARP_THREADS * this.ITEMS_PER_THREAD
//
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectWarpStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
//        let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//        let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
//        let warp_offset = wid * CUB_PTX_WARP_THREADS * this.ITEMS_PER_THREAD
//        let bounds = valid_items - warp_offset - tid
//
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do 
//            if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectWarpStriped(linear_tid:int, block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int, oob_default:'T) =
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
//        this.LoadDirectWarpStriped(linear_tid, block_itr, items, valid_items)
//
//    [<ReflectedDefinition>]
//    static member Create(real:RealTraits<'T>, _ITEMS_PER_THREAD:int) =
//        {   ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//            real = real }
//
//    [<ReflectedDefinition>]
//    static member Default(real:RealTraits<'T>) =
//        {   ITEMS_PER_THREAD = 128;
//            real = real }