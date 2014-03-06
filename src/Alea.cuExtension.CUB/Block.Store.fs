[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Store

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities


module Template =
    type BlockStoreAlgorithm =
        | BLOCK_STORE_DIRECT
        | BLOCK_STORE_VECTORIZE
        | BLOCK_STORE_TRANSPOSE
        | BLOCK_STORE_WARP_TRANSPOSE

    [<AutoOpen>]
    module Params =
        [<Record>]
        type API<'T> =
            {
                BLOCK_THREADS       : int
                ITEMS_PER_THREAD    : int
                ALGORITHM           : BlockStoreAlgorithm
                WARP_TIME_SLICING   : bool
            }

            [<ReflectedDefinition>]
            static member Init(block_threads, items_per_thread, algorithm, warp_time_slicing) =
                {
                    BLOCK_THREADS       = block_threads
                    ITEMS_PER_THREAD    = items_per_thread
                    ALGORITHM           = algorithm
                    WARP_TIME_SLICING   = warp_time_slicing
                }

            [<ReflectedDefinition>]
            static member Default(block_threads, items_per_thread) = API<'T>.Init(block_threads, items_per_thread, BLOCK_STORE_DIRECT, false)

    module TempStorage =
        [<Record>]
        type API<'T> =
            {
                BlockExchangeStorage    : Alea.cuExtension.CUB.Block.Exchange.Template._TempStorage<'T>
            }

            [<ReflectedDefinition>]
            static member Init(block_threads, items_per_thread, warp_time_slicing) =
                let tp = Alea.cuExtension.CUB.Block.Exchange.Template._TemplateParams<'T>.Init(block_threads, items_per_thread, warp_time_slicing)
                {
                    BlockExchangeStorage = Alea.cuExtension.CUB.Block.Exchange.Template._TempStorage<'T>.Uninitialized(tp)
                }

    type _TemplateParams<'T>    = Params.API<'T>
    type _TempStorage<'T>       = TempStorage.API<'T>



module private Internal =
    module Sig =
        module StoreDirectBlocked =
            type Default<'T>        = int -> deviceptr<'T> -> deviceptr<'T> -> unit
            type Guarded<'T>        = int -> deviceptr<'T> -> deviceptr<'T> -> int -> unit

        module StoreDirectBlockedVectorized =
            type Default<'T>        = StoreDirectBlocked.Default<'T>
            type Guarded<'T>        = StoreDirectBlocked.Guarded<'T>

        module StoreDirectStriped =
            type Default<'T>        = StoreDirectBlocked.Default<'T>
            type Guarded<'T>        = StoreDirectBlocked.Guarded<'T>
            
        module StoreDirectWarpStriped =
            type Default<'T>        = StoreDirectBlocked.Default<'T>
            type Guarded<'T>        = StoreDirectBlocked.Guarded<'T>
            
        module StoreInternal =
            type Default<'T>        = int -> deviceptr<'T> -> deviceptr<'T> -> unit
            type Guarded<'T>        = int -> deviceptr<'T> -> deviceptr<'T> -> int -> unit
            

module StoreDirectBlocked =
    open Template
    open Internal

    type API<'T> =
        {
            Default : Sig.StoreDirectBlocked.Default<'T>
            Guarded : Sig.StoreDirectBlocked.Guarded<'T>
        }


    let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
            for ITEM = 0 to tp.ITEMS_PER_THREAD - 1 do block_itr.[(linear_tid * tp.ITEMS_PER_THREAD) + ITEM] <- items.[ITEM]
        
    let [<ReflectedDefinition>] inline Guarded (tp:_TemplateParams<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
        for ITEM = 0 to tp.ITEMS_PER_THREAD - 1 do
            if ITEM + (linear_tid *tp.ITEMS_PER_THREAD) < valid_items then
                block_itr.[(linear_tid *tp.ITEMS_PER_THREAD) + ITEM] <- items.[ITEM]

    let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>)  =
        {
            Default =           Default tp
            Guarded =           Guarded tp
        }


module StoreDirectBlockedVectorized =
    open Template
    open Internal

    type API<'T> =
        {
            Default : Sig.StoreDirectBlockedVectorized.Default<'T>
            Guarded : Sig.StoreDirectBlockedVectorized.Guarded<'T>
        }
            
    let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>)
        (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
        let MAX_VEC_SIZE = (4,tp.ITEMS_PER_THREAD) ||> CUB_MIN
        let VEC_SIZE = if ((((MAX_VEC_SIZE - 1) &&& MAX_VEC_SIZE) = 0) && ((tp.ITEMS_PER_THREAD % MAX_VEC_SIZE) = 0)) then MAX_VEC_SIZE else 1
        let VECTORS_PER_THREAD = tp.ITEMS_PER_THREAD / VEC_SIZE
        ()


    let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>) =
        {
            Default =   Default tp
            Guarded =   (StoreDirectBlocked.api tp).Guarded
        }


module StoreDirectStriped =
    open Template
    open Internal

    type API<'T> =
        {
            Default : Sig.StoreDirectStriped.Default<'T>
            Guarded : Sig.StoreDirectStriped.Guarded<'T>
        }
            
    let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
        for ITEM = 0 to tp.ITEMS_PER_THREAD - 1 do block_itr.[(ITEM * tp.BLOCK_THREADS) + linear_tid] <- items.[ITEM]
        

    let [<ReflectedDefinition>] inline Guarded (tp:_TemplateParams<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
        for ITEM = 0 to tp.ITEMS_PER_THREAD - 1 do
            if ((ITEM * tp.BLOCK_THREADS) + linear_tid) < valid_items then
                block_itr.[(ITEM * tp.BLOCK_THREADS) + linear_tid] <- items.[ITEM]
    

    let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>) =
        {
            Default =           Default tp
            Guarded =           Guarded tp
        }


module StoreDirectWarpStriped =
    open Template
    open Internal

    type API<'T> =
        {
            Default : Sig.StoreDirectWarpStriped.Default<'T>
            Guarded : Sig.StoreDirectWarpStriped.Guarded<'T>
        }
            
    let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
        let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
        let wid = linear_tid >>> CUB_PTX_WARP_THREADS
        let warp_offset = wid * CUB_PTX_WARP_THREADS * tp.ITEMS_PER_THREAD
            
        for ITEM = 0 to tp.ITEMS_PER_THREAD - 1 do block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)] <- items.[ITEM]
            
    
    let [<ReflectedDefinition>] inline Guarded (tp:_TemplateParams<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
        let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
        let wid = linear_tid >>> CUB_PTX_WARP_THREADS
        let warp_offset = wid * CUB_PTX_WARP_THREADS *tp.ITEMS_PER_THREAD
            
        for ITEM = 0 to tp.ITEMS_PER_THREAD - 1 do
            if (warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS) < valid_items) then
                block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)] <- items.[ITEM]

    let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>) =
        {
            Default =           Default tp
            Guarded =           Guarded tp
        }

module private StoreInternal =
    open Template
    open Internal

    type API<'T> =
        {
            Default         : Sig.StoreInternal.Default<'T>
            Guarded         : Sig.StoreInternal.Guarded<'T>
        }

    module BlockStoreDirect =
        let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>)
            (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            (StoreDirectBlocked.api tp).Default linear_tid block_ptr items

        let [<ReflectedDefinition>] inline Guarded (tp:_TemplateParams<'T>)
            (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            (StoreDirectBlocked.api tp).Guarded linear_tid block_ptr items valid_items


    module BlockStoreVectorized =
        let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>)
            (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            (StoreDirectBlockedVectorized.api tp).Default linear_tid block_ptr items

        let [<ReflectedDefinition>] inline Guarded (tp:_TemplateParams<'T>)
            (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            (StoreDirectBlocked.api tp).Guarded linear_tid block_ptr items valid_items


            
    module BlockStoreTranspose =
        
        let [<ReflectedDefinition>] inline BlockedToStriped (tp:_TemplateParams<'T>)
            (temp_storage:_TempStorage<'T>) (linear_tid:int) =
                let bts =    
                    (   BlockExchange.api
                        <|  Alea.cuExtension.CUB.Block.Exchange.Template._TemplateParams<'T>.Init(tp.BLOCK_THREADS, tp.ITEMS_PER_THREAD, tp.WARP_TIME_SLICING)
                        <|  Alea.cuExtension.CUB.Block.Exchange.Template._ThreadFields<'T>.Init<'T>(temp_storage.BlockExchangeStorage, linear_tid, 0, 0, 0)
                    ).BlockedToStriped
                if tp.WARP_TIME_SLICING then bts.WithTimeslicing else bts.Default
        
        
        let [<ReflectedDefinition>] inline Default  (tp:_TemplateParams<'T>)
            (temp_storage:_TempStorage<'T>) (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            
            (StoreDirectStriped.api tp).Default linear_tid block_ptr items
            (BlockedToStriped tp temp_storage linear_tid) items

        let [<ReflectedDefinition>] inline Guarded  (tp:_TemplateParams<'T>)
            (temp_storage:_TempStorage<'T>) (linear_tid:int) 
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            
            (StoreDirectStriped.api tp).Guarded linear_tid block_ptr items valid_items
            (BlockedToStriped tp temp_storage linear_tid) items


    module BlockStoreWarpTranspose =
        
        let [<ReflectedDefinition>] inline WARP_THREADS block_threads =
            ((block_threads % CUB_PTX_WARP_THREADS) = 0) |> function
            | false -> failwith "BLOCK_THREADS must be a multiple of WARP_THREADS"
            | true -> CUB_PTX_WARP_THREADS

        let [<ReflectedDefinition>] inline BlockedToWarpStriped  (tp:_TemplateParams<'T>)
            (temp_storage:_TempStorage<'T>) (linear_tid:int) =
                let wbts =    
                    (   BlockExchange.api
                        <|  Alea.cuExtension.CUB.Block.Exchange.Template._TemplateParams<'T>.Init(tp.BLOCK_THREADS, tp.ITEMS_PER_THREAD, tp.WARP_TIME_SLICING)
                        <|  Alea.cuExtension.CUB.Block.Exchange.Template._ThreadFields<'T>.Init<'T>(temp_storage.BlockExchangeStorage, linear_tid, 0, 0, 0)
                    ).BlockedToWarpStriped
                if tp.WARP_TIME_SLICING then wbts.WithTimeslicing else wbts.Default

        let [<ReflectedDefinition>] inline Default  (tp:_TemplateParams<'T>)
            (temp_storage:_TempStorage<'T>) (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            (StoreDirectWarpStriped.api tp).Default linear_tid block_ptr items
            (BlockedToWarpStriped tp temp_storage linear_tid) items

        let [<ReflectedDefinition>] inline Guarded  (tp:_TemplateParams<'T>)
            (temp_storage:_TempStorage<'T>) (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            (StoreDirectWarpStriped.api tp).Guarded linear_tid block_ptr items valid_items
            (BlockedToWarpStriped tp temp_storage linear_tid) items

module BlockStore =
    open Template
    open Internal

    [<Record>]
    type API<'T> =
        {
            TempStorage     : _TempStorage<'T>
            Default         : Sig.StoreInternal.Default<'T>
            Guarded         : Sig.StoreInternal.Guarded<'T>
        }

        [<ReflectedDefinition>]
        static member Create(block_threads, items_per_thread, algorithm, warp_time_slicing) =
            let tp = _TemplateParams<'T>.Init(block_threads, items_per_thread, algorithm, warp_time_slicing)
            let temp_storage, _Default, _Guarded =
                let ts = _TempStorage<'T>.Init(block_threads, items_per_thread, warp_time_slicing)
                algorithm |> function
                | BLOCK_STORE_DIRECT ->
                    (
                        ts,
                        StoreInternal.BlockStoreDirect.Default tp,
                        StoreInternal.BlockStoreDirect.Guarded tp
                    )
                | BLOCK_STORE_VECTORIZE ->
                    (   
                        ts,
                        StoreInternal.BlockStoreVectorized.Default tp,                        
                        StoreInternal.BlockStoreVectorized.Guarded tp
                    )

                | BLOCK_STORE_TRANSPOSE ->
                    (   
                        ts,
                        StoreInternal.BlockStoreTranspose.Default tp ts,                        
                        StoreInternal.BlockStoreTranspose.Guarded tp ts
                    )

                | BLOCK_STORE_WARP_TRANSPOSE ->
                    (   
                        ts,
                        StoreInternal.BlockStoreWarpTranspose.Default tp ts,                    
                        StoreInternal.BlockStoreWarpTranspose.Guarded tp ts      
                    )
            {
                TempStorage = _TempStorage<'T>.Init(block_threads, items_per_thread, warp_time_slicing)
                Default     = _Default
                Guarded     = _Guarded
            }

//    let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>) =
//        let warp_time_slicing = if warp_time_slicing.IsNone then false else warp_time_slicing.Value
//        fun temp_storage linear_tid ->
//            let _Default, _Guarded =
//                algorithm |> function
//                | BLOCK_STORE_DIRECT -> 
//                    (   
//                        StoreInternal.BlockStoreDirect.Default
//                        <|||    (block_threads,tp.ITEMS_PER_THREAD, warp_time_slicing)
//                        <||     (temp_storage, linear_tid),
//                        StoreInternal.BlockStoreDirect.Guarded
//                        <|||    (block_threads,tp.ITEMS_PER_THREAD, warp_time_slicing)
//                        <||     (temp_storage, linear_tid)
//                    )
//
//                | BLOCK_STORE_VECTORIZE ->
//                    (   
//                        StoreInternal.BlockStoreVectorized.Default
//                        <|||    (block_threads,tp.ITEMS_PER_THREAD, warp_time_slicing)
//                        <||     (temp_storage, linear_tid),                        
//                        StoreInternal.BlockStoreVectorized.Guarded
//                        <|||    (block_threads,tp.ITEMS_PER_THREAD, warp_time_slicing)
//                        <||     (temp_storage, linear_tid)
//                    )
//
//                | BLOCK_STORE_TRANSPOSE ->
//                    (   
//                        StoreInternal.BlockStoreTranspose.Default
//                        <|||    (block_threads,tp.ITEMS_PER_THREAD, warp_time_slicing)
//                        <||     (temp_storage, linear_tid),                        
//                        StoreInternal.BlockStoreTranspose.Guarded
//                        <|||    (block_threads,tp.ITEMS_PER_THREAD, warp_time_slicing)
//                        <||     (temp_storage, linear_tid)
//                    )
//
//                | BLOCK_STORE_WARP_TRANSPOSE ->
//                    (   
//                        StoreInternal.BlockStoreWarpTranspose.Default
//                        <|||    (block_threads,tp.ITEMS_PER_THREAD, warp_time_slicing)
//                        <||     (temp_storage, linear_tid),                    
//                        StoreInternal.BlockStoreWarpTranspose.Guarded
//                        <|||    (block_threads,tp.ITEMS_PER_THREAD, warp_time_slicing)
//                        <||     (temp_storage, linear_tid)                    
//                    )
//
//            {Default = _Default; Guarded = _Guarded}
           


//module InternalStore =
//
//    let [<ReflectedDefinition>] api (block_threads:int) (items_per_thread:int) = 
//            {   StoreDirectBlocked           = (block_threads,tp.ITEMS_PER_THREAD) ||> StoreDirectBlocked.api;
//                StoreDirectBlockedVectorized = (block_threads,tp.ITEMS_PER_THREAD) ||> StoreDirectBlockedVectorized.api
//                StoreDirectStriped           = (block_threads,tp.ITEMS_PER_THREAD) ||> StoreDirectStriped.api
//                StoreDirectWarpStriped       = (block_threads,tp.ITEMS_PER_THREAD) ||> StoreDirectWarpStriped.api }
//
//
//    let store (block_threads:int) (items_per_thread:int) (algorithm:BlockStoreAlgorithm) =
//        let [<ReflectedDefinition>] api = (block_threads,tp.ITEMS_PER_THREAD) ||> api
//        algorithm |> function
//        | BLOCK_STORE_DIRECT ->         api.StoreDirectBlocked
//        | BLOCK_STORE_VECTORIZE ->      api.StoreDirectBlockedVectorized
//        | BLOCK_STORE_TRANSPOSE ->      api.StoreDirectStriped
//        | BLOCK_STORE_WARP_TRANSPOSE -> api.StoreDirectWarpStriped
//
//
//let inline blockStore (block_threads:int) (items_per_thread:int) (algorithm:BlockStoreAlgorithm) (warp_time_slicing:bool) =
//    let storeInternal = (block_threads,tp.ITEMS_PER_THREAD, algorithm) |||> InternalStore.store
//    { new BlockStoreAPI with
//        member this.Default =           cuda { return! storeInternal.Default         |> Compiler.DefineFunction}
//        member this.Guarded =           cuda { return! storeInternal.Guarded         |> Compiler.DefineFunction}
//    }
//let storeDirectBlocked (block_threads:int) (items_per_thread:int) =
//    fun (valid_items:int option) ->
//        valid_items |> function
//        | None ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//                for ITEM = 0 totp.ITEMS_PER_THREAD - 1 do block_itr.[(linear_tid *tp.ITEMS_PER_THREAD) + ITEM] <- items.[ITEM]
//        | Some valid_items ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//                for ITEM = 0 totp.ITEMS_PER_THREAD - 1 do
//                    if ITEM + (linear_tid *tp.ITEMS_PER_THREAD) < valid_items then
//                        block_itr.[(linear_tid *tp.ITEMS_PER_THREAD) + ITEM] <- items.[ITEM]
//
//let storeBlockedVectorized (block_threads:int) (items_per_thread:int) =
//    let MAX_VEC_SIZE = (4,tp.ITEMS_PER_THREAD) ||> CUB_MIN
//    let VEC_SIZE = if ((((MAX_VEC_SIZE - 1) &&& MAX_VEC_SIZE) = 0) && ((items_per_thread % MAX_VEC_SIZE) = 0)) then MAX_VEC_SIZE else 1
//    let VECTORS_PER_THREAD =tp.ITEMS_PER_THREAD / VEC_SIZE
//    fun _ ->
//        fun (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) -> ()
//
//let storeDirectStriped (block_threads:int) (items_per_thread:int) =
//    fun (valid_items:int option) ->
//        valid_items |> function
//        | None ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//               for ITEM = 0 totp.ITEMS_PER_THREAD - 1 do block_itr.[(ITEM * block_threads) + linear_tid] <- items.[ITEM]
//
//        | Some valid_items ->
//            fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//                for ITEM = 0 totp.ITEMS_PER_THREAD - 1 do
//                    if ((ITEM * block_threads) + linear_tid) < valid_items then
//                        block_itr.[(ITEM * block_threads) + linear_tid] <- items.[ITEM]
//
//
//let storeDirectWarpStriped (block_threads:int) (items_per_thread:int) =
//    fun (valid_items:int option) ->
//        fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
//            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//            let wid = linear_tid >>> CUB_PTX_WARP_THREADS
//            let warp_offset = wid * CUB_PTX_WARP_THREADS *tp.ITEMS_PER_THREAD
//
//            valid_items |> function
//            | None ->
//                for ITEM = 0 totp.ITEMS_PER_THREAD - 1 do block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)] <- items.[ITEM]
//            | Some valid_items ->
//                for ITEM = 0 totp.ITEMS_PER_THREAD - 1 do
//                    if (warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS) < valid_items) then
//                        block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)] <- items.[ITEM]
//
//
//let storeInternal (algorithm:BlockStoreAlgorithm) =
//    fun (block_threads:int) (items_per_thread:int) ->
//        algorithm |> function
//        | BLOCK_STORE_DIRECT ->         (block_threads,tp.ITEMS_PER_THREAD) ||> storeDirectBlocked
//        | BLOCK_STORE_VECTORIZE ->      (block_threads,tp.ITEMS_PER_THREAD) ||> storeBlockedVectorized
//        | BLOCK_STORE_TRANSPOSE ->      (block_threads,tp.ITEMS_PER_THREAD) ||> storeDirectStriped
//        | BLOCK_STORE_WARP_TRANSPOSE -> (block_threads,tp.ITEMS_PER_THREAD) ||> storeDirectWarpStriped
//
//
//let blockStore (block_threads:int) (items_per_thread:int) (algorithm:BlockStoreAlgorithm) (warp_time_slicing:bool) =
//    //let BlockExchange = BlockExchange.Create(block_threads,tp.ITEMS_PER_THREAD, warp_time_slicing)
//    
//    algorithm |> function
//    | BLOCK_STORE_DIRECT ->
//        fun _ (linear_tid:int) ->
//            fun (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int option) ->
//                algorithm |> storeInternal
//                <||     (block_threads,tp.ITEMS_PER_THREAD)
//                <|      (valid_items)
//                <|||    (linear_tid, block_itr, items)
//
//    | BLOCK_STORE_VECTORIZE ->
//        fun _ (linear_tid:int) ->
//            fun (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int option) ->
//                algorithm |> storeInternal
//                <||     (block_threads,tp.ITEMS_PER_THREAD)
//                <|      (valid_items)
//                <|||    (linear_tid, block_itr, items)
//    
//    | BLOCK_STORE_TRANSPOSE ->
//        let blockedToStriped = (block_threads,tp.ITEMS_PER_THREAD, warp_time_slicing) |||> Exchange.blockedToStriped
//        
//        fun (temp_storage:deviceptr<'T>) (linear_tid:int) ->
//            let blockedToStriped = (temp_storage, linear_tid) ||> blockedToStriped
//
//            fun (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int option) ->
//                items |> blockedToStriped
//                algorithm |> storeInternal
//                <||     (block_threads,tp.ITEMS_PER_THREAD)
//                <|      (valid_items)
//                <|||    (linear_tid, block_itr, items)    
//    
//    | BLOCK_STORE_WARP_TRANSPOSE ->   
//        let blockedToWarpStriped = (block_threads,tp.ITEMS_PER_THREAD, warp_time_slicing) |||> Exchange.blockedToWarpStriped
//        
//        fun (temp_storage:deviceptr<'T>) (linear_tid:int) ->
//            let blockedToWarpStriped = (temp_storage, linear_tid) ||> blockedToWarpStriped
//            
//            fun (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int option) ->
//                items |> blockedToWarpStriped
//                algorithm |> storeInternal
//                <||     (block_threads,tp.ITEMS_PER_THREAD)
//                <|      (valid_items)
//                <|||    (linear_tid, block_itr, items)
//
//[<Record>]
//type ThreadFields =
//    {
//        mutable temp_storage    : deviceptr<'T>
//        mutable linear_tid      : int
//    }
//
//    [<ReflectedDefinition>]
//    member this.Get() = (this.temp_storage, this.linear_tid)
//
//    [<ReflectedDefinition>]
//    static member Default() =
//        {
//            temp_storage    = PrivateStorage()
//            linear_tid      = threadIdx.x
//        }
//
////    static member Init(temp_storage:deviceptr<'T>) =
////        {
////            temp_storage    = temp_storage
////            linear_tid      = threadIdx.x
////        }
////
////    static member Init(linear_tid:int) =
////        {
////            temp_storage    = PrivateStorage()
////            linear_tid      = linear_tid
////        }
//    [<ReflectedDefinition>]
//    static member Init(temp_storage:deviceptr<'T>, linear_tid:int) =
//        {
//            temp_storage    = temp_storage
//            linear_tid      = linear_tid
//        }
//        
//
//[<Record>]
//type BlockStore =
//    {
//        // Template Parameters
//        BLOCK_THREADS       : int
//       tp.ITEMS_PER_THREAD    : int
//        ALGORITHM           : BlockStoreAlgorithm
//        WARP_TIME_SLICING   : bool
//        ThreadFields        : ThreadFields
//    }
//
//
//    [<ReflectedDefinition>]
//    member this.Initialize() =
//        this.ThreadFields.temp_storage    <- PrivateStorage()
//        this.ThreadFields.linear_tid      <- threadIdx.x
//        this
//
//    [<ReflectedDefinition>]
//    member this.Initialize(temp_storage:deviceptr<'T>) =
//        this.ThreadFields.temp_storage  <- temp_storage
//        this.ThreadFields.linear_tid    <- threadIdx.x
//        this
//
//    [<ReflectedDefinition>]
//    member this.Initialize(linear_tid:int) =
//        this.ThreadFields.temp_storage    <- PrivateStorage()
//        this.ThreadFields.linear_tid      <- linear_tid
//        this
//
//    [<ReflectedDefinition>]
//    member this.Initialize(temp_storage:deviceptr<'T>, linear_tid:int) =
//        this.ThreadFields.temp_storage    <- temp_storage
//        this.ThreadFields.linear_tid      <- linear_tid
//        this
//
//    [<ReflectedDefinition>]
//    member this.Store(block_itr:deviceptr<'T>, items:deviceptr<'T>) =
//        (blockStore this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items)
//            <|  None
//
//    
//    [<ReflectedDefinition>]
//    member this.Store(block_itr:deviceptr<'T>, items:deviceptr<'T>, valid_items:int) =
//        (blockStore this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items)
//            <|  Some valid_items
//
//    [<ReflectedDefinition>]
//    static member Create(block_threads:int,tp.ITEMS_PER_THREAD:int, algorithm:BlockStoreAlgorithm, warp_time_slicing:bool) =
//        {
//            BLOCK_THREADS       = block_threads
//           tp.ITEMS_PER_THREAD    =tp.ITEMS_PER_THREAD
//            ALGORITHM           = algorithm
//            WARP_TIME_SLICING   = warp_time_slicing
//            ThreadFields        = ThreadFields.Default()
//        }
//    
//    [<ReflectedDefinition>]
//    static member Create(block_threads:int,tp.ITEMS_PER_THREAD:int, algorithm:BlockStoreAlgorithm) =
//        {
//            BLOCK_THREADS       = block_threads
//           tp.ITEMS_PER_THREAD    =tp.ITEMS_PER_THREAD
//            ALGORITHM           = algorithm
//            WARP_TIME_SLICING   = false
//            ThreadFields        = ThreadFields.Default()
//        }
//    
//    [<ReflectedDefinition>]
//    static member Create(block_threads:int,tp.ITEMS_PER_THREAD:int) =
//        {
//            BLOCK_THREADS       = block_threads
//           tp.ITEMS_PER_THREAD    =tp.ITEMS_PER_THREAD
//            ALGORITHM           = BLOCK_STORE_DIRECT
//            WARP_TIME_SLICING   = false
//            ThreadFields        = ThreadFields.Default()
//        }