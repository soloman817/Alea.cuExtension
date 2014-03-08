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
        type API =
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
            static member Default(block_threads, items_per_thread) = API.Init(block_threads, items_per_thread, BLOCK_STORE_DIRECT, false)

    module TempStorage =
        [<Record>]
        type API<'T> =
            {
                BlockExchangeStorage    : Alea.cuExtension.CUB.Block.Exchange.Template._TempStorage<'T>
            }

            [<ReflectedDefinition>]
            static member Init(block_threads, items_per_thread, warp_time_slicing) =
                let template = Alea.cuExtension.CUB.Block.Exchange.Template._TemplateParams.Init(block_threads, items_per_thread, warp_time_slicing)
                {
                    BlockExchangeStorage = Alea.cuExtension.CUB.Block.Exchange.Template._TempStorage<'T>.Uninitialized(template)
                }

            [<ReflectedDefinition>] static member Init(p:Params.API) = API<'T>.Init(p.BLOCK_THREADS, p.ITEMS_PER_THREAD, p.WARP_TIME_SLICING)


    module ThreadFields =
        [<Record>]
        type API<'T> =
            {
                temp_storage : TempStorage.API<'T>
            }

            [<ReflectedDefinition>] static member Init(p:Params.API) = { temp_storage = TempStorage.API<'T>.Init(p)}
            

    type _TemplateParams        = Params.API
    type _TempStorage<'T>       = TempStorage.API<'T>
    type _ThreadFields<'T>      = ThreadFields.API<'T>

    [<Record>]
    type API<'T> =
        {
            mutable Params          : Params.API
            mutable ThreadFields    : ThreadFields.API<'T>
        }

        [<ReflectedDefinition>] 
        static member Init(block_threads, items_per_thread, algorithm, warp_time_slicing) =
            let p = Params.API.Init(block_threads, items_per_thread, algorithm, warp_time_slicing)
            let f = ThreadFields.API<'T>.Init(p)
            {
                Params = p
                ThreadFields = f
            }

        [<ReflectedDefinition>]
        static member Init(block_threads, items_per_thread) = API<'T>.Init(block_threads, items_per_thread, BLOCK_STORE_DIRECT, false)


type _Template<'T> = Template.API<'T>

module StoreDirectBlocked =
    open Template

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
            let p = template.Params
            for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM] <- items.[ITEM]
        
    let [<ReflectedDefinition>] inline Guarded (template:_Template<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
        let p = template.Params
        for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do
            if ITEM + (linear_tid * p.ITEMS_PER_THREAD) < valid_items then
                block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM] <- items.[ITEM]



module StoreDirectBlockedVectorized =
    open Template
            
    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
        let p = template.Params
        let MAX_VEC_SIZE = (4,p.ITEMS_PER_THREAD) ||> CUB_MIN
        let VEC_SIZE = if ((((MAX_VEC_SIZE - 1) &&& MAX_VEC_SIZE) = 0) && ((p.ITEMS_PER_THREAD % MAX_VEC_SIZE) = 0)) then MAX_VEC_SIZE else 1
        let VECTORS_PER_THREAD = p.ITEMS_PER_THREAD / VEC_SIZE
        ()



module StoreDirectStriped =
    open Template
            
    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
        let p = template.Params
        for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid] <- items.[ITEM]

    let [<ReflectedDefinition>] inline Guarded (template:_Template<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
        let p = template.Params
        for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do
            if ((ITEM * p.BLOCK_THREADS) + linear_tid) < valid_items then
                block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid] <- items.[ITEM]
    


module StoreDirectWarpStriped =
    open Template
            
    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
        let p = template.Params
        let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
        let wid = linear_tid >>> CUB_PTX_WARP_THREADS
        let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD
            
        for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)] <- items.[ITEM]
    
    let [<ReflectedDefinition>] inline Guarded (template:_Template<'T>)
        (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
        let p = template.Params
        let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
        let wid = linear_tid >>> CUB_PTX_WARP_THREADS
        let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD
            
        for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do
            if (warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS) < valid_items) then
                block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)] <- items.[ITEM]


module StoreInternal =
    open Template

    module BlockStoreDirect =
        let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
            _ (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            StoreDirectBlocked.Default template linear_tid block_ptr items

        let [<ReflectedDefinition>] inline Guarded (template:_Template<'T>)
            _ (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            StoreDirectBlocked.Guarded template linear_tid block_ptr items valid_items

    module BlockStoreVectorized =
        let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
            _ (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            StoreDirectBlockedVectorized.Default template linear_tid block_ptr items

        let [<ReflectedDefinition>] inline Guarded (template:_Template<'T>)
            _ (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            StoreDirectBlocked.Guarded template linear_tid block_ptr items valid_items
            
    module BlockStoreTranspose =
        let [<ReflectedDefinition>] inline BlockedToStriped (template:_Template<'T>)
            (temp_storage:_TempStorage<'T>) (linear_tid:int) =
            let p = template.Params
            if p.WARP_TIME_SLICING then
                BlockExchange.API<'T>.Init(p.BLOCK_THREADS, p.ITEMS_PER_THREAD, p.WARP_TIME_SLICING).BlockToStriped.WithTimeslicing
            else
                BlockExchange.API<'T>.Init(p.BLOCK_THREADS, p.ITEMS_PER_THREAD, p.WARP_TIME_SLICING).BlockToStriped.Default
         
        let [<ReflectedDefinition>] inline Default  (template:_Template<'T>)
            (temp_storage:_TempStorage<'T>) (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            
            StoreDirectStriped.Default template linear_tid block_ptr items
            BlockedToStriped template temp_storage linear_tid items

        let [<ReflectedDefinition>] inline Guarded  (template:_Template<'T>)
            (temp_storage:_TempStorage<'T>) (linear_tid:int) 
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            
            StoreDirectStriped.Guarded template linear_tid block_ptr items valid_items
            BlockedToStriped template temp_storage linear_tid items


    module BlockStoreWarpTranspose =
        let [<ReflectedDefinition>] inline WARP_THREADS block_threads =
            ((block_threads % CUB_PTX_WARP_THREADS) = 0) |> function
            | false -> failwith "BLOCK_THREADS must be a multiple of WARP_THREADS"
            | true -> CUB_PTX_WARP_THREADS

        let [<ReflectedDefinition>] inline BlockedToWarpStriped  (template:_Template<'T>)
            (temp_storage:_TempStorage<'T>) (linear_tid:int) =
            let p = template.Params
            if p.WARP_TIME_SLICING then 
                BlockExchange.API<'T>.Init(p.BLOCK_THREADS, p.ITEMS_PER_THREAD, p.WARP_TIME_SLICING).BlockToWarpStriped.WithTimeslicing
            else
                BlockExchange.API<'T>.Init(p.BLOCK_THREADS, p.ITEMS_PER_THREAD, p.WARP_TIME_SLICING).BlockToWarpStriped.Default


        let [<ReflectedDefinition>] inline Default  (template:_Template<'T>)
            (temp_storage:_TempStorage<'T>) (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            StoreDirectWarpStriped.Default template linear_tid block_ptr items
            BlockedToWarpStriped template temp_storage linear_tid items

        let [<ReflectedDefinition>] inline Guarded  (template:_Template<'T>)
            (temp_storage:_TempStorage<'T>) (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            StoreDirectWarpStriped.Guarded template linear_tid block_ptr items valid_items
            BlockedToWarpStriped template temp_storage linear_tid items


    [<Record>]
    type API<'T> =
        {
            Default : int -> deviceptr<'T> -> deviceptr<'T> -> unit
            Guarded : int -> deviceptr<'T> -> deviceptr<'T> -> int -> unit
        }

        [<ReflectedDefinition>]
        static member Init(template:_Template<'T>) =
            let temp_storage = template.ThreadFields.temp_storage
            let _Default = template.Params.ALGORITHM |> function
                | BLOCK_STORE_DIRECT ->          BlockStoreDirect.Default template temp_storage
                | BLOCK_STORE_VECTORIZE ->       BlockStoreVectorized.Default template temp_storage
                | BLOCK_STORE_TRANSPOSE ->       BlockStoreTranspose.Default template temp_storage
                | BLOCK_STORE_WARP_TRANSPOSE ->  BlockStoreWarpTranspose.Default template temp_storage

            let _Guarded = template.Params.ALGORITHM |> function
                | BLOCK_STORE_DIRECT ->          BlockStoreDirect.Guarded template temp_storage
                | BLOCK_STORE_VECTORIZE ->       BlockStoreVectorized.Guarded template temp_storage
                | BLOCK_STORE_TRANSPOSE ->       BlockStoreTranspose.Guarded template temp_storage
                | BLOCK_STORE_WARP_TRANSPOSE ->  BlockStoreWarpTranspose.Guarded template temp_storage

            { Default = _Default; Guarded = _Guarded }



module BlockStore =
    open Template
    open Internal

    [<Record>]
    type API<'T> =
        {
            template    : _Template<'T>
            Store       : StoreInternal.API<'T>
        }

        [<ReflectedDefinition>]
        static member Create(block_threads, items_per_thread, algorithm, warp_time_slicing) =
            let template = _Template<'T>.Init(block_threads, items_per_thread, algorithm, warp_time_slicing)
            {
                template    = template
                Store       = StoreInternal.API<'T>.Init(template)
            }

        [<ReflectedDefinition>] member this.Default = this.Store.Default
        [<ReflectedDefinition>] member this.Guarded = this.Store.Guarded

//    let [<ReflectedDefinition>] api (template:_Template<'T>) =
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
//       p.ITEMS_PER_THREAD    : int
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
//           p.ITEMS_PER_THREAD    =tp.ITEMS_PER_THREAD
//            ALGORITHM           = algorithm
//            WARP_TIME_SLICING   = warp_time_slicing
//            ThreadFields        = ThreadFields.Default()
//        }
//    
//    [<ReflectedDefinition>]
//    static member Create(block_threads:int,tp.ITEMS_PER_THREAD:int, algorithm:BlockStoreAlgorithm) =
//        {
//            BLOCK_THREADS       = block_threads
//           p.ITEMS_PER_THREAD    =tp.ITEMS_PER_THREAD
//            ALGORITHM           = algorithm
//            WARP_TIME_SLICING   = false
//            ThreadFields        = ThreadFields.Default()
//        }
//    
//    [<ReflectedDefinition>]
//    static member Create(block_threads:int,tp.ITEMS_PER_THREAD:int) =
//        {
//            BLOCK_THREADS       = block_threads
//           p.ITEMS_PER_THREAD    =tp.ITEMS_PER_THREAD
//            ALGORITHM           = BLOCK_STORE_DIRECT
//            WARP_TIME_SLICING   = false
//            ThreadFields        = ThreadFields.Default()
//        }