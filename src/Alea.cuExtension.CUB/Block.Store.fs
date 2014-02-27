[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Store

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

type BlockStoreAlgorithm =
    | BLOCK_STORE_DIRECT
    | BLOCK_STORE_VECTORIZE
    | BLOCK_STORE_TRANSPOSE
    | BLOCK_STORE_WARP_TRANSPOSE

type TemplateParameters =
    {
        BLOCK_THREADS       : int
        ITEMS_PER_THREAD    : int
        ALGORITHM           : BlockStoreAlgorithm
        WARP_TIME_SLICING   : bool
    }

    static member Default(block_threads, items_per_thread) =
        {
            BLOCK_THREADS       = block_threads
            ITEMS_PER_THREAD    = items_per_thread
            ALGORITHM           = BLOCK_STORE_DIRECT
            WARP_TIME_SLICING   = false
        }

module private Internal =
    module Sig =
        module StoreDirectBlocked =
            type DefaultExpr        = Expr<int -> deviceptr<int> -> deviceptr<int> -> unit>
            type GuardedExpr        = Expr<int -> deviceptr<int> -> deviceptr<int> -> int -> unit>

        module StoreDirectBlockedVectorized =
            type DefaultExpr        = Expr<int -> deviceptr<int> -> deviceptr<int> -> unit>
            type GuardedExpr        = Expr<int -> deviceptr<int> -> deviceptr<int> -> int -> unit>

        module StoreDirectStriped =
            type DefaultExpr        = Expr<int -> deviceptr<int> -> deviceptr<int> -> unit>
            type GuardedExpr        = Expr<int -> deviceptr<int> -> deviceptr<int> -> int -> unit>
            
        module StoreDirectWarpStriped =
            type DefaultExpr        = Expr<int -> deviceptr<int> -> deviceptr<int> -> unit>
            type GuardedExpr        = Expr<int -> deviceptr<int> -> deviceptr<int> -> int -> unit>
            
        module StoreInternal =
            type DefaultExpr        = Expr<deviceptr<int> -> deviceptr<int> -> unit>
            type GuardedExpr        = Expr<deviceptr<int> -> deviceptr<int> -> int -> unit>
            

module StoreDirectBlocked =
    open Internal

    type API =
        {
            Default : Sig.StoreDirectBlocked.DefaultExpr
            Guarded : Sig.StoreDirectBlocked.GuardedExpr
        }


    let private Default _ items_per_thread =
        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
            for ITEM = 0 to items_per_thread - 1 do block_itr.[(linear_tid * items_per_thread) + ITEM] <- items.[ITEM]
        @>

    let private Guarded _ items_per_thread =
        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
            for ITEM = 0 to items_per_thread - 1 do
                if ITEM + (linear_tid * items_per_thread) < valid_items then
                    block_itr.[(linear_tid * items_per_thread) + ITEM] <- items.[ITEM]
        @>

    let api block_threads items_per_thread  =
        {
            Default =           Default
                                <|| (block_threads, items_per_thread)

            Guarded =           Guarded
                                <|| (block_threads, items_per_thread)
        }


module StoreDirectBlockedVectorized =
    open Internal

    type API =
        {
            Default : Sig.StoreDirectBlockedVectorized.DefaultExpr
        }
            
    let private Default _ items_per_thread =
        let MAX_VEC_SIZE = (4, items_per_thread) ||> CUB_MIN
        let VEC_SIZE = if ((((MAX_VEC_SIZE - 1) &&& MAX_VEC_SIZE) = 0) && ((items_per_thread % MAX_VEC_SIZE) = 0)) then MAX_VEC_SIZE else 1
        let VECTORS_PER_THREAD = items_per_thread / VEC_SIZE
        
        <@ fun (linear_tid:int) (block_ptr:deviceptr<int>) (items:deviceptr<int>) -> () @>


    let api block_threads items_per_thread  =
        {
            Default =           Default
                                <|| (block_threads, items_per_thread)
        }


module StoreDirectStriped =
    open Internal

    type API =
        {
            Default : Sig.StoreDirectStriped.DefaultExpr
            Guarded : Sig.StoreDirectStriped.GuardedExpr
        }
            
    let private Default block_threads items_per_thread =
        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
            for ITEM = 0 to items_per_thread - 1 do block_itr.[(ITEM * block_threads) + linear_tid] <- items.[ITEM]
        @>

    let private Guarded block_threads items_per_thread =
        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
            for ITEM = 0 to items_per_thread - 1 do
                if ((ITEM * block_threads) + linear_tid) < valid_items then
                    block_itr.[(ITEM * block_threads) + linear_tid] <- items.[ITEM]
        @>

    let api block_threads items_per_thread  =
        {
            Default =           Default
                                <|| (block_threads, items_per_thread)

            Guarded =           Guarded
                                <|| (block_threads, items_per_thread)
        }


module StoreDirectWarpStriped =
    open Internal

    type API =
        {
            Default : Sig.StoreDirectWarpStriped.DefaultExpr
            Guarded : Sig.StoreDirectWarpStriped.GuardedExpr
        }
            
    let private Default _ items_per_thread =
        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * items_per_thread
            
            for ITEM = 0 to items_per_thread - 1 do block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)] <- items.[ITEM]
        @>    
            
    
    let private Guarded _ items_per_thread =
        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * items_per_thread
            
            for ITEM = 0 to items_per_thread - 1 do
                if (warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS) < valid_items) then
                    block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)] <- items.[ITEM]
        @>

    let api block_threads items_per_thread  =
        {
            Default =           Default
                                <|| (block_threads, items_per_thread)

            Guarded =           Guarded
                                <|| (block_threads, items_per_thread)
        }

module private StoreInternal =
    open Internal

    type API =
        {
            Default         : Sig.StoreInternal.DefaultExpr
            Guarded         : Sig.StoreInternal.GuardedExpr
        }

    module BlockStoreDirect =
        let Default block_threads items_per_thread _ =
            fun _ linear_tid ->
                let StoreDirectBlocked =
                    (   StoreDirectBlocked.api
                        <|| (block_threads, items_per_thread)
                    ).Default
                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) ->
                    %StoreDirectBlocked
                    <|||    (linear_tid, block_ptr, items)
                @>

        let Guarded block_threads items_per_thread _ =
            fun _ linear_tid ->
                let StoreDirectBlocked = 
                    (   StoreDirectBlocked.api
                        <|| (block_threads, items_per_thread)
                    ).Guarded
                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
                    %StoreDirectBlocked
                    <|  (linear_tid)
                    <|| (block_ptr, items)
                    <|  (valid_items)
                @>


    module BlockStoreVectorized =
        let Default block_threads items_per_thread _ =
            fun _ linear_tid ->
                let StoreDirectBlockedVectorized =  
                    (   StoreDirectBlockedVectorized.api
                        <|| (block_threads, items_per_thread)
                    ).Default
                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) ->
                    %StoreDirectBlockedVectorized
                    <|||    (linear_tid, block_ptr, items)
                @>

        let Guarded block_threads items_per_thread _ =
            fun _ linear_tid ->
                let StoreDirectBlocked =
                    (   StoreDirectBlocked.api
                        <|| (block_threads, items_per_thread)
                    ).Guarded
                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
                    %StoreDirectBlocked
                    <|  (linear_tid)
                    <|| (block_ptr, items)
                    <|  (valid_items)
                @>



    module BlockStoreTranspose =
        let private BlockedToStriped block_threads items_per_thread warp_time_slicing =
            fun temp_storage linear_tid ->
                let bts =    
                    (   BlockExchange.api
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid)
                        <|||    (0, 0, 0)
                    ).BlockedToStriped
                if warp_time_slicing then bts.WithTimeslicing else bts.Default
        
        
        let Default block_threads items_per_thread warp_time_slicing =
            fun temp_storage linear_tid ->
                
                let BlockedToStriped =  BlockedToStriped
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                
                let StoreDirectStriped =  
                    (   StoreDirectStriped.api
                        <|| (block_threads, items_per_thread)
                    ).Default

                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) ->
                    %StoreDirectStriped
                    <|||    (linear_tid, block_ptr, items)
                    
                    %BlockedToStriped
                    <|      (items)
                @>

        let Guarded block_threads items_per_thread warp_time_slicing =
            fun temp_storage linear_tid ->
                
                let BlockedToStriped =  BlockedToStriped
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                
                let StoreDirectStriped =  
                    (   StoreDirectStriped.api
                        <|| (block_threads, items_per_thread)
                    ).Guarded

                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
                    %StoreDirectStriped
                    <|  (linear_tid)
                    <|| (block_ptr, items)
                    <|  (valid_items)

                    %BlockedToStriped
                    <|      (items)
                @>


    module BlockStoreWarpTranspose =
        let private WARP_THREADS block_threads =
            ((block_threads % CUB_PTX_WARP_THREADS) = 0) |> function
            | false -> failwith "BLOCK_THREADS must be a multiple of WARP_THREADS"
            | true -> CUB_PTX_WARP_THREADS

        let private BlockedToWarpStriped block_threads items_per_thread warp_time_slicing =
            fun temp_storage linear_tid ->
                let wbts =    
                    (   BlockExchange.api
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid)
                        <|||    (0, 0, 0)
                    ).BlockedToWarpStriped
                if warp_time_slicing then wbts.WithTimeslicing else wbts.Default

        let Default block_threads items_per_thread warp_time_slicing =
            fun temp_storage linear_tid ->
                
                let BlockedToWarpStriped =  BlockedToWarpStriped
                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
                                            <||     (temp_storage, linear_tid)
                
                let StoreDirectWarpStriped =  
                    (   StoreDirectWarpStriped.api
                        <|| (block_threads, items_per_thread)
                    ).Default

                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) ->
                    %StoreDirectWarpStriped
                    <|||    (linear_tid, block_ptr, items)
                    
                    %BlockedToWarpStriped
                    <|      (items)
                @>

        let Guarded block_threads items_per_thread warp_time_slicing =
            fun temp_storage linear_tid ->
                
                let BlockedToWarpStriped =  BlockedToWarpStriped
                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
                                            <||     (temp_storage, linear_tid)
                
                let StoreDirectWarpStriped =  
                    (   StoreDirectWarpStriped.api
                        <|| (block_threads, items_per_thread)
                    ).Guarded

                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
                    %StoreDirectWarpStriped
                    <|  (linear_tid)
                    <|| (block_ptr, items)
                    <|  (valid_items)

                    %BlockedToWarpStriped
                    <|      (items)
                @>

module BlockStore =
    open Internal

    type API =
        {
            Default         : Sig.StoreInternal.DefaultExpr
            Guarded         : Sig.StoreInternal.GuardedExpr
        }

    let api block_threads items_per_thread algorithm (warp_time_slicing:bool option) =
        let warp_time_slicing = if warp_time_slicing.IsNone then false else warp_time_slicing.Value
        fun temp_storage linear_tid ->
            let _Default, _Guarded =
                algorithm |> function
                | BLOCK_STORE_DIRECT -> 
                    (   
                        StoreInternal.BlockStoreDirect.Default
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid),
                        StoreInternal.BlockStoreDirect.Guarded
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid)
                    )

                | BLOCK_STORE_VECTORIZE ->
                    (   
                        StoreInternal.BlockStoreVectorized.Default
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid),                        
                        StoreInternal.BlockStoreVectorized.Guarded
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid)
                    )

                | BLOCK_STORE_TRANSPOSE ->
                    (   
                        StoreInternal.BlockStoreTranspose.Default
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid),                        
                        StoreInternal.BlockStoreTranspose.Guarded
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid)
                    )

                | BLOCK_STORE_WARP_TRANSPOSE ->
                    (   
                        StoreInternal.BlockStoreWarpTranspose.Default
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid),                    
                        StoreInternal.BlockStoreWarpTranspose.Guarded
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid)                    
                    )

            {Default = _Default; Guarded = _Guarded}
           


//module InternalStore =
//
//    let api (block_threads:int) (items_per_thread:int) = 
//            {   StoreDirectBlocked           = (block_threads, items_per_thread) ||> StoreDirectBlocked.api;
//                StoreDirectBlockedVectorized = (block_threads, items_per_thread) ||> StoreDirectBlockedVectorized.api
//                StoreDirectStriped           = (block_threads, items_per_thread) ||> StoreDirectStriped.api
//                StoreDirectWarpStriped       = (block_threads, items_per_thread) ||> StoreDirectWarpStriped.api }
//
//
//    let store (block_threads:int) (items_per_thread:int) (algorithm:BlockStoreAlgorithm) =
//        let api = (block_threads, items_per_thread) ||> api
//        algorithm |> function
//        | BLOCK_STORE_DIRECT ->         api.StoreDirectBlocked
//        | BLOCK_STORE_VECTORIZE ->      api.StoreDirectBlockedVectorized
//        | BLOCK_STORE_TRANSPOSE ->      api.StoreDirectStriped
//        | BLOCK_STORE_WARP_TRANSPOSE -> api.StoreDirectWarpStriped
//
//
//let inline blockStore (block_threads:int) (items_per_thread:int) (algorithm:BlockStoreAlgorithm) (warp_time_slicing:bool) =
//    let storeInternal = (block_threads, items_per_thread, algorithm) |||> InternalStore.store
//    { new BlockStoreAPI with
//        member this.Default =           cuda { return! storeInternal.Default         |> Compiler.DefineFunction}
//        member this.Guarded =           cuda { return! storeInternal.Guarded         |> Compiler.DefineFunction}
//    }
//let storeDirectBlocked (block_threads:int) (items_per_thread:int) =
//    fun (valid_items:int option) ->
//        valid_items |> function
//        | None ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                for ITEM = 0 to items_per_thread - 1 do block_itr.[(linear_tid * items_per_thread) + ITEM] <- items.[ITEM]
//        | Some valid_items ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                for ITEM = 0 to items_per_thread - 1 do
//                    if ITEM + (linear_tid * items_per_thread) < valid_items then
//                        block_itr.[(linear_tid * items_per_thread) + ITEM] <- items.[ITEM]
//
//let storeBlockedVectorized (block_threads:int) (items_per_thread:int) =
//    let MAX_VEC_SIZE = (4, items_per_thread) ||> CUB_MIN
//    let VEC_SIZE = if ((((MAX_VEC_SIZE - 1) &&& MAX_VEC_SIZE) = 0) && ((items_per_thread % MAX_VEC_SIZE) = 0)) then MAX_VEC_SIZE else 1
//    let VECTORS_PER_THREAD = items_per_thread / VEC_SIZE
//    fun _ ->
//        fun (linear_tid:int) (block_ptr:deviceptr<int>) (items:deviceptr<int>) -> ()
//
//let storeDirectStriped (block_threads:int) (items_per_thread:int) =
//    fun (valid_items:int option) ->
//        valid_items |> function
//        | None ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//               for ITEM = 0 to items_per_thread - 1 do block_itr.[(ITEM * block_threads) + linear_tid] <- items.[ITEM]
//
//        | Some valid_items ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                for ITEM = 0 to items_per_thread - 1 do
//                    if ((ITEM * block_threads) + linear_tid) < valid_items then
//                        block_itr.[(ITEM * block_threads) + linear_tid] <- items.[ITEM]
//
//
//let storeDirectWarpStriped (block_threads:int) (items_per_thread:int) =
//    fun (valid_items:int option) ->
//        fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//            let wid = linear_tid >>> CUB_PTX_WARP_THREADS
//            let warp_offset = wid * CUB_PTX_WARP_THREADS * items_per_thread
//
//            valid_items |> function
//            | None ->
//                for ITEM = 0 to items_per_thread - 1 do block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)] <- items.[ITEM]
//            | Some valid_items ->
//                for ITEM = 0 to items_per_thread - 1 do
//                    if (warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS) < valid_items) then
//                        block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)] <- items.[ITEM]
//
//
//let storeInternal (algorithm:BlockStoreAlgorithm) =
//    fun (block_threads:int) (items_per_thread:int) ->
//        algorithm |> function
//        | BLOCK_STORE_DIRECT ->         (block_threads, items_per_thread) ||> storeDirectBlocked
//        | BLOCK_STORE_VECTORIZE ->      (block_threads, items_per_thread) ||> storeBlockedVectorized
//        | BLOCK_STORE_TRANSPOSE ->      (block_threads, items_per_thread) ||> storeDirectStriped
//        | BLOCK_STORE_WARP_TRANSPOSE -> (block_threads, items_per_thread) ||> storeDirectWarpStriped
//
//
//let blockStore (block_threads:int) (items_per_thread:int) (algorithm:BlockStoreAlgorithm) (warp_time_slicing:bool) =
//    //let BlockExchange = BlockExchange.Create(block_threads, items_per_thread, warp_time_slicing)
//    
//    algorithm |> function
//    | BLOCK_STORE_DIRECT ->
//        fun _ (linear_tid:int) ->
//            fun (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int option) ->
//                algorithm |> storeInternal
//                <||     (block_threads, items_per_thread)
//                <|      (valid_items)
//                <|||    (linear_tid, block_itr, items)
//
//    | BLOCK_STORE_VECTORIZE ->
//        fun _ (linear_tid:int) ->
//            fun (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int option) ->
//                algorithm |> storeInternal
//                <||     (block_threads, items_per_thread)
//                <|      (valid_items)
//                <|||    (linear_tid, block_itr, items)
//    
//    | BLOCK_STORE_TRANSPOSE ->
//        let blockedToStriped = (block_threads, items_per_thread, warp_time_slicing) |||> Exchange.blockedToStriped
//        
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            let blockedToStriped = (temp_storage, linear_tid) ||> blockedToStriped
//
//            fun (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int option) ->
//                items |> blockedToStriped
//                algorithm |> storeInternal
//                <||     (block_threads, items_per_thread)
//                <|      (valid_items)
//                <|||    (linear_tid, block_itr, items)    
//    
//    | BLOCK_STORE_WARP_TRANSPOSE ->   
//        let blockedToWarpStriped = (block_threads, items_per_thread, warp_time_slicing) |||> Exchange.blockedToWarpStriped
//        
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            let blockedToWarpStriped = (temp_storage, linear_tid) ||> blockedToWarpStriped
//            
//            fun (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int option) ->
//                items |> blockedToWarpStriped
//                algorithm |> storeInternal
//                <||     (block_threads, items_per_thread)
//                <|      (valid_items)
//                <|||    (linear_tid, block_itr, items)
//
//[<Record>]
//type ThreadFields =
//    {
//        mutable temp_storage    : deviceptr<int>
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
////    static member Init(temp_storage:deviceptr<int>) =
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
//    static member Init(temp_storage:deviceptr<int>, linear_tid:int) =
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
//        ITEMS_PER_THREAD    : int
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
//    member this.Initialize(temp_storage:deviceptr<int>) =
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
//    member this.Initialize(temp_storage:deviceptr<int>, linear_tid:int) =
//        this.ThreadFields.temp_storage    <- temp_storage
//        this.ThreadFields.linear_tid      <- linear_tid
//        this
//
//    [<ReflectedDefinition>]
//    member this.Store(block_itr:deviceptr<int>, items:deviceptr<int>) =
//        (blockStore this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items)
//            <|  None
//
//    
//    [<ReflectedDefinition>]
//    member this.Store(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int) =
//        (blockStore this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items)
//            <|  Some valid_items
//
//    [<ReflectedDefinition>]
//    static member Create(block_threads:int, items_per_thread:int, algorithm:BlockStoreAlgorithm, warp_time_slicing:bool) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = algorithm
//            WARP_TIME_SLICING   = warp_time_slicing
//            ThreadFields        = ThreadFields.Default()
//        }
//    
//    [<ReflectedDefinition>]
//    static member Create(block_threads:int, items_per_thread:int, algorithm:BlockStoreAlgorithm) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = algorithm
//            WARP_TIME_SLICING   = false
//            ThreadFields        = ThreadFields.Default()
//        }
//    
//    [<ReflectedDefinition>]
//    static member Create(block_threads:int, items_per_thread:int) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = BLOCK_STORE_DIRECT
//            WARP_TIME_SLICING   = false
//            ThreadFields        = ThreadFields.Default()
//        }