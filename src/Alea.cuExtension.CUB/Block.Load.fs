[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Load
    
open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

open Macro
open Vector


type BlockLoadAlgorithm =
    | BLOCK_LOAD_DIRECT
    | BLOCK_LOAD_VECTORIZE
    | BLOCK_LOAD_TRANSPOSE
    | BLOCK_LOAD_WARP_TRANSPOSE


type TemplateParameters =
    {
        BLOCK_THREADS       : int
        ITEMS_PER_THREAD    : int
        ALGORITHM           : BlockLoadAlgorithm
        WARP_TIME_SLICING   : bool
    }

    static member Default(block_threads, items_per_thread) =
        {
            BLOCK_THREADS       = block_threads
            ITEMS_PER_THREAD    = items_per_thread
            ALGORITHM           = BLOCK_LOAD_DIRECT
            WARP_TIME_SLICING   = false
        }




module private Internal =
    module Sig =
        module LoadDirectBlocked =
            type DefaultExpr        = Expr<int -> deviceptr<int> -> deviceptr<int> -> unit>
            type GuardedExpr        = Expr<int -> deviceptr<int> -> deviceptr<int> -> int -> unit>
            type GuardedWithOOBExpr = Expr<int -> deviceptr<int> -> deviceptr<int> -> int -> int -> unit>

        module LoadDirectBlockedVectorized =
            type DefaultExpr = Expr<int -> deviceptr<int> -> deviceptr<int> -> unit>

        module LoadDirectStriped =
            type DefaultExpr        = Expr<int -> deviceptr<int> -> deviceptr<int> -> unit>
            type GuardedExpr        = Expr<int -> deviceptr<int> -> deviceptr<int> -> int -> unit>
            type GuardedWithOOBExpr = Expr<int -> deviceptr<int> -> deviceptr<int> -> int -> int -> unit>

        module LoadDirectWarpStriped =
            type DefaultExpr        = Expr<int -> deviceptr<int> -> deviceptr<int> -> unit>
            type GuardedExpr        = Expr<int -> deviceptr<int> -> deviceptr<int> -> int -> unit>
            type GuardedWithOOBExpr = Expr<int -> deviceptr<int> -> deviceptr<int> -> int -> int -> unit>

        module LoadInternal =
            type DefaultExpr        = Expr<deviceptr<int> -> deviceptr<int> -> unit>
            type GuardedExpr        = Expr<deviceptr<int> -> deviceptr<int> -> int -> unit>
            type GuardedWithOOBExpr = Expr<deviceptr<int> -> deviceptr<int> -> int -> int -> unit>


module LoadDirectBlocked =
    open Internal

    type API =
        {
            Default         : Sig.LoadDirectBlocked.DefaultExpr
            Guarded         : Sig.LoadDirectBlocked.GuardedExpr
            GuardedWithOOB  : Sig.LoadDirectBlocked.GuardedWithOOBExpr
        }
    
    let private Default _ items_per_thread = 
        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
        @>

    let private Guarded _ items_per_thread =
        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
            let bounds = valid_items - (linear_tid * items_per_thread)
            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
        @>
        
    let private GuardedWithOOB _ items_per_thread =
        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) (oob_default:int) ->
            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- oob_default
            let bounds = valid_items - (linear_tid * items_per_thread)
            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
        @>


    let api block_threads items_per_thread  =
        {
            Default =           Default
                                <|| (block_threads, items_per_thread)

            Guarded =           Guarded
                                <|| (block_threads, items_per_thread)

            GuardedWithOOB =    GuardedWithOOB
                                <|| (block_threads, items_per_thread)
        }


module LoadDirectBlockedVectorized =
    open Internal

    type API =
        {
            Default : Sig.LoadDirectBlockedVectorized.DefaultExpr
        }

    let private Default items_per_thread =
        <@ fun (linear_tid:int) (block_ptr:deviceptr<int>) (items:deviceptr<int>) ->
            let MAX_VEC_SIZE = CUB_MIN 4 items_per_thread
            let VEC_SIZE = if (((MAX_VEC_SIZE - 1) &&& MAX_VEC_SIZE) = 0) && ((items_per_thread % MAX_VEC_SIZE) = 0) then MAX_VEC_SIZE else 1
            let VECTORS_PER_THREAD = items_per_thread / VEC_SIZE
            let ptr = (block_ptr + (linear_tid * VEC_SIZE * VECTORS_PER_THREAD)) |> __ptr_reinterpret

            let vec_items = __local__.Array<'Vector>(VECTORS_PER_THREAD) |> __array_to_ptr

            for ITEM = 0 to (VECTORS_PER_THREAD - 1) do vec_items.[ITEM] <- ptr.[ITEM]
            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- vec_items.[ITEM] //|> __ptr_to_obj
        @>

    let private Guarded x = <@ fun x y z w -> () @>
    let private GuardedWithOOB x = <@ fun x y z w u -> () @>


    let api _ items_per_thread  =
        {
            Default =           Default
                                <| items_per_thread

//            Guarded =           Guarded
//                                <| items_per_thread
//
//            GuradedWithOOB =    GuardedWithOOB
//                                <| items_per_thread
        }


module LoadDirectStriped =
    open Internal

    type API =
        {
            Default         : Sig.LoadDirectStriped.DefaultExpr
            Guarded         : Sig.LoadDirectStriped.GuardedExpr
            GuardedWithOOB  : Sig.LoadDirectStriped.GuardedWithOOBExpr
        }
    
    let private Default block_threads items_per_thread =
        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(ITEM * block_threads) + linear_tid]
        @>

    let private Guarded block_threads items_per_thread =
        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
            let bounds = valid_items - linear_tid
            for ITEM = 0 to (items_per_thread - 1) do 
                if (ITEM * block_threads < bounds) then items.[ITEM] <- block_itr.[(ITEM * block_threads) + linear_tid]
        @>

    let private GuardedWithOOB block_threads items_per_thread =
        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) (oob_default:int) ->
            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- oob_default
            let bounds = valid_items - linear_tid
            for ITEM = 0 to (items_per_thread - 1) do 
                if (ITEM * block_threads < bounds) then items.[ITEM] <- block_itr.[(ITEM * block_threads) + linear_tid]
        @>


    let api block_threads items_per_thread  =  
        {
            Default =           Default
                                <|| (block_threads, items_per_thread)

            Guarded =           Guarded
                                <|| (block_threads, items_per_thread)

            GuardedWithOOB =    GuardedWithOOB
                                <|| (block_threads, items_per_thread)
        }

    
module LoadDirectWarpStriped =
    open Internal

    type API =
        {
            Default         : Sig.LoadDirectWarpStriped.DefaultExpr
            Guarded         : Sig.LoadDirectWarpStriped.GuardedExpr
            GuardedWithOOB  : Sig.LoadDirectWarpStriped.GuardedWithOOBExpr
        }


    let private Default _ items_per_thread =
        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * items_per_thread

            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
        @>

    let private Guarded _ items_per_thread =
        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * items_per_thread
            let bounds = valid_items - warp_offset - tid

            for ITEM = 0 to (items_per_thread - 1) do 
                if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
        @>
        
    let private GuardedWithOOB _ items_per_thread =        
        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) (oob_default:int) ->
            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- oob_default
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * items_per_thread
            let bounds = valid_items - warp_offset - tid

            for ITEM = 0 to (items_per_thread - 1) do 
                if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
        @>

    
    let api block_threads items_per_thread =
        {
            Default =           Default
                                <|| (block_threads, items_per_thread)

            Guarded =           Guarded
                                <|| (block_threads, items_per_thread)

            GuardedWithOOB =    GuardedWithOOB
                                <|| (block_threads, items_per_thread)
        }


module private LoadInternal =
    open Internal

    type API =
        {
            Default         : Sig.LoadInternal.DefaultExpr
            Guarded         : Sig.LoadInternal.GuardedExpr
            GuardedWithOOB  : Sig.LoadInternal.GuardedWithOOBExpr
        }

    module BlockLoadDirect =
        let Default block_threads items_per_thread _ =
            fun _ linear_tid ->
                let LoadDirectBlocked =
                    (   LoadDirectBlocked.api
                        <|| (block_threads, items_per_thread)
                    ).Default
                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) ->
                    %LoadDirectBlocked
                    <|||    (linear_tid, block_ptr, items)
                @>

        let Guarded block_threads items_per_thread _ =
            fun _ linear_tid ->
                let LoadDirectBlocked = 
                    (   LoadDirectBlocked.api
                        <|| (block_threads, items_per_thread)
                    ).Guarded
                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
                    %LoadDirectBlocked
                    <|  (linear_tid)
                    <|| (block_ptr, items)
                    <|  (valid_items)
                @>

        let GuardedWithOOB block_threads items_per_thread _ =
            fun _ linear_tid ->
                let LoadDirectBlocked =
                    (   LoadDirectBlocked.api
                        <|| (block_threads, items_per_thread)
                    ).GuardedWithOOB
                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) (oob_default:int) ->
                    %LoadDirectBlocked
                    <|  (linear_tid)
                    <|| (block_ptr, items)
                    <|| (valid_items, oob_default)
                @>


    module BlockLoadVectorized =
        let Default block_threads items_per_thread _ =
            fun _ linear_tid ->
                let LoadDirectBlockedVectorized =  
                    (   LoadDirectBlockedVectorized.api
                        <|| (block_threads, items_per_thread)
                    ).Default
                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) ->
                    %LoadDirectBlockedVectorized
                    <|||    (linear_tid, block_ptr, items)
                @>

        let Guarded block_threads items_per_thread _ =
            fun _ linear_tid ->
                let LoadDirectBlocked =
                    (   LoadDirectBlocked.api
                        <|| (block_threads, items_per_thread)
                    ).Guarded
                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
                    %LoadDirectBlocked
                    <|  (linear_tid)
                    <|| (block_ptr, items)
                    <|  (valid_items)
                @>

        let GuardedWithOOB block_threads items_per_thread _ =
            fun _ linear_tid ->
                let LoadDirectBlocked =
                    (   LoadDirectBlocked.api
                        <|| (block_threads, items_per_thread)
                    ).GuardedWithOOB
                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) (oob_default:int) ->
                    %LoadDirectBlocked
                    <|  (linear_tid)
                    <|| (block_ptr, items)
                    <|| (valid_items, oob_default)
                @>


    module BlockLoadTranspose =
        let private StripedToBlocked block_threads items_per_thread warp_time_slicing =
            fun temp_storage linear_tid ->
                let stb =    
                    (   BlockExchange.api
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid)
                        <|||    (0, 0, 0)
                    ).StripedToBlocked
                if warp_time_slicing then stb.WithTimeslicing else stb.Default
        
        
        let Default block_threads items_per_thread warp_time_slicing =
            fun temp_storage linear_tid ->
                
                let StripedToBlocked =  StripedToBlocked
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                
                let LoadDirectStriped =  
                    (   LoadDirectStriped.api
                        <|| (block_threads, items_per_thread)
                    ).Default

                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) ->
                    %LoadDirectStriped
                    <|||    (linear_tid, block_ptr, items)
                    
                    %StripedToBlocked
                    <|      (items)
                @>

        let Guarded block_threads items_per_thread warp_time_slicing =
            fun temp_storage linear_tid ->
                
                let StripedToBlocked =  StripedToBlocked
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                
                let LoadDirectStriped =  
                    (   LoadDirectStriped.api
                        <|| (block_threads, items_per_thread)
                    ).Guarded

                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
                    %LoadDirectStriped
                    <|  (linear_tid)
                    <|| (block_ptr, items)
                    <|  (valid_items)

                    %StripedToBlocked
                    <|      (items)
                @>

        let GuardedWithOOB block_threads items_per_thread warp_time_slicing =
            fun temp_storage linear_tid ->
                
                let StripedToBlocked =  StripedToBlocked
                                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                                        <||     (temp_storage, linear_tid)
                
                let LoadDirectStriped =  
                    (   LoadDirectStriped.api
                        <|| (block_threads, items_per_thread)
                    ).GuardedWithOOB

                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) (oob_default:int) ->
                    %LoadDirectStriped
                    <|  (linear_tid)
                    <|| (block_ptr, items)
                    <|| (valid_items, oob_default)
                    %StripedToBlocked
                    <|      (items)
                @>


    module BlockLoadWarpTranspose =
        let private WARP_THREADS block_threads =
            ((block_threads % CUB_PTX_WARP_THREADS) = 0) |> function
            | false -> failwith "BLOCK_THREADS must be a multiple of WARP_THREADS"
            | true -> CUB_PTX_WARP_THREADS

        let private WarpStripedToBlocked block_threads items_per_thread warp_time_slicing =
            fun temp_storage linear_tid ->
                let wstb =    
                    (   BlockExchange.api
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid)
                        <|||    (0, 0, 0)
                    ).WarpStripedToBlocked
                if warp_time_slicing then wstb.WithTimeslicing else wstb.Default

        let Default block_threads items_per_thread warp_time_slicing =
            fun temp_storage linear_tid ->
                
                let WarpStripedToBlocked =  WarpStripedToBlocked
                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
                                            <||     (temp_storage, linear_tid)
                
                let LoadDirectWarpStriped =  
                    (   LoadDirectWarpStriped.api
                        <|| (block_threads, items_per_thread)
                    ).Default

                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) ->
                    %LoadDirectWarpStriped
                    <|||    (linear_tid, block_ptr, items)
                    
                    %WarpStripedToBlocked
                    <|      (items)
                @>

        let Guarded block_threads items_per_thread warp_time_slicing =
            fun temp_storage linear_tid ->
                
                let WarpStripedToBlocked =  WarpStripedToBlocked
                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
                                            <||     (temp_storage, linear_tid)
                
                let LoadDirectWarpStriped =  
                    (   LoadDirectWarpStriped.api
                        <|| (block_threads, items_per_thread)
                    ).Guarded

                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
                    %LoadDirectWarpStriped
                    <|  (linear_tid)
                    <|| (block_ptr, items)
                    <|  (valid_items)

                    %WarpStripedToBlocked
                    <|      (items)
                @>

        let GuardedWithOOB block_threads items_per_thread warp_time_slicing =
            fun temp_storage linear_tid ->
                
                let WarpStripedToBlocked =  WarpStripedToBlocked
                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
                                            <||     (temp_storage, linear_tid)
                
                let LoadDirectWarpStriped =  
                    (   LoadDirectWarpStriped.api
                        <|| (block_threads, items_per_thread)
                    ).GuardedWithOOB

                <@ fun (block_ptr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) (oob_default:int) ->
                    %LoadDirectWarpStriped
                    <|  (linear_tid)
                    <|| (block_ptr, items)
                    <|| (valid_items, oob_default)
                    %WarpStripedToBlocked
                    <|      (items)
                @>        


    //let api block_threads items_per_thread algorithm warp_time_slicing


module BlockLoad =
    open Internal

    type API =
        {
            Default         : Sig.LoadInternal.DefaultExpr
            Guarded         : Sig.LoadInternal.GuardedExpr
            GuardedWithOOB  : Sig.LoadInternal.GuardedWithOOBExpr
        }

    let api block_threads items_per_thread algorithm (warp_time_slicing:bool option) =
        let warp_time_slicing = if warp_time_slicing.IsNone then false else warp_time_slicing.Value
        fun temp_storage linear_tid ->
            let _Default, _Guarded, _GuardedWithOOB =
                algorithm |> function
                | BLOCK_LOAD_DIRECT -> 
                    (   
                        LoadInternal.BlockLoadDirect.Default
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid),
                        LoadInternal.BlockLoadDirect.Guarded
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid),
                        LoadInternal.BlockLoadDirect.GuardedWithOOB
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid)
                    )

                | BLOCK_LOAD_VECTORIZE ->
                    (   
                        LoadInternal.BlockLoadVectorized.Default
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid),                        
                        LoadInternal.BlockLoadVectorized.Guarded
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid),                        
                        LoadInternal.BlockLoadVectorized.GuardedWithOOB
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid)
                    )

                | BLOCK_LOAD_TRANSPOSE ->
                    (   
                        LoadInternal.BlockLoadTranspose.Default
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid),                        
                        LoadInternal.BlockLoadTranspose.Guarded
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid),                        
                        LoadInternal.BlockLoadTranspose.GuardedWithOOB
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid)
                    )

                | BLOCK_LOAD_WARP_TRANSPOSE ->
                    (   
                        LoadInternal.BlockLoadWarpTranspose.Default
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid),                    
                        LoadInternal.BlockLoadWarpTranspose.Guarded
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid),                    
                        LoadInternal.BlockLoadWarpTranspose.GuardedWithOOB
                        <|||    (block_threads, items_per_thread, warp_time_slicing)
                        <||     (temp_storage, linear_tid)                    
                    )

            {Default = _Default; Guarded = _Guarded; GuardedWithOOB = _GuardedWithOOB}
                
        
//    let api block_threads items_per_thread warp_time_slicing =
//        fun temp_storage linear_tid warp_lane warp_id warp_offset ->
//            {
//                BlockLoadDirect         =   BlockLoadDirect.api
//                                            <||| (block_threads, items_per_thread, warp_time_slicing)
//
//                BlockLoadVectorized     =   BlockLoadVectorized.api
//                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
//
//                BlockLoadTranspose      =   BlockLoadTranspose.api
//                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
//                                            <||     (temp_storage, linear_tid)
//                                            <|||    (warp_lane, warp_id, warp_offset)
//
//                BlockLoadWarpTranspose  =   BlockLoadWarpTranspose.api
//                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
//                                            <||     (temp_storage, linear_tid)
//                                            <|||    (warp_lane, warp_id, warp_offset)
//            }                    

//
//module BlockLoad =
//    open Internal
//
//    type API =
//        {
//            BlockLoadDirect         : LoadInternal.BlockLoadDirect.API
//            BlockLoadVectorized     : LoadInternal.BlockLoadVectorized.API
//            BlockLoadTranspose      : LoadInternal.BlockLoadTranspose.API
//            BlockLoadWarpTranspose  : LoadInternal.BlockLoadWarpTranspose.API
//        }
//
//    let private (|BlockLoadDirect|_|) threadFields templateParams =
//        let block_threads, items_per_thread, algorithm, warp_time_slicing = templateParams
//        let temp_storage, linear_tid, warp_lane, warp_id, warp_offset = threadFields
//        if algorithm = BlockLoadDirect then
//            (   LoadInternal.api
//                <|||    (block_threads, items_per_thread, warp_time_slicing)
//                <||     (temp_storage, linear_tid)
//                <|||    (warp_lane, warp_id, warp_offset)
//            ).BlockLoadDirect
//            |>      Some
//        else
//            None
//
//    let private (|BlockLoadVectorized|_|) threadFields templateParams =
//        let block_threads, items_per_thread, algorithm, warp_time_slicing = templateParams
//        let temp_storage, linear_tid, warp_lane, warp_id, warp_offset = threadFields
//        if algorithm = BlockLoadVectorized then
//            (   LoadInternal.api
//                <|||    (block_threads, items_per_thread, warp_time_slicing)
//                <||     (temp_storage, linear_tid)
//                <|||    (warp_lane, warp_id, warp_offset)
//            ).BlockLoadVectorized
//            |>      Some
//        else
//            None
//
//    let private (|BlockLoadTranspose|_|) threadFields templateParams =
//        let block_threads, items_per_thread, algorithm, warp_time_slicing = templateParams
//        let temp_storage, linear_tid, warp_lane, warp_id, warp_offset = threadFields
//        if algorithm = BlockLoadTranspose then
//            (   LoadInternal.api
//                <|||    (block_threads, items_per_thread, warp_time_slicing)
//                <||     (temp_storage, linear_tid)
//                <|||    (warp_lane, warp_id, warp_offset)
//            ).BlockLoadTranspose
//            |>      Some
//        else
//            None
//
//    let private (|BlockLoadWarpTranspose|_|) threadFields templateParams =
//        let block_threads, items_per_thread, algorithm, warp_time_slicing = templateParams
//        let temp_storage, linear_tid, warp_lane, warp_id, warp_offset = threadFields
//        if algorithm = BlockLoadWarpTranspose then
//            (   LoadInternal.api
//                <|||    (block_threads, items_per_thread, warp_time_slicing)
//                <||     (temp_storage, linear_tid)
//                <|||    (warp_lane, warp_id, warp_offset)
//            ).BlockLoadWarpTranspose
//            |>      Some
//        else
//            None
//
//
//    let private Default block_threads items_per_thread algorithm warp_time_slicing = 
//        fun temp_storage linear_tid warp_lane warp_id warp_offset ->
//            let templateParams = (block_threads, items_per_thread, algorithm, warp_time_slicing)
//            let threadFields = (temp_storage, linear_tid, warp_lane, warp_id, warp_offset)
//            let InternalLoad =
//                templateParams |> function
//                | BlockLoadDirect threadFields bld ->
//                    bld.LoadDirectBlocked.Default
//                | BlockLoadVectorized threadFields blv ->
//                    blv.LoadDirectBlockedVectorized.Default
//                | BlockLoadTranspose threadFields blt ->
//                    blt.LoadDirectStriped.Default
//                | BlockLoadWarpTranspose threadFields blwt ->
//                    blwt.LoadDirectWarpStriped.Default
//                | _ -> failwith "Invalid Template Parameters"
//
//
//
//
//            <@ fun _ -> () @>
//
//    let private Guarded block_threads items_per_thread algorithm warp_time_slicing =
//        <@ fun _ -> () @>
//
//    let private GuardedWithOOB block_threads items_per_thread algorithm warp_time_slicing =
//        <@ fun _ -> () @>

//module LoadInternal =
//    open Internal
//
//    module BlockLoadDirect =
//        type API =
//            {
//                LoadDirectBlocked : LoadDirectBlocked.API
//            }
//
//        let api _ items_per_thread _ = 
//            {
//                LoadDirectBlocked   =   LoadDirectBlocked.api
//                                        <|| (None, items_per_thread)
//            }
//
//    module BlockLoadVectorize =
//        type API =
//            {
//                LoadDirectBlockedVectorized : LoadDirectBlockedVectorized.API
//            }
//
//        let api _ items_per_thread _ =
//            {
//                LoadDirectBlockedVectorized =   LoadDirectBlockedVectorized.api
//                                                <|| (None, items_per_thread)
//            }
//
//    module BlockLoadTranspose =
//        type API =
//            {
//                LoadDirectStriped   : LoadDirectStriped.API
//                BlockExchange       : BlockExchange.API
//            }
//
//        let api block_threads items_per_thread warp_time_slicing = 
//            fun temp_storage linear_tid warp_lane warp_id warp_offset ->
//                {
//                    LoadDirectStriped   =   LoadDirectStriped.api
//                                            <|| (block_threads, items_per_thread)
//
//                    BlockExchange       =   BlockExchange.api
//                                            <|||    (block_threads, items_per_thread, warp_time_slicing)
//                                            <||     (temp_storage, linear_tid)
//                                            <|||    (warp_lane, warp_id, warp_offset)
//                }
//
//    module BlockLoadWarpTranspose =
//        type API =
//            {
//                LoadDirectWarpStriped   : LoadDirectWarpStriped.API
//                BlockExchange           : BlockExchange.API                
//            }
//
//        let api block_threads items_per_thread warp_time_slicing = 
//            fun temp_storage linear_tid warp_lane warp_id warp_offset ->
//                {
//                    LoadDirectWarpStriped   =   LoadDirectWarpStriped.api
//                                                <|| (block_threads, items_per_thread)
//
//                    BlockExchange           =   BlockExchange.api
//                                                <|||    (block_threads, items_per_thread, warp_time_slicing)
//                                                <||     (temp_storage, linear_tid)
//                                                <|||    (warp_lane, warp_id, warp_offset)
//                }
//                 
    //let load (block_threads:int) (items_per_thread:int) =
//    let private internalAPI = 
//            {   LoadDirectBlocked           = LoadDirectBlocked.api;
//                LoadDirectBlockedVectorized = LoadDirectBlockedVectorized.api
//                LoadDirectStriped           = LoadDirectStriped.api
//                LoadDirectWarpStriped       = LoadDirectWarpStriped.api }

//    let api block_threads items_per_thread algorithm = 
//            algorithm |> function
//            | BLOCK_LOAD_DIRECT ->           (block_threads, items_per_thread) ||> internalAPI.LoadDirectBlocked
//            | BLOCK_LOAD_VECTORIZE ->        (block_threads, items_per_thread) ||> internalAPI.LoadDirectBlockedVectorized
//            | BLOCK_LOAD_TRANSPOSE ->        (block_threads, items_per_thread) ||> internalAPI.LoadDirectStriped
//            | BLOCK_LOAD_WARP_TRANSPOSE ->   (block_threads, items_per_thread) ||> internalAPI.LoadDirectWarpStriped
//
//    let api (block_threads:int) (items_per_thread:int) = 
//            {   LoadDirectBlocked           = (block_threads, items_per_thread) ||> LoadDirectBlocked.api;
//                LoadDirectBlockedVectorized = (block_threads, items_per_thread) ||> LoadDirectBlockedVectorized.api
//                LoadDirectStriped           = (block_threads, items_per_thread) ||> LoadDirectStriped.api
//                LoadDirectWarpStriped       = (block_threads, items_per_thread) ||> LoadDirectWarpStriped.api }


//    let load (block_threads:int) (items_per_thread:int) (algorithm:BlockLoadAlgorithm) = //cuda {//(valid_items:int option) (oob_default:int option) =
//        let api = (block_threads, items_per_thread) ||> api
//
//        fun (valid_items:int option) (oob_default:int option) -> 
//            let Option = (valid_items, oob_default)
//            let (|DefaultOption|GuardedOption|GuardedWithOOBOption|) x = 
//                x |> function
//                | (None, None) -> DefaultOption
//                | (Some valid_items, None) -> GuardedOption
//                | (Some valid_items, Some oob_default) -> GuardedWithOOBOption
//                | _,_ -> DefaultOption
//
//            (algorithm, Option) |> function
//            | BLOCK_LOAD_DIRECT, DefaultOption ->           Load.Default(api.LoadDirectBlocked.Default)
//            | BLOCK_LOAD_VECTORIZE, DefaultOption ->        Load.Default(api.LoadDirectBlockedVectorized.Default)
//            | BLOCK_LOAD_TRANSPOSE, DefaultOption ->        Load.Default(api.LoadDirectStriped.Default)
//            | BLOCK_LOAD_WARP_TRANSPOSE, DefaultOption ->   Load.Default(api.LoadDirectWarpStriped.Default)
//
//            | BLOCK_LOAD_DIRECT, GuardedOption ->           Load.Guarded(api.LoadDirectBlocked.Guarded)
//            | BLOCK_LOAD_VECTORIZE, GuardedOption ->        Load.Guarded(api.LoadDirectBlockedVectorized.Guarded)
//            | BLOCK_LOAD_TRANSPOSE, GuardedOption ->        Load.Guarded(api.LoadDirectStriped.Guarded)
//            | BLOCK_LOAD_WARP_TRANSPOSE, GuardedOption ->   Load.Guarded(api.LoadDirectWarpStriped.Guarded)
//
//            | BLOCK_LOAD_DIRECT, GuardedWithOOBOption ->           Load.GuardedWithOOB(api.LoadDirectBlocked.GuardedWithOOB)
//            | BLOCK_LOAD_VECTORIZE, GuardedWithOOBOption ->        Load.GuardedWithOOB(api.LoadDirectBlockedVectorized.GuardedWithOOB)
//            | BLOCK_LOAD_TRANSPOSE, GuardedWithOOBOption ->        Load.GuardedWithOOB(api.LoadDirectStriped.GuardedWithOOB)
//            | BLOCK_LOAD_WARP_TRANSPOSE, GuardedWithOOBOption ->   Load.GuardedWithOOB(api.LoadDirectWarpStriped.GuardedWithOOB)
//    let load (block_threads:int) (items_per_thread:int) (algorithm:BlockLoadAlgorithm) = //cuda {//(valid_items:int option) (oob_default:int option) =
//        let api = (block_threads, items_per_thread) ||> api
//        algorithm |> function
//        | BLOCK_LOAD_DIRECT ->          api.LoadDirectBlocked
//        | BLOCK_LOAD_VECTORIZE ->       api.LoadDirectBlockedVectorized
//        | BLOCK_LOAD_TRANSPOSE ->       api.LoadDirectStriped
//        | BLOCK_LOAD_WARP_TRANSPOSE ->  api.LoadDirectWarpStriped
//
//
//let inline blockLoad (block_threads:int) (items_per_thread:int) (algorithm:BlockLoadAlgorithm) (warp_time_slicing:bool) = 
//    let loadInternal = (block_threads, items_per_thread, algorithm) |||> InternalLoad.load
//    { new BlockLoadAPI with
//        member this.Default =           cuda { return! loadInternal.Default         |> Compiler.DefineFunction}
//        member this.Guarded =           cuda { return! loadInternal.Guarded         |> Compiler.DefineFunction}
//        member this.GuardedWithOOB =    cuda { return! loadInternal.GuardedWithOOB  |> Compiler.DefineFunction}
//    }
//    
//
////let blockLoad (block_threads:int) (items_per_thread:int) (algorithm:BlockLoadAlgorithm) (warp_time_slicing:bool) = cuda {
//    //let loadInternal = (block_threads, items_per_thread, algorithm) |||> InternalLoad.load
////        algorithm |> function
////        | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->
////            ()
////
////        | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->
////            ()
////
////        | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->
////            let stripedToBlocked = (block_threads, items_per_thread, warp_time_slicing) |||> Exchange.stripedToBlocked
////            ()
////
////        | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->
////            let warpStripedToBlocked = (block_threads, items_per_thread, warp_time_slicing) |||> Exchange.warpStripedToBlocked
////            ()
//        //}
////let PrivateStorage() = __null()
////
////[<Record>]
////type ThreadFields =
////    {
////        mutable temp_storage : deviceptr<int>
////        mutable linear_tid : int
////    }
////
////    [<ReflectedDefinition>]
////    member this.Get() = (this.temp_storage, this.linear_tid)
////    
////    [<ReflectedDefinition>]
////    static member Init(temp_storage:deviceptr<int>, linear_tid:int) =
////        {
////            temp_storage = temp_storage
////            linear_tid = linear_tid
////        }
////
////    [<ReflectedDefinition>]
////    static member Default() =
////        {
////            temp_storage = __null()
////            linear_tid = 0
////        }
//
//
//
//[<Record>]
//type BlockLoad =
//    {
//        BLOCK_THREADS       : int
//        ITEMS_PER_THREAD    : int
//        [<RecordExcludedField>] ALGORITHM           : BlockLoadAlgorithm
//        WARP_TIME_SLICING   : bool
//        ThreadFields        : ThreadFields
//    }
//
//
//    [<ReflectedDefinition>]
//    member this.Initialize() =
//        this.ThreadFields.temp_storage <- PrivateStorage()
//        this.ThreadFields.linear_tid <- threadIdx.x
//        this
//    
//    [<ReflectedDefinition>]
//    member this.Initialize(temp_storage:deviceptr<int>) =
//        this.ThreadFields.temp_storage <- temp_storage
//        this.ThreadFields.linear_tid <- threadIdx.x
//        this
//    
//    [<ReflectedDefinition>]
//    member this.Initialize(linear_tid:int) =
//        this.ThreadFields.temp_storage <- PrivateStorage()
//        this.ThreadFields.linear_tid <- linear_tid
//        this
//
//    [<ReflectedDefinition>]
//    member this.Initialize(temp_storage:deviceptr<int>, linear_tid:int) =
//        this.ThreadFields.temp_storage <- temp_storage
//        this.ThreadFields.linear_tid <- linear_tid
//        this
//
//    [<ReflectedDefinition>]
//    member this.Load(block_itr:deviceptr<int>, items:deviceptr<int>) = 
//        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items) 
//            <|| (None, None)
//
//    [<ReflectedDefinition>]
//    member this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int) =
//        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items) 
//            <|| (Some valid_items, None)
//
//    [<ReflectedDefinition>]
//    member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int, oob_default:int) =
//        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items)
//            <|| (Some valid_items, Some oob_default)
//    
//    [<ReflectedDefinition>]
//    static member Create(block_threads, items_per_thread, algorithm, warp_time_slicing) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = algorithm
//            WARP_TIME_SLICING   = warp_time_slicing
//            ThreadFields        = ThreadFields.Default()
//        }
//
//    [<ReflectedDefinition>]
//    static member Create(block_threads, items_per_thread, algorithm) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = algorithm
//            WARP_TIME_SLICING   = false
//            ThreadFields        = ThreadFields.Default()
//        }
//
//    [<ReflectedDefinition>]
//    static member Create(block_threads, items_per_thread) = 
//        {
//            
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = BlockLoadAlgorithm.BLOCK_LOAD_DIRECT
//            WARP_TIME_SLICING   = false
//            ThreadFields        = ThreadFields.Default()
//        }


//[<ReflectedDefinition>]
//type BlockLoadAlgorithm =
//    | BLOCK_LOAD_DIRECT
//    | BLOCK_LOAD_VECTORIZE
//    | BLOCK_LOAD_TRANSPOSE
//    | BLOCK_LOAD_WARP_TRANSPOSE
//
//
//let loadDirectBlocked (block_threads:int) (items_per_thread:int) = 
//    fun (valid_items:int option) (oob_default:int option) ->
//        match valid_items, oob_default with
//        | None, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
//
//        | Some valid_items, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                let bounds = valid_items - (linear_tid * items_per_thread)
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
//                
//        | Some valid_items, Some oob_default ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- oob_default
//                let bounds = valid_items - (linear_tid * items_per_thread)
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
//
//        | _, _ ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) -> ()
//
//
//let loadDirectBlockedVectorized (block_threads:int) (items_per_thread:int) =
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
//let loadDirectStriped (block_threads:int) (items_per_thread:int) = 
//    fun (valid_items:int option) (oob_default:int option) ->
//        match valid_items, oob_default with
//        | None, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//               for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(ITEM * block_threads) + linear_tid]
//
//        | Some valid_items, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                let bounds = valid_items - linear_tid
//                for ITEM = 0 to (items_per_thread - 1) do 
//                    if (ITEM * block_threads < bounds) then items.[ITEM] <- block_itr.[(ITEM * block_threads) + linear_tid]
//                
//        | Some valid_items, Some oob_default ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- oob_default
//                let bounds = valid_items - linear_tid
//                for ITEM = 0 to (items_per_thread - 1) do 
//                    if (ITEM * block_threads < bounds) then items.[ITEM] <- block_itr.[(ITEM * block_threads) + linear_tid]
//
//        | _, _ ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) -> ()
//
//
//let loadDirectWarpStriped (block_threads:int) (items_per_thread:int) = 
//    fun (valid_items:int option) (oob_default:int option) ->
//        match valid_items, oob_default with
//        | None, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//                let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
//                let warp_offset = wid * CUB_PTX_WARP_THREADS * items_per_thread
//
//                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
//
//        | Some valid_items, None ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
//                let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//                let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
//                let warp_offset = wid * CUB_PTX_WARP_THREADS * items_per_thread
//                let bounds = valid_items - warp_offset - tid
//
//                for ITEM = 0 to (items_per_thread - 1) do 
//                    if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
//                
//        | Some valid_items, Some oob_default ->
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
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
//            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) -> ()
//
//
//let loadInternal (algorithm:BlockLoadAlgorithm) =
//    fun (block_threads:int) (items_per_thread:int) ->
//        algorithm |> function
//        | BLOCK_LOAD_DIRECT ->           (block_threads, items_per_thread) ||> loadDirectBlocked
//        | BLOCK_LOAD_VECTORIZE ->        (block_threads, items_per_thread) ||> loadDirectBlockedVectorized
//        | BLOCK_LOAD_TRANSPOSE ->        (block_threads, items_per_thread) ||> loadDirectStriped
//        | BLOCK_LOAD_WARP_TRANSPOSE ->   (block_threads, items_per_thread) ||> loadDirectWarpStriped
//
//
//let blockLoad (block_threads:int) (items_per_thread:int) (algorithm:BlockLoadAlgorithm) (warp_time_slicing:bool) =
//    algorithm |> function
//    | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->
//        fun _ linear_tid ->
//            fun (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int option) (oob_default:int option) ->
//                algorithm |> loadInternal 
//                <||     (items_per_thread, block_threads)
//                <||     (valid_items, oob_default)
//                <|||    (linear_tid, block_itr, items)
//
//    | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->
//        fun _ linear_tid ->    
//            fun (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int option) (oob_default:int option) ->
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
//            fun (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int option) (oob_default:int option) ->
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
//            fun (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int option) (oob_default:int option) ->
//                algorithm |> loadInternal 
//                <||     (items_per_thread, block_threads)
//                <||     (valid_items, oob_default)
//                <|||    (linear_tid, block_itr, items)
//                items |> warpStripedToBlocked
//
//let PrivateStorage() = __null()
//
//[<Record>]
//type ThreadFields =
//    {
//        mutable temp_storage : deviceptr<int>
//        mutable linear_tid : int
//    }
//
//    [<ReflectedDefinition>]
//    member this.Get() = (this.temp_storage, this.linear_tid)
//    
//    [<ReflectedDefinition>]
//    static member Init(temp_storage:deviceptr<int>, linear_tid:int) =
//        {
//            temp_storage = temp_storage
//            linear_tid = linear_tid
//        }
//
//    [<ReflectedDefinition>]
//    static member Default() =
//        {
//            temp_storage = __null()
//            linear_tid = 0
//        }
//
//
//
//[<Record>]
//type BlockLoad =
//    {
//        BLOCK_THREADS       : int
//        ITEMS_PER_THREAD    : int
//        [<RecordExcludedField>] ALGORITHM           : BlockLoadAlgorithm
//        WARP_TIME_SLICING   : bool
//        ThreadFields        : ThreadFields
//    }
//
//
//    [<ReflectedDefinition>]
//    member this.Initialize() =
//        this.ThreadFields.temp_storage <- PrivateStorage()
//        this.ThreadFields.linear_tid <- threadIdx.x
//        this
//    
//    [<ReflectedDefinition>]
//    member this.Initialize(temp_storage:deviceptr<int>) =
//        this.ThreadFields.temp_storage <- temp_storage
//        this.ThreadFields.linear_tid <- threadIdx.x
//        this
//    
//    [<ReflectedDefinition>]
//    member this.Initialize(linear_tid:int) =
//        this.ThreadFields.temp_storage <- PrivateStorage()
//        this.ThreadFields.linear_tid <- linear_tid
//        this
//
//    [<ReflectedDefinition>]
//    member this.Initialize(temp_storage:deviceptr<int>, linear_tid:int) =
//        this.ThreadFields.temp_storage <- temp_storage
//        this.ThreadFields.linear_tid <- linear_tid
//        this
//
//    [<ReflectedDefinition>]
//    member this.Load(block_itr:deviceptr<int>, items:deviceptr<int>) = 
//        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items) 
//            <|| (None, None)
//
//    [<ReflectedDefinition>]
//    member this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int) =
//        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items) 
//            <|| (Some valid_items, None)
//
//    [<ReflectedDefinition>]
//    member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int, oob_default:int) =
//        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//            <|| this.ThreadFields.Get()
//            <|| (block_itr, items)
//            <|| (Some valid_items, Some oob_default)
//    
//    [<ReflectedDefinition>]
//    static member Create(block_threads, items_per_thread, algorithm, warp_time_slicing) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = algorithm
//            WARP_TIME_SLICING   = warp_time_slicing
//            ThreadFields        = ThreadFields.Default()
//        }
//
//    [<ReflectedDefinition>]
//    static member Create(block_threads, items_per_thread, algorithm) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = algorithm
//            WARP_TIME_SLICING   = false
//            ThreadFields        = ThreadFields.Default()
//        }
//
//    [<ReflectedDefinition>]
//    static member Create(block_threads, items_per_thread) = 
//        {
//            
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            ALGORITHM           = BlockLoadAlgorithm.BLOCK_LOAD_DIRECT
//            WARP_TIME_SLICING   = false
//            ThreadFields        = ThreadFields.Default()
//        }


//
//let vars (temp_storage:deviceptr<int> option) (linear_tid:int option) =
//    match temp_storage, linear_tid with
//    | Some temp_storage, Some linear_tid -> temp_storage,       linear_tid
//    | None,              Some linear_tid -> privateStorage(),   linear_tid
//    | Some temp_storage, None ->            temp_storage,       threadIdx.x
//    | None,              None ->            privateStorage(),   threadIdx.x
//
//
//
//    [<Record>]
//    type LoadInternal =
//        {
//            mutable real : RealTraits<int>
//            mutable ITEMS_PER_THREAD : int option
//            mutable BLOCK_THREADS : int option
//            mutable ALGORITHM : BlockLoadAlgorithm
//            mutable temp_storage : deviceptr<int> option
//            mutable linear_tid : int option
//            mutable LoadDirectBlocked : LoadDirectBlocked<int> option
//            mutable LoadDirectBlockedVectorized : LoadDirectBlockedVectorized<int> option
//            mutable LoadDirectStriped : LoadDirectStriped<int> option
//        }
//
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>) =
//            match this.ALGORITHM with
//            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->
//                this.LoadDirectBlocked <- LoadDirectBlocked.Create(this.real, this.ITEMS_PER_THREAD.Value) |> Some
//            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE -> 
//                this.LoadDirectBlockedVectorized <- LoadDirectBlockedVectorized.Create(this.real, this.ITEMS_PER_THREAD.Value) |> Some
//            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE -> ()
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int) =
//            match this.ALGORITHM with
//            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE -> ()
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int, oob_default:int) =
//            match this.ALGORITHM with
//            | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE -> ()
//            | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE -> ()
//
//        [<ReflectedDefinition>]
//        static member inline Create(real:RealTraits<int>, _ALGORITHM:BlockLoadAlgorithm, linear_tid:int) =
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
//        static member inline Create(real:RealTraits<int>, _ALGORITHM:BlockLoadAlgorithm) =
//            {   real = real;
//                ITEMS_PER_THREAD = None;
//                BLOCK_THREADS = None;
//                ALGORITHM = _ALGORITHM;
//                temp_storage = None;
//                linear_tid = None;
//                LoadDirectBlocked = None;
//                LoadDirectStriped = None}
//
//
//    [<Record>]
//    type BlockLoad =
//        {
//            real : RealTraits<int>
//            mutable BLOCK_THREADS      : int
//            mutable ITEMS_PER_THREAD   : int
//            mutable ALGORITHM          : BlockLoadAlgorithm
//            mutable WARP_TIME_SLICING  : bool
//            TempStorage : Expr<unit -> deviceptr<int>> option
//            LoadInternal : LoadInternal<int> option
//        }
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>) = 
//            if this.LoadInternal.IsSome then this.LoadInternal.Value.Load(block_itr, items) else failwith "need to initialize LoadInternal"
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int) = 
//            if this.LoadInternal.IsSome then this.LoadInternal.Value.Load(block_itr, items, valid_items) else failwith "need to initialize LoadInternal"
//
//        [<ReflectedDefinition>]
//        member inline this.Load(block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int, oob_default:int) = 
//            if this.LoadInternal.IsSome then this.LoadInternal.Value.Load(block_itr, items, valid_items) else failwith "need to initialize LoadInternal"
//
//        [<ReflectedDefinition>]
//        static member Create(real:RealTraits<int>, _BLOCK_THREADS:int, _ITEMS_PER_THREAD:int, _ALGORITHM:BlockLoadAlgorithm, _WARP_TIME_SLICING:bool) =
//            {   real = real;
//                BLOCK_THREADS = _BLOCK_THREADS;
//                ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//                ALGORITHM = _ALGORITHM;
//                WARP_TIME_SLICING = _WARP_TIME_SLICING;
//                TempStorage = None;
//                LoadInternal = LoadInternal.Create(real, _ALGORITHM) |> Some}
//
//
//    type BlockLoad(?_ITEMS_PER_THREAD_:int,?BLOCK_THREADS:int) =
//        abstract LoadDirectBlocked : (int * deviceptr<int> * deviceptr<int>) -> unit
//        abstract LoadDirectBlocked : (int * deviceptr<int> * deviceptr<int> * int) -> unit
//        abstract LoadDirectBlocked : (int * deviceptr<int> * deviceptr<int> * int * 'T) -> unit
//        abstract LoadDirectBlockedVectorized : (int * deviceptr<int> * deviceptr<int>) -> unit
//        abstract LoadDirectStriped : (int * deviceptr<int> * deviceptr<int>) -> unit
//        abstract LoadDirectStriped : (int * deviceptr<int> * deviceptr<int> * int) -> unit
//        abstract LoadDirectStriped : (int * deviceptr<int> * deviceptr<int> * int * 'T) -> unit
//        abstract LoadDirectWarpStriped : (int * deviceptr<int> * deviceptr<int>) -> unit
//        abstract LoadDirectWarpStriped : (int * deviceptr<int> * deviceptr<int> * int) -> unit
//        abstract LoadDirectWarpStriped : (int * deviceptr<int> * deviceptr<int> * int * 'T) -> unit
//
//[<Record>]
//type LoadDirectBlocked =
//    {
//        ITEMS_PER_THREAD    : int
//        [<RecordExcludedField>] real : RealTraits<int>
//    }
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectBlocked(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>) =
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * this.ITEMS_PER_THREAD) + ITEM]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectBlocked(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int) =
//        let bounds = valid_items - (linear_tid * this.ITEMS_PER_THREAD)
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * this.ITEMS_PER_THREAD) + ITEM]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectBlocked(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int, oob_default:int) =
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
//        this.LoadDirectBlocked(linear_tid, block_itr, items, valid_items)
//
//    [<ReflectedDefinition>]
//    static member Create(real:RealTraits<int>, _ITEMS_PER_THREAD:int) =
//        {   ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//            real = real }
//
//    [<ReflectedDefinition>]
//    static member Default(real:RealTraits<int>) =
//        {   ITEMS_PER_THREAD = 128;
//            real = real }
//
//
//[<Record>]
//type LoadDirectBlockedVectorized =
//    {
//        ITEMS_PER_THREAD    : int
//        [<RecordExcludedField>] real : RealTraits<int>
//    }
//
//        
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectBlockedVectorized(linear_tid:int, block_ptr:deviceptr<int>, items:deviceptr<int>) =
//        let MAX_VEC_SIZE = CUB_MIN 4 this.ITEMS_PER_THREAD
//        let VEC_SIZE = if (((MAX_VEC_SIZE - 1) &&& MAX_VEC_SIZE) = 0) && ((this.ITEMS_PER_THREAD % MAX_VEC_SIZE) = 0) then MAX_VEC_SIZE else 1
//        let VECTORS_PER_THREAD = this.ITEMS_PER_THREAD / VEC_SIZE
//        let ptr = (block_ptr + (linear_tid * VEC_SIZE * VECTORS_PER_THREAD)) |> __ptr_reinterpret
//
//        let vec_items = __local__.Array<CubVector<int>>(VECTORS_PER_THREAD) |> __array_to_ptr
//
//        for ITEM = 0 to (VECTORS_PER_THREAD - 1) do vec_items.[ITEM] <- ptr.[ITEM]
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- vec_items.[ITEM].Ptr |> __ptr_to_obj
//
//    [<ReflectedDefinition>]
//    static member Create(real:RealTraits<int>, _ITEMS_PER_THREAD:int) =
//        {   ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//            real = real }
//
//    [<ReflectedDefinition>]
//    static member Default(real:RealTraits<int>) =
//        {   ITEMS_PER_THREAD = 128;
//            real = real }
//
//[<Record>]
//type LoadDirectStriped =
//    {
//        BLOCK_THREADS : int
//        ITEMS_PER_THREAD : int
//        [<RecordExcludedField>] real : RealTraits<int>
//    }
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectStriped(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>) =
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(ITEM * this.BLOCK_THREADS) + linear_tid]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectStriped(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int) =
//        let bounds = valid_items - linear_tid
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do 
//            if (ITEM * this.BLOCK_THREADS < bounds) then items.[ITEM] <- block_itr.[(ITEM * this.BLOCK_THREADS) + linear_tid]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectStriped(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int, oob_default:int) =
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
//        this.LoadDirectStriped(linear_tid, block_itr, items, valid_items)
//
//    [<ReflectedDefinition>]
//    static member Create(real:RealTraits<int>, _BLOCK_THREADS:int, _ITEMS_PER_THREAD:int) =
//        {   BLOCK_THREADS = _BLOCK_THREADS
//            ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//            real = real }
//
//    [<ReflectedDefinition>]
//    static member Default(real:RealTraits<int>) =
//        {   BLOCK_THREADS = 128;
//            ITEMS_PER_THREAD = 128;
//            real = real }
//
//
//[<Record>]
//type LoadDirectWarpStriped =
//    {
//        mutable ITEMS_PER_THREAD : int
//        [<RecordExcludedField>] real : RealTraits<int>
//    }
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectWarpStriped(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>) =
//        let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//        let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
//        let warp_offset = wid * CUB_PTX_WARP_THREADS * this.ITEMS_PER_THREAD
//
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectWarpStriped(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int) =
//        let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
//        let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
//        let warp_offset = wid * CUB_PTX_WARP_THREADS * this.ITEMS_PER_THREAD
//        let bounds = valid_items - warp_offset - tid
//
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do 
//            if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]
//
//    [<ReflectedDefinition>]
//    member inline this.LoadDirectWarpStriped(linear_tid:int, block_itr:deviceptr<int>, items:deviceptr<int>, valid_items:int, oob_default:int) =
//        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
//        this.LoadDirectWarpStriped(linear_tid, block_itr, items, valid_items)
//
//    [<ReflectedDefinition>]
//    static member Create(real:RealTraits<int>, _ITEMS_PER_THREAD:int) =
//        {   ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
//            real = real }
//
//    [<ReflectedDefinition>]
//    static member Default(real:RealTraits<int>) =
//        {   ITEMS_PER_THREAD = 128;
//            real = real }