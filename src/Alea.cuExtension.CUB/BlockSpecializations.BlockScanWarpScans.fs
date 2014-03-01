[<AutoOpen>]
module Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanWarpScans

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Warp



module TempStorage =
    [<Record>]
    type API =
        {
            mutable warp_scan               : Alea.cuExtension.CUB.Warp.Scan.TempStorage.API
            mutable warp_aggregates         : int[]
            mutable block_prefix            : int
        }

        [<ReflectedDefinition>]
        static member Uninitialized() =
            { 
                warp_scan       = Alea.cuExtension.CUB.Warp.Scan.TempStorage.uninitialized()
                warp_aggregates = Array.empty
                block_prefix    = 0
            }
    
    type TempStorage = API





module private Internal =
    //type TempStorage = TempStorage.API
    
    module Constants =
        let WARPS =
            fun block_threads ->
                (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS


    module internal Sig =
        module ApplyWarpAggregates =
            type DefaultExpr = Expr<Ref<int> -> int -> Ref<int> -> unit>
            type WithLaneValidationExpr = Expr<Ref<int> -> int -> Ref<int> -> bool -> unit>

        module ExclusiveScan =
            type WithAggregateExpr = Expr<int -> Ref<int> -> Ref<int> -> Ref<int> -> unit>
            type WithAggregateAndCallbackOpExpr = Expr<int -> Ref<int> -> Ref<int> -> Ref<int> -> Ref<int -> int> -> unit>

            module Identityless =
                type WithAggregateExpr = Expr<int -> Ref<int> -> Ref<int> -> unit>
                type WithAggregateAndCallbackOpExpr = Expr<int -> Ref<int> -> Ref<int> -> Ref<int -> int> -> unit>

        module ExclusiveSum =
            type WithAggregateExpr = Expr<int -> Ref<int> -> Ref<int> -> unit>
            type WithAggregateAndCallbackOpExpr = Expr<int -> Ref<int> -> Ref<int> -> Ref<int -> int> -> unit>
//            type WithAggregateExpr = Function<int -> Ref<int> -> Ref<int> -> unit>
//            type WithAggregateAndCallbackOpExpr = Function<int -> Ref<int> -> Ref<int> -> Ref<int -> int> -> unit>


        module InclusiveScan =
            type WithAggregateExpr = Expr<int -> Ref<int> -> Ref<int> -> unit>
            type WithAggregateAndCallbackOpExpr = Expr<int -> Ref<int> -> Ref<int> -> Ref<int -> int> -> unit>

        module InclusiveSum =
            type WithAggregateExpr = Expr<int -> Ref<int> -> Ref<int> -> unit>
            type WithAggregateAndCallbackOpExpr = Expr<int -> Ref<int> -> Ref<int> -> Ref<int -> int> -> unit>

    //type TempStorage = TempStorage.API

    let WarpScan block_threads scan_op =
        let WARPS = block_threads |> Constants.WARPS
        (WARPS, CUB_PTX_WARP_THREADS, scan_op) |||> WarpScan.api


module ApplyWarpAggregates =
    open Internal
    open TempStorage
    //type TempStorage = TempStorage.API

    type API =
        abstract    Default             : Sig.ApplyWarpAggregates.DefaultExpr
        abstract    WithLaneValidation  : Sig.ApplyWarpAggregates.WithLaneValidationExpr


    type [<Record>] TemplateParams =
        {
            BLOCK_THREADS   : int
            scan_op         : IScanOp
        }

        [<ReflectedDefinition>]
        static member Init(block_threads, scan_op) =
            {
                BLOCK_THREADS   = block_threads
                scan_op         = scan_op
            }


    type [<Record>] ThreadFields =
        {
            TemplateParams          : TemplateParams
            mutable temp_storage    : TempStorage
            mutable linear_tid      : int
            mutable warp_id         : int
            mutable lane_id         : int
        }

        [<ReflectedDefinition>]
        static member Init(template_params, temp_storage, linear_tid) =
            {
                TemplateParams = template_params
                temp_storage = temp_storage
                linear_tid = linear_tid
                warp_id = if template_params.BLOCK_THREADS <= CUB_PTX_WARP_THREADS then 0 else linear_tid / CUB_PTX_WARP_THREADS
                lane_id = if template_params.BLOCK_THREADS <= CUB_PTX_WARP_THREADS then linear_tid else linear_tid % CUB_PTX_WARP_THREADS
            }            

    let private Default (thread_fields:ThreadFields) =
        let lane_valid = true
        let WARPS = thread_fields.TemplateParams.BLOCK_THREADS |> Constants.WARPS
        let scan_op = thread_fields.TemplateParams.scan_op.op
        let temp_storage = thread_fields.temp_storage
        let linear_tid = thread_fields.linear_tid
        let warp_id = thread_fields.warp_id
        <@ fun (partial:Ref<int>) (warp_aggregate:int) (block_aggregate:Ref<int>) ->
            temp_storage.warp_aggregates.[warp_id] <- warp_aggregate

            __syncthreads()

            block_aggregate := temp_storage.warp_aggregates.[0]

            for WARP = 1 to WARPS - 1 do
                if warp_id = WARP then
                    partial := if lane_valid then (!block_aggregate, !partial) ||> %scan_op else !block_aggregate
                block_aggregate := (!block_aggregate, temp_storage.warp_aggregates.[WARP]) ||> %scan_op
        @>


    let private WithLaneValidation (thread_fields:ThreadFields) = 
        let WARPS           = thread_fields.TemplateParams.BLOCK_THREADS |> Constants.WARPS
        let scan_op         = thread_fields.TemplateParams.scan_op.op
        let temp_storage    = thread_fields.temp_storage
        let linear_tid      = thread_fields.linear_tid
        let warp_id         = thread_fields.warp_id
        <@ fun (partial:Ref<int>) (warp_aggregate:int) (block_aggregate:Ref<int>) (lane_valid:bool) ->
            temp_storage.warp_aggregates.[warp_id] <- warp_aggregate

            __syncthreads()

            block_aggregate := temp_storage.warp_aggregates.[0]

            for WARP = 1 to WARPS - 1 do
                if warp_id = WARP then
                    partial := if lane_valid then (!block_aggregate, !partial) ||> %scan_op else !block_aggregate
                block_aggregate := (!block_aggregate, temp_storage.warp_aggregates.[WARP]) ||> %scan_op
        @>
        
        

    let template (block_threads:int) (scan_op:IScanOp) = cuda {
        return fun temp_storage
        
        


        
        return fun (program:Program) -> 
            fun temp_storage warp_id ->
    
                { new API with
                    member this.Default             =   Default
                                                        <||     (block_threads, scan_op) 
                                                        <||     (temp_storage, warp_id)

                    member this.WithLaneValidation  =   WithLaneValidation
                                                        <||     (block_threads, scan_op) 
                                                        <||     (temp_storage, warp_id)
                }
    }
        
   


module ExclusiveSum =
    open Internal
    open TempStorage
    //type TempStorage = TempStorage.API

//    type API =
//        abstract    WithAggregate               : Sig.ExclusiveSum.WithAggregateExpr
//        abstract    WithAggregateAndCallbackOp  : Sig.ExclusiveSum.WithAggregateAndCallbackOpExpr

    type API =
        {
            WithAggregate               : Sig.ExclusiveSum.WithAggregateExpr
            WithAggregateAndCallbackOp  : Sig.ExclusiveSum.WithAggregateAndCallbackOpExpr
        }
        
        

    let private WithAggregate block_threads scan_op =
        
        fun (temp_storage:TempStorage) warp_id lane_id ->
            let WarpScan = (    WarpScan
                                <||     (block_threads, scan_op)
                                <|||    (temp_storage.warp_scan, warp_id, lane_id)
                            ).ExclusiveSum.WithAggregate
                        
            let ApplyWarpAggregates = ( ApplyWarpAggregates.api
                                        <||     (block_threads, scan_op)
                                        <||     (temp_storage, warp_id)
                                      ).WithLaneValidation

            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) ->
                let warp_aggregate = __local__.Variable()
                
                (input, output, warp_aggregate) |||> (%WarpScan)
                (%ApplyWarpAggregates)
                <|||    (output, !warp_aggregate, block_aggregate)
                <|      (lane_id > 0)
            @>

    let private WithAggregateAndCallbackOp block_threads scan_op =
        fun temp_storage warp_id lane_id ->
            let ExclusiveSum =  WithAggregate
                                <||     (block_threads, scan_op)
                                <|||    (temp_storage, warp_id, lane_id)
            let scan_op = scan_op.op
            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int -> int>) ->
                (input, output, block_aggregate) |||>  %ExclusiveSum

                if warp_id = 0 then 
                    let block_prefix = !block_aggregate |> !block_prefix_callback_op
                    if lane_id = 0 then
                        temp_storage.block_prefix <- block_prefix

                __syncthreads()

                output := (temp_storage.block_prefix, !output) ||> %scan_op
            @>

    let api block_threads scan_op =
        fun temp_storage warp_id lane_id -> 
            { 
                WithAggregate =                 WithAggregate
                                                <||     (block_threads, scan_op)
                                                <|||    (temp_storage, warp_id, lane_id)

                WithAggregateAndCallbackOp =    WithAggregateAndCallbackOp
                                                <||     (block_threads, scan_op)
                                                <|||    (temp_storage, warp_id, lane_id)
            }
        


module ExclusiveScan =
    open Internal
    
//    type API =
//        abstract    WithAggregate                       : Sig.ExclusiveScan.WithAggregateExpr
//        abstract    WithAggregate_NoID                  : Sig.ExclusiveScan.Identityless.WithAggregateExpr
//        abstract    WithAggregateAndCallbackOp          : Sig.ExclusiveScan.WithAggregateAndCallbackOpExpr
//        abstract    WithAggregateAndCallbackOp_NoID     : Sig.ExclusiveScan.Identityless.WithAggregateAndCallbackOpExpr
//        

    type API =
        {
            WithAggregate                       : Sig.ExclusiveScan.WithAggregateExpr
            WithAggregate_NoID                  : Sig.ExclusiveScan.Identityless.WithAggregateExpr
            WithAggregateAndCallbackOp          : Sig.ExclusiveScan.WithAggregateAndCallbackOpExpr
            WithAggregateAndCallbackOp_NoID     : Sig.ExclusiveScan.Identityless.WithAggregateAndCallbackOpExpr
        }


    let private WithAggregate block_threads scan_op =
        fun temp_storage warp_id lane_id ->
            <@ fun (input:int) (output:Ref<int>) (identity:Ref<int>) (block_aggregate:Ref<int>) -> () @>

    let private WithAggregateAndCallbackOp block_threads scan_op =
        fun temp_storage warp_id lane_id ->       
            <@ fun (input:int) (output:Ref<int>) (identity:Ref<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int -> int>) -> () @>


    module private Identityless =

        let WithAggregate block_threads scan_op =
            fun temp_storage warp_id lane_id ->    
                <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) -> () @>

        let WithAggregateAndCallbackOp block_threads scan_op =
            fun temp_storage warp_id lane_id ->            
                <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int -> int>) -> () @>


    let api block_threads scan_op =
        
        fun temp_storage warp_id lane_id ->
            { 
                WithAggregate =                     WithAggregate
                                                            <||     (block_threads, scan_op)
                                                            <|||    (temp_storage, warp_id, lane_id)

                WithAggregate_NoID =                Identityless.WithAggregate
                                                            <||     (block_threads, scan_op)
                                                            <|||    (temp_storage, warp_id, lane_id)

                WithAggregateAndCallbackOp =        WithAggregateAndCallbackOp
                                                            <||     (block_threads, scan_op)
                                                            <|||    (temp_storage, warp_id, lane_id)

                WithAggregateAndCallbackOp_NoID =   Identityless.WithAggregateAndCallbackOp
                                                            <||     (block_threads, scan_op)
                                                            <|||    (temp_storage, warp_id, lane_id)
            }
    

module InclusiveSum =
    open Internal
    
    type API =
        {
            WithAggregate               : Sig.InclusiveSum.WithAggregateExpr
            WithAggregateAndCallbackOp  : Sig.InclusiveSum.WithAggregateAndCallbackOpExpr
        }

    let private WithAggregate block_threads scan_op =
        fun temp_storage warp_id lane_id ->
            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) -> () @>

    let private WithAggregateAndCallbackOp block_threads scan_op =
        fun temp_storage warp_id lane_id ->
            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int->int>) -> () @>

    let api block_threads scan_op =
        fun temp_storage warp_id lane_id ->
            {
                WithAggregate =                 WithAggregate
                                                <||     (block_threads, scan_op)
                                                <|||    (temp_storage, warp_id, lane_id)

                WithAggregateAndCallbackOp =    WithAggregateAndCallbackOp
                                                <||     (block_threads, scan_op)
                                                <|||    (temp_storage, warp_id, lane_id)
            }


module InclusiveScan =
    open Internal
    
    type API =
        {
            WithAggregate               : Sig.InclusiveScan.WithAggregateExpr
            WithAggregateAndCallbackOp  : Sig.InclusiveScan.WithAggregateAndCallbackOpExpr
        }


    let private WithAggregate block_threads scan_op =
        fun temp_storage warp_id lane_id ->
            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) -> () @>

    let private WithAggregateAndCallbackOp block_threads scan_op =
        fun temp_storage warp_id lane_id ->
            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int->int>) -> () @>


    let api block_threads scan_op =
        fun temp_storage warp_id lane_id ->
            {
                WithAggregate =                 WithAggregate
                                                <||     (block_threads, scan_op)
                                                <|||    (temp_storage, warp_id, lane_id)

                WithAggregateAndCallbackOp =    WithAggregateAndCallbackOp
                                                <||     (block_threads, scan_op)
                                                <|||    (temp_storage, warp_id, lane_id)
            }


module BlockScanWarpScan =
    
//    type API =
//        abstract    InclusiveSum    : InclusiveSum.API
//        abstract    InclusiveScan   : InclusiveScan.API
//        abstract    ExclusiveSum    : ExclusiveSum.API
//        abstract    ExclusiveScan   : ExclusiveScan.API
    type API =
        {
            InclusiveSum    : InclusiveSum.API
            InclusiveScan   : InclusiveScan.API
            ExclusiveSum    : ExclusiveSum.API
            ExclusiveScan   : ExclusiveScan.API
        }

    let api block_threads scan_op =
        fun temp_storage warp_id lane_id ->
            {
                InclusiveSum    =   InclusiveSum.api
                                    <||     (block_threads, scan_op)
                                    <|||    (temp_storage, warp_id, lane_id)

                InclusiveScan   =   InclusiveScan.api
                                    <||     (block_threads, scan_op)
                                    <|||    (temp_storage, warp_id, lane_id)

                ExclusiveSum    =   ExclusiveSum.api
                                    <||     (block_threads, scan_op)
                                    <|||    (temp_storage, warp_id, lane_id)

                ExclusiveScan   =   ExclusiveScan.api
                                    <||     (block_threads, scan_op)
                                    <|||    (temp_storage, warp_id, lane_id)
            }
        



//let WARPS =
//    fun block_threads ->
//        (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS    
//
//
////type internal TemplateParameters =
////    {
////        BLOCK_THREADS : int
////    }
////
////    static member Init(block_threads:int) =
////        {
////            BLOCK_THREADS = block_threads
////        }
//
//
//type Constants =
//    {
//        WARPS : int
//        WarpScan : WarpScan
//    }
//
//    static member Init(block_threads) =
//        let warps = block_threads |> WARPS
//        {
//            WARPS = warps
//            WarpScan = WarpScan.Create(warps, CUB_PTX_WARP_THREADS)
//        }
//
//
//[<Record>] [<RequireQualifiedAccess>]
//type TempStorage =
//    {
//        mutable warp_scan : deviceptr<int>
//        mutable warp_aggregates : deviceptr<int>
//        mutable block_prefix : int
//    }
//
//
//    static member Init(warp_scan, warp_aggregates, block_prefix) =
//        {
//            warp_scan = warp_scan
//            warp_aggregates = warp_aggregates
//            block_prefix = block_prefix
//        }
//
//    static member Default() =
//        {
//            warp_scan       = __null()
//            warp_aggregates = __null()
//            block_prefix    = 0
//        }
//
//
//
//[<Record>]
//type ThreadFields =
//    {
//        mutable temp_storage : TempStorage
//        mutable linear_tid : int
//        mutable warp_id : int
//        mutable lane_id : int
//    }
//
//
//    static member Init(temp_storage, linear_tid, warp_id, lane_id) =
//        {
//            temp_storage = temp_storage
//            linear_tid = linear_tid
//            warp_id = warp_id
//            lane_id = lane_id
//        }
//
//    static member Init(block_threads, temp_storage, linear_tid) =
//        {
//            temp_storage = temp_storage
//            linear_tid = linear_tid
//            warp_id = if block_threads <= CUB_PTX_WARP_THREADS then 0 else linear_tid / CUB_PTX_WARP_THREADS
//            lane_id = if block_threads <= CUB_PTX_WARP_THREADS then linear_tid else linear_tid % CUB_PTX_WARP_THREADS
//        }
//
//
//let applyWarpAggregates block_threads = 
//    let WARPS = block_threads |> WARPS
//    fun (partial:Ref<int>) (scan_op:(int -> int -> int)) (warp_aggregate:int) (block_aggregate:Ref<int>) (lane_valid:bool option) ->
//        let lane_valid = if lane_valid.IsSome then lane_valid.Value else true
//        fun (temp_storage:TempStorage) (warp_id:int) ->
//            temp_storage.warp_aggregates.[warp_id] <- warp_aggregate
//
//            __syncthreads()
//
//            block_aggregate := temp_storage.warp_aggregates.[0]
//
//            for WARP = 1 to WARPS - 1 do
//                if warp_id = WARP then
//                    partial := if lane_valid then (!block_aggregate, !partial) ||> scan_op else !block_aggregate
//                block_aggregate := (!block_aggregate, temp_storage.warp_aggregates.[WARP]) ||> scan_op
//            
//     
//[<Record>]
//type BlockScanWarpScans =
//    {
//        BLOCK_THREADS       : int
//        Constants           : Constants
//        ThreadFields        : ThreadFields
//    }
//
//
//    member inline this.Initialize(temp_storage:deviceptr<int>, linear_tid) =
//        this.ThreadFields.temp_storage.warp_scan <- temp_storage
//        this.ThreadFields.linear_tid <- linear_tid
//        this
//
//    member this.ApplyWarpAggregates(partial:Ref<int>, scan_op:(int -> int -> int), warp_aggregate:int, block_aggregate:Ref<int>, ?lane_valid:bool) = 
//        applyWarpAggregates
//        <| this.BLOCK_THREADS
//        <| partial <| scan_op <| warp_aggregate <| block_aggregate <| lane_valid
//        <| this.ThreadFields.temp_storage <| this.ThreadFields.warp_id
//    
//        
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>) = 
//        let temp_storage = this.ThreadFields.temp_storage.warp_scan
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//
//        let warp_aggregate = __null() |> __ptr_to_ref
//        this.Constants.WarpScan.Initialize(temp_storage, warp_id, lane_id).ExclusiveScan(input, output, !identity, scan_op, warp_aggregate)
//        this.ApplyWarpAggregates(output, scan_op, !warp_aggregate, block_aggregate)
//
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:(int -> int -> int), block_aggregate:Ref<int>, block_prefix_callback_op:Ref<'T -> 'T>) = 
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//        let temp_storage = this.ThreadFields.temp_storage
//        let identity = identity |> __obj_to_ref
//            
//        this.ExclusiveScan(input, output, identity, scan_op, block_aggregate)
//        if warp_id = 0 then
//            let block_prefix = !block_aggregate |> !block_prefix_callback_op 
//            if lane_id = 0 then temp_storage.block_prefix <- block_prefix
//
//        __syncthreads()
//
//        output := (temp_storage.block_prefix, !output) ||> scan_op
//
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>) = 
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//
//        let warp_aggregate = __null() |> __ptr_to_ref
//        this.Constants.WarpScan.Initialize(temp_storage.warp_scan, warp_id, lane_id).ExclusiveScan(input, output, scan_op, warp_aggregate)
//            
//        this.ApplyWarpAggregates(output, scan_op, !warp_aggregate, block_aggregate, lane_id > 0)
//
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>, block_prefix_callback_op:Ref<'T -> 'T>) =
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//        let linear_tid = this.ThreadFields.linear_tid
//
//        this.ExclusiveScan(input, output, scan_op, block_aggregate)
//
//        if warp_id = 0 then
//            let block_prefix = !block_aggregate |> !block_prefix_callback_op
//            if lane_id = 0 then temp_storage.block_prefix <- block_prefix
//
//        __syncthreads()
//
//        output :=   if linear_tid = 0 then temp_storage.block_prefix 
//                    else (temp_storage.block_prefix, !output) ||> scan_op
//
//
//
//    member inline this.ExclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>) = 
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//            
//        let warp_aggregate = __null() |> __ptr_to_ref
//        this.Constants.WarpScan.Initialize(temp_storage.warp_scan, warp_id, lane_id).ExclusiveSum(input, output, warp_aggregate)
//        let inline sum x y = x + y
//        this.ApplyWarpAggregates(output, (+), !warp_aggregate, block_aggregate)
//
//    member this.ExclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>, block_prefix_callback_op:Ref<'T -> 'T>) = 
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//
//        this.ExclusiveSum(input, output, block_aggregate)
//
//        if warp_id = 0 then
//            let block_prefix = !block_aggregate |> !block_prefix_callback_op
//            if lane_id = 0 then temp_storage.block_prefix <- block_prefix 
//
//        __syncthreads()
//
//        output := (temp_storage.block_prefix, !output) ||> (+)
//
//
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>) = 
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//
//        let warp_aggregate = __null() |> __ptr_to_ref
//        this.Constants.WarpScan.Initialize(temp_storage.warp_scan, warp_id, lane_id).InclusiveScan(input, output, scan_op, warp_aggregate)
//
//        this.ApplyWarpAggregates(output, scan_op, !warp_aggregate, block_aggregate)
//
//
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>, block_prefix_callback_op:Ref<'T -> 'T>) =
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//
//        this.InclusiveScan(input, output, scan_op, block_aggregate)
//
//        if warp_id = 0 then
//            let block_prefix = !block_aggregate |> !block_prefix_callback_op
//            if lane_id = 0 then temp_storage.block_prefix <- block_prefix
//
//        __syncthreads()
//
//        output := (temp_storage.block_prefix, !output) ||> scan_op
//
//    member this.InclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>) = 
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//
//        let warp_aggregate = __null() |> __ptr_to_ref
//        this.Constants.WarpScan.Initialize(temp_storage.warp_scan, warp_id, lane_id).InclusiveSum(input, output, warp_aggregate)
//
//        this.ApplyWarpAggregates(output, (+), !warp_aggregate, block_aggregate)
//
//    member this.InclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) = 
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//
//        this.InclusiveSum(input, output, block_aggregate)
//
//        if warp_id = 0 then
//            let block_prefix = !block_aggregate |> !block_prefix_callback_op
//            if lane_id = 0 then temp_storage.block_prefix <- block_prefix
//
//        __syncthreads()
//
//        output := (temp_storage.block_prefix, !output) ||> (+)
//
//   
//
//
//    static member Create(block_threads:int, linear_tid:int) =
//        let c = block_threads |> Constants.Init
//        {   
//            BLOCK_THREADS = block_threads
//            Constants = c
//            ThreadFields = ThreadFields.Init(block_threads, TempStorage.Default(), linear_tid)
//        }
//
//    static member Create(block_threads:int) =
//        let c = block_threads |> Constants.Init
//        let tf = (block_threads, TempStorage.Default(), 0) |> ThreadFields.Init
//        {
//            BLOCK_THREADS = block_threads
//            Constants = c
//            ThreadFields = tf
//        }
//
//
//
//
//
//
//let exclusiveScan (block_threads:int) =
//    let WARPS = (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS
//    let WarpScan = WarpScan.Create(WARPS, CUB_PTX_WARP_THREADS)
//    fun (temp_storage:deviceptr<int>) (linear_tid:int) (warp_id:int) (lane_id:int) ->
//        fun (input:int) (output:Ref<int>) (identity:Ref<int> option) (scan_op:(int -> int -> int) option) (block_aggregate:Ref<int> option) (block_prefix_callback_op:Ref<int> option) ->
//            (identity, scan_op, block_aggregate, block_prefix_callback_op) |> function
//            | Some identity, Some scan_op, Some block_aggregate, None ->
//                let warp_aggregate = __local__.Variable()
//                //WarpScan
//                ()
//            | Some identity, Some scan_op, Some block_aggregate, Some block_prefix_callback_op -> ()
//            | None, Some scan_op, Some block_aggregate, None -> ()
//            | None, Some scan_op, Some block_aggregate, Some block_prefix_callback_op -> ()
//            | None, None, Some block_aggregate, None -> ()
//            | None, None, Some block_aggregate, Some block_prefix_callback_op -> ()
//            | _, _, _, _ -> ()
                