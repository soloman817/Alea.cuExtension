[<AutoOpen>]
module Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanWarpScans

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Warp



module Template =
    [<AutoOpen>]
    module Params =
        [<Record>]
        type API<'T> =
            {
                BLOCK_THREADS   : int
                scan_op         : IScanOp<'T>
            }

            [<ReflectedDefinition>]
            member this.Get() = (this.BLOCK_THREADS, this.scan_op)

            [<ReflectedDefinition>]
            static member inline Init(block_threads, scan_op) =
                {
                    BLOCK_THREADS   = block_threads
                    scan_op         = scan_op
                }

    [<AutoOpen>]
    module Constants =
        [<Record>]
        type API =
            {
                WARPS : int
            }

            [<ReflectedDefinition>]
            static member Init(block_threads) = { WARPS = (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS }

            [<ReflectedDefinition>]
            static member Init(tp:Params.API<'T>) = { WARPS = (tp.BLOCK_THREADS + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS }


    [<AutoOpen>]
    module TempStorage =
        [<Record>]
        type WarpAggregates<'T> =
            {
                mutable Ptr     : deviceptr<'T>
                mutable Length  : int
            }

            member this.Item
                with    [<ReflectedDefinition>] get (idx:int) = this.Ptr.[idx] 
                and     [<ReflectedDefinition>] set (idx:int) (v:'T) = this.Ptr.[idx] <- v

            [<ReflectedDefinition>]
            static member inline Uninitialized<'T>() = { Ptr = __null<'T>(); Length = 0}

            [<ReflectedDefinition>]
            static member inline Init<'T>(length:int) =
                let s = __shared__.Array<'T>(length)
                let ptr = s |> __array_to_ptr
                { Ptr = ptr; Length = length }

        [<Record>]
        type API<'T> =
            {
                mutable warp_scan               : Alea.cuExtension.CUB.Warp.Scan.Template._TempStorage<'T>
                mutable warp_aggregates         : WarpAggregates<'T>
                mutable block_prefix            : Ref<'T>
            }

            [<ReflectedDefinition>]
            static member inline Uninitialized<'T>() =
                { 
                    warp_scan       = Alea.cuExtension.CUB.Warp.Scan.Template._TempStorage<'T>.Uninitialized<'T>()
                    warp_aggregates = WarpAggregates<'T>.Uninitialized<'T>()
                    block_prefix    = __null() |> __ptr_to_ref
                }
    
        type TempStorage<'T> = API<'T>

    [<AutoOpen>]
    module ThreadFields =
        open TempStorage

        [<Record>]
        type API<'T> =
            {
                mutable temp_storage    : TempStorage<'T>
                mutable linear_tid      : int
                mutable warp_id         : int
                mutable lane_id         : int
            }

            [<ReflectedDefinition>]
            static member Init(temp_storage, linear_tid, warp_id, lane_id) =
                {
                    temp_storage    = temp_storage
                    linear_tid      = linear_tid
                    warp_id         = warp_id
                    lane_id         = lane_id
                }

            [<ReflectedDefinition>]
            static member Uninitialized<'T>() = API<'T>.Init(TempStorage<'T>.Uninitialized<'T>(),0,0,0)

    type _TemplateParams<'T>    = Params.API<'T>
    type _Constants             = Constants.API
    type _TempStorage<'T>       = TempStorage<'T>
    type _ThreadFields<'T>      = ThreadFields.API<'T>

    type API<'T> =
        {
            TemplateParams  : _TemplateParams<'T>
            Constants       : _Constants
            TempStorage     : _TempStorage<'T>
            ThreadFields    : _ThreadFields<'T>
        }


module private Internal =
    open Template

    module internal Sig =
        module ApplyWarpAggregates =
            type Default<'T>            = Ref<'T> -> 'T -> Ref<'T> -> unit
            type WithLaneValidation<'T> = Ref<'T> -> 'T -> Ref<'T> -> bool -> unit

        module ExclusiveScan =
            type WithAggregate<'T>              = 'T -> Ref<'T> -> Ref<'T> -> Ref<'T> -> unit
            type WithAggregateAndCallbackOp<'T> = 'T -> Ref<'T> -> Ref<'T> -> Ref<'T> -> Ref<'T -> 'T> -> unit

            module Identityless =
                type WithAggregate<'T>              = 'T -> Ref<'T> -> Ref<'T> -> unit
                type WithAggregateAndCallbackOp<'T> = 'T -> Ref<'T> -> Ref<'T> -> Ref<'T -> 'T> -> unit

        module ExclusiveSum =
            type WithAggregate<'T>              = 'T -> Ref<'T> -> Ref<'T> -> unit
            type WithAggregateAndCallbackOp<'T> = 'T -> Ref<'T> -> Ref<'T> -> Ref<'T -> 'T> -> unit

        module InclusiveScan =
            type WithAggregate<'T>              = 'T -> Ref<'T> -> Ref<'T> -> unit
            type WithAggregateAndCallbackOp<'T> = 'T -> Ref<'T> -> Ref<'T> -> Ref<'T -> 'T> -> unit

        module InclusiveSum =
            type WithAggregate<'T>              = 'T -> Ref<'T> -> Ref<'T> -> unit
            type WithAggregateAndCallbackOp<'T> = 'T -> Ref<'T> -> Ref<'T> -> Ref<'T -> 'T> -> unit


    let [<ReflectedDefinition>] inline WarpScan (tp:_TemplateParams<'T>) =
        let WARPS = (_Constants.Init(tp)).WARPS
        Alea.cuExtension.CUB.Warp.Scan.Template._TemplateParams<'T>.Init(WARPS, CUB_PTX_WARP_THREADS, tp.scan_op)
        |>  WarpScan.api
        

module ApplyWarpAggregates =
    open Template
    open Internal

    [<Record>]
    type API<'T> =
        {
            Default             : Sig.ApplyWarpAggregates.Default<'T>
            WithLaneValidation  : Sig.ApplyWarpAggregates.WithLaneValidation<'T>
        }


    let [<ReflectedDefinition>] inline private Default (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (partial:Ref<'T>) (warp_aggregate:'T) (block_aggregate:Ref<'T>) =
        let c = _Constants.Init tp.BLOCK_THREADS
        let scan_op = tp.scan_op.op
        let lane_valid = true
        tf.temp_storage.warp_aggregates.[tf.warp_id] <- warp_aggregate
        __syncthreads()
        block_aggregate := tf.temp_storage.warp_aggregates.[0]
        
        for WARP = 1 to c.WARPS - 1 do
            if tf.warp_id = WARP then
                partial := if lane_valid then (!block_aggregate, !partial) ||> %scan_op else !block_aggregate
            block_aggregate := (!block_aggregate, tf.temp_storage.warp_aggregates.[WARP]) ||> %scan_op


    let [<ReflectedDefinition>] inline private WithLaneValidation (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (partial:Ref<'T>) (warp_aggregate:'T) (block_aggregate:Ref<'T>) (lane_valid:bool) = 
        let scan_op = tp.scan_op.op
        let c = _Constants.Init tp
        tf.temp_storage.warp_aggregates.[tf.warp_id] <- warp_aggregate

        __syncthreads()

        block_aggregate := tf.temp_storage.warp_aggregates.[0]

        for WARP = 1 to c.WARPS - 1 do
            if tf.warp_id = WARP then
                partial := if lane_valid then (!block_aggregate, !partial) ||> %scan_op else !block_aggregate
            block_aggregate := (!block_aggregate, tf.temp_storage.warp_aggregates.[WARP]) ||> %scan_op
        
        

    let [<ReflectedDefinition>] inline api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
        {
            Default             =   Default tp tf
            WithLaneValidation  =   WithLaneValidation tp tf
        }
        
   


module ExclusiveSum =
    open Template
    open Internal

    type API<'T> =
        {
            WithAggregate               : Sig.ExclusiveSum.WithAggregate<'T>
            WithAggregateAndCallbackOp  : Sig.ExclusiveSum.WithAggregateAndCallbackOp<'T>
        }
        
    let [<ReflectedDefinition>] inline private WithAggregate (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) =
        let warp_aggregate = __local__.Variable()
        (WarpScan tp).ExclusiveSum.WithAggregate input output warp_aggregate
        (ApplyWarpAggregates.api tp tf).WithLaneValidation output !warp_aggregate block_aggregate (tf.lane_id > 0)


    let [<ReflectedDefinition>] inline private WithAggregateAndCallbackOp (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
        let scan_op = tp.scan_op.op
        (WithAggregate tp tf) input output block_aggregate

        if tf.warp_id = 0 then 
            let block_prefix = !block_aggregate |> !block_prefix_callback_op
            if tf.lane_id = 0 then
                tf.temp_storage.block_prefix <- ref block_prefix

        __syncthreads()

        output := (!tf.temp_storage.block_prefix, !output) ||> %scan_op


    let [<ReflectedDefinition>] inline api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
        { 
            WithAggregate =                 WithAggregate tp tf
            WithAggregateAndCallbackOp =    WithAggregateAndCallbackOp tp tf
        }
        


module ExclusiveScan =
    open Template
    open Internal


    type API<'T> =
        {
            WithAggregate                       : Sig.ExclusiveScan.WithAggregate<'T>
            WithAggregate_NoID                  : Sig.ExclusiveScan.Identityless.WithAggregate<'T>
            WithAggregateAndCallbackOp          : Sig.ExclusiveScan.WithAggregateAndCallbackOp<'T>
            WithAggregateAndCallbackOp_NoID     : Sig.ExclusiveScan.Identityless.WithAggregateAndCallbackOp<'T>
        }


    let [<ReflectedDefinition>] inline private WithAggregate (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline private WithAggregateAndCallbackOp  (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)       
        (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()


    module private Identityless =
        let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams<'T>)
            (tf:_ThreadFields<'T>)
            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (tp:_TemplateParams<'T>)
            (tf:_ThreadFields<'T>)
            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()


    let [<ReflectedDefinition>] inline api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
        { 
            WithAggregate =                     WithAggregate tp tf
            WithAggregate_NoID =                Identityless.WithAggregate tp tf
            WithAggregateAndCallbackOp =        WithAggregateAndCallbackOp tp tf
            WithAggregateAndCallbackOp_NoID =   Identityless.WithAggregateAndCallbackOp tp tf
        }
    

module InclusiveSum =
    open Template
    open Internal
    
    type API<'T> =
        {
            WithAggregate               : Sig.InclusiveSum.WithAggregate<'T>
            WithAggregateAndCallbackOp  : Sig.InclusiveSum.WithAggregateAndCallbackOp<'T>
        }

    let [<ReflectedDefinition>] inline private WithAggregate (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline private WithAggregateAndCallbackOp (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()

    let [<ReflectedDefinition>] inline api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
        {
            WithAggregate =                 WithAggregate tp tf
            WithAggregateAndCallbackOp =    WithAggregateAndCallbackOp tp tf
        }


module InclusiveScan =
    open Template
    open Internal
    
    type API<'T> =
        {
            WithAggregate               : Sig.InclusiveScan.WithAggregate<'T>
            WithAggregateAndCallbackOp  : Sig.InclusiveScan.WithAggregateAndCallbackOp<'T>
        }


    let [<ReflectedDefinition>] inline private WithAggregate (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline private WithAggregateAndCallbackOp (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()

    let [<ReflectedDefinition>] inline api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
        {
            WithAggregate =                 WithAggregate tp tf
            WithAggregateAndCallbackOp =    WithAggregateAndCallbackOp tp tf
        }


module BlockScanWarpScan =
    open Template

    type API<'T> =
        {
            InclusiveSum    : InclusiveSum.API<'T>
            InclusiveScan   : InclusiveScan.API<'T>
            ExclusiveSum    : ExclusiveSum.API<'T>
            ExclusiveScan   : ExclusiveScan.API<'T>
        }

    let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
        {
            InclusiveSum    =   InclusiveSum.api tp tf
            InclusiveScan   =   InclusiveScan.api tp tf
            ExclusiveSum    =   ExclusiveSum.api tp tf
            ExclusiveScan   =   ExclusiveScan.api tp tf
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
//        fun (input:'T) (output:Ref<'T>) (identity:Ref<int> option) (scan_op:(int -> int -> int) option) (block_aggregate:Ref<int> option) (block_prefix_callback_op:Ref<int> option) ->
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
                