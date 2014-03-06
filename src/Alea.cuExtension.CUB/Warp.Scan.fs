[<AutoOpen>]
module Alea.cuExtension.CUB.Warp.Scan

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Common

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Warp.WarpSpecializations

type private InternalWarpScan<'T> =
    | WarpScanShfl of WarpScanShfl.API<'T>
    | WarpScanSmem of WarpScanSmem.API<'T>

module Template =
    [<AutoOpen>]
    module Params =
        [<Record>]
        type API =
            {
                LOGICAL_WARPS           : int
                LOGICAL_WARP_THREADS    : int
            }

            [<ReflectedDefinition>]
            member this.Get() = (this.LOGICAL_WARPS, this.LOGICAL_WARP_THREADS)

            [<ReflectedDefinition>]
            static member Init(logical_warps, logical_warp_threads) =
                {
                    LOGICAL_WARPS           = logical_warps
                    LOGICAL_WARP_THREADS    = logical_warp_threads                    
                }

            [<ReflectedDefinition>]
            static member Default() =
                {
                    LOGICAL_WARPS           = 1
                    LOGICAL_WARP_THREADS    = CUB_PTX_WARP_THREADS
                }

    [<AutoOpen>]
    module Constants =
        [<Record>]
        type API =
            {
                POW_OF_TWO : bool
            }

            [<ReflectedDefinition>]
            static member Init(tp:Params.API) = {POW_OF_TWO = ((tp.LOGICAL_WARP_THREADS &&& (tp.LOGICAL_WARP_THREADS - 1)) = 0)}


    [<AutoOpen>]
    module TempStorage =
        [<Record>]
        type API<'T> =
            {
                mutable Ptr     : deviceptr<'T>
                mutable Length  : int
            }

            member this.Item
                with    [<ReflectedDefinition>] get (idx:int) = this.Ptr.[idx]
                and     [<ReflectedDefinition>] set (idx:int) (v:'T) = this.Ptr.[idx] <- v

            [<ReflectedDefinition>]
            static member inline Uninitialized<'T>() = { Ptr = __null<'T>(); Length = 0}


    [<AutoOpen>]
    module ThreadFields =
        [<Record>]
        type API<'T> =
            {
                mutable temp_storage    : TempStorage.API<'T>
                mutable warp_id         : int
                mutable lane_id         : int
            }
                        
            [<ReflectedDefinition>] 
            member this.Get() = (this.temp_storage, this.warp_id, this.lane_id)


            [<ReflectedDefinition>]
            static member inline Init(logical_warps, logical_warp_threads) =
                {
                    temp_storage    = TempStorage.API<'T>.Uninitialized<'T>()
                    warp_id         = if logical_warps = 1 then 0 else threadIdx.x / logical_warp_threads
                    lane_id         = if logical_warps = 1 || logical_warp_threads = CUB_PTX_WARP_THREADS then __ptx__.LaneId() else threadIdx.x % logical_warp_threads
                }

            [<ReflectedDefinition>] 
            static member inline Init(tp:Params.API) = API<'T>.Init(tp.LOGICAL_WARPS, tp.LOGICAL_WARP_THREADS)



    type _TemplateParams    = Params.API
    type _Constants         = Constants.API
    type _TempStorage<'T>   = TempStorage.API<'T>
    type _ThreadFields<'T>  = ThreadFields.API<'T>



module private Internal =
    open Template
        
    module Sig =
        module InclusiveSum =
            type Default<'T>                    = 'T -> Ref<'T> -> unit
            type WithAggregate<'T>              = 'T -> Ref<'T> -> Ref<'T> -> unit
            type WithAggregateAndCallbackOp<'T> = 'T -> Ref<'T> -> Ref<'T> -> Ref<'T -> 'T> -> unit

        module ExclusiveSum =
            type Default<'T>                    = InclusiveSum.Default<'T>
            type WithAggregate<'T>              = InclusiveSum.WithAggregate<'T>
            type WithAggregateAndCallbackOp<'T> = InclusiveSum.WithAggregateAndCallbackOp<'T>

        module InclusiveScan =
            type Default<'T>                    = InclusiveSum.Default<'T>
            type WithAggregate<'T>              = InclusiveSum.WithAggregate<'T>
            type WithAggregateAndCallbackOp<'T> = InclusiveSum.WithAggregateAndCallbackOp<'T>

        module ExclusiveScan =
            type Default<'T>                    = 'T -> Ref<'T> -> 'T -> unit
            type WithAggregate<'T>              = 'T -> Ref<'T> -> 'T -> Ref<'T> -> unit
            type WithAggregateAndCallbackOp<'T> = 'T -> Ref<'T> -> 'T -> Ref<'T> -> Ref<'T -> 'T> -> unit

            module Identityless =
                type Default<'T>                    = ExclusiveSum.Default<'T>
                type WithAggregate<'T>              = ExclusiveSum.WithAggregate<'T>
                type WithAggregateAndCallbackOp<'T> = ExclusiveSum.WithAggregateAndCallbackOp<'T>


    let pickScanKind (tp:_TemplateParams) =
        let POW_OF_TWO = (_Constants.Init tp).POW_OF_TWO
        ((CUB_PTX_VERSION >= 300) && ((tp.LOGICAL_WARPS = 1) || POW_OF_TWO))

    let (|WarpScanShfl|_|) (tp:_TemplateParams) =
        if pickScanKind tp then
            let tp = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanShfl.Template._TemplateParams<'T>.Init(tp.LOGICAL_WARPS, tp.LOGICAL_WARP_THREADS, tp.scan_op)
            WarpScanShfl.api tp
            |>      Some
        else
            None 

    let (|WarpScanSmem|_|) (tp:_TemplateParams) =
        if pickScanKind tp |> not then
            let tp = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem.Template._TemplateParams<'T>.Init(tp.LOGICAL_WARPS, tp.LOGICAL_WARP_THREADS, tp.scan_op)
            WarpScanSmem.api tp
            |>      Some
        else
            None


        
    module WarpScan =
        open Template

        module InclusiveSum =
//            type API<'T> =
//                {
//                    Default                     : Sig.InclusiveSum.Default<'T>
//                    WithAggregate               : Sig.InclusiveSum.WithAggregate<'T>
//                    WithAggregateAndCallbackOp  : Sig.InclusiveSum.WithAggregateAndCallbackOp<'T>
//                }
            
            let [<ReflectedDefinition>] inline Default (tp:_TemplateParams)
                (tf:_ThreadFields<'T>) (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>) =
                tp |> function
                | WarpScanShfl wsShfl ->
                    let tf = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanShfl.Template._ThreadFields<'T>.Init(tf.warp_id, tf.lane_id)
                    (wsShfl tf).InclusiveSum.Default
                | WarpScanSmem wsSmem ->
                    let c = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem.WarpScanSmem.Constants.Init tp.LOGICAL_WARP_THREADS
                    let tf = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem.Template._ThreadFields<'T>.Init(tp.LOGICAL_WARPS, c.WARP_SMEM_ELEMENTS, tf.warp_id, tf.lane_id)
                    (wsSmem tf).InclusiveSum.Default
                | _ -> failwith "Invalid Template Parameters"

                <|| (input, output)

            let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams)
                (tf:_ThreadFields<'T>) (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
                tp |> function
                | WarpScanShfl wsShfl ->
                    let tf = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanShfl.Template._ThreadFields<'T>.Init<'T>(tf.warp_id, tf.lane_id)
                    (wsShfl tf).InclusiveSum.Generic
                | WarpScanSmem wsSmem ->
                    let c = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem.WarpScanSmem.Constants.Init tp.LOGICAL_WARP_THREADS
                    let tf = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem.Template._ThreadFields<'T>.Init<'T>(tp.LOGICAL_WARPS, c.WARP_SMEM_ELEMENTS, tf.warp_id, tf.lane_id)
                    (wsSmem tf).InclusiveSum.WithAggregate
                | _ -> failwith "Invalid Template Parameters"
            
                <|||    (input, output, warp_aggregate)
            

            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (tp:_TemplateParams)
                (tf:_ThreadFields<'T>) (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) (warp_prefix_op:Ref<'T -> 'T>) =
                let InclusiveSum =  WithAggregate tp tf scan_op input output warp_aggregate
                let prefix = __local__.Variable()
                prefix := !warp_aggregate |> !warp_prefix_op

                prefix :=
                    tp |> function
                    | WarpScanShfl wsShfl ->
                        let tf = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanShfl.Template._ThreadFields<'T>.Init(tf.warp_id, tf.lane_id )
                        (wsShfl tf).Broadcast
                    | WarpScanSmem wsSmem ->
                        let c = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem.WarpScanSmem.Constants.Init tp.LOGICAL_WARP_THREADS
                        let tf = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem.Template._ThreadFields<'T>.Init<'T>(tp.LOGICAL_WARPS, c.WARP_SMEM_ELEMENTS, tf.warp_id, tf.lane_id)
                        (wsSmem tf).Broadcast
                    | _ -> failwith "Invalid Template Parameters"
                    <|| (!prefix, 0)
                
                output := !prefix + !output
            
            [<Record>]
            type API<'T> =
                {
                    tp : _TemplateParams
                    tf  : _ThreadFields<'T>
                }

                [<ReflectedDefinition>]
                static member Init(tp) =
                    {
                        tp = tp
                        tf = _ThreadFields<'T>.Init(tp.LOGICAL_WARPS, tp.LOGICAL_WARP_THREADS)
                    }

                [<ReflectedDefinition>]
                member this.Default = Default this.tp

                [<ReflectedDefinition>]
                member this.WithAggregate = WithAggregate this.tp

//                [<ReflectedDefinition>]
//                member this.WithAggregateAndCallbackOp =  WithAggregateAndCallbackOp this.tp this.tf

//            let [<ReflectedDefinition>] api (tp:_TemplateParams)
//                (scan_op:'T -> 'T -> 'T) =
//                    {
//                        Default                         =   Default tp scan_op
//                        WithAggregate                   =   WithAggregate tp scan_op
//                        WithAggregateAndCallbackOp      =   WithAggregateAndCallbackOp tp scan_op
//                    }

        module ExclusiveSum =
            let [<ReflectedDefinition>] inline Default (tp:_TemplateParams)
                (tf:_ThreadFields<'T>) (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>) =
                    let inclusive = __local__.Variable()
                    InclusiveSum.Default tp tf scan_op input inclusive
                    output := !inclusive - input
                
               

            let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams)
                (tf:_ThreadFields<'T>) (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
                    let inclusive = __local__.Variable()
                    InclusiveSum.WithAggregate tp tf scan_op input inclusive warp_aggregate
                    output := !inclusive - input


            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (tp:_TemplateParams)
                (tf:_ThreadFields<'T>) (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) (warp_prefix_op:Ref<'T -> 'T>) =
                    let inclusive = __local__.Variable()
                    InclusiveSum.WithAggregateAndCallbackOp tp tf scan_op input inclusive warp_aggregate warp_prefix_op
                    output := !inclusive - input
                    
//            type API<'T> =
//                {
//                    tp  : _TemplateParams
//                    tf  : _ThreadFields<'T>
//                }
//
//                [<ReflectedDefinition>]
//                static member Create(tp, tf) = {tp = tp; tf = tf}
//        
//                [<ReflectedDefinition>]
//                member inline this.Default = Default this.tp this.tf
//
//                [<ReflectedDefinition>]
//                member this.WithAggregate = WithAggregate this.tp this.tf
//            let [<ReflectedDefinition>] api (tp:_TemplateParams)
//                (scan_op:'T -> 'T -> 'T) =
//                    {
//                        Default =                       Default tp scan_op
//                        WithAggregate =                 WithAggregate tp scan_op
//                        WithAggregateAndCallbackOp =    WithAggregateAndCallbackOp tp scan_op
//                    }


module private InclusiveSum =
    open Template
    open Internal

//    type API<'T> =
//        {
//            Default                     : Sig.InclusiveSum.Default<'T>
//            WithAggregate               : Sig.InclusiveSum.WithAggregate<'T>
//            WithAggregateAndCallbackOp  : Sig.InclusiveSum.WithAggregateAndCallbackOp<'T>
//        }

    let [<ReflectedDefinition>] inline Default (tp:_TemplateParams)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp  (tp:_TemplateParams)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()


//    let [<ReflectedDefinition>] api  (tp:_TemplateParams)
//        (scan_op:'T -> 'T -> 'T) =
//            {
//                Default                     =   Default tp scan_op
//                WithAggregate               =   WithAggregate tp scan_op
//                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp tp scan_op
//            }
        



module private ExclusiveSum =
    open Template
    open Internal

//    type API<'T> =
//        {
//            Default                     : Sig.ExclusiveSum.Default<'T>
//            WithAggregate               : Sig.ExclusiveSum.WithAggregate<'T>
//            WithAggregateAndCallbackOp  : Sig.ExclusiveSum.WithAggregateAndCallbackOp<'T>
//        }

    let [<ReflectedDefinition>] inline Default (tp:_TemplateParams)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (tp:_TemplateParams)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()


//    let [<ReflectedDefinition>] api  (tp:_TemplateParams)
//        (scan_op:'T -> 'T -> 'T) =
//            {
//                Default                     =   Default tp scan_op
//                WithAggregate               =   WithAggregate tp scan_op
//                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp tp scan_op
//            }


module private ExclusiveScan =
    open Template
    open Internal

//    type API<'T> =
//        {
//            Default                         : Sig.ExclusiveScan.Default<'T>
//            Default_NoID                    : Sig.ExclusiveScan.Identityless.Default<'T>
//            WithAggregate                   : Sig.ExclusiveScan.WithAggregate<'T>
//            WithAggregate_NoID              : Sig.ExclusiveScan.Identityless.WithAggregate<'T>
//            WithAggregateAndCallbackOp      : Sig.ExclusiveScan.WithAggregateAndCallbackOp<'T>
//            WithAggregateAndCallbackOp_NoID : Sig.ExclusiveScan.Identityless.WithAggregateAndCallbackOp<'T>
//        }

    let [<ReflectedDefinition>] inline Default (tp:_TemplateParams)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (identity:'T) = ()

    let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (identity:'T) (block_aggregate:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (tp:_TemplateParams)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (identity:'T) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()

    module Identityless =
        let [<ReflectedDefinition>] inline Default (tp:_TemplateParams)
            (scan_op:'T -> 'T -> 'T)
            (input:'T) (output:Ref<'T>) = ()

        let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams)
            (scan_op:'T -> 'T -> 'T)
            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (tp:_TemplateParams)
            (scan_op:'T -> 'T -> 'T)
            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()


//    let [<ReflectedDefinition>] api (tp:_TemplateParams)
//        (scan_op:'T -> 'T -> 'T) =
//            {
//                Default                          = Default tp scan_op
//                Default_NoID                     = Identityless.Default tp scan_op
//                WithAggregate                    = WithAggregate tp scan_op
//                WithAggregate_NoID               = Identityless.WithAggregate tp scan_op      
//                WithAggregateAndCallbackOp       = WithAggregateAndCallbackOp tp scan_op
//                WithAggregateAndCallbackOp_NoID  = Identityless.WithAggregateAndCallbackOp tp scan_op
//            }


module private InclusiveScan =
    open Template
    open Internal

//    type API<'T> =
//        {
//            Default                     : Sig.InclusiveScan.Default<'T>
//            WithAggregate               : Sig.InclusiveScan.WithAggregate<'T>
//            WithAggregateAndCallbackOp  : Sig.InclusiveScan.WithAggregateAndCallbackOp<'T>
//        }

    let [<ReflectedDefinition>] inline Default (tp:_TemplateParams)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp  (tp:_TemplateParams)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()


//    let [<ReflectedDefinition>] api  (tp:_TemplateParams)
//        (scan_op:'T -> 'T -> 'T) =
//            {
//                Default                     =   Default tp scan_op
//                WithAggregate               =   WithAggregate tp scan_op
//                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp tp scan_op
//            }


module WarpScan =
    open Template

    module InclusiveSum = 
        let [<ReflectedDefinition>] inline Default (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let tp = _TemplateParams.Init(logical_warps, logical_warp_threads)
            InclusiveSum.Default tp scan_op

        let [<ReflectedDefinition>] inline WithAggregate (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let tp = _TemplateParams.Init(logical_warps, logical_warp_threads)
            InclusiveSum.WithAggregate tp scan_op

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let tp = _TemplateParams.Init(logical_warps, logical_warp_threads)
            InclusiveSum.WithAggregateAndCallbackOp tp scan_op

    module InclusiveScan =
        let [<ReflectedDefinition>] inline Default (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let tp = _TemplateParams.Init(logical_warps, logical_warp_threads)
            InclusiveScan.Default tp scan_op

        let [<ReflectedDefinition>] inline WithAggregate (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let tp = _TemplateParams.Init(logical_warps, logical_warp_threads)
            InclusiveScan.WithAggregate tp scan_op

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) =  
            let tp = _TemplateParams.Init(logical_warps, logical_warp_threads)
            InclusiveScan.WithAggregateAndCallbackOp tp scan_op

    module ExclusiveSum =
        let [<ReflectedDefinition>] inline Default (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let tp = _TemplateParams.Init(logical_warps, logical_warp_threads)
            ExclusiveSum.Default tp scan_op

        let [<ReflectedDefinition>] inline WithAggregate (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let tp = _TemplateParams.Init(logical_warps, logical_warp_threads)
            ExclusiveSum.WithAggregate tp scan_op

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) =  
            let tp = _TemplateParams.Init(logical_warps, logical_warp_threads)
            ExclusiveSum.WithAggregateAndCallbackOp tp scan_op

    module ExclusiveScan =
        let [<ReflectedDefinition>] inline Default (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let tp = _TemplateParams.Init(logical_warps, logical_warp_threads)
            ExclusiveScan.Default tp scan_op

        let [<ReflectedDefinition>] inline WithAggregate (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let tp = _TemplateParams.Init(logical_warps, logical_warp_threads)
            ExclusiveScan.WithAggregate tp scan_op

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) =  
            let tp = _TemplateParams.Init(logical_warps, logical_warp_threads)
            ExclusiveScan.WithAggregateAndCallbackOp tp scan_op

//    [<Record>]
//    type API<'T> =
//        {
//            Constants       : _Constants
//            ThreadFields    : _ThreadFields<'T>
//            InclusiveSum    : InclusiveSum.API<'T>
//            InclusiveScan   : InclusiveScan.API<'T>
//            ExclusiveSum    : ExclusiveSum.API<'T>
//            ExclusiveScan   : ExclusiveScan.API<'T>
//        }

//        [<ReflectedDefinition>]
//        member this.Init(temp_storage) = 
//            this.ThreadFields.temp_storage <- temp_storage
//            this
//                    
//        [<ReflectedDefinition>]
//        member this.Init(warp_id, lane_id) = 
//            this.ThreadFields.warp_id <- warp_id
//            this.ThreadFields.lane_id <- lane_id
//            this
//
//        [<ReflectedDefinition>]
//        member this.Init(temp_storage, warp_id, lane_id) = 
//            this.ThreadFields.temp_storage  <- temp_storage
//            this.ThreadFields.warp_id       <- warp_id
//            this.ThreadFields.lane_id       <- lane_id
//            this
//
//        [<ReflectedDefinition>]
//        static member Create(logical_warps, logical_warp_threads, scan_op) =
//            let tp = _TemplateParams<'T>.Init(logical_warps, logical_warp_threads)
//            let tf = _ThreadFields<'T>.Init(tp)
//            {
//                Constants       =   _Constants.Init tp
//                ThreadFields    =   tf
//                InclusiveSum    =   InclusiveSum.api tp scan_op                                    
//                InclusiveScan   =   InclusiveScan.api tp scan_op
//                ExclusiveSum    =   ExclusiveSum.api tp scan_op
//                ExclusiveScan   =   ExclusiveScan.api tp scan_op
//            }
//
//        [<ReflectedDefinition>]
//        static member Create(tp:_TemplateParams) = API<'T>.Create(tp.LOGICAL_WARPS, tp.LOGICAL_WARP_THREADS)
//
//
//
//    let [<ReflectedDefinition>] api (tp:_TemplateParams) =
//        let tf = _ThreadFields<'T>.Init(tp)
//        {
//            Constants       =   _Constants.Init tp
//            ThreadFields    =   tf
//            InclusiveSum    =   InclusiveSum.api tp scan_op                                    
//            InclusiveScan   =   InclusiveScan.api tp scan_op
//            ExclusiveSum    =   ExclusiveSum.api tp scan_op
//            ExclusiveScan   =   ExclusiveScan.api tp scan_op
//        }