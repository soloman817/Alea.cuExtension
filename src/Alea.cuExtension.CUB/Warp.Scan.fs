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
        type API<'T> =
            {
                LOGICAL_WARPS           : int
                LOGICAL_WARP_THREADS    : int
                scan_op                 : IScanOp<'T> //('T -> 'T -> 'T)
            }

            [<ReflectedDefinition>]
            member this.Get() = (this.LOGICAL_WARPS, this.LOGICAL_WARP_THREADS, this.scan_op)

            [<ReflectedDefinition>]
            static member Init(logical_warps, logical_warp_threads, scan_op) =
                {
                    LOGICAL_WARPS           = logical_warps
                    LOGICAL_WARP_THREADS    = logical_warp_threads
                    scan_op                 = scan_op
                }

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

    type TempStorage<'T> = TempStorage.API<'T>

    [<AutoOpen>]
    module ThreadFields =
        [<Record>]
        type API<'T> =
            {
                mutable temp_storage    : TempStorage<'T>
                mutable warp_id         : int
                mutable lane_id         : int
            }
                        
            [<ReflectedDefinition>] 
            member this.Get() = (this.temp_storage, this.warp_id, this.lane_id)

            [<ReflectedDefinition>]
            static member inline Init(logical_warps, logical_warp_threads, scan_op) =
                {
                    temp_storage    = TempStorage<'T>.Uninitialized<'T>()
                    warp_id         = if logical_warps = 1 then 0 else threadIdx.x / logical_warp_threads
                    lane_id         = if logical_warps = 1 || logical_warp_threads = CUB_PTX_WARP_THREADS then __ptx__.LaneId() else threadIdx.x % logical_warp_threads
                }

            [<ReflectedDefinition>] 
            static member inline Init(tp:Params.API<'T>) = API<'T>.Init(tp.LOGICAL_WARPS, tp.LOGICAL_WARP_THREADS, tp.scan_op)


    type _TemplateParams<'T>    = Params.API<'T>
    type _TempStorage<'T>       = TempStorage.API<'T>
    type _ThreadFields<'T>      = ThreadFields.API<'T>



module private Internal =
    open Template

    module Constants =
        let POW_OF_TWO =
            fun logical_warp_threads -> ((logical_warp_threads &&& (logical_warp_threads - 1)) = 0)

    
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


    let pickScanKind (tp:_TemplateParams<'T>) =
        let POW_OF_TWO = tp.LOGICAL_WARP_THREADS |> Constants.POW_OF_TWO
        ((CUB_PTX_VERSION >= 300) && ((tp.LOGICAL_WARPS = 1) || POW_OF_TWO))

    let (|WarpScanShfl|_|) (tp:_TemplateParams<'T>) =
        if pickScanKind tp then
            let tp = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanShfl.Template._TemplateParams<'T>.Init(tp.LOGICAL_WARPS, tp.LOGICAL_WARP_THREADS, tp.scan_op)
            WarpScanShfl.api tp
            |>      Some
        else
            None 

    let (|WarpScanSmem|_|) (tp:_TemplateParams<'T>) =
        if pickScanKind tp |> not then
            let tp = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem.Template._TemplateParams<'T>.Init(tp.LOGICAL_WARPS, tp.LOGICAL_WARP_THREADS, tp.scan_op)
            WarpScanSmem.api tp
            |>      Some
        else
            None


        
    module WarpScan =
        open Template

        module InclusiveSum =
            type API<'T> =
                {
                    Default                     : Sig.InclusiveSum.Default<'T>
                    WithAggregate               : Sig.InclusiveSum.WithAggregate<'T>
                    WithAggregateAndCallbackOp  : Sig.InclusiveSum.WithAggregateAndCallbackOp<'T>
                }
            
            let [<ReflectedDefinition>] inline private Default (tp:_TemplateParams<'T>)
                (tf:_ThreadFields<'T>)
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

            let [<ReflectedDefinition>] inline private WithAggregate (tp:_TemplateParams<'T>)
                (tf:_ThreadFields<'T>)
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
            

            let [<ReflectedDefinition>] inline private WithAggregateAndCallbackOp (tp:_TemplateParams<'T>)
                (tf:_ThreadFields<'T>)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) (warp_prefix_op:Ref<'T -> 'T>) =
                
                let InclusiveSum =  WithAggregate tp tf input output warp_aggregate
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
            

            let [<ReflectedDefinition>] inline api (tp:_TemplateParams<'T>)
                (tf:_ThreadFields<'T>) =
                    {
                        Default                         =   Default tp tf
                        WithAggregate                   =   WithAggregate tp tf
                        WithAggregateAndCallbackOp      =   WithAggregateAndCallbackOp tp tf
                    }

        module ExclusiveSum =
            type API<'T> =
                {
                    Default                     : Sig.ExclusiveSum.Default<'T>
                    WithAggregate               : Sig.ExclusiveSum.WithAggregate<'T>
                    WithAggregateAndCallbackOp  : Sig.ExclusiveSum.WithAggregateAndCallbackOp<'T>
                }


            let [<ReflectedDefinition>] inline private Default (tp:_TemplateParams<'T>)
                (tf:_ThreadFields<'T>)
                (input:'T) (output:Ref<'T>) =
                    let inclusive = __local__.Variable()
                    (InclusiveSum.api tp tf).Default input inclusive
                    output := !inclusive - input
                
               

            let [<ReflectedDefinition>] inline private WithAggregate (tp:_TemplateParams<'T>)
                (tf:_ThreadFields<'T>)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
                    let inclusive = __local__.Variable()
                    (InclusiveSum.api tp tf).WithAggregate input inclusive warp_aggregate
                    output := !inclusive - input


            let [<ReflectedDefinition>] inline private WithAggregateAndCallbackOp (tp:_TemplateParams<'T>)
                (tf:_ThreadFields<'T>)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) (warp_prefix_op:Ref<'T -> 'T>) =
                    let inclusive = __local__.Variable()
                    (InclusiveSum.api tp tf).WithAggregateAndCallbackOp input inclusive warp_aggregate warp_prefix_op
                    output := !inclusive - input
                    

            let [<ReflectedDefinition>] inline api (tp:_TemplateParams<'T>)
                (tf:_ThreadFields<'T>) =
                    {
                        Default =                       Default tp tf
                        WithAggregate =                 WithAggregate tp tf
                        WithAggregateAndCallbackOp =    WithAggregateAndCallbackOp tp tf
                    }


module InclusiveSum =
    open Template
    open Internal

    type API<'T> =
        {
            Default                     : Sig.InclusiveSum.Default<'T>
            WithAggregate               : Sig.InclusiveSum.WithAggregate<'T>
            WithAggregateAndCallbackOp  : Sig.InclusiveSum.WithAggregateAndCallbackOp<'T>
        }

    let [<ReflectedDefinition>] inline private Default (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline private WithAggregate (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline private WithAggregateAndCallbackOp  (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()


    let [<ReflectedDefinition>] inline api  (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
            {
                Default                     =   Default tp tf
                WithAggregate               =   WithAggregate tp tf
                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp tp tf
            }
        



module ExclusiveSum =
    open Template
    open Internal

    type API<'T> =
        {
            Default                     : Sig.ExclusiveSum.Default<'T>
            WithAggregate               : Sig.ExclusiveSum.WithAggregate<'T>
            WithAggregateAndCallbackOp  : Sig.ExclusiveSum.WithAggregateAndCallbackOp<'T>
        }

    let [<ReflectedDefinition>] inline private Default (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline private WithAggregate (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline private WithAggregateAndCallbackOp (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()


    let [<ReflectedDefinition>] inline api  (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
            {
                Default                     =   Default tp tf
                WithAggregate               =   WithAggregate tp tf
                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp tp tf
            }


module ExclusiveScan =
    open Template
    open Internal

    type API<'T> =
        {
            Default                         : Sig.ExclusiveScan.Default<'T>
            Default_NoID                    : Sig.ExclusiveScan.Identityless.Default<'T>
            WithAggregate                   : Sig.ExclusiveScan.WithAggregate<'T>
            WithAggregate_NoID              : Sig.ExclusiveScan.Identityless.WithAggregate<'T>
            WithAggregateAndCallbackOp      : Sig.ExclusiveScan.WithAggregateAndCallbackOp<'T>
            WithAggregateAndCallbackOp_NoID : Sig.ExclusiveScan.Identityless.WithAggregateAndCallbackOp<'T>
        }

    let [<ReflectedDefinition>] inline private Default (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (identity:'T) = ()

    let [<ReflectedDefinition>] inline private WithAggregate (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (identity:'T) (block_aggregate:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline private WithAggregateAndCallbackOp (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (identity:'T) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()

    module private Identityless =
        let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>)
            (tf:_ThreadFields<'T>)
            (input:'T) (output:Ref<'T>) = ()

        let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams<'T>)
            (tf:_ThreadFields<'T>)
            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (tp:_TemplateParams<'T>)
            (tf:_ThreadFields<'T>)
            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()


    let [<ReflectedDefinition>] inline api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
            {
                Default                          = Default tp tf
                Default_NoID                     = Identityless.Default tp tf
                WithAggregate                    = WithAggregate tp tf
                WithAggregate_NoID               = Identityless.WithAggregate tp tf      
                WithAggregateAndCallbackOp       = WithAggregateAndCallbackOp tp tf
                WithAggregateAndCallbackOp_NoID  = Identityless.WithAggregateAndCallbackOp tp tf
            }


module InclusiveScan =
    open Template
    open Internal

    type API<'T> =
        {
            Default                     : Sig.InclusiveScan.Default<'T>
            WithAggregate               : Sig.InclusiveScan.WithAggregate<'T>
            WithAggregateAndCallbackOp  : Sig.InclusiveScan.WithAggregateAndCallbackOp<'T>
        }

    let [<ReflectedDefinition>] inline private Default (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline private WithAggregate (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline private WithAggregateAndCallbackOp  (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()


    let [<ReflectedDefinition>] inline api  (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
            {
                Default                     =   Default tp tf
                WithAggregate               =   WithAggregate tp tf
                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp tp tf
            }


module WarpScan =
    open Template

    type API<'T> =
        {
            InclusiveSum    : InclusiveSum.API<'T>
            InclusiveScan   : InclusiveScan.API<'T>
            ExclusiveSum    : ExclusiveSum.API<'T>
            ExclusiveScan   : ExclusiveScan.API<'T>
        }


    let [<ReflectedDefinition>] inline api (tp:_TemplateParams<'T>) =
        let tf = _ThreadFields<'T>.Init(tp)
        {
            InclusiveSum    =   InclusiveSum.api tp tf                                    
            InclusiveScan   =   InclusiveScan.api tp tf
            ExclusiveSum    =   ExclusiveSum.api tp tf
            ExclusiveScan   =   ExclusiveScan.api tp tf
        }