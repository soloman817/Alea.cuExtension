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
            static member Init(tp:Params.API) = {POW_OF_TWO = ((LOGICAL_WARP_THREADS &&& (LOGICAL_WARP_THREADS - 1)) = 0)}


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
            static member inline Init(tp:Params.API) = API<'T>.Init(tp.LOGICAL_WARPS, LOGICAL_WARP_THREADS)



    type _TemplateParams    = Params.API
    type _Constants         = Constants.API
    type _TempStorage<'T>   = TempStorage.API<'T>
    type _ThreadFields<'T>  = ThreadFields.API<'T>

    type API<'T> =
        {
            mutable Params          : Params.API
            mutable Constants       : Constants.API
            mutable ThreadFields    : ThreadFields.API<'T>
        }

type _Template<'T> = Template.API<'T>

module private Internal =
    open Template

    let pickScanKind (template:_Template<'T>) =
        let POW_OF_TWO = template.Constants.POW_OF_TWO
        ((CUB_PTX_VERSION >= 300) && ((template.Params.LOGICAL_WARPS = 1) || POW_OF_TWO))

    let (|WarpScanShfl|_|) (template:_Template<'T>) =
        if pickScanKind template then
            let template = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanShfl._Template<'T>.Init(template.Params.LOGICAL_WARPS, template.Params.LOGICAL_WARP_THREADS)
            WarpScanShfl.API<'T>.Init(template)
            |>      Some
        else
            None 

    let (|WarpScanSmem|_|) (template:_Template<'T>) =
        if pickScanKind template |> not then
            let template = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem._Template<'T>.Init(template.Params.LOGICAL_WARPS, template.Params.LOGICAL_WARP_THREADS)
            WarpScanSmem.API<'T>.Init(template)
            |>      Some
        else
            None


        
    module WarpScan =
        open Template

        module InclusiveSum =
            
            let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
                (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>) =
                template |> function
                | WarpScanShfl wsShfl ->    wsShfl.InclusiveSum.Default
                | WarpScanSmem wsSmem ->    wsSmem.InclusiveSum.Default scan_op
                | _ -> failwith "Invalid Template Parameters"


            let [<ReflectedDefinition>] inline WithAggregate (template:_Template<'T>)
                 (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
                template |> function
                | WarpScanShfl wsShfl ->
                    let tf = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanShfl.Template._ThreadFields<'T>.Init<'T>(tf.warp_id, tf.lane_id)
                    (wsShfl tf).InclusiveSum.Generic
                | WarpScanSmem wsSmem ->
                    let c = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem.WarpScanSmem.Constants.Init LOGICAL_WARP_THREADS
                    let tf = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem.Template._ThreadFields<'T>.Init<'T>(tp.LOGICAL_WARPS, c.WARP_SMEM_ELEMENTS, tf.warp_id, tf.lane_id)
                    (wsSmem tf).InclusiveSum.WithAggregate
                | _ -> failwith "Invalid Template Parameters"
            
                <|||    (input, output, warp_aggregate)
            

            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (template:_Template<'T>)
                 (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) (warp_prefix_op:Ref<'T -> 'T>) =
                let InclusiveSum =  WithAggregate template tf scan_op input output warp_aggregate
                let prefix = __local__.Variable()
                prefix := !warp_aggregate |> !warp_prefix_op

                prefix :=
                    template |> function
                    | WarpScanShfl wsShfl ->
                        let tf = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanShfl.Template._ThreadFields<'T>.Init(tf.warp_id, tf.lane_id )
                        (wsShfl tf).Broadcast
                    | WarpScanSmem wsSmem ->
                        let c = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem.WarpScanSmem.Constants.Init LOGICAL_WARP_THREADS
                        let tf = Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem.Template._ThreadFields<'T>.Init<'T>(tp.LOGICAL_WARPS, c.WARP_SMEM_ELEMENTS, tf.warp_id, tf.lane_id)
                        (wsSmem tf).Broadcast
                    | _ -> failwith "Invalid Template Parameters"
                    <|| (!prefix, 0)
                
                output := !prefix + !output
            
            [<Record>]
            type API<'T> =
                {
                    template : _TemplateParams
                    tf  : _ThreadFields<'T>
                }

                [<ReflectedDefinition>]
                static member Init(tp) =
                    {
                        template = template
                        tf = _ThreadFields<'T>.Init(tp.LOGICAL_WARPS, LOGICAL_WARP_THREADS)
                    }

                [<ReflectedDefinition>]
                member this.Default = Default this.tp

                [<ReflectedDefinition>]
                member this.WithAggregate = WithAggregate this.tp



        module ExclusiveSum =
            let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
                 (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>) =
                    let inclusive = __local__.Variable()
                    InclusiveSum.Default template tf scan_op input inclusive
                    output := !inclusive - input
                
               

            let [<ReflectedDefinition>] inline WithAggregate (template:_Template<'T>)
                 (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
                    let inclusive = __local__.Variable()
                    InclusiveSum.WithAggregate template tf scan_op input inclusive warp_aggregate
                    output := !inclusive - input


            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (template:_Template<'T>)
                 (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) (warp_prefix_op:Ref<'T -> 'T>) =
                    let inclusive = __local__.Variable()
                    InclusiveSum.WithAggregateAndCallbackOp template tf scan_op input inclusive warp_aggregate warp_prefix_op
                    output := !inclusive - input
                    
//            type API<'T> =
//                {
//                    template  : _TemplateParams
//                    tf  : _ThreadFields<'T>
//                }
//
//                [<ReflectedDefinition>]
//                static member Create(tp, tf) = {tp = template; tf = tf}
//        
//                [<ReflectedDefinition>]
//                member inline this.Default = Default this.tp this.tf
//
//                [<ReflectedDefinition>]
//                member this.WithAggregate = WithAggregate this.tp this.tf
//            let [<ReflectedDefinition>] api (template:_Template<'T>)
//                (scan_op:'T -> 'T -> 'T) =
//                    {
//                        Default =                       Default template scan_op
//                        WithAggregate =                 WithAggregate template scan_op
//                        WithAggregateAndCallbackOp =    WithAggregateAndCallbackOp template scan_op
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

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline WithAggregate (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp  (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()


//    let [<ReflectedDefinition>] api  (template:_Template<'T>)
//        (scan_op:'T -> 'T -> 'T) =
//            {
//                Default                     =   Default template scan_op
//                WithAggregate               =   WithAggregate template scan_op
//                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp template scan_op
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

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline WithAggregate (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()


//    let [<ReflectedDefinition>] api  (template:_Template<'T>)
//        (scan_op:'T -> 'T -> 'T) =
//            {
//                Default                     =   Default template scan_op
//                WithAggregate               =   WithAggregate template scan_op
//                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp template scan_op
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

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (identity:'T) = ()

    let [<ReflectedDefinition>] inline WithAggregate (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (identity:'T) (block_aggregate:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (identity:'T) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()

    module Identityless =
        let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
            (scan_op:'T -> 'T -> 'T)
            (input:'T) (output:Ref<'T>) = ()

        let [<ReflectedDefinition>] inline WithAggregate (template:_Template<'T>)
            (scan_op:'T -> 'T -> 'T)
            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (template:_Template<'T>)
            (scan_op:'T -> 'T -> 'T)
            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()


//    let [<ReflectedDefinition>] api (template:_Template<'T>)
//        (scan_op:'T -> 'T -> 'T) =
//            {
//                Default                          = Default template scan_op
//                Default_NoID                     = Identityless.Default template scan_op
//                WithAggregate                    = WithAggregate template scan_op
//                WithAggregate_NoID               = Identityless.WithAggregate template scan_op      
//                WithAggregateAndCallbackOp       = WithAggregateAndCallbackOp template scan_op
//                WithAggregateAndCallbackOp_NoID  = Identityless.WithAggregateAndCallbackOp template scan_op
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

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline WithAggregate (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

    let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp  (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()


//    let [<ReflectedDefinition>] api  (template:_Template<'T>)
//        (scan_op:'T -> 'T -> 'T) =
//            {
//                Default                     =   Default template scan_op
//                WithAggregate               =   WithAggregate template scan_op
//                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp template scan_op
//            }


module WarpScan =
    open Template

    module InclusiveSum = 
        let [<ReflectedDefinition>] inline Default (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let template = _TemplateParams.Init(logical_warps, logical_warp_threads)
            InclusiveSum.Default template scan_op

        let [<ReflectedDefinition>] inline WithAggregate (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let template = _TemplateParams.Init(logical_warps, logical_warp_threads)
            InclusiveSum.WithAggregate template scan_op

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let template = _TemplateParams.Init(logical_warps, logical_warp_threads)
            InclusiveSum.WithAggregateAndCallbackOp template scan_op

    module InclusiveScan =
        let [<ReflectedDefinition>] inline Default (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let template = _TemplateParams.Init(logical_warps, logical_warp_threads)
            InclusiveScan.Default template scan_op

        let [<ReflectedDefinition>] inline WithAggregate (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let template = _TemplateParams.Init(logical_warps, logical_warp_threads)
            InclusiveScan.WithAggregate template scan_op

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) =  
            let template = _TemplateParams.Init(logical_warps, logical_warp_threads)
            InclusiveScan.WithAggregateAndCallbackOp template scan_op

    module ExclusiveSum =
        let [<ReflectedDefinition>] inline Default (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let template = _TemplateParams.Init(logical_warps, logical_warp_threads)
            ExclusiveSum.Default template scan_op

        let [<ReflectedDefinition>] inline WithAggregate (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let template = _TemplateParams.Init(logical_warps, logical_warp_threads)
            ExclusiveSum.WithAggregate template scan_op

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) =  
            let template = _TemplateParams.Init(logical_warps, logical_warp_threads)
            ExclusiveSum.WithAggregateAndCallbackOp template scan_op

    module ExclusiveScan =
        let [<ReflectedDefinition>] inline Default (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let template = _TemplateParams.Init(logical_warps, logical_warp_threads)
            ExclusiveScan.Default template scan_op

        let [<ReflectedDefinition>] inline WithAggregate (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
            let template = _TemplateParams.Init(logical_warps, logical_warp_threads)
            ExclusiveScan.WithAggregate template scan_op

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) =  
            let template = _TemplateParams.Init(logical_warps, logical_warp_threads)
            ExclusiveScan.WithAggregateAndCallbackOp template scan_op

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
//            let template = _TemplateParams<'T>.Init(logical_warps, logical_warp_threads)
//            let tf = _ThreadFields<'T>.Init(tp)
//            {
//                Constants       =   _Constants.Init template
//                ThreadFields    =   tf
//                InclusiveSum    =   InclusiveSum.api template scan_op                                    
//                InclusiveScan   =   InclusiveScan.api template scan_op
//                ExclusiveSum    =   ExclusiveSum.api template scan_op
//                ExclusiveScan   =   ExclusiveScan.api template scan_op
//            }
//
//        [<ReflectedDefinition>]
//        static member Create(template:_Template<'T>) = API<'T>.Create(tp.LOGICAL_WARPS, LOGICAL_WARP_THREADS)
//
//
//
//    let [<ReflectedDefinition>] api (template:_Template<'T>) =
//        let tf = _ThreadFields<'T>.Init(tp)
//        {
//            Constants       =   _Constants.Init template
//            ThreadFields    =   tf
//            InclusiveSum    =   InclusiveSum.api template scan_op                                    
//            InclusiveScan   =   InclusiveScan.api template scan_op
//            ExclusiveSum    =   ExclusiveSum.api template scan_op
//            ExclusiveScan   =   ExclusiveScan.api template scan_op
//        }