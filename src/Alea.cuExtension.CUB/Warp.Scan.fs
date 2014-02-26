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

type private InternalWarpScan =
    | WarpScanShfl of WarpScanShfl.API
    | WarpScanSmem of WarpScanSmem.API

module private Internal =
    module Constants =
        let POW_OF_TWO =
            fun logical_warp_threads -> ((logical_warp_threads &&& (logical_warp_threads - 1)) = 0)

    
    module Sig =
        module InclusiveSum =
            type DefaultExpr                    = Expr<int -> Ref<int> -> unit>
            type WithAggregateExpr              = Expr<int -> Ref<int> -> Ref<int> -> unit>
            type WithAggregateAndCallbackOpExpr = Expr<int -> Ref<int> -> Ref<int> -> Ref<int -> int> -> unit>

        module ExclusiveSum =
            type DefaultExpr                    = Expr<int -> Ref<int> -> unit>
            type WithAggregateExpr              = Expr<int -> Ref<int> -> Ref<int> -> unit>
            type WithAggregateAndCallbackOpExpr = Expr<int -> Ref<int> -> Ref<int> -> Ref<int -> int> -> unit>

        module InclusiveScan =
            type DefaultExpr                    = Expr<int -> Ref<int> -> unit>
            type WithAggregateExpr              = Expr<int -> Ref<int> -> Ref<int> -> unit>
            type WithAggregateAndCallbackOpExpr = Expr<int -> Ref<int> -> Ref<int> -> Ref<int -> int> -> unit>

        module ExclusiveScan =
            type DefaultExpr                    = Expr<int -> Ref<int> -> int -> unit>
            type WithAggregateExpr              = Expr<int -> Ref<int> -> int -> Ref<int> -> unit>
            type WithAggregateAndCallbackOpExpr = Expr<int -> Ref<int> -> int -> Ref<int> -> Ref<int -> int> -> unit>

            module Identityless =
                type DefaultExpr                    = Expr<int -> Ref<int> -> unit>
                type WithAggregateExpr              = Expr<int -> Ref<int> -> Ref<int> -> unit>
                type WithAggregateAndCallbackOpExpr = Expr<int -> Ref<int> -> Ref<int> -> Ref<int -> int> -> unit>


    let pickScanKind logical_warps logical_warp_threads =
        let POW_OF_TWO = logical_warp_threads |> Constants.POW_OF_TWO
        ((CUB_PTX_VERSION >= 300) && ((logical_warps = 1) || POW_OF_TWO))

    let (|WarpScanShfl|_|) templateParams =
        let logical_warps, logical_warp_threads, scan_op = templateParams
        
        if (logical_warps, logical_warp_threads) ||> pickScanKind then
            WarpScanShfl.api
            <|||    (logical_warps, logical_warp_threads, scan_op)
            |>      Some
        else
            None 

    let (|WarpScanSmem|_|) templateParams =
        let logical_warps, logical_warp_threads, scan_op = templateParams

        if (logical_warps, logical_warp_threads) ||> pickScanKind |> not then
            WarpScanSmem.api
            <|||    (logical_warps, logical_warp_threads, scan_op)
            |>      Some
        else
            None

    
    module WarpScan =
        module InclusiveSum =
            type API =
                {
                    Default                     : Sig.InclusiveSum.DefaultExpr
                    WithAggregate               : Sig.InclusiveSum.WithAggregateExpr
                    WithAggregateAndCallbackOp  : Sig.InclusiveSum.WithAggregateAndCallbackOpExpr
                }

            let private Default logical_warps logical_warp_threads scan_op =
                let templateParams = (logical_warps, logical_warp_threads, scan_op)
                fun temp_storage warp_id lane_id ->
                    let InternalWarpScan =
                        templateParams |> function
                        | WarpScanShfl wsShfl ->
                            let scan = (temp_storage, warp_id, lane_id) |||> wsShfl
                            scan.InclusiveSum.Default
                        | WarpScanSmem wsSmem ->
                            let scan = (temp_storage, warp_id, lane_id) |||> wsSmem
                            scan.InclusiveSum.Default
                        | _ -> failwith "Invalid Template Parameters"

                    <@ fun (input:int) (output:Ref<int>) -> 
                        (input, output) ||> %InternalWarpScan 
                    @>

            let private WithAggregate logical_warps logical_warp_threads scan_op =
                let templateParams = (logical_warps, logical_warp_threads, scan_op)
                fun temp_storage warp_id lane_id ->
                    let InternalWarpScan =
                        templateParams |> function
                        | WarpScanShfl wsShfl ->
                            let scan = (temp_storage, warp_id, lane_id) |||> wsShfl
                            scan.InclusiveSum.Generic
                        | WarpScanSmem wsSmem ->
                            let scan = (temp_storage, warp_id, lane_id) |||> wsSmem
                            scan.InclusiveSum.WithAggregate
                        | _ -> failwith "Invalid Template Parameters"

                    <@ fun (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) -> 
                        (input, output, warp_aggregate) |||> %InternalWarpScan 
                    @>

            let private WithAggregateAndCallbackOp logical_warps logical_warp_threads scan_op =
                let templateParams = (logical_warps, logical_warp_threads, scan_op)
                fun temp_storage warp_id lane_id ->
                    let InclusiveSum =  WithAggregate
                                        <|||    (logical_warps, logical_warp_threads, scan_op)
                                        <|||    (temp_storage, warp_id, lane_id)

                    let Broadcast =
                        templateParams |> function
                        | WarpScanShfl wsShfl ->
                            let scan = (temp_storage, warp_id, lane_id) |||> wsShfl
                            scan.Broadcast
                        | WarpScanSmem wsSmem ->
                            let scan = (temp_storage, warp_id, lane_id) |||> wsSmem
                            scan.Broadcast
                        | _ -> failwith "Invalid Template Parameters"

                    <@ fun (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) (warp_prefix_op:Ref<int -> int>) -> 
                        (input, output, warp_aggregate) |||> %InclusiveSum
                        let prefix = __local__.Variable()
                        prefix := !warp_aggregate |> !warp_prefix_op
                        prefix := (!prefix, 0) ||> %Broadcast
                        output := !prefix + !output
                    @>

            let api logical_warps logical_warp_threads scan_op =
                fun temp_storage warp_id lane_id ->
                    {
                        Default                         =   Default
                                                            <|||    (logical_warps, logical_warp_threads, scan_op)
                                                            <|||    (temp_storage, warp_id, lane_id)

                        WithAggregate                   =   WithAggregate
                                                            <|||    (logical_warps, logical_warp_threads, scan_op)
                                                            <|||    (temp_storage, warp_id, lane_id)

                        WithAggregateAndCallbackOp      =   WithAggregateAndCallbackOp
                                                            <|||    (logical_warps, logical_warp_threads, scan_op)
                                                            <|||    (temp_storage, warp_id, lane_id)
                    }

        module ExclusiveSum =
            type API =
                {
                    Default : Expr<int -> Ref<int> -> unit>
                    WithAggregate : Expr<int -> Ref<int> -> Ref<int> -> unit>
                    WithAggregateAndCallbackOp : Expr<int -> Ref<int> -> Ref<int> -> Ref<int -> int> -> unit>
                }


            let private Default logical_warps logical_warp_threads scan_op =
                fun temp_storage warp_id lane_id ->
                    let InclusiveSum =  (   InclusiveSum.api
                                            <|||    (temp_storage, warp_id, lane_id)
                                            <|||    (logical_warps, logical_warp_threads, scan_op)).Default
                    <@ fun (input:int) (output:Ref<int>) ->
                        let inclusive = __local__.Variable()
                        (input, inclusive) ||> %InclusiveSum
                        output := !inclusive - input
                    @>
               

            let private WithAggregate logical_warps logical_warp_threads scan_op =
                fun temp_storage warp_id lane_id ->
                    let InclusiveSum =  (   InclusiveSum.api
                                            <|||    (temp_storage, warp_id, lane_id)
                                            <|||    (logical_warps, logical_warp_threads, scan_op)).WithAggregate
                    <@ fun (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) ->
                        let inclusive = __local__.Variable()
                        (input, inclusive, warp_aggregate) |||> %InclusiveSum
                        output := !inclusive - input
                    @>


            let private WithAggregateAndCallbackOp logical_warps logical_warp_threads scan_op =
                fun temp_storage warp_id lane_id ->
                    let InclusiveSum =  (   InclusiveSum.api
                                            <|||    (temp_storage, warp_id, lane_id)
                                            <|||    (logical_warps, logical_warp_threads, scan_op)).WithAggregateAndCallbackOp
                    <@ fun (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) (warp_prefix_op:Ref<int -> int>)->
                        let inclusive = __local__.Variable()
                        %InclusiveSum
                        <|| (input, inclusive)
                        <|| (warp_aggregate, warp_prefix_op)
                      
                        output := !inclusive - input
                    @>

            let api logical_warps logical_warp_threads scan_op =
                fun temp_storage warp_id lane_id ->
                    {
                        Default =                       Default
                                                        <|||    (logical_warps, logical_warp_threads, scan_op)
                                                        <|||    (temp_storage, warp_id, lane_id)

                        WithAggregate =                 WithAggregate
                                                        <|||    (logical_warps, logical_warp_threads, scan_op)
                                                        <|||    (temp_storage, warp_id, lane_id)

                        WithAggregateAndCallbackOp =    WithAggregateAndCallbackOp
                                                        <|||    (logical_warps, logical_warp_threads, scan_op)
                                                        <|||    (temp_storage, warp_id, lane_id)
                    }


module InclusiveSum =
    open Internal

    type API =
        {
            Default                     : Sig.InclusiveSum.DefaultExpr
            WithAggregate               : Sig.InclusiveSum.WithAggregateExpr
            WithAggregateAndCallbackOp  : Sig.InclusiveSum.WithAggregateAndCallbackOpExpr
        }

    let private Default logical_warps logical_warp_threads (scan_op:IScanOp) =
        fun temp_storage warp_id lane_id ->
            <@ fun (input:int) (output:Ref<int>) -> () @>

    let private WithAggregate logical_warps logical_warp_threads (scan_op:IScanOp) =
        fun temp_storage warp_id lane_id ->    
            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) -> () @>

    let private WithAggregateAndCallbackOp logical_warps logical_warp_threads (scan_op:IScanOp) =
        fun temp_storage warp_id lane_id ->    
           <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int -> int>) -> () @>


    let api logical_warps logical_warp_threads scan_op =
        fun temp_storage warp_id lane_id ->
            {
                Default                     =   Default
                                                <|||    (logical_warps, logical_warp_threads, scan_op)
                                                <|||    (temp_storage, warp_id, lane_id)
                
                WithAggregate               =   WithAggregate
                                                <|||    (logical_warps, logical_warp_threads, scan_op)
                                                <|||    (temp_storage, warp_id, lane_id)

                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp
                                                <|||    (logical_warps, logical_warp_threads, scan_op)
                                                <|||    (temp_storage, warp_id, lane_id)
            }
        



module ExclusiveSum =
    open Internal

    type API =
        {
            Default                     : Sig.ExclusiveSum.DefaultExpr
            WithAggregate               : Sig.ExclusiveSum.WithAggregateExpr
            WithAggregateAndCallbackOp  : Sig.ExclusiveSum.WithAggregateAndCallbackOpExpr
        }

    let private Default logical_warps logical_warp_threads (scan_op:IScanOp) =
        fun temp_storage warp_id lane_id ->
            <@ fun (input:int) (output:Ref<int>) -> () @>

    let private WithAggregate logical_warps logical_warp_threads (scan_op:IScanOp) =
        fun temp_storage warp_id lane_id ->
            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) -> () @>

    let private WithAggregateAndCallbackOp logical_warps logical_warp_threads (scan_op:IScanOp) =
        fun temp_storage warp_id lane_id ->
            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int -> int>) -> () @>


    let api logical_warps logical_warp_threads scan_op =
        fun temp_storage warp_id lane_id ->
            {
                Default                     =   Default
                                                <|||    (logical_warps, logical_warp_threads, scan_op)
                                                <|||    (temp_storage, warp_id, lane_id)
                
                WithAggregate               =   WithAggregate
                                                <|||    (logical_warps, logical_warp_threads, scan_op)
                                                <|||    (temp_storage, warp_id, lane_id)
                                                                                    
                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp
                                                <|||    (logical_warps, logical_warp_threads, scan_op)
                                                <|||    (temp_storage, warp_id, lane_id)
            }


module ExclusiveScan =
    open Internal

    type API =
        {
            Default                         : Sig.ExclusiveScan.DefaultExpr
            Default_NoID                    : Sig.ExclusiveScan.Identityless.DefaultExpr
            WithAggregate                   : Sig.ExclusiveScan.WithAggregateExpr
            WithAggregate_NoID              : Sig.ExclusiveScan.Identityless.WithAggregateExpr
            WithAggregateAndCallbackOp      : Sig.ExclusiveScan.WithAggregateAndCallbackOpExpr
            WithAggregateAndCallbackOp_NoID : Sig.ExclusiveScan.Identityless.WithAggregateAndCallbackOpExpr
        }

    let private Default logical_warps logical_warp_threads (scan_op:IScanOp) =
        fun temp_storage warp_id lane_id ->
            <@ fun (input:int) (output:Ref<int>) (identity:int) -> () @>

    let private WithAggregate logical_warps logical_warp_threads (scan_op:IScanOp) =
        fun temp_storage warp_id lane_id ->
            <@ fun (input:int) (output:Ref<int>) (identity:int) (block_aggregate:Ref<int>) -> () @>

    let private WithAggregateAndCallbackOp logical_warps logical_warp_threads (scan_op:IScanOp) =
        fun temp_storage warp_id lane_id ->
            <@ fun (input:int) (output:Ref<int>) (identity:int) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int -> int>) -> () @>

    module private Identityless =
        let Default logical_warps logical_warp_threads (scan_op:IScanOp) =
            fun temp_storage warp_id lane_id ->
                <@ fun (input:int) (output:Ref<int>) -> () @>

        let WithAggregate logical_warps logical_warp_threads (scan_op:IScanOp) =
            fun temp_storage warp_id lane_id ->
                <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) -> () @>

        let WithAggregateAndCallbackOp logical_warps logical_warp_threads (scan_op:IScanOp) =
            fun temp_storage warp_id lane_id ->
                <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int -> int>) -> () @>



    let api logical_warps logical_warp_threads scan_op =
        fun temp_storage warp_id lane_id ->
            {
                Default                     =       Default
                                                    <|||    (logical_warps, logical_warp_threads, scan_op)
                                                    <|||    (temp_storage, warp_id, lane_id)

                Default_NoID                =       Identityless.Default
                                                    <|||    (logical_warps, logical_warp_threads, scan_op)
                                                    <|||    (temp_storage, warp_id, lane_id)
                
                WithAggregate               =       WithAggregate
                                                    <|||    (logical_warps, logical_warp_threads, scan_op)
                                                    <|||    (temp_storage, warp_id, lane_id)

                WithAggregate_NoID          =       Identityless.WithAggregate
                                                    <|||    (logical_warps, logical_warp_threads, scan_op)
                                                    <|||    (temp_storage, warp_id, lane_id)
                                                                                    
                WithAggregateAndCallbackOp  =       WithAggregateAndCallbackOp
                                                    <|||    (logical_warps, logical_warp_threads, scan_op)
                                                    <|||    (temp_storage, warp_id, lane_id)

                WithAggregateAndCallbackOp_NoID  =  Identityless.WithAggregateAndCallbackOp
                                                    <|||    (logical_warps, logical_warp_threads, scan_op)
                                                    <|||    (temp_storage, warp_id, lane_id)
            }


module InclusiveScan =
    open Internal

    type API =
        {
            Default                     : Sig.InclusiveScan.DefaultExpr
            WithAggregate               : Sig.InclusiveScan.WithAggregateExpr
            WithAggregateAndCallbackOp  : Sig.InclusiveScan.WithAggregateAndCallbackOpExpr
        }


    let private Default logical_warps logical_warp_threads (scan_op:IScanOp) =
        fun temp_storage warp_id lane_id ->
            <@ fun (input:int) (output:Ref<int>) -> () @>

    let private WithAggregate logical_warps logical_warp_threads (scan_op:IScanOp) =
        fun temp_storage warp_id lane_id ->
            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) -> () @>

    let private WithAggregateAndCallbackOp logical_warps logical_warp_threads (scan_op:IScanOp) =
        fun temp_storage warp_id lane_id ->
            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int -> int>) -> () @>

    
    let api logical_warps logical_warp_threads scan_op =
        fun temp_storage warp_id lane_id ->
            {
                Default                     =   Default
                                                <|||    (logical_warps, logical_warp_threads, scan_op)
                                                <|||    (temp_storage, warp_id, lane_id)
                
                WithAggregate               =   WithAggregate
                                                <|||    (logical_warps, logical_warp_threads, scan_op)
                                                <|||    (temp_storage, warp_id, lane_id)
                                                                                    
                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp
                                                <|||    (logical_warps, logical_warp_threads, scan_op)
                                                <|||    (temp_storage, warp_id, lane_id)
            }


module WarpScan =

    type API =
        {
            InclusiveSum    : InclusiveSum.API
            InclusiveScan   : InclusiveScan.API
            ExclusiveSum    : ExclusiveSum.API
            ExclusiveScan   : ExclusiveScan.API
        }


    let api logical_warps logical_warp_threads scan_op =
        fun temp_storage warp_id lane_id ->
            {
                InclusiveSum    =   InclusiveSum.api
                                    <|||    (logical_warps, logical_warp_threads, scan_op)
                                    <|||    (temp_storage, warp_id, lane_id)
                                    
                                    
                InclusiveScan   =   InclusiveScan.api
                                    <|||    (logical_warps, logical_warp_threads, scan_op)
                                    <|||    (temp_storage, warp_id, lane_id)
                                    
                ExclusiveSum    =   ExclusiveSum.api
                                    <|||    (logical_warps, logical_warp_threads, scan_op)
                                    <|||    (temp_storage, warp_id, lane_id)

                ExclusiveScan   =   ExclusiveScan.api
                                    <|||    (logical_warps, logical_warp_threads, scan_op)
                                    <|||    (temp_storage, warp_id, lane_id)
            }