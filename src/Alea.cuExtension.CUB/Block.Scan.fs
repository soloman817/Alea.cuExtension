﻿[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Scan

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Block.BlockSpecializations


type ReductionOpKind =
    | ReduceByKey
    | SegmentedOp

//type ReduceByKeyOp<'K,'V> =
//    abstract op : Expr<KeyValuePair<'K,'V> -> KeyValuePair<'K,'V> -> KeyValuePair<'K,'V>>

//let reductionOp (kind:ReductionOpKind) (op:('V -> 'V -> 'V)) =
//    match kind with
//    | ReduceByKey ->
//        fun (first:KeyValuePair<'K,'V>, second:KeyValuePair<'K,'V>) ->
//            KeyValuePair<'K,'V>(second.Key,
//                if second.Key <> first.Key then second.Value else (first.Value, second.Value) ||> op )
//    | SegmentedOp ->
//        fun (first:KeyValuePair<'K,'V>, second:KeyValuePair<'K,'V>) ->
//            if second.Key > 0G then KeyValuePair<'K,'V>(first.Key + second.Key, second.Value)
//            else KeyValuePair<'K,'V>(first.Key + second.Key, (first.Value, second.Value) ||> op)


type BlockScanAlgorithm =
    | BLOCK_SCAN_RAKING
    | BLOCK_SCAN_RAKING_MEMOIZE
    | BLOCK_SCAN_WARP_SCANS


module TempStorage =
    type API =
        abstract    BlockScanWarpScan   : Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanWarpScans.TempStorage.API
        abstract    BlockScanRaking     : Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanRaking.TempStorage.API
        

    let uninitialized grid_elements =
        { new API with
            member this.BlockScanWarpScan = Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanWarpScans.TempStorage.TempStorage.Uninitialized
            member this.BlockScanRaking = grid_elements |> Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanRaking.TempStorage.unitialized
        }


module private Internal =
    //type TempStorage = TempStorage.API

    module Constants =
        let SAFE_ALGORITHM = 
            fun block_threads algorithm ->
                if (algorithm = BLOCK_SCAN_WARP_SCANS) && ((block_threads % CUB_PTX_WARP_THREADS) <> 0) then
                    BLOCK_SCAN_RAKING
                else
                    algorithm

    module Sig =
        module SingleDatumPerThread =
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

        module MultipleDataPerThread =
            module InclusiveSum = ()
            module ExclusiveSum = 
                type DefaultExpr                    = Expr<deviceptr<int> -> deviceptr<int> -> unit>
                type WithAggregateExpr              = Expr<deviceptr<int> -> deviceptr<int> -> Ref<int> -> unit>
                type WithAggregateAndCallbackOpExpr = Expr<deviceptr<int> -> deviceptr<int> -> Ref<int> -> Ref<int -> int> -> unit>
            module InclusiveScan = ()
            module ExclusiveScan =
                ()
                module Identityless = ()

    module BlockScan =
        let pickScanKind block_threads algorithm =
            let SAFE_ALGORITHM = (block_threads, algorithm) ||> Constants.SAFE_ALGORITHM
            SAFE_ALGORITHM = BLOCK_SCAN_WARP_SCANS

        let (|BlockScanWarpScan|_|) templateParams =
            let block_threads, algorithm, scan_op = templateParams
            if (block_threads, algorithm) ||> pickScanKind then 
                BlockScanWarpScan.api 
                <|| (block_threads, scan_op)
                |> Some
            else
                None

        let (|BlockScanRaking|_|) templateParams =
            let block_threads, algorithm, scan_op = templateParams
            let SAFE_ALGORITHM = (block_threads, algorithm) ||> Constants.SAFE_ALGORITHM
            if (block_threads, algorithm) ||> pickScanKind |> not then
                BlockScanRaking.api
                <||| (block_threads, (SAFE_ALGORITHM = BLOCK_SCAN_RAKING_MEMOIZE), scan_op)
                |> Some
            else
                None



module ExclusiveSum =
    open Internal
    open BlockScan
    open TempStorage
    //type TempStorage = TempStorage.API

    module SingleDatumPerThread =
        type API =
            abstract    Default                     : Sig.SingleDatumPerThread.ExclusiveSum.DefaultExpr
            abstract    WithAggregate               : Sig.SingleDatumPerThread.ExclusiveSum.WithAggregateExpr
            abstract    WithAggregateAndCallbackOp  : Sig.SingleDatumPerThread.ExclusiveSum.WithAggregateAndCallbackOpExpr
            
        
        let private Default block_threads algorithm scan_op =
            let templateParams = (block_threads, algorithm, scan_op)
            
            fun (temp_storage:TempStorage.API) linear_tid cached_segment ->
                let InternalBlockScan =
                    templateParams |> function
                    | BlockScanWarpScan bsws ->
                        (   bsws
                            <|||    (temp_storage.BlockScanWarpScan, linear_tid, 0)
                        ).ExclusiveSum.WithAggregate
                    | BlockScanRaking bsr ->
                        (   bsr
                            <|||    (temp_storage.BlockScanRaking, linear_tid, cached_segment)
                        ).ExclusiveSum.WithAggregate
                    | _ -> failwith "Invalid Template Parameters"
                <@ fun (input:'T) (output:Ref<'T>) ->
                    let block_aggregate = __local__.Variable()
                    %InternalBlockScan
                    <|| (input, output)
                    <|  block_aggregate
                @>

        let private WithAggregate block_threads algorithm scan_op =
            fun temp_storage linear_tid cached_segment ->
                <@ fun (input:'T) (output:Ref<'T>) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp block_threads algorithm scan_op =
            fun temp_storage linear_tid cached_segment ->
                <@ fun (input:'T) (output:Ref<'T>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int -> int>) -> () @>

        let api block_threads algorithm scan_op =
            fun temp_storage linear_tid cached_segment ->
                { new API with
                    member this.Default                     =   Default
                                                                <|||    (block_threads, algorithm, scan_op)
                                                                <|||    (temp_storage, linear_tid, cached_segment)

                    member this.WithAggregate               =   WithAggregate
                                                                <|||    (block_threads, algorithm, scan_op)
                                                                <|||    (temp_storage, linear_tid, cached_segment)

                    member this.WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp
                                                                <|||    (block_threads, algorithm, scan_op)
                                                                <|||    (temp_storage, linear_tid, cached_segment)
                }

    module MultipleDataPerThread =
        type API =
            abstract    Default                     : Sig.MultipleDataPerThread.ExclusiveSum.DefaultExpr
            abstract    WithAggregate               : Sig.MultipleDataPerThread.ExclusiveSum.WithAggregateExpr
            abstract    WithAggregateAndCallbackOp  : Sig.MultipleDataPerThread.ExclusiveSum.WithAggregateAndCallbackOpExpr
            

        let private Default block_threads algorithm scan_op =
            fun items_per_thread ->
                fun temp_storage linear_tid cached_segment ->
                    <@ fun (input:deviceptr<int>) (output:deviceptr<int>) -> () @>

        let private WithAggregate block_threads algorithm scan_op =
            fun items_per_thread ->
                fun temp_storage linear_tid cached_segment ->
                    <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp block_threads algorithm scan_op =
            fun items_per_thread ->
                fun temp_storage linear_tid cached_segment ->
                    <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int -> int>) -> () @>

        let api block_threads algorithm scan_op =
            fun items_per_thread ->
                fun temp_storage linear_tid cached_segment ->
                    { new API with
                        member this.Default                     =   Default
                                                                    <|||    (block_threads, algorithm, scan_op)
                                                                    <|      (items_per_thread)
                                                                    <|||    (temp_storage, linear_tid, cached_segment)

                        member this.WithAggregate               =   WithAggregate
                                                                    <|||    (block_threads, algorithm, scan_op)
                                                                    <|      (items_per_thread)
                                                                    <|||    (temp_storage, linear_tid, cached_segment)

                        member this.WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp
                                                                    <|||    (block_threads, algorithm, scan_op)
                                                                    <|      (items_per_thread)
                                                                    <|||    (temp_storage, linear_tid, cached_segment)
                    }

    type API =
        {
            SingleDatumPerThread    : SingleDatumPerThread.API
            MultipleDataPerThread   : MultipleDataPerThread.API
        }

    let api block_threads algorithm scan_op =
        fun items_per_thread ->
            fun temp_storage linear_tid cached_segment ->
                items_per_thread |> function
                | None ->
                    {
                        SingleDatumPerThread    =   SingleDatumPerThread.api
                                                    <|||    (block_threads, algorithm, scan_op)
                                                    <|||    (temp_storage, linear_tid, cached_segment)

                        MultipleDataPerThread   =   MultipleDataPerThread.api
                                                    <|||    (block_threads, algorithm, scan_op)
                                                    <|      (4)
                                                    <|||    (temp_storage, linear_tid, cached_segment)

                    }

                | Some items_per_thread ->
                    {
                        SingleDatumPerThread    =   SingleDatumPerThread.api
                                                    <|||    (block_threads, algorithm, scan_op)
                                                    <|||    (temp_storage, linear_tid, cached_segment)

                        MultipleDataPerThread   =   MultipleDataPerThread.api
                                                    <|||    (block_threads, algorithm, scan_op)
                                                    <|      (items_per_thread)
                                                    <|||    (temp_storage, linear_tid, cached_segment)

                    }

module ExclusiveScan =
//                                    input -> &output -> identity -> scanop
//    type DefaultFunctionExpr = Expr<int -> Ref<int> -> int -> ScanOp -> unit>
//                                       input -> &output -> &identity -> scanop -> &block_aggregate
//    type WithAggregateFunctionExpr = Expr<int -> Ref<int> -> Ref<int> -> ScanOp -> Ref<int> -> unit>
//                                                         input -> &output -> &identity -> scanop -> &block_aggregate -> &block_prefix_callback_op
//    type WithAggregateAndCallbackOpFunctionExpr = Expr<int -> Ref<int> -> Ref<int> -> ScanOp -> Ref<int> -> Ref<int> -> unit>
    module SingleDatumPerThread =
        let private Default =
            <@ fun (input:'T) (output:Ref<'T>) (identity:'T) (scan_op:IScanOp<'T>) -> () @>

        let private WithAggregate =
            <@ fun (input:'T) (output:Ref<'T>) (identity:Ref<int>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp =
            <@ fun (input:'T) (output:Ref<'T>) (identity:Ref<int>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>


        module Identityless =
            let private Default =
                <@ fun (input:'T) (output:Ref<'T>) (scan_op:IScanOp<'T>) -> () @>

            let private WithAggregate =
                <@ fun (input:'T) (output:Ref<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<int>) -> () @>

            let private WithAggregateAndCallbackOp =
                <@ fun (input:'T) (output:Ref<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>

    module MultipleDataPerThread =
        let private Default items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (identity:Ref<int>) (scan_op:IScanOp<'T>) -> () @>

        let private WithAggregate items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (identity:Ref<int>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (identity:Ref<int>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>


        module Identityless =
            let private Default items_per_thread =
                <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:IScanOp<'T>) -> () @>

            let private WithAggregate items_per_thread =
                <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<int>) -> () @>

            let private WithAggregateAndCallbackOp items_per_thread =
                <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>


module InclusiveSum =

    module SingleDatumPerThread =
        let private Default =
            <@ fun (input:'T) (output:Ref<'T>) -> () @>

        let private WithAggregate =
            <@ fun (input:'T) (output:Ref<'T>) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp =
            <@ fun (input:'T) (output:Ref<'T>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>


    module MultipleDataPerThread =
        let private Default items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) -> () @>

        let private WithAggregate items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>


module InclusiveScan =

    module SingleDatumPerThread =
        let private Default =
            <@ fun (input:'T) (output:Ref<'T>) (scan_op:IScanOp<'T>) -> () @>

        let private WithAggregate =
            <@ fun (input:'T) (output:Ref<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp =
            <@ fun (input:'T) (output:Ref<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>


    module MultipleDataPerThread =
        let private Default items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:IScanOp<'T>) -> () @>

        let private WithAggregate items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>


module BlockScan =
    //open Internal

    type API =
        abstract ExclusiveSum    : ExclusiveSum.API
    

    let api block_threads algorithm scan_op = cuda {
        
        return fun (program:Program) ->
            let worker = program.Worker
            
            fun temp_storage linear_tid cached_segment ->
                { new API with
                    member this.ExclusiveSum    =   ExclusiveSum.api
                                                    <|||    (block_threads, algorithm, scan_op)
                                                    <|      (None)
                                                    <|||    (temp_storage, linear_tid, cached_segment)
                }
        }
        
    
     
//let inline InternalBlockScan() =
//let inline InternalBlockScan() =
//    fun block_threads algorithm ->
//        let SAFE_ALGORITHM = (block_threads, algorithm) ||> SAFE_ALGORITHM
//        match SAFE_ALGORITHM with
//        | BLOCK_SCAN_WARP_SCANS -> 
//            (block_threads |> BlockScanWarpScans.BlockScanWarpScans.Create |> Some, None)
//        | _ -> 
//            (None, (block_threads, (SAFE_ALGORITHM = BLOCK_SCAN_RAKING_MEMOIZE)) |> BlockScanRaking.BlockScanRaking.Create |> Some)

//let exclusiveSum<int> (block_threads:int) (algorithm:BlockScanAlgorithm) () =

// public (ctors)
//blockscan() temp_storage(privateStorage()) linear_tid(threadIdx.x)
//blockscan(temp_storage:Ref<'TempStorage>) temp_storage(temp_storage.alias()) linear_tid(linear_tid)
//blockscan(linear_tid) temp_storage(privateStorage()) linear_tid(linear_tid)
//blockscan(&temp_storage, linear_tid) temp_storage(temp_storage.alias()) linear_tid(linear_tid)
//type InternalBlockScan =
//    | BlockScanWarpScans
//    | BlockScanRaking
//
//type internal TemplateParameters =
//    {
//        BLOCK_THREADS : int
//        ALGORITHM : BlockScanAlgorithm
//    }

//
//
//type Constants =
//    {
//        SAFE_ALGORITHM : BlockScanAlgorithm
//    }
//
//    static member Init(block_threads, algorithm) =
//        {
//            SAFE_ALGORITHM = 
//                if ((algorithm = BLOCK_SCAN_WARP_SCANS) && ((block_threads % CUB_PTX_WARP_THREADS) <> 0)) then
//                    BLOCK_SCAN_RAKING
//                else
//                    algorithm
//        }
//
//
//
//[<Record>]
//type InternalBlockScan =
//    {   
//        SAFE_ALGORITHM          : BlockScanAlgorithm
//        BlockScanWarpScans      : BlockScanWarpScans
//        BlockScanRaking         : BlockScanRaking
//    }
//
//
//    member this.ExclusiveSum(temp_storage, linear_tid, a, b, c, d) =
//        if this.SAFE_ALGORITHM = BLOCK_SCAN_WARP_SCANS then
//            this.BlockScanWarpScans.Initialize(temp_storage, linear_tid).ExclusiveSum(a,b,c,d)
//        else
//            this.BlockScanRaking.Initialize(temp_storage, linear_tid).ExclusiveSum(a,b,c,d)
//
//    member this.ExclusiveSum(temp_storage, linear_tid, a, b, c) =
//        if this.SAFE_ALGORITHM = BLOCK_SCAN_WARP_SCANS then
//            this.BlockScanWarpScans.Initialize(temp_storage, linear_tid).ExclusiveSum(a,b,c)
//        else
//            this.BlockScanRaking.Initialize(temp_storage, linear_tid).ExclusiveSum(a,b,c)
//
//    member this.GetScanner(c:Constants) = //, temp_storage, linear_tid) =
//        if c.SAFE_ALGORITHM = BLOCK_SCAN_WARP_SCANS then
//            (Some this.BlockScanWarpScans, None)
//        else
//            (None, Some this.BlockScanRaking)
//
//    member this.GetStorage(c:Constants) =
//        if c.SAFE_ALGORITHM = BLOCK_SCAN_WARP_SCANS then
//            (Some this.BlockScanWarpScans.ThreadFields.temp_storage, None)
//        else
//            (None, Some this.BlockScanRaking.ThreadFields.temp_storage)
//
//    static member Init(block_threads:int, c:Constants) =
//        {
//            SAFE_ALGORITHM = c.SAFE_ALGORITHM
//            BlockScanWarpScans = block_threads |> BlockScanWarpScans.Create
//            BlockScanRaking = (block_threads, (c.SAFE_ALGORITHM = BLOCK_SCAN_RAKING_MEMOIZE)) |> BlockScanRaking.Create
//        }
//
//
//[<Record>] [<RequireQualifiedAccess>]
//type ThreadFields =
//    {
//        mutable temp_storage : deviceptr<int>
//        mutable linear_tid : int
//    }
//
//    static member Init(temp_storage:deviceptr<int>, linear_tid:int) =
//        {
//            temp_storage = temp_storage
//            linear_tid = linear_tid
//        }
//
//[<Record>]
//type BlockScan =
//    {
//        / Template Parameters
//        BLOCK_THREADS       : int
//        ALGORITHM           : BlockScanAlgorithm
//        ///////////////////////////////////////////
//        Constants           : Constants
//        InternalBlockScan   : InternalBlockScan
//        ThreadFields        : ThreadFields
//        ThreadScan          : ThreadScan
//    }
//
//    member this.Initialize() =
//        this.ThreadFields.temp_storage  <- PrivateStorage()
//        this.ThreadFields.linear_tid    <- threadIdx.x
//        this
//
//    member this.Initialize(temp_storage:deviceptr<int>) =
//        this.ThreadFields.temp_storage  <- temp_storage
//        this.ThreadFields.linear_tid    <- threadIdx.x
//        this
//
//    member this.Initialize(linear_tid:int) =
//        this.ThreadFields.temp_storage  <- PrivateStorage()
//        this.ThreadFields.linear_tid    <- linear_tid
//        this
//
//    member this.Initialize(temp_storage:deviceptr<int>, linear_tid:int) =
//        this.ThreadFields.temp_storage  <- temp_storage
//        this.ThreadFields.linear_tid    <- linear_tid
//        this
//
//    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     exclusive prefix sum operations
//    member inline this.ExclusiveSum(input:int, output:Ref<int>) =
//         localize thread fields
//        let temp_storage = this.ThreadFields.temp_storage
//        let linear_tid = this.ThreadFields.linear_tid
//
//        let block_aggregate = __local__.Variable()
//        this.InternalBlockScan.ExclusiveSum(temp_storage, linear_tid, input, output, block_aggregate)
//
//            
//
//    member inline this.ExclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>) =
//         localize thread fields
//        let temp_storage = this.ThreadFields.temp_storage
//        let linear_tid = this.ThreadFields.linear_tid
//        
//        this.InternalBlockScan.ExclusiveSum(temp_storage, linear_tid, input, output, block_aggregate)
//
//    member this.ExclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) =
//        let temp_storage = this.ThreadFields.temp_storage
//        let linear_tid = this.ThreadFields.linear_tid
//
//        this.InternalBlockScan.ExclusiveSum(temp_storage, linear_tid, input, output, block_aggregate, block_prefix_callback_op)
//
//     exclusive prefix sum operations (multiple data per thread)
//    <items_per_thread>
//    member inline this.ExclusiveSum(items_per_thread:int, input:deviceptr<int>, output:deviceptr<int>) =
//        let scan_op = (+)
//
//        let thread_partial = 
//            ThreadReduce
//            <| items_per_thread
//            <||| (input, scan_op, None)
//
//         Exclusive threadblock-scan
//        this.ExclusiveSum(thread_partial, ref thread_partial)
//
//         Exclusive scan in registers with prefix
//        ThreadScanExclusive
//        <| items_per_thread
//        <|| (input, output)
//        <| scan_op
//        <| thread_partial
//        <| None
//    
//    member inline this.ExclusiveSum(items_per_thread:int, input:deviceptr<int>, output:deviceptr<int>, block_aggregate:Ref<int>) =
//        let scan_op = (+)
//        let thread_partial = 
//            ThreadReduce
//            <| items_per_thread
//            <||| (input, scan_op, None)
//
//         Exclusive threadblock-scan
//        this.ExclusiveSum(thread_partial, ref thread_partial, block_aggregate)
//
//         Exclusive scan in registers with prefix
//        ThreadScanExclusive
//        <| items_per_thread
//        <|| (input, output)
//        <| scan_op 
//        <| thread_partial
//        <| None
//    
//    member inline this.ExclusiveSum(items_per_thread:int, input:deviceptr<int>, output:deviceptr<int>, block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) =
//        let scan_op = (+)
//        let thread_partial = 
//            ThreadReduce
//            <| items_per_thread
//            <||| (input, scan_op, None)
//
//         Exclusive threadblock-scan
//        this.ExclusiveSum(thread_partial, ref thread_partial, block_aggregate, block_prefix_callback_op)
//
//         Exclusive scan in registers with prefix
//        ThreadScanExclusive
//        <| items_per_thread
//        <|| (input, output)
//        <| scan_op
//        <| thread_partial
//        <| None
//    
//
//    // exclusive prefix scan operations
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:(int -> int -> int)) =
//        let bswc, bsr = this.InternalBlockScan.GetScanner(this.Constants)
//        (bswc, bsr) |> function
//        | Some bswc, None ->
//            bswc.Initialize(temp_storage, linear_tid).ExclusiveSum(input, output, block_aggregate)
//        | None, Some bsr ->
//            bsr.Initialize(temp_storage, linear_tid).ExclusiveSum(input, output, block_aggregate)
//        | _, _ ->        
//        InternalBlockScan(temp_storage, linear_tid).ExclusiveScan(input, output, identity, scan_op, block_aggregate)
//
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>) =
//        let bswc, bsr = this.InternalBlockScan.GetScanner(this.Constants)
//        (bswc, bsr) |> function
//        | Some bswc, None ->
//            bswc.Initialize(temp_storage, linear_tid).ExclusiveSum(input, output, block_aggregate)
//        | None, Some bsr ->
//            bsr.Initialize(temp_storage, linear_tid).ExclusiveSum(input, output, block_aggregate)
//        | _, _ ->        
//        
//        InternalBlockScan(temp_storage, linear_tid).ExclusiveScan(input, output, identity, scan_op, block_aggregate, block_prefix_callback_op)
//    
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:(int -> int -> int), block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) =
//        let thread_partial = ThreadReduce(input, scan_op)
//
//        // Exclusive threadblock-scan
//        ExclusiveScan(thread_partial, thread_partial, identity, scan_op)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial)
//    
//    // exclusive prefix scan operations (identityless, single datum per thread)
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int)) =
//        let thread_partial = ThreadReduce(input, scan_op)
//
//        // Exclusive threadblock-scan
//        ExclusiveScan(thread_partial, thread_partial, identity, scan_op, block_aggregate)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial)
//    
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>) =
//        let thread_partial = ThreadReduce(input, scan_op)
//
//        // Exclusive threadblock-scan
//        this.ExclusiveScan(thread_partial, thread_partial, identity, scan_op, block_aggregate, block_prefix_callback_op)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial)
//
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) =
//        let block_aggregate = __null() |> __ptr_to_ref
//        this.InternalBlockScan(temp_storage, linear_tid).ExclusiveScan(input, output, scan_op, block_aggregate)
//
//
//    // exclusive prefix scan operations (multiple data per thread)
//    //<items_per_thread>
//    member this.ExclusiveScan(input:deviceptr<int>, output:deviceptr<int>, identity:Ref<int>, scan_op:(int -> int -> int)) =
//        this.InternalBlockScan(temp_storage, linear_tid).ExclusiveScan(input, output, scan_op, block_aggregate)
//
//    member this.ExclusiveScan(input:deviceptr<int>, output:deviceptr<int>, identity:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>) =
//        InternalBlockScan(temp_storage, linear_tid).ExclusiveScan(input, output, scan_op, block_aggregate, block_prefix_callback_op)
//
//    member this.ExclusiveScan(input:deviceptr<int>, output:deviceptr<int>, identity:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) =
//        let thread_partial = ThreadReduce(input, scan_op)
//
//        // Exclusive threadblock-scan
//        this.ExclusiveScan(thread_partial, thread_partial, scan_op)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial, (linear_tid != 0))
//
//
//    // exclusive prefix scan operations (identityless, multiple data per thread)
//    //<items_per_thread>
//    member this.ExclusiveScan(input:deviceptr<int>, output:deviceptr<int>, scan_op:(int -> int -> int)) =
//        // Reduce consecutive thread items in registers
//        let thread_partial = ThreadReduce(input, scan_op)
//
//        // Exclusive threadblock-scan
//        this.ExclusiveScan(thread_partial, thread_partial, scan_op, block_aggregate)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial, (linear_tid != 0))
//
//    member this.ExclusiveScan(input:deviceptr<int>, output:deviceptr<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>) =
//        // Reduce consecutive thread items in registers
//        let thread_partial = ThreadReduce(input, scan_op)
//
//        // Exclusive threadblock-scan
//        this.ExclusiveScan(thread_partial, thread_partial, scan_op, block_aggregate, block_prefix_callback_op)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial)
//
//    member this.ExclusiveScan(input:deviceptr<int>, output:deviceptr<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) =
//        let block_aggregate = __null() |> __ptr_to_ref
//        this.InternalBlockScan(temp_storage, linear_tid).InclusiveSum(input, output, block_aggregate)
//
//
//    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    // inclusive prefix sum operations
//    member this.InclusiveSum(input:int, output:Ref<int>) =
//        this.InternalBlockScan(temp_storage, linear_tid).InclusiveSum(input, output, block_aggregate)
//
//    member this.InclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>) =
//        this.InternalBlockScan(temp_storage, linear_tid).InclusiveSum(input, output, block_aggregate, block_prefix_callback_op)
//
//    member this.InclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) =
//        fun (items_per_thread:int) ->
//            if (items_per_thread = 1) then
//                this.InclusiveSum(input[0], output[0])
//            else
//                // Reduce consecutive thread items in registers
//                let scan_op = (+)
//                let thread_partial = ThreadReduce(input, scan_op)
//
//                // Exclusive threadblock-scan
//                this.ExclusiveSum(thread_partial, thread_partial)
//
//                // Inclusive scan in registers with prefix
//                ThreadScanInclusive(input, output, scan_op, thread_partial, (linear_tid != 0))
//
//    // inclusive prefix sum operations (multiple data per thread)
//    //<items_per_thread>
//    member this.InclusiveSum(input:deviceptr<int>, output:deviceptr<int>) =
//        fun (items_per_thread:int) ->
//            if (items_per_thread = 1) then
//                this.InclusiveSum(input[0], output[0], block_aggregate)
//            else
//                // Reduce consecutive thread items in registers
//                let scan_op = (+)
//                let thread_partial = ThreadReduce(input, scan_op)
//
//                // Exclusive threadblock-scan
//                this.ExclusiveSum(thread_partial, thread_partial, block_aggregate)
//
//                // Inclusive scan in registers with prefix
//                ThreadScanInclusive(input, output, scan_op, thread_partial, (linear_tid != 0))
//
//        
//    member this.InclusiveSum(input:deviceptr<int>, output:deviceptr<int>, block_aggregate:Ref<int>) =
//        fun (items_per_thread:int) ->
//            if (items_per_thread = 1) then
//                this.InclusiveSum(input[0], output[0], block_aggregate, block_prefix_callback_op)
//            else
//                // Reduce consecutive thread items in registers
//                let scan_op = (+)
//                let thread_partial = ThreadReduce(input, scan_op)
//
//                // Exclusive threadblock-scan
//                this.ExclusiveSum(thread_partial, thread_partial, block_aggregate, block_prefix_callback_op)
//
//                // Inclusive scan in registers with prefix
//                ThreadScanInclusive(input, output, scan_op, thread_partial)
//
//    
//    member this.InclusiveSum(input:deviceptr<int>, output:deviceptr<int>, block_prefix_callback_op:Ref<int -> int>) =
//        let block_aggregate = __nul() |> __ptr_to_ref
//        this.InclusiveScan(input, output, scan_op, block_aggregate)
//    
//   
//    // inclusive prefix scan operations
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int)) =
//        let block_aggregate = __null() |> __ptr_to_ref
//        this.InclusiveScan(input, output, scan_op, block_aggregate)
//    
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>) =
//        InternalBlockScan(temp_storage, linear_tid).InclusiveScan(input, output, scan_op, block_aggregate)
//
//
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) =
//        InternalBlockScan(temp_storage, linear_tid).InclusiveScan(input, output, scan_op, block_aggregate, block_prefix_callback_op)
//    
//    // inclusive scan operations (multiple data per thread)
//    //<items_per_thread>
//    member this.InclusiveScan(input:deviceptr<int>, output:deviceptr<int>, scan_op:(int -> int -> int)) =
//        fun (items_per_thread:int) ->
//           if (items_per_thread = 1) then
//                this.InclusiveScan(input[0], output[0], scan_op)
//            else
//                // Reduce consecutive thread items in registers
//                let thread_partial = ThreadReduce(input, scan_op)
//
//                // Exclusive threadblock-scan
//                this.ExclusiveScan(thread_partial, thread_partial, scan_op)
//
//                // Inclusive scan in registers with prefix
//                ThreadScanInclusive(input, output, scan_op, thread_partial, (linear_tid != 0))
//
//        
//    member this.InclusiveScan(input:deviceptr<int>, output:deviceptr<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>) =
//        fun (items_per_thread:int) ->
//            if (items_per_thread = 1) then
//                this.InclusiveScan(input[0], output[0], scan_op, block_aggregate)
//            else
//                // Reduce consecutive thread items in registers
//                let thread_partial = ThreadReduce(input, scan_op)
//
//                // Exclusive threadblock-scan
//                this.ExclusiveScan(thread_partial, thread_partial, scan_op, block_aggregate)
//
//                // Inclusive scan in registers with prefix
//                ThreadScanInclusive(input, output, scan_op, thread_partial, (linear_tid != 0))
//        
//
//    member this.InclusiveScan(input:deviceptr<int>, output:deviceptr<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) =
//        fun (items_per_thread:int) ->
//            if (items_per_thread = 1) then
//                this.InclusiveScan(input[0], output[0], scan_op, block_aggregate, block_prefix_callback_op)
//            else
//                // Reduce consecutive thread items in registers
//                let thread_partial = ThreadReduce(input, scan_op)
//
//                // Exclusive threadblock-scan
//                this.ExclusiveScan(thread_partial, thread_partial, scan_op, block_aggregate, block_prefix_callback_op)
//
//                // Inclusive scan in registers with prefix
//                ThreadScanInclusive(input, output, scan_op, thread_partial)
//
//    static member Create(block_threads:int, algorithm:BlockScanAlgorithm)
//
//    static member Create(block_threads:int, algorithm:BlockScanAlgorithm, items_per_thread:int) =
//        let c = (block_threads, algorithm) |> Constants.Init
//        {
//            BLOCK_THREADS = block_threads
//            ALGORITHM = algorithm
//            Constants = c
//            InternalBlockScan = (block_threads, c) |> InternalBlockScan.Init
//            ThreadFields = ThreadFields.Init(__null(), threadIdx.x)
//            ThreadScan = items_per_thread |> ThreadScan.Create
//        }
//
//    static member Create(block_threads:int, items_per_thread:int) =
//        let c = (block_threads, BLOCK_SCAN_RAKING) |> Constants.Init
//        {
//            BLOCK_THREADS       = block_threads
//            ALGORITHM           = BLOCK_SCAN_RAKING
//            Constants           = c
//            InternalBlockScan   = (block_threads, c) |> InternalBlockScan.Init
//            ThreadFields        = ThreadFields.Init(__null(), threadIdx.x)
//            ThreadScan          = items_per_thread |> ThreadScan.Create
//        }
//
//    static member Create(block_threads:int) =
//        let c = (block_threads, BLOCK_SCAN_RAKING) |> Constants.Init
//        {
//            BLOCK_THREADS       = block_threads
//            ALGORITHM           = BLOCK_SCAN_RAKING
//            Constants           = c
//            InternalBlockScan   = (block_threads, c) |> InternalBlockScan.Init
//            ThreadFields        = ThreadFields.Init(__null(), threadIdx.x)
//            ThreadScan          = 1 |> ThreadScan.Create
//        }
//
//    static member Create(block_threads:int, algorithm:BlockScanAlgorithm) =
//        let c = (block_threads, algorithm) |> Constants.Init
//        {
//            BLOCK_THREADS = block_threads
//            ALGORITHM = algorithm
//            Constants = c
//            InternalBlockScan = (block_threads, c) |> InternalBlockScan.Init
//            ThreadFields = ThreadFields.Init(__null(), threadIdx.x)
//            ThreadScan = 1 |> ThreadScan.Create
//        }

//
//   
//
//
//
//
////module ExclusiveScan =
////
////    module STSD =
////        // exclusive prefix sum operations
////        let exclusiveSum =
////            fun (input:'T) (output:Ref<'T>) (block_aggregate:Ref<int> option) (block_prefix_callback_op:Ref<int -> int> option) -> ()
////
////        // exclusive prefix scan operations
////        let exclusiveScan =
////            fun (input:'T) (output:Ref<'T>) (identity:Ref<int>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<int> option) (block_prefix_callback_op:Ref<int -> int> option) -> ()
////
////
////        module Identityless =
////            // exclusive prefix scan operations (identityless, single datum per thread)
////            let exclusiveScan =
////                fun (input:'T) (output:Ref<'T>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<int> option) (block_prefix_callback_op:Ref<int -> int> option) -> ()   
////
////
////    module STMD =
////        // exclusive prefix sum operations (multiple data per thread)
////        let exclusiveSum items_per_thread =
////            fun (input:deviceptr<int>) (output:deviceptr<int>) (block_aggregate:Ref<int> option) (block_prefix_callback_op:Ref<int -> int> option) -> ()
////
////        // exclusive prefix scan operations (multiple data per thread)
////        let exclusiveScan items_per_thread =
////            fun (input:deviceptr<int>) (output:deviceptr<int>) (identity:Ref<int>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<int> option) (block_prefix_callback_op:Ref<int -> int> option) -> ()
////        
////
////        module Identityless =
////            // exclusive prefix scan operations (identityless, multiple data per thread)
////            let exclusiveScan_noId items_per_thread =
////                fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<int> option) (block_prefix_callback_op:Ref<int -> int> option) -> ()
////
////
////module InclusiveScan =
////    
////    module STSD =
////        // inclusive prefix sum operations
////        let inclusiveSum =
////            fun (input:'T) (output:Ref<'T>) (block_aggregate:Ref<int> option) (block_prefix_callback_op:Ref<int -> int> option) -> ()
////
////        // inclusive prefix scan operations
////        let inclusiveScan =
////            fun (input:'T) (output:Ref<'T>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<int> option) (block_prefix_callback_op:Ref<int -> int> option) -> ()
////        
////
////    module STMD =
////        // inclusive prefix sum operations (multiple data per thread)
////        let inclusiveSum items_per_thread =
////            fun (input:deviceptr<int>) (output:deviceptr<int>) (block_aggregate:Ref<int> option) (block_prefix_callback_op:Ref<int -> int> option) -> ()
////        
////        // inclusive scan operations (multiple data per thread)
////        let inclusiveScan items_per_thread =
////            fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<int> option) (block_prefix_callback_op:Ref<int -> int> option) -> ()
////        
////        
////type API =
////    {
////        ExclusiveScan : Expr
////        InclusiveScan : Expr
////    }
////
////let inline BlockScan () 
//
