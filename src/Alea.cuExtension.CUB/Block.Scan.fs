[<AutoOpen>]
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
//    abstract op : <'T><KeyValuePair<'K,'V> -> KeyValuePair<'K,'V> -> KeyValuePair<'K,'V>>

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

module Template =
    type BlockScanAlgorithm =
        | BLOCK_SCAN_RAKING
        | BLOCK_SCAN_RAKING_MEMOIZE
        | BLOCK_SCAN_WARP_SCANS

    [<AutoOpen>]
    module Params =
        [<Record>]
        type API<'T> =
            {
                BLOCK_THREADS   : int
                ALGORITHM       : BlockScanAlgorithm                
            }

            [<ReflectedDefinition>]
            static member Init(block_threads, algorithm, scan_op) =
                {
                    BLOCK_THREADS   = block_threads
                    ALGORITHM       = algorithm
                }

    [<AutoOpen>]
    module Constants =
        [<Record>]
        type API =
            {
                SAFE_ALGORITHM : BlockScanAlgorithm
            }

            [<ReflectedDefinition>]
            static member Init(tp:Params.API<'T>) =
                {
                    SAFE_ALGORITHM = if ((tp.ALGORITHM = BLOCK_SCAN_WARP_SCANS) && (tp.BLOCK_THREADS % CUB_PTX_WARP_THREADS <> 0)) then BLOCK_SCAN_RAKING else tp.ALGORITHM
                }

    [<AutoOpen>]
    module TempStorage =
        [<Record>]
        type API<'T> =
            {
                BlockScanWarpScan   : Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanWarpScans.Template._TempStorage<'T>
                BlockScanRaking     : Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanRaking.Template._TempStorage<'T>
            }

            [<ReflectedDefinition>]
            static member Init(block_threads, memoize, scan_op) =
                {
                    BlockScanWarpScan   = Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanWarpScans.Template._TempStorage.Uninitialized()
                    BlockScanRaking     = Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanRaking.Template._TempStorage.Init(block_threads, memoize, scan_op)
                }

            [<ReflectedDefinition>]
            static member Init(tp:Params.API<'T>) = 
                let c = Constants.API.Init tp
                API<'T>.Init(tp.BLOCK_THREADS, (c.SAFE_ALGORITHM = BLOCK_SCAN_RAKING_MEMOIZE))
                

    [<AutoOpen>]
    module ThreadFields =
        [<Record>]
        type API<'T> =
            {
                mutable temp_storage    : TempStorage.API<'T>
                mutable linear_tid      : int
            }

            [<ReflectedDefinition>]
            static member Init(temp_storage, linear_tid) =
                {
                    temp_storage = temp_storage
                    linear_tid = linear_tid
                }

            [<ReflectedDefinition>]
            static member Init(tp:Params.API<'T>, linear_tid) = API<'T>.Init(TempStorage.API<'T>.Init(tp), linear_tid)

    type _TemplateParams<'T>    = Params.API<'T>
    type _Constants             = Constants.API
    type _TempStorage<'T>       = TempStorage.API<'T>
    type _ThreadFields<'T>      = ThreadFields.API<'T>


module private Internal =
    open Template

    module Sig =
        module SingleDatumPerThread =
            module InclusiveSum =
                type Default<'T>                    = 'T -> Ref<'T> -> unit
                type WithAggregate<'T>              = 'T -> Ref<'T> -> Ref<'T> -> unit
                type WithAggregateAndCallbackOp<'T> = 'T -> Ref<'T> -> Ref<'T> -> Ref<'T -> 'T> -> unit

            module ExclusiveSum =
                type Default<'T>                    = 'T -> Ref<'T> -> unit
                type WithAggregate<'T>              = 'T -> Ref<'T> -> Ref<'T> -> unit
                type WithAggregateAndCallbackOp<'T> = 'T -> Ref<'T> -> Ref<'T> -> Ref<'T -> 'T> -> unit

            module InclusiveScan =
                type Default<'T>                    = 'T -> Ref<'T> -> unit
                type WithAggregate<'T>              = 'T -> Ref<'T> -> Ref<'T> -> unit
                type WithAggregateAndCallbackOp<'T> = 'T -> Ref<'T> -> Ref<'T> -> Ref<'T -> 'T> -> unit

            module ExclusiveScan =
                type Default<'T>                    = 'T -> Ref<'T> -> int -> unit
                type WithAggregate<'T>              = 'T -> Ref<'T> -> int -> Ref<'T> -> unit
                type WithAggregateAndCallbackOp<'T> = 'T -> Ref<'T> -> int -> Ref<'T> -> Ref<'T -> 'T> -> unit

                module Identityless =
                    type Default<'T>                    = 'T -> Ref<'T> -> unit
                    type WithAggregate<'T>              = 'T -> Ref<'T> -> Ref<'T> -> unit
                    type WithAggregateAndCallbackOp<'T> = 'T -> Ref<'T> -> Ref<'T> -> Ref<'T -> 'T> -> unit

        module MultipleDataPerThread =
            module InclusiveSum = ()
            module ExclusiveSum = 
                type Default<'T>                    = deviceptr<'T> -> deviceptr<'T> -> unit
                type WithAggregate<'T>              = deviceptr<'T> -> deviceptr<'T> -> Ref<'T> -> unit
                type WithAggregateAndCallbackOp<'T> = deviceptr<'T> -> deviceptr<'T> -> Ref<'T> -> Ref<'T -> 'T> -> unit
            module InclusiveScan = ()
            module ExclusiveScan =
                ()
                module Identityless = ()
    
    [<AutoOpen>]
    module BlockScan =
        let pickScanKind (tp:_TemplateParams<'T>) =
            let SAFE_ALGORITHM = (_Constants.Init tp).SAFE_ALGORITHM
            SAFE_ALGORITHM = BLOCK_SCAN_WARP_SCANS

        let (|BlockScanWarpScan|_|) (tp:_TemplateParams<'T>) =
            if pickScanKind tp then
                BlockScanWarpScan.api
                <|  Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanWarpScans.Template._TemplateParams<'T>.Init(tp.BLOCK_THREADS, tp.scan_op)
                |> Some
            else
                None

        let (|BlockScanRaking|_|) (tp:_TemplateParams<'T>) =
            let SAFE_ALGORITHM = (_Constants.Init tp).SAFE_ALGORITHM
            if pickScanKind tp |> not then
                BlockScanRaking.api
                <|  Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanRaking.Template._TemplateParams<'T>.Init(tp.BLOCK_THREADS, (SAFE_ALGORITHM = BLOCK_SCAN_RAKING_MEMOIZE), tp.scan_op)
                |> Some
            else
                None



module ExclusiveSum =
    open Template
    open Internal

    module SingleDatumPerThread =
        [<Record>]
        type API<'T> =
            {
                Default                     : Sig.SingleDatumPerThread.ExclusiveSum.Default<'T>
                WithAggregate               : Sig.SingleDatumPerThread.ExclusiveSum.WithAggregate<'T>
                WithAggregateAndCallbackOp  : Sig.SingleDatumPerThread.ExclusiveSum.WithAggregateAndCallbackOp<'T>
            }
        
        let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>)
            (tf:_ThreadFields<'T>) (input:'T) (output:Ref<'T>) =
            let c = _Constants.Init tp
            let InternalBlockScan =
                tp |> function
                | BlockScanWarpScan bsws ->
                    (   bsws
                        <|  Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanWarpScans.Template._ThreadFields<'T>.Init(tf.temp_storage.BlockScanWarpScan, tf.linear_tid, 0, 0)
                    ).ExclusiveSum.WithAggregate
                | BlockScanRaking bsr ->
                    (   bsr
                        <|  Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanRaking.Template._ThreadFields<'T>.Init(tp.BLOCK_THREADS, (c.SAFE_ALGORITHM = BLOCK_SCAN_RAKING_MEMOIZE), tp.scan_op, tf.temp_storage.BlockScanRaking, tf.linear_tid)
                    ).ExclusiveSum.WithAggregate
                | _ -> failwith "Invalid Template Parameters"
                
            let block_aggregate = __local__.Variable()
            InternalBlockScan input output block_aggregate


        let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams<'T>)
            (tf:_ThreadFields<'T>)
            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (tp:_TemplateParams<'T>)
            (tf:_ThreadFields<'T>)
            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()

        let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>)
            (tf:_ThreadFields<'T>) =
                {
                    Default                     =   Default tp tf
                    WithAggregate               =   WithAggregate tp tf
                    WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp tp tf
                }

    module MultipleDataPerThread =
        [<Record>]
        type API<'T> =
            {
                Default                     : Sig.MultipleDataPerThread.ExclusiveSum.Default<'T>
                WithAggregate               : Sig.MultipleDataPerThread.ExclusiveSum.WithAggregate<'T>
                WithAggregateAndCallbackOp  : Sig.MultipleDataPerThread.ExclusiveSum.WithAggregateAndCallbackOp<'T>
            }

        let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>) (items_per_thread:int)
            (tf:_ThreadFields<'T>)
            (input:deviceptr<'T>) (output:deviceptr<'T>) =
            
//            let thread_partial = __local__.Variable<'T>((ThreadReduce.api items_per_thread tp.scan_op).Default input)
//            (SingleDatumPerThread.api tp tf).Default !thread_partial thread_partial
//            (ThreadScanExclusive.api items_per_thread tp.scan_op).WithApplyPrefixDefault input output !thread_partial
//            |> ignore
            ()

        let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams<'T>) (items_per_thread:int)
            (tf:_ThreadFields<'T>)
            (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) = ()

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (tp:_TemplateParams<'T>) (items_per_thread:int)
            (tf:_ThreadFields<'T>)
            (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()

        let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>) (items_per_thread:int)
            (tf:_ThreadFields<'T>) =
            {
                Default                     =   Default tp items_per_thread tf
                WithAggregate               =   WithAggregate tp items_per_thread tf
                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp tp items_per_thread tf
            }
    
    [<Record>]
    type API<'T> =
        {
            SingleDatumPerThread    : SingleDatumPerThread.API<'T>
            MultipleDataPerThread   : MultipleDataPerThread.API<'T>
        }

    let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>) (items_per_thread:int)
        (tf:_ThreadFields<'T>) =
        {
            SingleDatumPerThread    =   SingleDatumPerThread.api tp tf
            MultipleDataPerThread   =   MultipleDataPerThread.api tp items_per_thread tf
        }

module ExclusiveScan =
    open Template
    open Internal

    module SingleDatumPerThread =
        let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>) =
            <@ fun (input:'T) (output:Ref<'T>) (identity:'T) (scan_op:IScanOp<'T>) -> () @>

        let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams<'T>) =
            <@ fun (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<'T>) -> () @>

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (tp:_TemplateParams<'T>) =
            <@ fun (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) -> () @>


        module Identityless =
            let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>) =
                <@ fun (input:'T) (output:Ref<'T>) (scan_op:IScanOp<'T>) -> () @>

            let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams<'T>) =
                <@ fun (input:'T) (output:Ref<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<'T>) -> () @>

            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (tp:_TemplateParams<'T>) =
                <@ fun (input:'T) (output:Ref<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) -> () @>

    module MultipleDataPerThread =
        let [<ReflectedDefinition>] inline Default items_per_thread =
            <@ fun (input:deviceptr<'T>) (output:deviceptr<'T>) (identity:Ref<'T>) (scan_op:IScanOp<'T>) -> () @>

        let [<ReflectedDefinition>] inline WithAggregate items_per_thread =
            <@ fun (input:deviceptr<'T>) (output:deviceptr<'T>) (identity:Ref<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<'T>) -> () @>

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp items_per_thread =
            <@ fun (input:deviceptr<'T>) (output:deviceptr<'T>) (identity:Ref<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) -> () @>


        module Identityless =
            let [<ReflectedDefinition>] inline Default items_per_thread =
                <@ fun (input:deviceptr<'T>) (output:deviceptr<'T>) (scan_op:IScanOp<'T>) -> () @>

            let [<ReflectedDefinition>] inline WithAggregate items_per_thread =
                <@ fun (input:deviceptr<'T>) (output:deviceptr<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<'T>) -> () @>

            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp items_per_thread =
                <@ fun (input:deviceptr<'T>) (output:deviceptr<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) -> () @>


module InclusiveSum =
    open Template
    open Internal

    module SingleDatumPerThread =
        let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>) =
            <@ fun (input:'T) (output:Ref<'T>) -> () @>

        let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams<'T>) =
            <@ fun (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) -> () @>

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (tp:_TemplateParams<'T>) =
            <@ fun (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) -> () @>


    module MultipleDataPerThread =
        let [<ReflectedDefinition>] inline Default items_per_thread =
            <@ fun (input:deviceptr<'T>) (output:deviceptr<'T>) -> () @>

        let [<ReflectedDefinition>] inline WithAggregate items_per_thread =
            <@ fun (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) -> () @>

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp items_per_thread =
            <@ fun (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) -> () @>


module InclusiveScan =
    open Template
    open Internal

    module SingleDatumPerThread =
        let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>) =
            <@ fun (input:'T) (output:Ref<'T>) (scan_op:IScanOp<'T>) -> () @>

        let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams<'T>) =
            <@ fun (input:'T) (output:Ref<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<'T>) -> () @>

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (tp:_TemplateParams<'T>) =
            <@ fun (input:'T) (output:Ref<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) -> () @>


    module MultipleDataPerThread =
        let [<ReflectedDefinition>] inline Default items_per_thread =
            <@ fun (input:deviceptr<'T>) (output:deviceptr<'T>) (scan_op:IScanOp<'T>) -> () @>

        let [<ReflectedDefinition>] inline WithAggregate items_per_thread =
            <@ fun (input:deviceptr<'T>) (output:deviceptr<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<'T>) -> () @>

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp items_per_thread =
            <@ fun (input:deviceptr<'T>) (output:deviceptr<'T>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) -> () @>


module BlockScan =
    open Template

    [<Record>]
    type API<'T> =
        {
            ExclusiveSum    : ExclusiveSum.API<'T>
        }

        [<ReflectedDefinition>]
        static member Create(block_threads:int, algorithm:BlockScanAlgorithm, items_per_thread:int, scan_op:IScanOp<'T>) =
            let tp = _TemplateParams<'T>.Init(block_threads, algorithm, scan_op)
            let tf = _ThreadFields<'T>.Init(tp, 0)
            {
                ExclusiveSum = ExclusiveSum.api tp items_per_thread tf
            }

        [<ReflectedDefinition>]
        static member Create(block_threads:int, algorithm:BlockScanAlgorithm, scan_op:IScanOp<'T>) = API<'T>.Create(block_threads, algorithm, 1, scan_op)

        [<ReflectedDefinition>]
        static member Create(block_threads:int, items_per_thread:int, scan_op:IScanOp<'T>) = API<'T>.Create(block_threads, BLOCK_SCAN_RAKING, items_per_thread, scan_op )

        [<ReflectedDefinition>]
        static member Create(block_threads:int, scan_op:IScanOp<'T>) = API<'T>.Create(block_threads, 1, scan_op)
//        [<ReflectedDefinition>]
//        static member Create(block_threads:int, algorithm:BlockScanAlgorithm, items_per_thread:int, id:'T) = API<'T>.Create(block_threads, algorithm, 1, (scan_op ADD id))


    let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>) (items_per_thread:int)
        (tf:_ThreadFields<'T>) = 
        {
            ExclusiveSum = ExclusiveSum.api tp items_per_thread tf
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
//    member inline this.ExclusiveSum(input:int, output:Ref<'T>) =
//         localize thread fields
//        let temp_storage = this.ThreadFields.temp_storage
//        let linear_tid = this.ThreadFields.linear_tid
//
//        let block_aggregate = __local__.Variable()
//        this.InternalBlockScan.ExclusiveSum(temp_storage, linear_tid, input, output, block_aggregate)
//
//            
//
//    member inline this.ExclusiveSum(input:int, output:Ref<'T>, block_aggregate:Ref<'T>) =
//         localize thread fields
//        let temp_storage = this.ThreadFields.temp_storage
//        let linear_tid = this.ThreadFields.linear_tid
//        
//        this.InternalBlockScan.ExclusiveSum(temp_storage, linear_tid, input, output, block_aggregate)
//
//    member this.ExclusiveSum(input:int, output:Ref<'T>, block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T -> 'T>) =
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
//    member inline this.ExclusiveSum(items_per_thread:int, input:deviceptr<int>, output:deviceptr<int>, block_aggregate:Ref<'T>) =
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
//    member inline this.ExclusiveSum(items_per_thread:int, input:deviceptr<int>, output:deviceptr<int>, block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T -> 'T>) =
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
//    member this.ExclusiveScan(input:int, output:Ref<'T>, identity:int, scan_op:(int -> int -> int)) =
//        let bswc, bsr = this.InternalBlockScan.GetScanner(this.Constants)
//        (bswc, bsr) |> function
//        | Some bswc, None ->
//            bswc.Initialize(temp_storage, linear_tid).ExclusiveSum(input, output, block_aggregate)
//        | None, Some bsr ->
//            bsr.Initialize(temp_storage, linear_tid).ExclusiveSum(input, output, block_aggregate)
//        | _, _ ->        
//        InternalBlockScan(temp_storage, linear_tid).ExclusiveScan(input, output, identity, scan_op, block_aggregate)
//
//    member this.ExclusiveScan(input:int, output:Ref<'T>, identity:Ref<'T>, scan_op:(int -> int -> int), block_aggregate:Ref<'T>) =
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
//    member this.ExclusiveScan(input:int, output:Ref<'T>, identity:int, scan_op:(int -> int -> int), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T -> 'T>) =
//        let thread_partial = ThreadReduce(input, scan_op)
//
//        // Exclusive threadblock-scan
//        ExclusiveScan(thread_partial, thread_partial, identity, scan_op)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial)
//    
//    // exclusive prefix scan operations (identityless, single datum per thread)
//    member this.ExclusiveScan(input:int, output:Ref<'T>, scan_op:(int -> int -> int)) =
//        let thread_partial = ThreadReduce(input, scan_op)
//
//        // Exclusive threadblock-scan
//        ExclusiveScan(thread_partial, thread_partial, identity, scan_op, block_aggregate)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial)
//    
//    member this.ExclusiveScan(input:int, output:Ref<'T>, scan_op:(int -> int -> int), block_aggregate:Ref<'T>) =
//        let thread_partial = ThreadReduce(input, scan_op)
//
//        // Exclusive threadblock-scan
//        this.ExclusiveScan(thread_partial, thread_partial, identity, scan_op, block_aggregate, block_prefix_callback_op)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial)
//
//    member this.ExclusiveScan(input:int, output:Ref<'T>, scan_op:(int -> int -> int), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T -> 'T>) =
//        let block_aggregate = __null() |> __ptr_to_ref
//        this.InternalBlockScan(temp_storage, linear_tid).ExclusiveScan(input, output, scan_op, block_aggregate)
//
//
//    // exclusive prefix scan operations (multiple data per thread)
//    //<items_per_thread>
//    member this.ExclusiveScan(input:deviceptr<int>, output:deviceptr<int>, identity:Ref<'T>, scan_op:(int -> int -> int)) =
//        this.InternalBlockScan(temp_storage, linear_tid).ExclusiveScan(input, output, scan_op, block_aggregate)
//
//    member this.ExclusiveScan(input:deviceptr<int>, output:deviceptr<int>, identity:Ref<'T>, scan_op:(int -> int -> int), block_aggregate:Ref<'T>) =
//        InternalBlockScan(temp_storage, linear_tid).ExclusiveScan(input, output, scan_op, block_aggregate, block_prefix_callback_op)
//
//    member this.ExclusiveScan(input:deviceptr<int>, output:deviceptr<int>, identity:Ref<'T>, scan_op:(int -> int -> int), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T -> 'T>) =
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
//    member this.ExclusiveScan(input:deviceptr<int>, output:deviceptr<int>, scan_op:(int -> int -> int), block_aggregate:Ref<'T>) =
//        // Reduce consecutive thread items in registers
//        let thread_partial = ThreadReduce(input, scan_op)
//
//        // Exclusive threadblock-scan
//        this.ExclusiveScan(thread_partial, thread_partial, scan_op, block_aggregate, block_prefix_callback_op)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial)
//
//    member this.ExclusiveScan(input:deviceptr<int>, output:deviceptr<int>, scan_op:(int -> int -> int), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T -> 'T>) =
//        let block_aggregate = __null() |> __ptr_to_ref
//        this.InternalBlockScan(temp_storage, linear_tid).InclusiveSum(input, output, block_aggregate)
//
//
//    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    // inclusive prefix sum operations
//    member this.InclusiveSum(input:int, output:Ref<'T>) =
//        this.InternalBlockScan(temp_storage, linear_tid).InclusiveSum(input, output, block_aggregate)
//
//    member this.InclusiveSum(input:int, output:Ref<'T>, block_aggregate:Ref<'T>) =
//        this.InternalBlockScan(temp_storage, linear_tid).InclusiveSum(input, output, block_aggregate, block_prefix_callback_op)
//
//    member this.InclusiveSum(input:int, output:Ref<'T>, block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T -> 'T>) =
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
//    member this.InclusiveSum(input:deviceptr<int>, output:deviceptr<int>, block_aggregate:Ref<'T>) =
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
//    member this.InclusiveSum(input:deviceptr<int>, output:deviceptr<int>, block_prefix_callback_op:Ref<'T -> 'T>) =
//        let block_aggregate = __nul() |> __ptr_to_ref
//        this.InclusiveScan(input, output, scan_op, block_aggregate)
//    
//   
//    // inclusive prefix scan operations
//    member this.InclusiveScan(input:int, output:Ref<'T>, scan_op:(int -> int -> int)) =
//        let block_aggregate = __null() |> __ptr_to_ref
//        this.InclusiveScan(input, output, scan_op, block_aggregate)
//    
//    member this.InclusiveScan(input:int, output:Ref<'T>, scan_op:(int -> int -> int), block_aggregate:Ref<'T>) =
//        InternalBlockScan(temp_storage, linear_tid).InclusiveScan(input, output, scan_op, block_aggregate)
//
//
//    member this.InclusiveScan(input:int, output:Ref<'T>, scan_op:(int -> int -> int), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T -> 'T>) =
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
//    member this.InclusiveScan(input:deviceptr<int>, output:deviceptr<int>, scan_op:(int -> int -> int), block_aggregate:Ref<'T>) =
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
//    member this.InclusiveScan(input:deviceptr<int>, output:deviceptr<int>, scan_op:(int -> int -> int), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T -> 'T>) =
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
////            fun (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) -> ()
////
////        // exclusive prefix scan operations
////        let exclusiveScan =
////            fun (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) -> ()
////
////
////        module Identityless =
////            // exclusive prefix scan operations (identityless, single datum per thread)
////            let exclusiveScan =
////                fun (input:'T) (output:Ref<'T>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) -> ()   
////
////
////    module STMD =
////        // exclusive prefix sum operations (multiple data per thread)
////        let exclusiveSum items_per_thread =
////            fun (input:deviceptr<int>) (output:deviceptr<int>) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) -> ()
////
////        // exclusive prefix scan operations (multiple data per thread)
////        let exclusiveScan items_per_thread =
////            fun (input:deviceptr<int>) (output:deviceptr<int>) (identity:Ref<'T>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) -> ()
////        
////
////        module Identityless =
////            // exclusive prefix scan operations (identityless, multiple data per thread)
////            let exclusiveScan_noId items_per_thread =
////                fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) -> ()
////
////
////module InclusiveScan =
////    
////    module STSD =
////        // inclusive prefix sum operations
////        let inclusiveSum =
////            fun (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) -> ()
////
////        // inclusive prefix scan operations
////        let inclusiveScan =
////            fun (input:'T) (output:Ref<'T>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) -> ()
////        
////
////    module STMD =
////        // inclusive prefix sum operations (multiple data per thread)
////        let inclusiveSum items_per_thread =
////            fun (input:deviceptr<int>) (output:deviceptr<int>) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) -> ()
////        
////        // inclusive scan operations (multiple data per thread)
////        let inclusiveScan items_per_thread =
////            fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) -> ()
////        
////        
////type API =
////    {
////        ExclusiveScan : <'T>
////        InclusiveScan : <'T>
////    }
////
////let inline BlockScan () 
//
