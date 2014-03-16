[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Scan

open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Core.Operators

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
//        fun (first:KeyValuePair<'K,'V>, second:KeyValuePair<'K,'V>) =
//            KeyValuePair<'K,'V>(second.Key,
//                if second.Key <> first.Key then second.Value else (first.Value, second.Value) ||> op )
//    | SegmentedOp ->
//        fun (first:KeyValuePair<'K,'V>, second:KeyValuePair<'K,'V>) =
//            if second.Key > 0G then KeyValuePair<'K,'V>(first.Key + second.Key, second.Value)
//            else KeyValuePair<'K,'V>(first.Key + second.Key, (first.Value, second.Value) ||> op)
type BlockScanAlgorithm =
    | BLOCK_SCAN_RAKING         = 0
    | BLOCK_SCAN_RAKING_MEMOIZE = 1
    | BLOCK_SCAN_WARP_SCANS     = 2



module BlockScan =
   
    type Params = { BLOCK_THREADS : int; ALGORITHM : BlockScanAlgorithm }
                    static member Init(block_threads, algorithm) 
                        = { BLOCK_THREADS = block_threads; ALGORITHM = algorithm }

    type Constants = 
        { SAFE_ALGORITHM : BlockScanAlgorithm }

        static member Init(p:Params) =
            let safe_algorithm = 
                if ((p.ALGORITHM = BlockScanAlgorithm.BLOCK_SCAN_WARP_SCANS) && (p.BLOCK_THREADS % CUB_PTX_WARP_THREADS <> 0)) then 
                    BlockScanAlgorithm.BLOCK_SCAN_RAKING 
                else p.ALGORITHM
            { SAFE_ALGORITHM = safe_algorithm }

    type ScanKind =
        | WarpScans = 0
        | Raking    = 1


    type HostApi =
        {   
            ScanKind                    : ScanKind
            BlockScanWarpScansHostApi   : BlockScanWarpScans.HostApi
            BlockScanRakingHostApi      : BlockScanRaking.HostApi             
            Params                      : Params
            Constants                   : Constants
        }

        static member Init(block_threads, algorithm) =
            let p = Params.Init(block_threads, algorithm)
            let c = Constants.Init(p)
            let bsws_h = BlockScanWarpScans.HostApi.Init(p.BLOCK_THREADS)
            let bsr_h = BlockScanRaking.HostApi.Init(p.BLOCK_THREADS, (c.SAFE_ALGORITHM = BlockScanAlgorithm.BLOCK_SCAN_RAKING_MEMOIZE))
            let kind = if c.SAFE_ALGORITHM = BlockScanAlgorithm.BLOCK_SCAN_WARP_SCANS then ScanKind.WarpScans else ScanKind.Raking
            { ScanKind = kind; BlockScanWarpScansHostApi = bsws_h; BlockScanRakingHostApi = bsr_h; Params = p; Constants = c }

    
    type TempStorage<'T> = BlockScanWarpScans.TempStorage<'T>

    module InternalBlockScan =
//        module BlockScanRaking =
//            module ExclusiveSum = ()
//            module ExclusiveScan = ()
//            module InclusiveSum = ()

        module ExclusiveSum =
            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) =
                let warp_id = BlockScanWarpScans.warp_id h.Params.BLOCK_THREADS linear_tid
                let lane_id = BlockScanWarpScans.lane_id h.Params.BLOCK_THREADS linear_tid
                BlockScanWarpScans.ExclusiveSum.WithAggregate h.BlockScanWarpScansHostApi scan_op
                    temp_storage linear_tid warp_id lane_id
                    input output block_aggregate


            let [<ReflectedDefinition>] inline WithAggregateInt (h:HostApi)
                (temp_storage:TempStorage<int>) (linear_tid:int)
                (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) =
                BlockScanWarpScans.IntApi.Init(h.BlockScanWarpScansHostApi, temp_storage, linear_tid).ExclusiveSum(
                    h.BlockScanWarpScansHostApi, input, output, block_aggregate)
    
            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (input:'T) (output:Ref<'T>) =
                let block_aggregate = __local__.Variable<'T>()
                WithAggregate h scan_op temp_storage linear_tid input output block_aggregate

            let [<ReflectedDefinition>] inline DefaultInt (h:HostApi)
                (temp_storage:TempStorage<int>) (linear_tid:int)
                (input:int) (output:Ref<int>) =
                let block_aggregate = __local__.Variable<int>()
                BlockScanWarpScans.IntApi.Init(h.BlockScanWarpScansHostApi, temp_storage, linear_tid).ExclusiveSum(
                    h.BlockScanWarpScansHostApi, input, output, block_aggregate)

//            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T)
//                (temp_storage:TempStorage<'T>) (linear_tid:int)
//                (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
//                BlockScanWarpScans.ExclusiveSum.WithAggregateAndCallbackOp h.BlockScanWarpScansHostApi scan_op ws_d


        module ExclusiveScan =
            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) =
                let warp_id = BlockScanWarpScans.warp_id h.Params.BLOCK_THREADS linear_tid
                let lane_id = BlockScanWarpScans.lane_id h.Params.BLOCK_THREADS linear_tid
                BlockScanWarpScans.ExclusiveScan.WithAggregate h.BlockScanWarpScansHostApi scan_op 
                    temp_storage linear_tid warp_id lane_id
                    input output identity block_aggregate
    
            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (input:'T) (output:Ref<'T>) (identity:Ref<'T>) =
                let block_aggregate = __local__.Variable<'T>()
                WithAggregate h scan_op temp_storage linear_tid input output identity block_aggregate
    
    
//            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T)
//                (temp_storage:TempStorage<'T>) (linear_tid:int)
//                (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
//                BlockScanWarpScans.ExclusiveScan.WithAggregateAndCallbackOp h.BlockScanWarpScansHostApi scan_op w
    
            module Identityless =
                let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
                    (temp_storage:TempStorage<'T>) (linear_tid:int)
                    (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) =
                    let warp_id = BlockScanWarpScans.warp_id h.Params.BLOCK_THREADS linear_tid
                    let lane_id = BlockScanWarpScans.lane_id h.Params.BLOCK_THREADS linear_tid
                    BlockScanWarpScans.ExclusiveScan.Identityless.WithAggregate h.BlockScanWarpScansHostApi scan_op
                        temp_storage linear_tid warp_id lane_id
                        input output block_aggregate
    
                let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
                    (temp_storage:TempStorage<'T>) (linear_tid:int)
                    (input:'T) (output:Ref<'T>) =
                    let block_aggregate = __local__.Variable<'T>()
                    WithAggregate h scan_op temp_storage linear_tid input output block_aggregate
    
    
//                let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T)
//                    (temp_storage:TempStorage<'T>) (linear_tid:int)
//                    (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
//                    BlockScanWarpScans.ExclusiveScan.Identityless.WithAggregateAndCallbackOp h.BlockScanWarpScansHostApi scan_op ws_d


    module ExclusiveSum =
        open InternalBlockScan

        module SingleDatumPerThread =        
            let [<ReflectedDefinition>] inline WithAggregateInt (h:HostApi)
                (temp_storage:TempStorage<int>) (linear_tid:int)
                (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) =
                ExclusiveSum.WithAggregateInt h temp_storage linear_tid input output block_aggregate

            let [<ReflectedDefinition>] inline DefaultInt (h:HostApi)
                (temp_storage:TempStorage<int>) (linear_tid:int)
                (input:int) (output:Ref<int>) =
                let block_aggregate = __local__.Variable<int>()
                ExclusiveSum.DefaultInt h temp_storage linear_tid input output
            
            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) =
                InternalBlockScan.ExclusiveSum.WithAggregate h scan_op temp_storage linear_tid input output block_aggregate
            

            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (input:'T) (output:Ref<'T>) =
                let block_aggregate = __local__.Variable()
                WithAggregate h scan_op temp_storage linear_tid input output block_aggregate
            

//            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T)
//                (temp_storage:TempStorage<'T>) (linear_tid:int)
//                        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
//                    ()
//            


        module MultipleDataPerThread =
            let [<ReflectedDefinition>] inline WithAggregateInt (h:HostApi) (items_per_thread:int)
                (temp_storage:TempStorage<int>) (linear_tid:int)
                (input:deviceptr<int>) (output:deviceptr<int>) (block_aggregate:Ref<int>) =
                let thread_partial = __local__.Variable<int>(ThreadReduce.DefaultInt items_per_thread input)
                SingleDatumPerThread.WithAggregateInt h temp_storage linear_tid !thread_partial thread_partial block_aggregate
                let x = ThreadScanExclusive.WithApplyPrefixDefaultInt items_per_thread input output !thread_partial
                ()

            let [<ReflectedDefinition>] inline DefaultInt (h:HostApi) (items_per_thread:int)
                (temp_storage:TempStorage<int>) (linear_tid:int)
                (input:deviceptr<int>) (output:deviceptr<int>) =
                let thread_partial = __local__.Variable<int>(ThreadReduce.DefaultInt items_per_thread input)
                SingleDatumPerThread.DefaultInt h temp_storage linear_tid !thread_partial thread_partial
                let x = ThreadScanExclusive.WithApplyPrefixDefaultInt items_per_thread input output !thread_partial
                ()

            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) =
            
                let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
                
                SingleDatumPerThread.WithAggregate h scan_op temp_storage linear_tid !thread_partial thread_partial block_aggregate
                
                ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
            

            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (input:deviceptr<'T>) (output:deviceptr<'T>) =
                
                let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
                
                SingleDatumPerThread.Default h scan_op temp_storage linear_tid !thread_partial thread_partial

                ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
            



//            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
//                (temp_storage:TempStorage<'T>) (linear_tid:int)
//                (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
//                ()
            
    //
    //
    module ExclusiveScan =
    
        module SingleDatumPerThread =
            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) =
                InternalBlockScan.ExclusiveScan.WithAggregate h scan_op temp_storage linear_tid input output identity block_aggregate
                
         
            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (input:'T) (output:Ref<'T>) (identity:Ref<'T>) =
                let block_aggregate = __local__.Variable<'T>()
                WithAggregate h scan_op temp_storage linear_tid input output identity block_aggregate
                
    
//            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T)
//                (temp_storage:TempStorage<'T>) (linear_tid:int)
//                (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
//                ()
                
    
            module Identityless =
                let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
                    (temp_storage:TempStorage<'T>) (linear_tid:int)
                    (input:'T) (output:Ref<'T>) =
                    InternalBlockScan.ExclusiveScan.Identityless.Default h scan_op temp_storage linear_tid input output
                    
    
                let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
                    (temp_storage:TempStorage<'T>) (linear_tid:int)
                    (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) =
                    InternalBlockScan.ExclusiveScan.Identityless.WithAggregate h scan_op temp_storage linear_tid input output block_aggregate
                    
                        
    
//                let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T)
//                    (temp_storage:TempStorage<'T>) (linear_tid:int)
//                    (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
//                    InternalBlockScan.ExclusiveScan.Identityless.WithAggregateAndCallbackOp h scan_op temp_storage linear_tid input output block_aggregate block_prefix_callback_op
//                    
    
        module MultipleDataPerThread =
            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (input:deviceptr<'T>) (output:deviceptr<'T>) (identity:Ref<'T>) =
                    
                let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
    
                SingleDatumPerThread.Default h scan_op temp_storage linear_tid !thread_partial thread_partial identity
    
                ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
                
    
            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (input:deviceptr<'T>) (output:deviceptr<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) =
                    
                let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
    
                SingleDatumPerThread.WithAggregate h scan_op temp_storage linear_tid !thread_partial thread_partial identity block_aggregate
    
                ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
                
    
//            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
//                (temp_storage:TempStorage<'T>) (linear_tid:int)
//                (input:deviceptr<'T>) (output:deviceptr<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
//                
//                let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
//    
//                SingleDatumPerThread.WithAggregateAndCallbackOp h scan_op temp_storage linear_tid !thread_partial thread_partial identity block_aggregate block_prefix_callback_op
//    
//                ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
//                
                
    
    
            module Identityless =
                let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
                    (temp_storage:TempStorage<'T>) (linear_tid:int)
                    (input:deviceptr<'T>) (output:deviceptr<'T>) =
                    let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
    
                    SingleDatumPerThread.Identityless.Default h scan_op temp_storage linear_tid !thread_partial thread_partial
    
                    ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
                
                    
    
                let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int) 
                    (temp_storage:TempStorage<'T>) (linear_tid:int)
                    (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) =
                    let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
    
                    SingleDatumPerThread.Identityless.WithAggregate h scan_op temp_storage linear_tid !thread_partial thread_partial block_aggregate
    
                    ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
                
                    
    
//                let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
//                    (temp_storage:TempStorage<'T>) (linear_tid:int)
//                    (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
//                    let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
//    
//                    SingleDatumPerThread.Identityless.WithAggregateAndCallbackOp h scan_op temp_storage linear_tid !thread_partial thread_partial block_aggregate block_prefix_callback_op
//    
//                    ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
                
                


    module InclusiveSum =
        open Template
    
        module SingleDatumPerThread =
            let [<ReflectedDefinition>] inline Default  (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>) = ()
    
            let [<ReflectedDefinition>] inline WithAggregate  (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()
    
            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp  (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) = ()
    
    
        module MultipleDataPerThread =
            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
                (input:deviceptr<'T>) (output:deviceptr<'T>) = ()
    
            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
                (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) = ()
    
            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
                (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) = ()


    module InclusiveScan =
        open Template
    
        module SingleDatumPerThread =
            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>)  = ()
    
            let [<ReflectedDefinition>] inline WithAggregate  (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>)  (block_aggregate:Ref<'T>) = ()
    
            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (input:'T) (output:Ref<'T>)  (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) = ()
    
    
        module MultipleDataPerThread =
            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
                (input:deviceptr<'T>) (output:deviceptr<'T>) = ()
    
            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
                (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) = ()
    
            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
                (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) = ()


    let [<ReflectedDefinition>] inline PrivateStorage<'T>(h:HostApi) = TempStorage<'T>.Uninitialized(h.BlockScanWarpScansHostApi)
//        if h.ScanKind = ScanKind.WarpScans then 
//            BlockScanWarpScans.TempStorage<'T>.Uninitialized(h.BlockScanWarpScansHostApi)
//        else
//            BlockScanRaking.TempStorage<'T>.Uninitialized(h.BlockScanRakingHostApi)

    [<Record>]
    type IntApi =
        { mutable temp_storage : TempStorage<int>; mutable linear_tid : int}

        [<ReflectedDefinition>] static member Init(h:HostApi) 
            = { temp_storage = PrivateStorage<int>(h);  linear_tid = threadIdx.x }

        [<ReflectedDefinition>] static member Init(h:HostApi, temp_storage:TempStorage<int>) 
            = { temp_storage = temp_storage;  linear_tid = threadIdx.x }

        [<ReflectedDefinition>] static member Init(h:HostApi, linear_tid:int) 
            = { temp_storage = PrivateStorage<int>(h);  linear_tid = linear_tid }

        [<ReflectedDefinition>] static member Init(h:HostApi, temp_storage:TempStorage<int>, linear_tid:int) 
            = { temp_storage = temp_storage;  linear_tid = linear_tid }



        [<ReflectedDefinition>] 
        member this.ExclusiveSum(h, input:int, output:Ref<int>) = 
            ExclusiveSum.SingleDatumPerThread.DefaultInt h this.temp_storage this.linear_tid input output
        
        [<ReflectedDefinition>]
        member this.ExclusiveSum(h, input:int, output:Ref<int>, block_aggregate:Ref<int>) = 
            ExclusiveSum.SingleDatumPerThread.WithAggregateInt h this.temp_storage this.linear_tid input output block_aggregate
        


        [<ReflectedDefinition>]
        member this.ExclusiveSum(h, items_per_thread, input:deviceptr<int>, output:deviceptr<int>) =
            ExclusiveSum.MultipleDataPerThread.DefaultInt h items_per_thread this.temp_storage this.linear_tid input output
        
        [<ReflectedDefinition>] 
        member this.ExclusiveSum(h, items_per_thread, input:deviceptr<int>, output:deviceptr<int>, block_aggregate:Ref<int>) =
            ExclusiveSum.MultipleDataPerThread.WithAggregateInt h items_per_thread this.temp_storage this.linear_tid input output block_aggregate





//    [<Record>]
//    type API<'T> =
//        { mutable temp_storage : TempStorage<'T>; mutable linear_tid : int}
//
//        [<ReflectedDefinition>] static member Init(h:HostApi) 
//            = { temp_storage = PrivateStorage<'T>(h);  linear_tid = threadIdx.x }
//
//        [<ReflectedDefinition>] static member Init(h:HostApi, temp_storage:TempStorage<'T>) 
//            = { temp_storage = temp_storage;  linear_tid = threadIdx.x }
//
//        [<ReflectedDefinition>] static member Init(h:HostApi, linear_tid:int) 
//            = { temp_storage = PrivateStorage<'T>(h);  linear_tid = linear_tid }
//
//        [<ReflectedDefinition>] static member Init(h:HostApi, temp_storage:TempStorage<'T>, linear_tid:int) 
//            = { temp_storage = temp_storage;  linear_tid = linear_tid }
//
//
//
//        [<ReflectedDefinition>] member this.ExclusiveSum(h, scan_op, input:'T, output:Ref<'T>) 
//            = ExclusiveSum.SingleDatumPerThread.Default h scan_op this.DeviceApi input output
//        [<ReflectedDefinition>] member this.ExclusiveSum(h, scan_op, input:'T, output:Ref<'T>, block_aggregate:Ref<'T>)
//            = ExclusiveSum.SingleDatumPerThread.WithAggregate h scan_op this.DeviceApi input output block_aggregate
//        
//        [<ReflectedDefinition>] member this.ExclusiveSum(h, scan_op, items_per_thread, input:deviceptr<'T>, output:deviceptr<'T>)
//            = ExclusiveSum.MultipleDataPerThread.Default h scan_op items_per_thread this.DeviceApi input output
//        [<ReflectedDefinition>] member this.ExclusiveSum(h, scan_op, items_per_thread, input:deviceptr<'T>, output:deviceptr<'T>, block_aggregate:Ref<'T>)
//            = ExclusiveSum.MultipleDataPerThread.WithAggregate h scan_op items_per_thread this.DeviceApi input output block_aggregate
////
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op:'T -> 'T -> 'T, input:'T, output:Ref<'T>, identity:'T)
//            = ExclusiveScan.SingleDatumPerThread.Default h scan_op this.DeviceApi input output (__local__.Variable<'T>(identity))
//        
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input:'T, output:Ref<'T>, identity:Ref<'T>, block_aggregate:Ref<'T>) 
//            = ExclusiveScan.SingleDatumPerThread.WithAggregate h scan_op this.DeviceApi input output identity block_aggregate
//        
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op:'T -> 'T -> 'T, items_per_thread:int, input:deviceptr<'T>, output:deviceptr<'T>, identity:'T) 
//            = ExclusiveScan.MultipleDataPerThread.Default h scan_op items_per_thread this.DeviceApi input output (__local__.Variable<'T>(identity))
//        
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, items_per_thread, input:deviceptr<'T>, output:deviceptr<'T>, identity:Ref<'T>, block_aggregate)
//            = ExclusiveScan.MultipleDataPerThread.WithAggregate h scan_op items_per_thread this.DeviceApi input output identity block_aggregate
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input:'T, output:Ref<'T>) 
//            = ExclusiveScan.SingleDatumPerThread.Identityless.Default h scan_op this.DeviceApi input output
//        
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input:'T, output:Ref<'T>, block_aggregate:Ref<'T>) 
//            = ExclusiveScan.SingleDatumPerThread.Identityless.WithAggregate h scan_op this.DeviceApi input output block_aggregate
//        
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, items_per_thread, input:deviceptr<'T>, output:deviceptr<'T>) 
//            = ExclusiveScan.MultipleDataPerThread.Identityless.Default h scan_op items_per_thread this.DeviceApi input output
//        
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, items_per_thread, input:deviceptr<'T>, output:deviceptr<'T>, block_aggregate:Ref<'T>) 
//            = ExclusiveScan.MultipleDataPerThread.Identityless.WithAggregate h scan_op items_per_thread this.DeviceApi input output block_aggregate

//module Template =
//    [<AutoOpen>]
//    module Host =
//        module Params =
//            type API =
//                {
//                    BLOCK_THREADS   : int
//                    ALGORITHM       : BlockScanAlgorithm                
//                }
//                                
//                static member Init(block_threads, algorithm) = { BLOCK_THREADS = block_threads; ALGORITHM = algorithm }
//
//        module Constants =
//            type API =
//                {
//                    SAFE_ALGORITHM : BlockScanAlgorithm
//                }
//
//                static member Init(p:Params.API) =
//                    let safe_algorithm = 
//                        if ((p.ALGORITHM = BlockScanAlgorithm.BLOCK_SCAN_WARP_SCANS) && (p.BLOCK_THREADS % CUB_PTX_WARP_THREADS <> 0)) then 
//                            BlockScanAlgorithm.BLOCK_SCAN_RAKING 
//                        else p.ALGORITHM
//                    {
//                        SAFE_ALGORITHM = safe_algorithm
//                    }
//        ///@TODO
////        module Specialization =
////            type InternalBlockScan =
////                | BlockScanWarpScans of BlockScanWarpScans.HostApi
////                | BlockScanRaking of BlockScanRaking.HostApi
////
////            type API =
////                {
////                    InternalBlockScan : InternalBlockScan
////                }
////
////                static member Init(p:Params.API, c:Constants.API) =
////                    let ibs = 
////                        if c.SAFE_ALGORITHM = BlockScanAlgorithm.BLOCK_SCAN_WARP_SCANS then
////                            BlockScanWarpScans(BlockScanWarpScans.HostApi.Init(p.BLOCK_THREADS))
////                        else
////                            BlockScanRaking(BlockScanRaking.HostApi.Init(p.BLOCK_THREADS, (c.SAFE_ALGORITHM = BlockScanAlgorithm.BLOCK_SCAN_RAKING_MEMOIZE)))
////                    {
////                        InternalBlockScan = ibs
////                   }
//        type ScanKind =
//            | WarpScans = 0
//            | Raking    = 1
//
//
//        type API =
//            {   
//                ScanKind                    : ScanKind
//                BlockScanWarpScansHostApi   : BlockScanWarpScans.HostApi
//                BlockScanRakingHostApi      : BlockScanRaking.HostApi             
//                Params                      : Params.API
//                Constants                   : Constants.API                
//            }
//
//            static member Init(block_threads, algorithm) =
//                let p = Params.API.Init(block_threads, algorithm)
//                let c = Constants.API.Init(p)
//                let bsws_h = BlockScanWarpScans.HostApi.Init(p.BLOCK_THREADS)
//                let bsr_h = BlockScanRaking.HostApi.Init(p.BLOCK_THREADS, (c.SAFE_ALGORITHM = BlockScanAlgorithm.BLOCK_SCAN_RAKING_MEMOIZE))
//                let kind = if c.SAFE_ALGORITHM = BlockScanAlgorithm.BLOCK_SCAN_WARP_SCANS then ScanKind.WarpScans else ScanKind.Raking
//                { ScanKind = kind; BlockScanWarpScansHostApi = bsws_h; BlockScanRakingHostApi = bsr_h; Params = p; Constants = c }
//
//
//    module Device =
//        module TempStorage =
//            [<Record>] 
//            type API<'T> = 
//                { BlockScanWarpScans : BlockScanWarpScans.TempStorage<'T>;  BlockScanRaking : BlockScanRaking.TempStorage<'T> }
//                
//                [<ReflectedDefinition>] static member Init(h:Host.API) 
//                    = {     BlockScanWarpScans  = BlockScanWarpScans.TempStorage<'T>.Uninitialized(h.BlockScanWarpScansHostApi);
//                            BlockScanRaking     = BlockScanRaking.TempStorage<'T>.Uninitialized(h.BlockScanRakingHostApi)}
////            type [<Record>] API<'T> = BlockScanRaking.TempStorage<'T>
//
////        let [<ReflectedDefinition>] inline PrivateStorage<'T>(h:Host.API) = TempStorage.API<'T>.Init(h)
////                
////        [<Record>]
////        type API<'T> =
////            { mutable temp_storage : TempStorage.API<'T>; mutable linear_tid : int}
////
////            [<ReflectedDefinition>] static member Init(h:Host.API) 
////                = { temp_storage = PrivateStorage<'T>(h);  linear_tid = threadIdx.x }
////
////            [<ReflectedDefinition>] static member Init(h:Host.API, temp_storage:TempStorage.API<'T>) 
////                = { temp_storage = temp_storage;  linear_tid = threadIdx.x }
////
////            [<ReflectedDefinition>] static member Init(h:Host.API, linear_tid:int) 
////                = { temp_storage = PrivateStorage<'T>(h);  linear_tid = linear_tid }
////
////            [<ReflectedDefinition>] static member Init(h:Host.API, temp_storage:TempStorage.API<'T>, linear_tid:int) 
////                = { temp_storage = temp_storage;  linear_tid = linear_tid }
//
//
//
//    type _TemplateParams        = Host.Params.API
//    type _Constants             = Host.Constants.API
//    type HostApi               = Host.API
//    
//    type _TempStorage<'T>       = Device.TempStorage.API<'T>
//
//
//module InternalBlockScan =
//    open Template
//
//    module ExclusiveSum =
////        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (temp_storage:TempStorage<'T>) (linear_tid:int)
////            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) =
//////            h.ScanKind |> function
//////            | ScanKind.WarpScans ->
////            BlockScanWarpScans.ExclusiveSum.WithAggregate h.BlockScanWarpScansHostApi scan_op ws_d
////                input output block_aggregate
//////            | _ ->
////                let bsr_d = BlockScanRaking.DeviceApi<'T>.Init(h.BlockScanRakingHostApi, d.temp_storage.BlockScanRaking, d.linear_tid)
////                BlockScanRaking.ExclusiveSum.WithAggregate h.BlockScanRakingHostApi scan_op bsr_d
////                    input output block_aggregate
//        let [<ReflectedDefinition>] inline WithAggregateInt (h:HostApi)
//            (temp_storage:TempStorage<'T>) (linear_tid:int)
//            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) =
////            h.ScanKind |> function
////            | ScanKind.WarpScans ->
//            BlockScanWarpScans.ExclusiveSum.WithAggregateInt h.BlockScanWarpScansHostApi ws_d
//                input output block_aggregate
//
////
////        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (temp_storage:TempStorage<'T>) (linear_tid:int)
////            (input:'T) (output:Ref<'T>) =
////            let block_aggregate = __local__.Variable<'T>()
////            WithAggregate h scan_op temp_storage linear_tid input output block_aggregate
//
//        let [<ReflectedDefinition>] inline DefaultInt (h:HostApi)
//            (temp_storage:TempStorage<'T>) (linear_tid:int)
//            (input:'T) (output:Ref<'T>) =
//            let block_aggregate = __local__.Variable<'T>()
//            WithAggregateInt h temp_storage linear_tid input output block_aggregate
//
//
////        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (temp_storage:TempStorage<'T>) (linear_tid:int)
////            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
////                            h.ScanKind |> function
////            | ScanKind.WarpScans ->
////                let ws_d = BlockScanWarpScans.DeviceApi<'T>.Init(h.BlockScanWarpScansHostApi, d.temp_storage.BlockScanWarpScans, d.linear_tid)
////                BlockScanWarpScans.ExclusiveSum.WithAggregateAndCallbackOp h.BlockScanWarpScansHostApi scan_op ws_d
////                    input output block_aggregate block_prefix_callback_op
////            | _ ->
////                let bsr_d = BlockScanRaking.DeviceApi<'T>.Init(h.BlockScanRakingHostApi, d.temp_storage.BlockScanRaking, d.linear_tid)
////                BlockScanRaking.ExclusiveSum.WithAggregateAndCallbackOp h.BlockScanRakingHostApi scan_op bsr_d
////                    input output block_aggregate block_prefix_callback_op
//
////    module ExclusiveScan =
////        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (temp_storage:TempStorage<'T>) (linear_tid:int)
////            (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) =
////            h.ScanKind |> function
////            | ScanKind.WarpScans ->
////                let ws_d = BlockScanWarpScans.DeviceApi<'T>.Init(h.BlockScanWarpScansHostApi, d.temp_storage.BlockScanWarpScans, d.linear_tid)
////                BlockScanWarpScans.ExclusiveScan.WithAggregate h.BlockScanWarpScansHostApi scan_op ws_d
////                    input output identity block_aggregate
////            | _ ->
////                let bsr_d = BlockScanRaking.DeviceApi<'T>.Init(h.BlockScanRakingHostApi, d.temp_storage.BlockScanRaking, d.linear_tid)
////                BlockScanRaking.ExclusiveScan.WithAggregate h.BlockScanRakingHostApi scan_op bsr_d
////                    input output identity block_aggregate
////
////        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (temp_storage:TempStorage<'T>) (linear_tid:int)
////            (input:'T) (output:Ref<'T>) (identity:Ref<'T>) =
////            let block_aggregate = __local__.Variable<'T>()
////            WithAggregate h scan_op temp_storage linear_tid input output identity block_aggregate
////
////
////        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (temp_storage:TempStorage<'T>) (linear_tid:int)
////            (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
////                            h.ScanKind |> function
////            | ScanKind.WarpScans ->
////                let ws_d = BlockScanWarpScans.DeviceApi<'T>.Init(h.BlockScanWarpScansHostApi, d.temp_storage.BlockScanWarpScans, d.linear_tid)
////                BlockScanWarpScans.ExclusiveScan.WithAggregateAndCallbackOp h.BlockScanWarpScansHostApi scan_op ws_d
////                    input output identity block_aggregate block_prefix_callback_op
////            | _ ->
////                let bsr_d = BlockScanRaking.DeviceApi<'T>.Init(h.BlockScanRakingHostApi, d.temp_storage.BlockScanRaking, d.linear_tid)
////                BlockScanRaking.ExclusiveScan.WithAggregateAndCallbackOp h.BlockScanRakingHostApi scan_op bsr_d
////                    input output identity block_aggregate block_prefix_callback_op
////
////        module Identityless =
////            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
////                (temp_storage:TempStorage<'T>) (linear_tid:int)
////                (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) =
////                h.ScanKind |> function
////                | ScanKind.WarpScans ->
////                    let ws_d = BlockScanWarpScans.DeviceApi<'T>.Init(h.BlockScanWarpScansHostApi, d.temp_storage.BlockScanWarpScans, d.linear_tid)
////                    BlockScanWarpScans.ExclusiveScan.Identityless.WithAggregate h.BlockScanWarpScansHostApi scan_op ws_d
////                        input output block_aggregate
////                | _ ->
////                    let bsr_d = BlockScanRaking.DeviceApi<'T>.Init(h.BlockScanRakingHostApi, d.temp_storage.BlockScanRaking, d.linear_tid)
////                    BlockScanRaking.ExclusiveScan.Identityless.WithAggregate h.BlockScanRakingHostApi scan_op bsr_d
////                        input output block_aggregate
////
////            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
////                (temp_storage:TempStorage<'T>) (linear_tid:int)
////                (input:'T) (output:Ref<'T>) =
////                let block_aggregate = __local__.Variable<'T>()
////                WithAggregate h scan_op temp_storage linear_tid input output block_aggregate
////
////
////            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T)
////                (temp_storage:TempStorage<'T>) (linear_tid:int)
////                (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
////                                h.ScanKind |> function
////                | ScanKind.WarpScans ->
////                    let ws_d = BlockScanWarpScans.DeviceApi<'T>.Init(h.BlockScanWarpScansHostApi, d.temp_storage.BlockScanWarpScans, d.linear_tid)
////                    BlockScanWarpScans.ExclusiveScan.Identityless.WithAggregateAndCallbackOp h.BlockScanWarpScansHostApi scan_op ws_d
////                        input output block_aggregate block_prefix_callback_op
////                | _ ->
////                    let bsr_d = BlockScanRaking.DeviceApi<'T>.Init(h.BlockScanRakingHostApi, d.temp_storage.BlockScanRaking, d.linear_tid)
////                    BlockScanRaking.ExclusiveScan.Identityless.WithAggregateAndCallbackOp h.BlockScanRakingHostApi scan_op bsr_d
////                        input output block_aggregate block_prefix_callback_op
//
//
//module ExclusiveSum =
//    open Template
//
//    module SingleDatumPerThread =        
//    
////        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (temp_storage:TempStorage<'T>) (linear_tid:int)
////            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) =
////            InternalBlockScan.ExclusiveSum.WithAggregate h scan_op temp_storage linear_tid input output block_aggregate
////            
////
////        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (temp_storage:TempStorage<'T>) (linear_tid:int)
////            (input:'T) (output:Ref<'T>) =
////            let block_aggregate = __local__.Variable()
////            WithAggregate h scan_op temp_storage linear_tid input output block_aggregate
////            
////
////        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (temp_storage:TempStorage<'T>) (linear_tid:int)
////                    (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
////                ()
//// 
// 
//        let [<ReflectedDefinition>] inline WithAggregateInt (h:HostApi)
//            (temp_storage:TempStorage<int>) (linear_tid:int)
//            (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) =
//            InternalBlockScan.ExclusiveSum.WithAggregateInt h temp_storage linear_tid input output block_aggregate
//            
//
//        let [<ReflectedDefinition>] inline DefaultInt (h:HostApi) (scan_op:'T -> 'T -> 'T)
//            (temp_storage:TempStorage<int>) (linear_tid:int)
//            (input:int) (output:Ref<int>) =
//            let block_aggregate = __local__.Variable<int>()
//            WithAggregateInt h temp_storage linear_tid input output block_aggregate
//            
//
//
//    module MultipleDataPerThread =
//        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
//            (temp_storage:TempStorage<'T>) (linear_tid:int)
//            (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) =
//            
//            let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
//                
//            SingleDatumPerThread.WithAggregate h scan_op temp_storage linear_tid !thread_partial thread_partial block_aggregate
//                
//            ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
//            
//
//        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
//            (temp_storage:TempStorage<'T>) (linear_tid:int)
//            (input:deviceptr<'T>) (output:deviceptr<'T>) =
//                
//            let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
//                
//            SingleDatumPerThread.Default h scan_op temp_storage linear_tid !thread_partial thread_partial
//
//            ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
//            
//
//
//
//        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
//            (temp_storage:TempStorage<'T>) (linear_tid:int)
//            (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
//            ()
//            
////
//////
////module ExclusiveScan =
////    open Template
////
////    module SingleDatumPerThread =
////        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (temp_storage:TempStorage<'T>) (linear_tid:int)
////            (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) =
////            InternalBlockScan.ExclusiveScan.WithAggregate h scan_op temp_storage linear_tid input output identity block_aggregate
////            
////     
////        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (temp_storage:TempStorage<'T>) (linear_tid:int)
////            (input:'T) (output:Ref<'T>) (identity:Ref<'T>) =
////            let block_aggregate = __local__.Variable<'T>()
////            WithAggregate h scan_op temp_storage linear_tid input output identity block_aggregate
////            
////
////        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (temp_storage:TempStorage<'T>) (linear_tid:int)
////            (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
////            ()
////            
////
////        module Identityless =
////            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
////                (temp_storage:TempStorage<'T>) (linear_tid:int)
////                (input:'T) (output:Ref<'T>) =
////                InternalBlockScan.ExclusiveScan.Identityless.Default h scan_op temp_storage linear_tid input output
////                
////
////            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
////                (temp_storage:TempStorage<'T>) (linear_tid:int)
////                (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) =
////                InternalBlockScan.ExclusiveScan.Identityless.WithAggregate h scan_op temp_storage linear_tid input output block_aggregate
////                
////                    
////
////            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T)
////                (temp_storage:TempStorage<'T>) (linear_tid:int)
////                (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
////                InternalBlockScan.ExclusiveScan.Identityless.WithAggregateAndCallbackOp h scan_op temp_storage linear_tid input output block_aggregate block_prefix_callback_op
////                
////
////    module MultipleDataPerThread =
////        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
////            (temp_storage:TempStorage<'T>) (linear_tid:int)
////            (input:deviceptr<'T>) (output:deviceptr<'T>) (identity:Ref<'T>) =
////                
////            let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
////
////            SingleDatumPerThread.Default h scan_op temp_storage linear_tid !thread_partial thread_partial identity
////
////            ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
////            
////
////        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
////            (temp_storage:TempStorage<'T>) (linear_tid:int)
////            (input:deviceptr<'T>) (output:deviceptr<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) =
////                
////            let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
////
////            SingleDatumPerThread.WithAggregate h scan_op temp_storage linear_tid !thread_partial thread_partial identity block_aggregate
////
////            ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
////            
////
////        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
////            (temp_storage:TempStorage<'T>) (linear_tid:int)
////            (input:deviceptr<'T>) (output:deviceptr<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
////            
////            let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
////
////            SingleDatumPerThread.WithAggregateAndCallbackOp h scan_op temp_storage linear_tid !thread_partial thread_partial identity block_aggregate block_prefix_callback_op
////
////            ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
////            
////            
////
////
////        module Identityless =
////            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
////                (temp_storage:TempStorage<'T>) (linear_tid:int)
////                (input:deviceptr<'T>) (output:deviceptr<'T>) =
////                let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
////
////                SingleDatumPerThread.Identityless.Default h scan_op temp_storage linear_tid !thread_partial thread_partial
////
////                ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
////            
////                
////
////            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int) 
////                (temp_storage:TempStorage<'T>) (linear_tid:int)
////                (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) =
////                let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
////
////                SingleDatumPerThread.Identityless.WithAggregate h scan_op temp_storage linear_tid !thread_partial thread_partial block_aggregate
////
////                ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
////            
////                
////
////            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
////                (temp_storage:TempStorage<'T>) (linear_tid:int)
////                (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
////                let thread_partial = __local__.Variable<'T>(ThreadReduce.Default items_per_thread scan_op input)
////
////                SingleDatumPerThread.Identityless.WithAggregateAndCallbackOp h scan_op temp_storage linear_tid !thread_partial thread_partial block_aggregate block_prefix_callback_op
////
////                ThreadScanExclusive.WithApplyPrefixDefault items_per_thread scan_op input output !thread_partial 
////            
//                
//
//
////module InclusiveSum =
////    open Template
////
////    module SingleDatumPerThread =
////        let [<ReflectedDefinition>] inline Default  (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (input:'T) (output:Ref<'T>) = ()
////
////        let [<ReflectedDefinition>] inline WithAggregate  (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = ()
////
////        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp  (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) = ()
////
////
////    module MultipleDataPerThread =
////        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
////            (scan_op:'T -> 'T -> 'T)
////            (input:deviceptr<'T>) (output:deviceptr<'T>) = ()
////
////        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
////            (scan_op:'T -> 'T -> 'T)
////            (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) = ()
////
////        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
////            (scan_op:'T -> 'T -> 'T)
////            (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) = ()
//
//
////module InclusiveScan =
////    open Template
////
////    module SingleDatumPerThread =
////        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (input:'T) (output:Ref<'T>)  = ()
////
////        let [<ReflectedDefinition>] inline WithAggregate  (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (input:'T) (output:Ref<'T>)  (block_aggregate:Ref<'T>) = ()
////
////        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (input:'T) (output:Ref<'T>)  (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) = ()
////
////
////    module MultipleDataPerThread =
////        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
////            (scan_op:'T -> 'T -> 'T)
////            (input:deviceptr<'T>) (output:deviceptr<'T>) = ()
////
////        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
////            (scan_op:'T -> 'T -> 'T)
////            (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) = ()
////
////        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
////            (scan_op:'T -> 'T -> 'T)
////            (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T>) = ()
//v

//        module ExclusiveSum =
//            module SDPT = ()
//            module MDPT =
//                let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int) (d:DeviceApi<'T>) (input:deviceptr<'T>) (output:deviceptr<'T>) =
//                    ExclusiveSum.MultipleDataPerThread.Default h scan_op items_per_thread temp_storage linear_tid input output
//
//                let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) (items_per_thread:int) (d:DeviceApi<'T>) (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T>) =
//                    ExclusiveSum.MultipleDataPerThread.WithAggregate h scan_op items_per_thread temp_storage linear_tid input output block_aggregate
//        
//        module ExclusiveScan =
//            module Identityless = ()



//    module ExclusiveSum =
//        module SingleDatumPerThread =
//            type FunctionApi<'T> = Template.ExclusiveSum.SingleDatumPerThread._FunctionApi<'T>
//            
//            let [<ReflectedDefinition>] inline template<'T> (block_threads:int) (algorithm:BlockScanAlgorithm) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
//                let h = HostApi.Init(block_threads, algorithm)
//                let! dfault = (h, scan_op) ||> ExclusiveSum.SingleDatumPerThread.Default          |> Compiler.DefineFunction
//                let! waggr  = (h, scan_op) ||> ExclusiveSum.SingleDatumPerThread.WithAggregate    |> Compiler.DefineFunction
//                    
//                return h, {
//                    Default         = dfault
//                    WithAggregate   = waggr
//                }}
//
//        module MultipleDataPerThread =
//            type FunctionApi<'T> = Template.ExclusiveSum.MultipleDataPerThread._FunctionApi<'T>
//
//            let [<ReflectedDefinition>] inline template<'T> (block_threads:int) (algorithm:BlockScanAlgorithm) (scan_op:'T -> 'T -> 'T) (items_per_thread:int) : Template<HostApi*FunctionApi<'T>> = cuda {
//                let h = HostApi.Init(block_threads, algorithm)
//                let! dfault = (h, scan_op, items_per_thread) |||> ExclusiveSum.MultipleDataPerThread.Default          |> Compiler.DefineFunction
//                let! waggr  = (h, scan_op, items_per_thread) |||> ExclusiveSum.MultipleDataPerThread.WithAggregate    |> Compiler.DefineFunction
//                    
//                return h, {
//                    Default         = dfault
//                    WithAggregate   = waggr
//                }}
//        
//        type FunctionApi<'T> =
//            {
//                SingleDatumPerThread    : SingleDatumPerThread.FunctionApi<'T>
//                MultipleDataPerThread   : MultipleDataPerThread.FunctionApi<'T>
//            }
//            
//        let [<ReflectedDefinition>] inline template<'T> (block_threads:int) (algorithm:BlockScanAlgorithm) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
//            : Template<HostApi*FunctionApi<'T>> = cuda {
//
//            let! h, sdpt = SingleDatumPerThread.template<'T> block_threads algorithm scan_op
//            let! _, mdpt = MultipleDataPerThread.template<'T> block_threads algorithm scan_op items_per_thread
//
//            return
//                h, {
//                    SingleDatumPerThread    = sdpt
//                    MultipleDataPerThread   = mdpt
//                }
//            }
//
//    module ExclusiveScan =
//        module SingleDatumPerThread =
//            type FunctionApi<'T> = Template.ExclusiveScan.SingleDatumPerThread._FunctionApi<'T>
//            
//            let [<ReflectedDefinition>] inline template<'T> (block_threads:int) (algorithm:BlockScanAlgorithm) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
//                let h = HostApi.Init(block_threads, algorithm)
//                let! dfault = (h, scan_op) ||> ExclusiveScan.SingleDatumPerThread.Default          |> Compiler.DefineFunction
//                let! waggr  = (h, scan_op) ||> ExclusiveScan.SingleDatumPerThread.WithAggregate    |> Compiler.DefineFunction
//                    
//                return h, {
//                    Default         = dfault
//                    WithAggregate   = waggr
//                }}
//
//        module MultipleDataPerThread =
//            type FunctionApi<'T> = Template.ExclusiveScan.MultipleDataPerThread._FunctionApi<'T>
//
//            let [<ReflectedDefinition>] inline template<'T> (block_threads:int) (algorithm:BlockScanAlgorithm) (scan_op:'T -> 'T -> 'T) (items_per_thread:int) : Template<HostApi*FunctionApi<'T>> = cuda {
//                let h = HostApi.Init(block_threads, algorithm)
//                let! dfault = (h, scan_op, items_per_thread) |||> ExclusiveScan.MultipleDataPerThread.Default          |> Compiler.DefineFunction
//                let! waggr  = (h, scan_op, items_per_thread) |||> ExclusiveScan.MultipleDataPerThread.WithAggregate    |> Compiler.DefineFunction
//                    
//                return h, {
//                    Default         = dfault
//                    WithAggregate   = waggr
//                }}
//        
//        type FunctionApi<'T> =
//            {
//                SingleDatumPerThread    : SingleDatumPerThread.FunctionApi<'T>
//                MultipleDataPerThread   : MultipleDataPerThread.FunctionApi<'T>
//            }
//            
//        let [<ReflectedDefinition>] inline template<'T> (block_threads:int) (algorithm:BlockScanAlgorithm) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
//            : Template<HostApi*FunctionApi<'T>> = cuda {
//
//            let! h, sdpt = SingleDatumPerThread.template<'T> block_threads algorithm scan_op
//            let! _, mdpt = MultipleDataPerThread.template<'T> block_threads algorithm scan_op items_per_thread
//
//            return
//                h, {
//                    SingleDatumPerThread    = sdpt
//                    MultipleDataPerThread   = mdpt
//                }
//            }
//                
//    type FunctionApi<'T> =
//        {
//            ExclusiveSum    : ExclusiveSum.FunctionApi<'T>
//            ExclusiveScan   : ExclusiveScan.FunctionApi<'T>
//        } 
//
//            
//    let [<ReflectedDefinition>] inline template<'T> (block_threads:int) (algorithm:BlockScanAlgorithm) (scan_op:'T -> 'T -> 'T) (items_per_thread:int)
//        : Template<HostApi*FunctionApi<'T>> = cuda {
//
//        let! h, exlsum  = ExclusiveSum.template<'T> block_threads algorithm scan_op items_per_thread
//        let! _, exlscan = ExclusiveScan.template<'T> block_threads algorithm scan_op items_per_thread
//
//        return
//            h, {
//                ExclusiveSum    = exlsum
//                ExclusiveScan   = exlscan
//            }
//        }
//     
//let [<ReflectedDefinition>] inline InternalBlockScan() =
//let [<ReflectedDefinition>] inline InternalBlockScan() =
//    fun block_threads algorithm ->
//        let SAFE_ALGORITHM = (block_threads, algorithm) ||> SAFE_ALGORITHM
//        match SAFE_ALGORITHM with
//        | BLOCK_SCAN_WARP_SCANS -> 
//            (block_threads |> BlockScanWarpScans.BlockScanWarpScans.Init |> Some, None)
//        | _ -> 
//            (None, (block_threads, (SAFE_ALGORITHM = BLOCK_SCAN_RAKING_MEMOIZE)) |> BlockScanRaking.BlockScanRaking.Init |> Some)

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
//            BlockScanWarpScans = block_threads |> BlockScanWarpScans.Init
//            BlockScanRaking = (block_threads, (c.SAFE_ALGORITHM = BLOCK_SCAN_RAKING_MEMOIZE)) |> BlockScanRaking.Init
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
//        let scan_op = scan_op
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
//        let scan_op = scan_op
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
//        let scan_op = scan_op
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
//        fun (items_per_thread:int) =
//            if (items_per_thread = 1) then
//                this.InclusiveSum(input[0], output[0])
//            else
//                // Reduce consecutive thread items in registers
//                let scan_op = scan_op
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
//        fun (items_per_thread:int) =
//            if (items_per_thread = 1) then
//                this.InclusiveSum(input[0], output[0], block_aggregate)
//            else
//                // Reduce consecutive thread items in registers
//                let scan_op = scan_op
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
//        fun (items_per_thread:int) =
//            if (items_per_thread = 1) then
//                this.InclusiveSum(input[0], output[0], block_aggregate, block_prefix_callback_op)
//            else
//                // Reduce consecutive thread items in registers
//                let scan_op = scan_op
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
//        fun (items_per_thread:int) =
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
//        fun (items_per_thread:int) =
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
//        fun (items_per_thread:int) =
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
//    static member Init(block_threads:int, algorithm:BlockScanAlgorithm)
//
//    static member Init(block_threads:int, algorithm:BlockScanAlgorithm, items_per_thread:int) =
//        let c = (block_threads, algorithm) |> Constants.Init
//        {
//            BLOCK_THREADS = block_threads
//            ALGORITHM = algorithm
//            Constants = c
//            InternalBlockScan = (block_threads, c) |> InternalBlockScan.Init
//            ThreadFields = ThreadFields.Init(__null(), threadIdx.x)
//            ThreadScan = items_per_thread |> ThreadScan.Init
//        }
//
//    static member Init(block_threads:int, items_per_thread:int) =
//        let c = (block_threads, BLOCK_SCAN_RAKING) |> Constants.Init
//        {
//            BLOCK_THREADS       = block_threads
//            ALGORITHM           = BLOCK_SCAN_RAKING
//            Constants           = c
//            InternalBlockScan   = (block_threads, c) |> InternalBlockScan.Init
//            ThreadFields        = ThreadFields.Init(__null(), threadIdx.x)
//            ThreadScan          = items_per_thread |> ThreadScan.Init
//        }
//
//    static member Init(block_threads:int) =
//        let c = (block_threads, BLOCK_SCAN_RAKING) |> Constants.Init
//        {
//            BLOCK_THREADS       = block_threads
//            ALGORITHM           = BLOCK_SCAN_RAKING
//            Constants           = c
//            InternalBlockScan   = (block_threads, c) |> InternalBlockScan.Init
//            ThreadFields        = ThreadFields.Init(__null(), threadIdx.x)
//            ThreadScan          = 1 |> ThreadScan.Init
//        }
//
//    static member Init(block_threads:int, algorithm:BlockScanAlgorithm) =
//        let c = (block_threads, algorithm) |> Constants.Init
//        {
//            BLOCK_THREADS = block_threads
//            ALGORITHM = algorithm
//            Constants = c
//            InternalBlockScan = (block_threads, c) |> InternalBlockScan.Init
//            ThreadFields = ThreadFields.Init(__null(), threadIdx.x)
//            ThreadScan = 1 |> ThreadScan.Init
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
////            fun (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) = ()
////
////        // exclusive prefix scan operations
////        let exclusiveScan =
////            fun (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) = ()
////
////
////        module Identityless =
////            // exclusive prefix scan operations (identityless, single datum per thread)
////            let exclusiveScan =
////                fun (input:'T) (output:Ref<'T>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) = ()   
////
////
////    module STMD =
////        // exclusive prefix sum operations (multiple data per thread)
////        let exclusiveSum items_per_thread =
////            fun (input:deviceptr<int>) (output:deviceptr<int>) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) = ()
////
////        // exclusive prefix scan operations (multiple data per thread)
////        let exclusiveScan items_per_thread =
////            fun (input:deviceptr<int>) (output:deviceptr<int>) (identity:Ref<'T>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) = ()
////        
////
////        module Identityless =
////            // exclusive prefix scan operations (identityless, multiple data per thread)
////            let exclusiveScan_noId items_per_thread =
////                fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) = ()
////
////
////module InclusiveScan =
////    
////    module STSD =
////        // inclusive prefix sum operations
////        let inclusiveSum =
////            fun (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) = ()
////
////        // inclusive prefix scan operations
////        let inclusiveScan =
////            fun (input:'T) (output:Ref<'T>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) = ()
////        
////
////    module STMD =
////        // inclusive prefix sum operations (multiple data per thread)
////        let inclusiveSum items_per_thread =
////            fun (input:deviceptr<int>) (output:deviceptr<int>) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) = ()
////        
////        // inclusive scan operations (multiple data per thread)
////        let inclusiveScan items_per_thread =
////            fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:(int -> int -> int)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'T -> 'T> option) = ()
////        
////        
////type API =
////    {
////        ExclusiveScan : <'T>
////        InclusiveScan : <'T>
////    }
////
////let [<ReflectedDefinition>] inline BlockScan () 
//
