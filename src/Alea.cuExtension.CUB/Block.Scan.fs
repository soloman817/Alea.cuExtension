[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Scan

//open Microsoft.FSharp.Quotations
//
//open Alea.CUDA
//open Alea.CUDA.Utilities
//open Alea.cuExtension.CUB.Common
//open Alea.cuExtension.CUB.Utilities
//
//open ScanOperators

//
//
//type ReductionOpKind =
//    | ReduceByKey
//    | SegmentedOp
//
////type ReduceByKeyOp<'K,'V> =
////    abstract op : Expr<KeyValuePair<'K,'V> -> KeyValuePair<'K,'V> -> KeyValuePair<'K,'V>>
//
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
//
//
//type BlockScanAlgorithm =
//    | BLOCK_SCAN_RAKING
//    | BLOCK_SCAN_RAKING_MEMOIZE
//    | BLOCK_SCAN_WARP_SCANS
//
//
//let SAFE_ALGORITHM = 
//    fun block_threads algorithm ->
//        if (algorithm = BLOCK_SCAN_WARP_SCANS) && ((block_threads % CUB_PTX_WARP_THREADS) <> 0) then
//            BLOCK_SCAN_RAKING
//        else
//            algorithm
//
//
//let InternalBlockScan =
//    fun block_threads algorithm ->
//        match (block_threads, algorithm) ||> SAFE_ALGORITHM with
//        | BLOCK_SCAN_WARP_SCANS -> BlockScanWarpScans()
//        | _ -> BlockScanRaking()
//
//
//
//// public (ctors)
////blockscan() temp_storage(privateStorage()) linear_tid(threadIdx.x)
////blockscan(temp_storage:Ref<'TempStorage>) temp_storage(temp_storage.alias()) linear_tid(linear_tid)
////blockscan(linear_tid) temp_storage(privateStorage()) linear_tid(linear_tid)
////blockscan(&temp_storage, linear_tid) temp_storage(temp_storage.alias()) linear_tid(linear_tid)
//
//
//
//[<Record>]
//type BlockScan<'T> =
//    {
//        x : int
//    }
//    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    // exclusive prefix sum operations
//    member this.ExclusiveSum(input:'T, output:Ref<'T>) = ()
//        let block_aggregate = __null() |> __ptr_to_ref
//        InternalBlockScan(temp_storage, linear_tid).ExclusiveSum(input, output, block_aggregate)
//
//    member this.ExclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>) = ()
//        InternalBlockScan(temp_storage, linear_tid).ExclusiveSum(input, output, block_aggregate)
//
//    member this.ExclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = ()
//        InternalBlockScan(temp_storage, linear_tid).ExclusiveSum(input, output, block_aggregate, block_prefix_callback_op)
//
//    // exclusive prefix sum operations (multiple data per thread)
//    //<items_per_thread>
//    member this.ExclusiveSum(input:deviceptr<'T>, output:deviceptr<'T>) = ()
//        let scan_op = (+)
//        let thread_partial = ThreadReduce(input, scan_op) |> __obj_to_ref
//
//        // Exclusive threadblock-scan
//        this.ExclusiveSum(thread_partial, thread_partial)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial)
//    
//    member this.ExclusiveSum(input:deviceptr<'T>, output:deviceptr<'T>, block_aggregate:Ref<'T>) = ()
//        let scan_op = (+)
//        let thread_partial = ThreadReduce(input, scan_op) |> __obj_to_ref
//
//        // Exclusive threadblock-scan
//        this.ExclusiveSum(thread_partial, thread_partial, block_aggregate)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial)
//    
//    member this.ExclusiveSum(input:deviceptr<'T>, output:deviceptr<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = ()
//        let scan_op = (+)
//        let thread_partial = ThreadReduce(input, scan_op) |> __obj_to_ref
//
//        // Exclusive threadblock-scan
//        this.ExclusiveSum(thread_partial, thread_partial, block_aggregate, block_prefix_callback_op)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial)
//    
//    member this.ExclusiveSum(input:deviceptr<'T>, output:deviceptr<'T>, block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = ()
//        let block_aggregate = __null() |> __ptr_to_ref
//        InternalBlockScan(temp_storage, linear_tid).ExclusiveScan(input, output, identity, scan_op, block_aggregate)
//    
//    // exclusive prefix scan operations
//    member this.ExclusiveScan(input:'T, output:Ref<'T>, identity:'T, scan_op:('T -> 'T -> 'T)) = ()
//        InternalBlockScan(temp_storage, linear_tid).ExclusiveScan(input, output, identity, scan_op, block_aggregate)
//
//    member this.ExclusiveScan(input:'T, output:Ref<'T>, identity:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) = ()
//        InternalBlockScan(temp_storage, linear_tid).ExclusiveScan(input, output, identity, scan_op, block_aggregate, block_prefix_callback_op)
//    
//    member this.ExclusiveScan(input:'T, output:Ref<'T>, identity:'T, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = ()
//        let thread_partial = ThreadReduce(input, scan_op)
//
//        // Exclusive threadblock-scan
//        ExclusiveScan(thread_partial, thread_partial, identity, scan_op)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial)
//    
//    // exclusive prefix scan operations (identityless, single datum per thread)
//    member this.ExclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T)) = ()
//        let thread_partial = ThreadReduce(input, scan_op)
//
//        // Exclusive threadblock-scan
//        ExclusiveScan(thread_partial, thread_partial, identity, scan_op, block_aggregate)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial)
//    
//    member this.ExclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) = ()
//        let thread_partial = ThreadReduce(input, scan_op)
//
//        // Exclusive threadblock-scan
//        this.ExclusiveScan(thread_partial, thread_partial, identity, scan_op, block_aggregate, block_prefix_callback_op)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial)
//
//    member this.ExclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = ()
//        let block_aggregate = __null() |> __ptr_to_ref
//        InternalBlockScan(temp_storage, linear_tid).ExclusiveScan(input, output, scan_op, block_aggregate)
//
//
//    // exclusive prefix scan operations (multiple data per thread)
//    //<items_per_thread>
//    member this.ExclusiveScan(input:deviceptr<'T>, output:deviceptr<'T>, identity:Ref<'T>, scan_op:('T -> 'T -> 'T)) = ()
//        InternalBlockScan(temp_storage, linear_tid).ExclusiveScan(input, output, scan_op, block_aggregate)
//
//    member this.ExclusiveScan(input:deviceptr<'T>, output:deviceptr<'T>, identity:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) = ()
//        InternalBlockScan(temp_storage, linear_tid).ExclusiveScan(input, output, scan_op, block_aggregate, block_prefix_callback_op)
//
//    member this.ExclusiveScan(input:deviceptr<'T>, output:deviceptr<'T>, identity:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = ()
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
//    member this.ExclusiveScan(input:deviceptr<'T>, output:deviceptr<'T>, scan_op:('T -> 'T -> 'T)) = ()
//        // Reduce consecutive thread items in registers
//        let thread_partial = ThreadReduce(input, scan_op)
//
//        // Exclusive threadblock-scan
//        this.ExclusiveScan(thread_partial, thread_partial, scan_op, block_aggregate)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial, (linear_tid != 0))
//
//    member this.ExclusiveScan(input:deviceptr<'T>, output:deviceptr<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) = ()
//        // Reduce consecutive thread items in registers
//        let thread_partial = ThreadReduce(input, scan_op)
//
//        // Exclusive threadblock-scan
//        this.ExclusiveScan(thread_partial, thread_partial, scan_op, block_aggregate, block_prefix_callback_op)
//
//        // Exclusive scan in registers with prefix
//        ThreadScanExclusive(input, output, scan_op, thread_partial)
//
//    member this.ExclusiveScan(input:deviceptr<'T>, output:deviceptr<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = ()
//        let block_aggregate = __null() |> __ptr_to_ref
//        InternalBlockScan(temp_storage, linear_tid).InclusiveSum(input, output, block_aggregate)
//
//
//    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    // inclusive prefix sum operations
//    member this.InclusiveSum(input:'T, output:Ref<'T>) = ()
//        InternalBlockScan(temp_storage, linear_tid).InclusiveSum(input, output, block_aggregate)
//
//    member this.InclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>) = ()
//        InternalBlockScan(temp_storage, linear_tid).InclusiveSum(input, output, block_aggregate, block_prefix_callback_op)
//
//    member this.InclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = ()
//        if (ITEMS_PER_THREAD = 1) then
//            InclusiveSum(input[0], output[0])
//        else
//            // Reduce consecutive thread items in registers
//            let scan_op = (+)
//            let thread_partial = ThreadReduce(input, scan_op)
//
//            // Exclusive threadblock-scan
//            this.ExclusiveSum(thread_partial, thread_partial)
//
//            // Inclusive scan in registers with prefix
//            ThreadScanInclusive(input, output, scan_op, thread_partial, (linear_tid != 0))
//
//    // inclusive prefix sum operations (multiple data per thread)
//    //<items_per_thread>
//    member this.InclusiveSum(input:deviceptr<'T>, output:deviceptr<'T>) = ()
//        if (ITEMS_PER_THREAD = 1) then
//            this.InclusiveSum(input[0], output[0], block_aggregate)
//        else
//            // Reduce consecutive thread items in registers
//            let scan_op = (+)
//            let thread_partial = ThreadReduce(input, scan_op)
//
//            // Exclusive threadblock-scan
//            this.ExclusiveSum(thread_partial, thread_partial, block_aggregate)
//
//            // Inclusive scan in registers with prefix
//            ThreadScanInclusive(input, output, scan_op, thread_partial, (linear_tid != 0))
//
//        
//    member this.InclusiveSum(input:deviceptr<'T>, output:deviceptr<'T>, block_aggregate:Ref<'T>) = ()
//        if (ITEMS_PER_THREAD = 1) then
//            this.InclusiveSum(input[0], output[0], block_aggregate, block_prefix_callback_op)
//        else
//            // Reduce consecutive thread items in registers
//            let scan_op = (+)
//            let thread_partial = ThreadReduce(input, scan_op)
//
//            // Exclusive threadblock-scan
//            this.ExclusiveSum(thread_partial, thread_partial, block_aggregate, block_prefix_callback_op)
//
//            // Inclusive scan in registers with prefix
//            ThreadScanInclusive(input, output, scan_op, thread_partial)
//
//    
//    member this.InclusiveSum(input:deviceptr<'T>, output:deviceptr<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = ()
//        let block_aggregate = __nul() |> __ptr_to_ref
//        this.InclusiveScan(input, output, scan_op, block_aggregate)
//    
//   
//    // inclusive prefix scan operations
//    member this.InclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T)) = ()
//        let block_aggregate = __null() |> __ptr_to_ref
//        this.InclusiveScan(input, output, scan_op, block_aggregate)
//    
//    member this.InclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) = ()
//        InternalBlockScan(temp_storage, linear_tid).InclusiveScan(input, output, scan_op, block_aggregate)
//
//
//    member this.InclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = ()
//        InternalBlockScan(temp_storage, linear_tid).InclusiveScan(input, output, scan_op, block_aggregate, block_prefix_callback_op)
//    
//    // inclusive scan operations (multiple data per thread)
//    //<items_per_thread>
//    member this.InclusiveScan(input:deviceptr<'T>, output:deviceptr<'T>, scan_op:('T -> 'T -> 'T)) = ()
//       if (ITEMS_PER_THREAD = 1) then
//            this.InclusiveScan(input[0], output[0], scan_op)
//        else
//            // Reduce consecutive thread items in registers
//            let thread_partial = ThreadReduce(input, scan_op)
//
//            // Exclusive threadblock-scan
//            this.ExclusiveScan(thread_partial, thread_partial, scan_op)
//
//            // Inclusive scan in registers with prefix
//            ThreadScanInclusive(input, output, scan_op, thread_partial, (linear_tid != 0))
//
//        
//    member this.InclusiveScan(input:deviceptr<'T>, output:deviceptr<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) = ()
//        if (ITEMS_PER_THREAD = 1) then
//            this.InclusiveScan(input[0], output[0], scan_op, block_aggregate)
//        else
//            // Reduce consecutive thread items in registers
//            let thread_partial = ThreadReduce(input, scan_op)
//
//            // Exclusive threadblock-scan
//            this.ExclusiveScan(thread_partial, thread_partial, scan_op, block_aggregate)
//
//            // Inclusive scan in registers with prefix
//            ThreadScanInclusive(input, output, scan_op, thread_partial, (linear_tid != 0))
//        
//
//    member this.InclusiveScan(input:deviceptr<'T>, output:deviceptr<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = ()
//        if (ITEMS_PER_THREAD = 1) then
//            this.InclusiveScan(input[0], output[0], scan_op, block_aggregate, block_prefix_callback_op)
//        else
//            // Reduce consecutive thread items in registers
//            let thread_partial = ThreadReduce(input, scan_op)
//
//            // Exclusive threadblock-scan
//            this.ExclusiveScan(thread_partial, thread_partial, scan_op, block_aggregate, block_prefix_callback_op)
//
//            // Inclusive scan in registers with prefix
//            ThreadScanInclusive(input, output, scan_op, thread_partial)
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
////            fun (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'BlockPrefixCallbackOp> option) -> ()
////
////        // exclusive prefix scan operations
////        let exclusiveScan =
////            fun (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (scan_op:('T -> 'T -> 'T)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'BlockPrefixCallbackOp> option) -> ()
////
////
////        module Identityless =
////            // exclusive prefix scan operations (identityless, single datum per thread)
////            let exclusiveScan =
////                fun (input:'T) (output:Ref<'T>) (scan_op:('T -> 'T -> 'T)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'BlockPrefixCallbackOp> option) -> ()   
////
////
////    module STMD =
////        // exclusive prefix sum operations (multiple data per thread)
////        let exclusiveSum items_per_thread =
////            fun (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'BlockPrefixCallbackOp> option) -> ()
////
////        // exclusive prefix scan operations (multiple data per thread)
////        let exclusiveScan items_per_thread =
////            fun (input:deviceptr<'T>) (output:deviceptr<'T>) (identity:Ref<'T>) (scan_op:('T -> 'T -> 'T)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'BlockPrefixCallbackOp> option) -> ()
////        
////
////        module Identityless =
////            // exclusive prefix scan operations (identityless, multiple data per thread)
////            let exclusiveScan_noId items_per_thread =
////                fun (input:deviceptr<'T>) (output:deviceptr<'T>) (scan_op:('T -> 'T -> 'T)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'BlockPrefixCallbackOp> option) -> ()
////
////
////module InclusiveScan =
////    
////    module STSD =
////        // inclusive prefix sum operations
////        let inclusiveSum =
////            fun (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'BlockPrefixCallbackOp> option) -> ()
////
////        // inclusive prefix scan operations
////        let inclusiveScan =
////            fun (input:'T) (output:Ref<'T>) (scan_op:('T -> 'T -> 'T)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'BlockPrefixCallbackOp> option) -> ()
////        
////
////    module STMD =
////        // inclusive prefix sum operations (multiple data per thread)
////        let inclusiveSum items_per_thread =
////            fun (input:deviceptr<'T>) (output:deviceptr<'T>) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'BlockPrefixCallbackOp> option) -> ()
////        
////        // inclusive scan operations (multiple data per thread)
////        let inclusiveScan items_per_thread =
////            fun (input:deviceptr<'T>) (output:deviceptr<'T>) (scan_op:('T -> 'T -> 'T)) (block_aggregate:Ref<'T> option) (block_prefix_callback_op:Ref<'BlockPrefixCallbackOp> option) -> ()
////        
////        
////type API<'T> =
////    {
////        ExclusiveScan : Expr
////        InclusiveScan : Expr
////    }
////
////let inline BlockScan () 
//
