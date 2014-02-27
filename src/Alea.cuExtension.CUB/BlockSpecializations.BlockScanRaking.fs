[<AutoOpen>]
module Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanRaking

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Warp
open Alea.cuExtension.CUB.Block



module TempStorage =
    type API =
        {
            warp_scan               :   Alea.cuExtension.CUB.Warp.Scan.TempStorage.API
            raking_grid             :   deviceptr<int>
            mutable block_aggregate :   int
        }

    let unitialized grid_elements =
        {
            warp_scan       = Alea.cuExtension.CUB.Warp.Scan.TempStorage.uninitialized()
            raking_grid     = (grid_elements |> Alea.cuExtension.CUB.Block.RakingLayout.TempStorage.initialize)
            block_aggregate = 0
        }


module private Internal =
    type TempStorage = TempStorage.API

    let BlockRakingLayout =
        fun block_threads ->
            (block_threads, None) ||> BlockRakingLayout.api

    module Constants =
        let WARPS =
            fun block_threads ->
                (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS

        let RAKING_THREADS = 
            fun block_threads ->
                (block_threads |> BlockRakingLayout).Constants.RAKING_THREADS
            
        let SEGMENT_LENGTH =
            fun block_threads ->
                (block_threads |> BlockRakingLayout).Constants.SEGMENT_LENGTH

        let WARP_SYNCHRONOUS = 
            fun block_threads ->
                block_threads = (block_threads |> RAKING_THREADS)

    module Sig =
        module Upsweep =
            type DefaultExpr = Expr<unit -> int>

        module GuardedReduce =
            type DefaultExpr = Expr<deviceptr<int> -> int -> int>

        module ExclusiveDownsweep =
            type DefaultExpr        = Expr<int -> unit>
            type WithApplyPrefix    = Expr<int -> bool -> unit>

        module ExclusiveScan =
            type WithAggregateExpr              = Expr<int -> Ref<int> -> Ref<int> -> Ref<int> -> unit>
            type WithAggregateAndCallbackOpExpr = Expr<int -> Ref<int> -> Ref<int> -> Ref<int> -> Ref<int -> int> -> unit>

            module Identityless =
                type WithAggregateExpr              = Expr<int -> Ref<int> -> Ref<int> -> unit>
                type WithAggregateAndCallbackOpExpr = Expr<int -> Ref<int> -> Ref<int> -> Ref<int -> int> -> unit>

        module ExclusiveSum =
            type WithAggregateExpr              = Expr<int -> Ref<int> -> Ref<int> -> unit>
            type WithAggregateAndCallbackOpExpr = Expr<int -> Ref<int> -> Ref<int> -> Ref<int -> int> -> unit>

        module InclusiveScan =
            type WithAggregateExpr              = Expr<int -> Ref<int> -> Ref<int> -> unit>
            type WithAggregateAndCallbackOpExpr = Expr<int -> Ref<int> -> Ref<int> -> Ref<int> -> unit>

        module InclusiveSum =
            type WithAggregateExpr              = Expr<int -> Ref<int> -> Ref<int> -> unit>
            type WithAggregateAndCallbackOpExpr = Expr<int -> Ref<int> -> Ref<int> -> Ref<int> -> unit>



    let WarpScan block_threads scan_op =
        fun temp_storage warp_id lane_id ->
            let RAKING_THREADS = block_threads |> Constants.RAKING_THREADS
            WarpScan.api
            <|||    (1, RAKING_THREADS, scan_op)
            <|||    (temp_storage, warp_id, lane_id)

module GuardedReduce =
    open Internal
    // Need to do Attribute unrolling stuff
    type API =
        {
            Default : Sig.GuardedReduce.DefaultExpr
        }

    let private Default block_threads (scan_op:IScanOp) =
        let SEGMENT_LENGTH = block_threads |> Constants.SEGMENT_LENGTH
        let UNGUARDED      = (block_threads |> BlockRakingLayout).Constants.UNGUARDED
        let scan_op = scan_op.op
        fun _ linear_tid ->
            <@ fun (raking_ptr:deviceptr<int>) (raking_partial:int) ->
                let mutable raking_partial = raking_partial
                for i = 0 to (SEGMENT_LENGTH - 1) do
                    if (UNGUARDED <> 0) || (((linear_tid * SEGMENT_LENGTH) + i) < block_threads) then
                        let addend = raking_ptr.[i]
                        raking_partial <- (raking_partial, addend) ||> %scan_op
                raking_partial
            @>

    let api block_threads scan_op =
        fun temp_storage linear_tid ->
            {
                Default =   Default
                            <|| (block_threads, scan_op)
                            <|| (temp_storage, linear_tid)
            }


module Upsweep = 
    open Internal

    type API =
        {
            Default : Sig.Upsweep.DefaultExpr
        }

    let private Default block_threads memoize scan_op =
        let SEGMENT_LENGTH = block_threads |> Constants.SEGMENT_LENGTH
        fun (temp_storage:TempStorage) linear_tid (cached_segment:deviceptr<int>) ->
            let RakingPtr = (block_threads |> BlockRakingLayout).RakingPtr.Default
            let GuardedReduce = (GuardedReduce.api
                                 <||    (block_threads, scan_op)
                                 <||    (temp_storage, linear_tid)).Default
            <@ fun _ ->
                let smem_raking_ptr =   %RakingPtr
                                        <|| (temp_storage.raking_grid, linear_tid)
                
                let mutable raking_ptr = __local__.Array(SEGMENT_LENGTH) |> __array_to_ptr

                if memoize then 
                    for i = 0 to (SEGMENT_LENGTH - 1) do cached_segment.[i] <- smem_raking_ptr.[i]
                    raking_ptr <- cached_segment
                else
                    raking_ptr <- smem_raking_ptr

                let raking_partial = raking_ptr.[0]

                %GuardedReduce
                <|| (raking_ptr, raking_partial)
            @>

    let api block_threads memoize scan_op =
        fun temp_storage linear_tid cached_segment ->
            {
                Default =   Default
                            <|||    (block_threads, memoize, scan_op)
                            <|||    (temp_storage, linear_tid, cached_segment)
            }

module ExclusiveDownsweep =
    open Internal

    type API =
        {
            Default         : Sig.ExclusiveDownsweep.DefaultExpr
            WithApplyPrefix : Sig.ExclusiveDownsweep.WithApplyPrefix
        }

    let private WithApplyPrefix block_threads memoize scan_op =
        let SEGMENT_LENGTH      = block_threads |> Constants.SEGMENT_LENGTH
        let ThreadScanExclusive = ( ThreadScanExclusive.api
                                    <|| (SEGMENT_LENGTH, scan_op)).WithApplyPrefix 
        
        fun (temp_storage:TempStorage) linear_tid cached_segment ->
            let RakingPtr = (block_threads |> BlockRakingLayout).RakingPtr.Default
            <@ fun (raking_partial:int) (apply_prefix:bool) ->
                let smem_raking_ptr =   %RakingPtr
                                        <|| (temp_storage.raking_grid, linear_tid)
                
                let raking_ptr = if memoize then cached_segment else smem_raking_ptr
                
                %ThreadScanExclusive
                <|| (raking_ptr, raking_ptr)
                <|  (raking_partial)
                <|  (apply_prefix)
                |> ignore

                if memoize then
                    for i = 0 to (SEGMENT_LENGTH - 1) do smem_raking_ptr.[i] <- cached_segment.[i]
            
            @>

    let private Default block_threads memoize scan_op =
        let SEGMENT_LENGTH      = block_threads |> Constants.SEGMENT_LENGTH
        let ThreadScanExclusive = ( ThreadScanExclusive.api
                                    <|| (SEGMENT_LENGTH, scan_op)).WithApplyPrefix 
        fun temp_storage linear_tid cached_segment ->
            let WithApplyPrefix =   WithApplyPrefix
                                    <|||    (block_threads, memoize, scan_op)
                                    <|||    (temp_storage, linear_tid, cached_segment)
            
            <@ fun (raking_partial:int) ->
                (raking_partial, true) ||> %WithApplyPrefix
            @>

    let api block_threads memoize scan_op =
        fun temp_storage linear_tid cached_segment ->
            {
                Default         =   Default
                                    <|||    (block_threads, memoize, scan_op)
                                    <|||    (temp_storage, linear_tid, cached_segment)

                WithApplyPrefix =   WithApplyPrefix
                                    <|||    (block_threads, memoize, scan_op)
                                    <|||    (temp_storage, linear_tid, cached_segment)
            }

module InclusiveDownsweep = ()

module ExclusiveScan =
    open Internal

    type API =
        {
            WithAggregate               : Sig.ExclusiveScan.WithAggregateExpr
            WithAggregateAndCallbackOp  : Sig.ExclusiveScan.WithAggregateAndCallbackOpExpr
        }

    let private WithAggregate block_threads memoize scan_op =
        let RAKING_THREADS      = block_threads |> Constants.RAKING_THREADS 
        let WARP_SYNCHRONOUS    = block_threads |> Constants.WARP_SYNCHRONOUS
        let BlockRakingLayout   = block_threads |> BlockRakingLayout

        fun (temp_storage:TempStorage) linear_tid cached_segment ->
            let WarpScan = 
                (   Internal.WarpScan
                    <||  (block_threads, scan_op)
                    <||| (temp_storage.warp_scan, 0, linear_tid)
                ).ExclusiveScan.WithAggregate
            
            let PlacementPtr   = BlockRakingLayout.PlacementPtr.Default
            
            let Upsweep =   
                (   Upsweep.api
                    <|||    (block_threads, memoize, scan_op)
                    <|||    (temp_storage, linear_tid, cached_segment)
                ).Default
            
            let ExclusiveDownsweep =
                (   ExclusiveDownsweep.api
                    <|||    (block_threads, memoize, scan_op)
                    <|||    (temp_storage, linear_tid, cached_segment)
                ).Default

            <@ fun (input:int) (output:Ref<int>) (identity:Ref<int>) (block_aggregate:Ref<int>) ->
                if WARP_SYNCHRONOUS then
                    %WarpScan
                    <|| (input, output)
                    <|  !identity
                    <|  block_aggregate
                else
                    let placement_ptr = %PlacementPtr
                                        <|| (temp_storage.raking_grid, linear_tid)
                    placement_ptr.[0] <- input

                    __syncthreads()

                    if linear_tid < RAKING_THREADS then
                        let raking_partial = (%Upsweep)()
                        %WarpScan
                        <|| (raking_partial, ref raking_partial)
                        <|  !identity
                        <|  ref temp_storage.block_aggregate

                        raking_partial |> %ExclusiveDownsweep

                    __syncthreads()

                    output := placement_ptr.[0]

                    block_aggregate := temp_storage.block_aggregate
            @>

    let private WithAggregateAndCallbackOp block_threads memoize scan_op =
        fun temp_storage linear_tid cached_segment ->
            <@ fun (input:int) (output:Ref<int>) (identity:Ref<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int -> int>) ->
                ()
            @>

    let api block_threads memoize scan_op =
        fun temp_storage linear_tid cached_segment ->
            {
                WithAggregate               =   WithAggregate
                                                <|||    (block_threads, memoize, scan_op)
                                                <|||    (temp_storage, linear_tid, cached_segment)

                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp
                                                <|||    (block_threads, memoize, scan_op)
                                                <|||    (temp_storage, linear_tid, cached_segment)
            }

module ExclusiveSum =
    open Internal

    type API =
        {
            WithAggregate               : Sig.ExclusiveSum.WithAggregateExpr
            WithAggregateAndCallbackOp  : Sig.ExclusiveSum.WithAggregateAndCallbackOpExpr
        }

    let private WithAggregate block_threads memoize scan_op =
        let WARP_SYNCHRONOUS    = block_threads |> Constants.WARP_SYNCHRONOUS
        let RAKING_THREADS      = block_threads |> Constants.RAKING_THREADS
        
        fun (temp_storage:TempStorage) linear_tid cached_segment ->
            let WarpScan =
                (   Internal.WarpScan
                    <||     (block_threads, scan_op)
                    <|||    (temp_storage.warp_scan, 0, linear_tid)
                ).ExclusiveSum.WithAggregate

            let PlacementPtr =
                (   Internal.BlockRakingLayout
                    <|  (block_threads)
                ).PlacementPtr.Default

            let Upsweep =
                (   Upsweep.api
                    <|||    (block_threads, memoize, scan_op)
                    <|||    (temp_storage, linear_tid, cached_segment)
                ).Default

            let ExclusiveDownsweep =
                (   ExclusiveDownsweep.api
                    <|||    (block_threads, memoize, scan_op)
                    <|||    (temp_storage, linear_tid, cached_segment)
                ).Default

            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) ->
                if WARP_SYNCHRONOUS then
                    %WarpScan
                    <|| (input, output)
                    <|  (block_aggregate)
                else
                    let placement_ptr = %PlacementPtr
                                        <|| (temp_storage.raking_grid, linear_tid)
                    placement_ptr.[0] <- input

                    __syncthreads()

                    if linear_tid < RAKING_THREADS then
                        let raking_partial = (%Upsweep)()
                        %WarpScan
                        <|| (raking_partial, ref raking_partial)
                        <|  ref temp_storage.block_aggregate

                        raking_partial |> %ExclusiveDownsweep

                    __syncthreads()

                    output := placement_ptr.[0]

                    block_aggregate := temp_storage.block_aggregate
            @>

    let private WithAggregateAndCallbackOp block_threads memoize scan_op =
        fun temp_storage linear_tid cached_segment ->
            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int -> int>) -> () @>

    let api block_threads memoize scan_op =
        fun temp_storage linear_tid cached_segment ->
            {
                WithAggregate               =   WithAggregate
                                                <|||    (block_threads, memoize, scan_op)
                                                <|||    (temp_storage, linear_tid, cached_segment)


                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp
                                                <|||    (block_threads, memoize, scan_op)
                                                <|||    (temp_storage, linear_tid, cached_segment)
            }


module BlockScanRaking =
    open Internal

    type API =
        {
            ExclusiveSum    : ExclusiveSum.API
            ExclusiveScan   : ExclusiveScan.API
        }

    let api block_threads memoize scan_op =
        fun temp_storage linear_tid cached_segment ->
            {
                ExclusiveSum    =   ExclusiveSum.api
                                    <|||    (block_threads, memoize, scan_op)
                                    <|||    (temp_storage, linear_tid, cached_segment)

                ExclusiveScan   =   ExclusiveScan.api
                                    <|||    (block_threads, memoize, scan_op)
                                    <|||    (temp_storage, linear_tid, cached_segment)

            }

//    member this.GuardedReduce(raking_ptr:deviceptr<int>, scan_op:IScanOp, raking_partial:int) = //, iteration:bool) =    
//        // localize template params & constants
//        let UNGUARDED       = this.BlockRakingLayout.Constants.UNGUARDED
//        let SEGMENT_LENGTH  = this.BlockRakingLayout.Constants.SEGMENT_LENGTH
//        let BLOCK_THREADS = this.TemplateParameters.BLOCK_THREADS
//        // localize thread fields
//        let linear_tid = this.ThreadFields.linear_tid
//
//        let mutable raking_partial = raking_partial
//        for ITERATION = 0 to (SEGMENT_LENGTH - 1) do
//            if ((UNGUARDED) || (((linear_tid * SEGMENT_LENGTH) + ITERATION) < BLOCK_THREADS)) then
//                let addend = raking_ptr.[ITERATION]
//                raking_partial <- (raking_partial, addend) ||> scan_op
//        
//        raking_partial
//    
//    
//    /// Performs upsweep raking reduction, returning the aggregate
//    //template <typename ScanOp>
//    member inline this.Upsweep(scan_op:IScanOp) =
//        // localize template params & constants
//        let SEGMENT_LENGTH = this.Constants.SEGMENT_LENGTH
//        let MEMOIZE = this.TemplateParameters.MEMOIZE
//        // localize thread fields
//        let temp_storage = !(this.ThreadFields.temp_storage)
//        let linear_tid = this.ThreadFields.linear_tid
//        let cached_segment = this.ThreadFields.cached_segment
//
//        let smem_raking_ptr = this.BlockRakingLayout.RakingPtr <|| (temp_storage.raking_grid, linear_tid) //rakingPtr()
//        let mutable raking_ptr = __null()
//
//        if MEMOIZE then
//            // Copy data into registers
//            //#pragma unroll
//            for i = 0 to (SEGMENT_LENGTH - 1) do cached_segment.[i] <- smem_raking_ptr.[i]
//            
//            raking_ptr <- cached_segment |> __array_to_ptr
//        
//        else
//            raking_ptr <- smem_raking_ptr
//        
//
//        let raking_partial = raking_ptr.[0]
//
//        this.GuardedReduce(raking_ptr, scan_op, raking_partial) //, Int2Type<1>())
//    
//
//
//    /// Performs exclusive downsweep raking scan
//    //template <typename ScanOp>
//    member inline this.ExclusiveDownsweep(scan_op:IScanOp, raking_partial:int, ?apply_prefix:bool) =
//        let apply_prefix = if apply_prefix.IsSome then apply_prefix.Value else true
//
//        // localize template params & constants
//        let MEMOIZE = this.TemplateParameters.MEMOIZE
//        let SEGMENT_LENGTH = this.Constants.SEGMENT_LENGTH
//        // localize thread fields
//        let temp_storage = !(this.ThreadFields.temp_storage)
//        let linear_tid = this.ThreadFields.linear_tid
//        let cached_segment = this.ThreadFields.cached_segment
//
//        let smem_raking_ptr = this.BlockRakingLayout.RakingPtr <|| (temp_storage.raking_grid, linear_tid)
//
//        let raking_ptr = if (MEMOIZE) then cached_segment |> __array_to_ptr else smem_raking_ptr
//
//        //SEGMENT_LENGTH |> ThreadScanExclusive<int> <| raking_ptr <| raking_ptr <| scan_op <| raking_partial <| apply_prefix
//        this.ThreadScan.Initialize(SEGMENT_LENGTH).Exclusive(raking_ptr, raking_ptr, scan_op, raking_partial, apply_prefix)
//
//        if (MEMOIZE) then
//            // Copy data back to smem
//            for i = 0 to (SEGMENT_LENGTH - 1) do smem_raking_ptr.[i] <- cached_segment.[i]
//
//
//    /// Performs inclusive downsweep raking scan
//    //template <typename ScanOp>
//    member this.InclusiveDownsweep(scan_op:IScanOp, raking_partial:int, ?apply_prefix:bool) =
//        // localize template params & constants
//        let MEMOIZE = this.TemplateParameters.MEMOIZE
//        let SEGMENT_LENGTH = this.Constants.SEGMENT_LENGTH
//        // localize thread fields
//        let temp_storage = !(this.ThreadFields.temp_storage)
//        let linear_tid = this.ThreadFields.linear_tid
//        let cached_segment = this.ThreadFields.cached_segment
//                        
//        let smem_raking_ptr = this.BlockRakingLayout.RakingPtr <|| (temp_storage.raking_grid, linear_tid) //(temp_storage.raking_grid.temp_storage, linear_tid)
//
//        let raking_ptr = if (MEMOIZE) then cached_segment |> __array_to_ptr else smem_raking_ptr
//
//        this.ThreadScan.Initialize(SEGMENT_LENGTH).Inclusive(raking_ptr, raking_ptr, scan_op, raking_partial, apply_prefix.Value)
//
//        if (MEMOIZE) then
//            // Copy data back to smem
//            for i = 0 to (SEGMENT_LENGTH - 1) do smem_raking_ptr.[i] <- cached_segment.[i]
//type TempStorage =
//    {
//        warp_scan : deviceptr<int>
//        raking_grid : deviceptr<int>
//        block_aggregate : int
//    }
//
//module ExclusiveScan =
//    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
//    //template <typename ScanOp>
//    
//    let private Default raking_threads warp_synchronous =
//        let RAKING_THREADS = raking_threads
//        let WARP_SYNCHRONOUS = warp_synchronous
//        let WarpScan = Alea.cuExtension.CUB.Warp.Scan.ExclusiveScan
//        fun (temp_storage:TempStorage) (linear_tid:int) ->
//            <@ fun (input:int) (output:Ref<int>) (identity:Ref<int>) (scan_op:IScanOp) (block_aggregate:Ref<int>) ->
//                
//
//                if (WARP_SYNCHRONOUS) then
//                    // Short-circuit directly to warp scan
//                    this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                        input,
//                        output,
//                        !identity,
//                        scan_op,
//                        block_aggregate)
//                else
//                    // Place thread partial into shared memory raking grid
//                    let placement_ptr = this.BlockRakingLayout.PlacementPtr <||| (temp_storage.raking_grid, linear_tid, None)
//                    placement_ptr.[0] <- input
//
//                    __syncthreads()
//
//                    // Reduce parallelism down to just raking threads
//                    if (linear_tid < RAKING_THREADS) then
//                        // Raking upsweep reduction in grid
//                        let raking_partial = this.Upsweep(scan_op)
//
//                        // Exclusive warp synchronous scan
//                        this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                            raking_partial,
//                            raking_partial |> __obj_to_ref,
//                            !identity,
//                            scan_op,
//                            temp_storage.block_aggregate |> __obj_to_ref)
//
//                        // Exclusive raking downsweep scan
//                        this.ExclusiveDownsweep(scan_op, raking_partial)
//            
//                    __syncthreads()
//
//                    // Grab thread prefix from shared memory
//                    output := placement_ptr.[0]
//
//                    // Retrieve block aggregate
//                    block_aggregate := temp_storage.block_aggregate
//            @>
//        
//    
//    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
//    //template <
//    //    typename        ScanOp,
//    //    typename        BlockPrefixCallbackOp>
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:IScanOp, block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) =
//        // localize template params & constants
//        let WARP_SYNCHRONOUS = this.Constants.WARP_SYNCHRONOUS
//        let RAKING_THREADS = this.Constants.RAKING_THREADS
//        // localize thread fields
//        let temp_storage = !(this.ThreadFields.temp_storage)
//        let linear_tid = this.ThreadFields.linear_tid
//
//
//        if (WARP_SYNCHRONOUS) then
//            // Short-circuit directly to warp scan
//            this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                input,
//                output,
//                identity,
//                scan_op,
//                block_aggregate,
//                block_prefix_callback_op)        
//        else        
//            // Place thread partial into shared memory raking grid
//            let placement_ptr = this.BlockRakingLayout.PlacementPtr <||| (temp_storage.raking_grid, linear_tid, None)
//            placement_ptr.[0] <- input
//
//            __syncthreads()
//
//            // Reduce parallelism down to just raking threads
//            if (linear_tid < RAKING_THREADS) then
//                // Raking upsweep reduction in grid
//                let raking_partial = this.Upsweep(scan_op)
//
//                // Exclusive warp synchronous scan
//                this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                    raking_partial,
//                    raking_partial |> __obj_to_ref,
//                    identity,
//                    scan_op,
//                    temp_storage.block_aggregate |> __obj_to_ref,
//                    block_prefix_callback_op)
//
//                // Exclusive raking downsweep scan
//                this.ExclusiveDownsweep(scan_op, raking_partial)
//
//            __syncthreads()
//
//            // Grab thread prefix from shared memory
//            output := placement_ptr.[0]
//
//            // Retrieve block aggregate
//            block_aggregate := temp_storage.block_aggregate
//        
//
//    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is undefined.
//    //template <typename ScanOp>
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:IScanOp, block_aggregate:Ref<int>) =
//        // localize template params & constants
//        let WARP_SYNCHRONOUS = this.Constants.WARP_SYNCHRONOUS
//        let RAKING_THREADS = this.Constants.RAKING_THREADS
//        // localize thread fields
//        let temp_storage = !(this.ThreadFields.temp_storage)
//        let linear_tid = this.ThreadFields.linear_tid
//
//        if (WARP_SYNCHRONOUS) then
//            // Short-circuit directly to warp scan
//            this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                input,
//                output,
//                scan_op,
//                block_aggregate)
//        else
//            // Place thread partial into shared memory raking grid
//            let placement_ptr = this.BlockRakingLayout.PlacementPtr <||| (temp_storage.raking_grid, linear_tid, None)
//            placement_ptr.[0] <- input
//
//            __syncthreads()
//
//            // Reduce parallelism down to just raking threads
//            if (linear_tid < RAKING_THREADS) then
//                // Raking upsweep reduction in grid
//                let raking_partial = this.Upsweep(scan_op)
//
//                // Exclusive warp synchronous scan
//                this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                    raking_partial,
//                    raking_partial |> __obj_to_ref,
//                    scan_op,
//                    temp_storage.block_aggregate |> __obj_to_ref)
//
//                // Exclusive raking downsweep scan
//                this.ExclusiveDownsweep(scan_op, raking_partial, (linear_tid <> 0))
//            
//
//            __syncthreads()
//
//            // Grab thread prefix from shared memory
//            output := placement_ptr.[0]
//
//            // Retrieve block aggregate
//            block_aggregate := temp_storage.block_aggregate
//        
//    
//    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
//    //template <
//    //    typename ScanOp,
//    //    typename BlockPrefixCallbackOp>
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:IScanOp, block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) =
//        // localize template params & constants
//        let WARP_SYNCHRONOUS = this.Constants.WARP_SYNCHRONOUS
//        let RAKING_THREADS = this.Constants.RAKING_THREADS
//        // localize thread fields
//        let temp_storage = !(this.ThreadFields.temp_storage)
//        let linear_tid = this.ThreadFields.linear_tid
//            
//        if (WARP_SYNCHRONOUS) then
//            // Short-circuit directly to warp scan
//            this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                input,
//                output,
//                scan_op,
//                block_aggregate,
//                block_prefix_callback_op)
//        else       
//            // Place thread partial into shared memory raking grid
//            let placement_ptr = this.BlockRakingLayout.PlacementPtr <||| (temp_storage.raking_grid, linear_tid, None)
//            placement_ptr.[0] <- input
//
//            __syncthreads()
//
//            // Reduce parallelism down to just raking threads
//            if (linear_tid < RAKING_THREADS) then
//                // Raking upsweep reduction in grid
//                let raking_partial = this.Upsweep(scan_op)
//
//                // Exclusive warp synchronous scan
//                this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                    raking_partial,
//                    raking_partial |> __obj_to_ref,
//                    scan_op,
//                    temp_storage.block_aggregate |> __obj_to_ref,
//                    block_prefix_callback_op)
//
//                // Exclusive raking downsweep scan
//                this.ExclusiveDownsweep(scan_op, raking_partial)
//            
//
//            __syncthreads()
//
//            // Grab thread prefix from shared memory
//            output := placement_ptr.[0]
//
//            // Retrieve block aggregate
//            block_aggregate := temp_storage.block_aggregate


//type TemplateParameters =
//    {
//        BLOCK_THREADS : int
//        MEMOIZE : bool        
//    }
//
//    static member Init(block_threads, memoize) =
//        {
//            BLOCK_THREADS = block_threads
//            MEMOIZE = memoize
//        }
//
//
//type Constants =
//    {
//        WARPS : int
//        RAKING_THREADS : int
//        SEGMENT_LENGTH : int
//        WARP_SYNCHRONOUS : bool
//    }
//    
//    static member Init(block_threads) =
//        let BlockRakingLayout = block_threads |> BlockRakingLayout.Init 
//        let raking_threads, segment_length =  BlockRakingLayout.Constants.RAKING_THREADS, BlockRakingLayout.Constants.SEGMENT_LENGTH
//        {
//            WARPS               = block_threads |> WARPS
//            RAKING_THREADS      = raking_threads
//            SEGMENT_LENGTH      = segment_length
//            WARP_SYNCHRONOUS    = (block_threads, raking_threads) ||> WARP_SYNCHRONOUS
//        }
//
//
//[<Record>] [<RequireQualifiedAccess>]
//type TempStorage =
//    {
//        mutable warp_scan : deviceptr<int> //Alea.cuExtension.CUB.Warp.Scan.TempStorage //deviceptr<int> //Alea.cuExtension.CUB.Warp.Scan.ITempStorage
//        mutable raking_grid : deviceptr<int> //Alea.cuExtension.CUB.Block.RakingLayout.ITempStorage //deviceptr<int> //Alea.cuExtension.CUB.Block.RakingLayout.ITempStorage
//        mutable block_aggregate : int
//    }
//
//    static member Init(grid_elements:int) =
//        {
//            warp_scan = __null()
//            raking_grid = grid_elements |> RakingLayout.tempStorage <| ()
//            block_aggregate = __null() |> __ptr_to_obj
//        }
//
//    
//[<Record>]
//type ThreadFields =
//    {
//        mutable temp_storage : Ref<TempStorage>
//        //mutable temp_storage : deviceptr<int>
//        mutable linear_tid : int
//        mutable cached_segment : int[]
//    }
//
//    static member Init(c:Constants, temp_storage:Ref<TempStorage>, linear_tid) =
////    static member Create(c:Constants, temp_storage:deviceptr<int>, linear_tid) =
//        {
//            temp_storage = temp_storage
//            linear_tid = linear_tid
//            cached_segment = __local__.Array(c.SEGMENT_LENGTH)
//        }
//
//    static member Init(c:Constants) =
//        {
//            temp_storage = ref (0 |> TempStorage.Init)
//            linear_tid = 0
//            cached_segment = __local__.Array(c.SEGMENT_LENGTH)
//        }
//    
//        
//[<Record>]
//type BlockScanRaking =
//    {
//        TemplateParameters  : TemplateParameters
//        BlockRakingLayout   : BlockRakingLayout //<int>
//        Constants           : Constants //<int>
//        ThreadScan          : ThreadScan //<int>
//        WarpScan            : WarpScan //<int>
//        //TempStorage         : TempStorage
//        ThreadFields        : ThreadFields //<int>
//    }
//
//
//    member this.Initialize(temp_storage:Ref<TempStorage>, linear_tid) =
//    //member this.Initialize(temp_storage:deviceptr<int>, linear_tid) =
//        this.ThreadFields.temp_storage <- temp_storage
//        this.ThreadFields.linear_tid <- linear_tid
//        this
//
//    member this.Initialize(temp_storage:deviceptr<int>, linear_tid) =
//        this.ThreadFields.temp_storage.contents.warp_scan <- temp_storage
//        this.ThreadFields.linear_tid <- linear_tid
//        this
//
//    /// Templated reduction
//    //template <int ITERATION, typename ScanOp>
//    member this.GuardedReduce(raking_ptr:deviceptr<int>, scan_op:IScanOp, raking_partial:int) = //, iteration:bool) =    
//        // localize template params & constants
//        let UNGUARDED       = this.BlockRakingLayout.Constants.UNGUARDED
//        let SEGMENT_LENGTH  = this.BlockRakingLayout.Constants.SEGMENT_LENGTH
//        let BLOCK_THREADS = this.TemplateParameters.BLOCK_THREADS
//        // localize thread fields
//        let linear_tid = this.ThreadFields.linear_tid
//
//        let mutable raking_partial = raking_partial
//        for ITERATION = 0 to (SEGMENT_LENGTH - 1) do
//            if ((UNGUARDED) || (((linear_tid * SEGMENT_LENGTH) + ITERATION) < BLOCK_THREADS)) then
//                let addend = raking_ptr.[ITERATION]
//                raking_partial <- (raking_partial, addend) ||> scan_op
//        
//        raking_partial
//    
//    
//    /// Performs upsweep raking reduction, returning the aggregate
//    //template <typename ScanOp>
//    member inline this.Upsweep(scan_op:IScanOp) =
//        // localize template params & constants
//        let SEGMENT_LENGTH = this.Constants.SEGMENT_LENGTH
//        let MEMOIZE = this.TemplateParameters.MEMOIZE
//        // localize thread fields
//        let temp_storage = !(this.ThreadFields.temp_storage)
//        let linear_tid = this.ThreadFields.linear_tid
//        let cached_segment = this.ThreadFields.cached_segment
//
//        let smem_raking_ptr = this.BlockRakingLayout.RakingPtr <|| (temp_storage.raking_grid, linear_tid) //rakingPtr()
//        let mutable raking_ptr = __null()
//
//        if MEMOIZE then
//            // Copy data into registers
//            //#pragma unroll
//            for i = 0 to (SEGMENT_LENGTH - 1) do cached_segment.[i] <- smem_raking_ptr.[i]
//            
//            raking_ptr <- cached_segment |> __array_to_ptr
//        
//        else
//            raking_ptr <- smem_raking_ptr
//        
//
//        let raking_partial = raking_ptr.[0]
//
//        this.GuardedReduce(raking_ptr, scan_op, raking_partial) //, Int2Type<1>())
//    
//
//
//    /// Performs exclusive downsweep raking scan
//    //template <typename ScanOp>
//    member inline this.ExclusiveDownsweep(scan_op:IScanOp, raking_partial:int, ?apply_prefix:bool) =
//        let apply_prefix = if apply_prefix.IsSome then apply_prefix.Value else true
//
//        // localize template params & constants
//        let MEMOIZE = this.TemplateParameters.MEMOIZE
//        let SEGMENT_LENGTH = this.Constants.SEGMENT_LENGTH
//        // localize thread fields
//        let temp_storage = !(this.ThreadFields.temp_storage)
//        let linear_tid = this.ThreadFields.linear_tid
//        let cached_segment = this.ThreadFields.cached_segment
//
//        let smem_raking_ptr = this.BlockRakingLayout.RakingPtr <|| (temp_storage.raking_grid, linear_tid)
//
//        let raking_ptr = if (MEMOIZE) then cached_segment |> __array_to_ptr else smem_raking_ptr
//
//        //SEGMENT_LENGTH |> ThreadScanExclusive<int> <| raking_ptr <| raking_ptr <| scan_op <| raking_partial <| apply_prefix
//        this.ThreadScan.Initialize(SEGMENT_LENGTH).Exclusive(raking_ptr, raking_ptr, scan_op, raking_partial, apply_prefix)
//
//        if (MEMOIZE) then
//            // Copy data back to smem
//            for i = 0 to (SEGMENT_LENGTH - 1) do smem_raking_ptr.[i] <- cached_segment.[i]
//
//
//    /// Performs inclusive downsweep raking scan
//    //template <typename ScanOp>
//    member this.InclusiveDownsweep(scan_op:IScanOp, raking_partial:int, ?apply_prefix:bool) =
//        // localize template params & constants
//        let MEMOIZE = this.TemplateParameters.MEMOIZE
//        let SEGMENT_LENGTH = this.Constants.SEGMENT_LENGTH
//        // localize thread fields
//        let temp_storage = !(this.ThreadFields.temp_storage)
//        let linear_tid = this.ThreadFields.linear_tid
//        let cached_segment = this.ThreadFields.cached_segment
//                        
//        let smem_raking_ptr = this.BlockRakingLayout.RakingPtr <|| (temp_storage.raking_grid, linear_tid) //(temp_storage.raking_grid.temp_storage, linear_tid)
//
//        let raking_ptr = if (MEMOIZE) then cached_segment |> __array_to_ptr else smem_raking_ptr
//
//        this.ThreadScan.Initialize(SEGMENT_LENGTH).Inclusive(raking_ptr, raking_ptr, scan_op, raking_partial, apply_prefix.Value)
//
//        if (MEMOIZE) then
//            // Copy data back to smem
//            for i = 0 to (SEGMENT_LENGTH - 1) do smem_raking_ptr.[i] <- cached_segment.[i]
//
//
//    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
//    //template <typename ScanOp>
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:Ref<int>, scan_op:IScanOp, block_aggregate:Ref<int>) =
//        // localize template params & constants
//        let RAKING_THREADS = this.Constants.RAKING_THREADS
//        let WARP_SYNCHRONOUS = this.Constants.WARP_SYNCHRONOUS
//        // localize thread fields
//        let temp_storage = !(this.ThreadFields.temp_storage)
//        let linear_tid = this.ThreadFields.linear_tid
//            
//
//        if (WARP_SYNCHRONOUS) then
//            // Short-circuit directly to warp scan
//            this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                input,
//                output,
//                !identity,
//                scan_op,
//                block_aggregate)
//        else
//            // Place thread partial into shared memory raking grid
//            let placement_ptr = this.BlockRakingLayout.PlacementPtr <||| (temp_storage.raking_grid, linear_tid, None)
//            placement_ptr.[0] <- input
//
//            __syncthreads()
//
//            // Reduce parallelism down to just raking threads
//            if (linear_tid < RAKING_THREADS) then
//                // Raking upsweep reduction in grid
//                let raking_partial = this.Upsweep(scan_op)
//
//                // Exclusive warp synchronous scan
//                this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                    raking_partial,
//                    raking_partial |> __obj_to_ref,
//                    !identity,
//                    scan_op,
//                    temp_storage.block_aggregate |> __obj_to_ref)
//
//                // Exclusive raking downsweep scan
//                this.ExclusiveDownsweep(scan_op, raking_partial)
//            
//            __syncthreads()
//
//            // Grab thread prefix from shared memory
//            output := placement_ptr.[0]
//
//            // Retrieve block aggregate
//            block_aggregate := temp_storage.block_aggregate
//        
//    
//    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
//    //template <
//    //    typename        ScanOp,
//    //    typename        BlockPrefixCallbackOp>
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:IScanOp, block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) =
//        // localize template params & constants
//        let WARP_SYNCHRONOUS = this.Constants.WARP_SYNCHRONOUS
//        let RAKING_THREADS = this.Constants.RAKING_THREADS
//        // localize thread fields
//        let temp_storage = !(this.ThreadFields.temp_storage)
//        let linear_tid = this.ThreadFields.linear_tid
//
//
//        if (WARP_SYNCHRONOUS) then
//            // Short-circuit directly to warp scan
//            this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                input,
//                output,
//                identity,
//                scan_op,
//                block_aggregate,
//                block_prefix_callback_op)        
//        else        
//            // Place thread partial into shared memory raking grid
//            let placement_ptr = this.BlockRakingLayout.PlacementPtr <||| (temp_storage.raking_grid, linear_tid, None)
//            placement_ptr.[0] <- input
//
//            __syncthreads()
//
//            // Reduce parallelism down to just raking threads
//            if (linear_tid < RAKING_THREADS) then
//                // Raking upsweep reduction in grid
//                let raking_partial = this.Upsweep(scan_op)
//
//                // Exclusive warp synchronous scan
//                this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                    raking_partial,
//                    raking_partial |> __obj_to_ref,
//                    identity,
//                    scan_op,
//                    temp_storage.block_aggregate |> __obj_to_ref,
//                    block_prefix_callback_op)
//
//                // Exclusive raking downsweep scan
//                this.ExclusiveDownsweep(scan_op, raking_partial)
//
//            __syncthreads()
//
//            // Grab thread prefix from shared memory
//            output := placement_ptr.[0]
//
//            // Retrieve block aggregate
//            block_aggregate := temp_storage.block_aggregate
//        
//
//    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is undefined.
//    //template <typename ScanOp>
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:IScanOp, block_aggregate:Ref<int>) =
//        // localize template params & constants
//        let WARP_SYNCHRONOUS = this.Constants.WARP_SYNCHRONOUS
//        let RAKING_THREADS = this.Constants.RAKING_THREADS
//        // localize thread fields
//        let temp_storage = !(this.ThreadFields.temp_storage)
//        let linear_tid = this.ThreadFields.linear_tid
//
//        if (WARP_SYNCHRONOUS) then
//            // Short-circuit directly to warp scan
//            this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                input,
//                output,
//                scan_op,
//                block_aggregate)
//        else
//            // Place thread partial into shared memory raking grid
//            let placement_ptr = this.BlockRakingLayout.PlacementPtr <||| (temp_storage.raking_grid, linear_tid, None)
//            placement_ptr.[0] <- input
//
//            __syncthreads()
//
//            // Reduce parallelism down to just raking threads
//            if (linear_tid < RAKING_THREADS) then
//                // Raking upsweep reduction in grid
//                let raking_partial = this.Upsweep(scan_op)
//
//                // Exclusive warp synchronous scan
//                this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                    raking_partial,
//                    raking_partial |> __obj_to_ref,
//                    scan_op,
//                    temp_storage.block_aggregate |> __obj_to_ref)
//
//                // Exclusive raking downsweep scan
//                this.ExclusiveDownsweep(scan_op, raking_partial, (linear_tid <> 0))
//            
//
//            __syncthreads()
//
//            // Grab thread prefix from shared memory
//            output := placement_ptr.[0]
//
//            // Retrieve block aggregate
//            block_aggregate := temp_storage.block_aggregate
//        
//    
//    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
//    //template <
//    //    typename ScanOp,
//    //    typename BlockPrefixCallbackOp>
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:IScanOp, block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) =
//        // localize template params & constants
//        let WARP_SYNCHRONOUS = this.Constants.WARP_SYNCHRONOUS
//        let RAKING_THREADS = this.Constants.RAKING_THREADS
//        // localize thread fields
//        let temp_storage = !(this.ThreadFields.temp_storage)
//        let linear_tid = this.ThreadFields.linear_tid
//            
//        if (WARP_SYNCHRONOUS) then
//            // Short-circuit directly to warp scan
//            this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                input,
//                output,
//                scan_op,
//                block_aggregate,
//                block_prefix_callback_op)
//        else       
//            // Place thread partial into shared memory raking grid
//            let placement_ptr = this.BlockRakingLayout.PlacementPtr <||| (temp_storage.raking_grid, linear_tid, None)
//            placement_ptr.[0] <- input
//
//            __syncthreads()
//
//            // Reduce parallelism down to just raking threads
//            if (linear_tid < RAKING_THREADS) then
//                // Raking upsweep reduction in grid
//                let raking_partial = this.Upsweep(scan_op)
//
//                // Exclusive warp synchronous scan
//                this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
//                    raking_partial,
//                    raking_partial |> __obj_to_ref,
//                    scan_op,
//                    temp_storage.block_aggregate |> __obj_to_ref,
//                    block_prefix_callback_op)
//
//                // Exclusive raking downsweep scan
//                this.ExclusiveDownsweep(scan_op, raking_partial)
//            
//
//            __syncthreads()
//
//            // Grab thread prefix from shared memory
//            output := placement_ptr.[0]
//
//            // Retrieve block aggregate
//            block_aggregate := temp_storage.block_aggregate
//       d

//    let WarpScan = ()
//
//
//    [<Record>]
//    type TempStorage =
//        
//            warp_scan : deviceptr<int>
//            raking_grid : deviceptr<int>
//            block_aggregate : int
//        
//
//        static member Create(warp_scan, raking_grid, block_aggregate) =
//            
//                warp_scan = warp_scan
//                raking_grid = raking_grid
//                block_aggregate = block_aggregate
//                  
//   
//    
//    let guardedReduce =
//        fun block_threads ->
//            fun unguarded segment_length ->
//                fun linear_tid ->
//                    fun warps raking_threads segment_length warp_synchronous ->
//                        fun iteration ->
//                            fun (raking_ptr:deviceptr<int>) (scan_op:IScanOp) (raking_partial:int) ->
//                                let mutable raking_partial = raking_partial
//                                if unguarded || (((linear_tid * segment_length) + iteration) < block_threads) then
//                                    let addend = raking_ptr.[iteration]
//                                    raking_partial <- (raking_partial, addend) ||> scan_op
//
//    let upsweep =
//        fun (scan_op:IScanOp) ->
//            let smem_raking_ptr = ()
//            ()
//        
//
//    let blockScanRaking block_threads memoize =
//        let BlockRakingLayout = block_threads |> BlockRakingLayout
//        ()
//    
//        
//    //let guardedReduce (iteration:int) (scanOp:)
//
//
//    [<Record>]
//    type ThreadFields =
//        
//            temp_storage    : TempStorage
//            linear_tid      : int
//            cached_segment  : deviceptr<int>
//        
//
//        static member Create(temp_storage, linear_tid) =
//            
//                temp_storage = temp_storage
//                linear_tid = linear_tid
//                cached_segment = __null()
//            
//
//
//    [<Record>]
//    type BlockScanRaking =
//        
//            BLOCK_THREADS : int
//            MEMOIZE : bool
//        
//
//       