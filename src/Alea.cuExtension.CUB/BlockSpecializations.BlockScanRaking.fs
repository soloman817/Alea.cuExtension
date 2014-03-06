[<AutoOpen>]
module Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanRaking

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Warp
open Alea.cuExtension.CUB.Block


module Template =   
    
    [<AutoOpen>]
    module Params =
        [<Record>]
        type API<'T> =
            {
                BLOCK_THREADS   : int
                MEMOIZE         : bool   
            }

            [<ReflectedDefinition>]
            static member Init(block_threads, memoize, scan_op) =
                {
                    BLOCK_THREADS   = block_threads
                    MEMOIZE         = memoize
                }

    let [<ReflectedDefinition>] inline BlockRakingLayout (tp:Params.API) = BlockRakingLayout.API<'T>.Init(tp.BLOCK_THREADS)

    [<AutoOpen>]
    module Constants =
        [<Record>]
        type API =
            {
                WARPS               : int
                RAKING_THREADS      : int
                SEGMENT_LENGTH      : int
                WARP_SYNCHRONOUS    : bool
            }
            
            [<ReflectedDefinition>]
            static member Init(tp:Params.API) =
                let warps               = (tp.BLOCK_THREADS + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS
                let raking_threads      = (BlockRakingLayout template).Constants.RAKING_THREADS
                let segment_length      = (BlockRakingLayout template).Constants.SEGMENT_LENGTH
                let warp_synchronous    = (tp.BLOCK_THREADS = raking_threads)
                {
                    WARPS               = warps
                    RAKING_THREADS      = raking_threads
                    SEGMENT_LENGTH      = segment_length
                    WARP_SYNCHRONOUS    = warp_synchronous
                }

    let [<ReflectedDefinition>] inline WarpScan (tp:Params.API) = 
        let c = Constants.API.Init template
        WarpScan.API<'T>.Create(1, c.RAKING_THREADS)

    [<AutoOpen>]
    module TempStorage =
        [<Record>]
        type API<'T> =
            {
                mutable warp_scan               :   Alea.cuExtension.CUB.Warp.Scan.Template._TempStorage<'T>
                mutable raking_grid             :   Alea.cuExtension.CUB.Block.RakingLayout.Template._TempStorage<'T>
                mutable block_aggregate         :   'T
            }

            [<ReflectedDefinition>]
            static member inline Init(block_threads, memoize, scan_op) =
                let template = Params.API.Init(block_threads, memoize, scan_op)
                let WarpScan = WarpScan template
                let BlockRakingLayout = BlockRakingLayout template
                let b_a = __shared__.Variable<'T>()
                {
                    warp_scan       = WarpScan.ThreadFields.temp_storage
                    raking_grid     = BlockRakingLayout.TempStorage
                    block_aggregate = !b_a
                }

            [<ReflectedDefinition>]
            static member inline Init(tp:Params.API) = API<'T>.Init(tp.BLOCK_THREADS, template.MEMOIZE, template.scan_op)


    [<AutoOpen>]
    module ThreadFields =
        [<Record>]
        type API<'T> =
            {
                mutable temp_storage    : TempStorage.API<'T>
                mutable linear_tid      : int
                mutable cached_segment  : 'T[]
            }

            [<ReflectedDefinition>]
            static member Init(block_threads, memoize, scan_op, temp_storage, linear_tid) =
                let template = Params.API.Init(block_threads, memoize, scan_op)
                let c = Constants.API.Init template
                let cs = __local__.Array<'T>(c.SEGMENT_LENGTH)
                {
                    temp_storage    = temp_storage
                    linear_tid      = linear_tid
                    cached_segment  = cs
                }

            [<ReflectedDefinition>]
            static member Init(tp:Params.API, temp_storage, linear_tid) = API<'T>.Init(tp.BLOCK_THREADS, template.MEMOIZE, template.scan_op, temp_storage, linear_tid)

            [<ReflectedDefinition>]
            static member Init(tp:Params.API) = API<'T>.Init(tp, (TempStorage.API<'T>.Init template), 0)

                        


    type _TemplateParams<'T>    = Params.API
    type _Constants             = Constants.API
    type _TempStorage<'T>       = TempStorage.API<'T>
    type _ThreadFields<'T>      = ThreadFields.API<'T>

module private Internal =
    open Template


    module Sig =
        module Upsweep =
            type Default<'T> = unit -> 'T

        module GuardedReduce =
            type Default<'T> = deviceptr<'T> -> 'T -> 'T

        module ExclusiveDownsweep =
            type Default<'T>            = 'T -> unit
            type WithApplyPrefix<'T>    = 'T -> bool -> unit

        module ExclusiveScan =
            type WithAggregate<'T>              = 'T -> Ref<'T> -> Ref<'T> -> Ref<'T> -> unit
            type WithAggregateAndCallbackOp<'T> = 'T -> Ref<'T> -> Ref<'T> -> Ref<'T> -> Ref<'T -> 'T> -> unit

            module Identityless =
                type WithAggregate<'T>              = 'T -> Ref<'T> -> Ref<'T> -> unit
                type WithAggregateAndCallbackOp<'T> = 'T -> Ref<'T> -> Ref<'T> -> Ref<'T -> 'T> -> unit

        module ExclusiveSum =
            type WithAggregate<'T>              = ExclusiveScan.Identityless.WithAggregate<'T>
            type WithAggregateAndCallbackOp<'T> = ExclusiveScan.Identityless.WithAggregateAndCallbackOp<'T>

        module InclusiveScan =
            type WithAggregate<'T>              = ExclusiveSum.WithAggregate<'T>
            type WithAggregateAndCallbackOp<'T> = ExclusiveSum.WithAggregateAndCallbackOp<'T>

        module InclusiveSum =
            type WithAggregate<'T>              = InclusiveScan.WithAggregate<'T>
            type WithAggregateAndCallbackOp<'T> = InclusiveScan.WithAggregateAndCallbackOp<'T>


module GuardedReduce =
    open Template
    open Internal
    // Need to do Attribute unrolling stuff
    type API<'T> =
        {
            Default : Sig.GuardedReduce.Default<'T>
        }

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        
        (raking_ptr:deviceptr<'T>) (raking_partial:'T) =
        let c = _Constants.Init template
        let UNGUARDED = (BlockRakingLayout template).Constants.UNGUARDED
        let mutable raking_partial = raking_partial
        let scan_op = template.scan_op.op

        for i = 0 to (c.SEGMENT_LENGTH - 1) do
            if UNGUARDED || (((tf.linear_tid * c.SEGMENT_LENGTH) + i) < template.BLOCK_THREADS) then
                let addend = raking_ptr.[i]
                raking_partial <- (raking_partial, addend) ||> scan_op
        raking_partial


    let [<ReflectedDefinition>] api (template:_Template<'T>)
         =
            {
                Default =   Default template tf
            }


module Upsweep =
    open Template 
    open Internal

    type API<'T> =
        {
            Default : Sig.Upsweep.Default<'T>
        }

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
         () =
        
        let c = _Constants.Init template
        
        let smem_raking_ptr =   (BlockRakingLayout template).RakingPtr.Default tf.temp_storage.raking_grid tf.linear_tid
        let mutable raking_ptr = __local__.Array(c.SEGMENT_LENGTH) |> __array_to_ptr

        if template.MEMOIZE then 
            for i = 0 to (c.SEGMENT_LENGTH - 1) do tf.cached_segment.[i] <- smem_raking_ptr.[i]
            raking_ptr <- tf.cached_segment |> __array_to_ptr
        else
            raking_ptr <- smem_raking_ptr

        let raking_partial = raking_ptr.[0]
        (GuardedReduce.api template tf).Default raking_ptr raking_partial

    
    let [<ReflectedDefinition>] api (template:_Template<'T>)
         =
            let d = Default template tf
            {
                Default =   Default template tf
            }

module ExclusiveDownsweep =
    open Template
    open Internal

    type API<'T> =
        {
            Default         : Sig.ExclusiveDownsweep.Default<'T>
            WithApplyPrefix : Sig.ExclusiveDownsweep.WithApplyPrefix<'T>
        }

    let [<ReflectedDefinition>] inline WithApplyPrefix (template:_Template<'T>)
        
        (raking_partial:'T) (apply_prefix:bool) =
        let c = _Constants.Init template
//        let ThreadScanExclusive = (ThreadScanExclusive.api c.SEGMENT_LENGTH template.scan_op).WithApplyPrefix
//       
//        let smem_raking_ptr =   (BlockRakingLayout template).RakingPtr.Default tf.temp_storage.raking_grid tf.linear_tid
//                
//        let raking_ptr = if template.MEMOIZE then tf.cached_segment |> __array_to_ptr else smem_raking_ptr
//                
//        ThreadScanExclusive
//        <|| (raking_ptr, raking_ptr)
//        <|  (raking_partial)
//        <|  (apply_prefix)
//        |> ignore
//
//        if template.MEMOIZE then
//            for i = 0 to (c.SEGMENT_LENGTH - 1) do smem_raking_ptr.[i] <- tf.cached_segment.[i]
//            
        ()

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>) 
        
        (raking_partial:'T) =
        WithApplyPrefix template tf raking_partial true


    let [<ReflectedDefinition>] api (template:_Template<'T>)
         =
            {
                Default         =   Default template tf
                WithApplyPrefix =   WithApplyPrefix template tf
            }

module InclusiveDownsweep = ()

module ExclusiveScan =
    open Template
    open Internal

    type API<'T> =
        {
            WithAggregate               : Sig.ExclusiveScan.WithAggregate<'T>
            WithAggregateAndCallbackOp  : Sig.ExclusiveScan.WithAggregateAndCallbackOp<'T>
        }

    let [<ReflectedDefinition>] inline WithAggregate (template:_Template<'T>)
        
        (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) =
        let c = _Constants.Init template

        let WarpScan            = (WarpScan template).Init(tf.temp_storage.warp_scan, 0, tf.linear_tid).ExclusiveScan.WithAggregate
        let PlacementPtr        = (BlockRakingLayout template).PlacementPtr.Default
        let Upsweep             = (Upsweep.api template tf).Default
        let ExclusiveDownsweep  = (ExclusiveDownsweep.api template tf).Default
                
        if c.WARP_SYNCHRONOUS then
            WarpScan input output !identity block_aggregate
        else
            let placement_ptr = PlacementPtr tf.temp_storage.raking_grid tf.linear_tid
            placement_ptr.[0] <- input

            __syncthreads()

            if tf.linear_tid < c.RAKING_THREADS then
                let raking_partial = Upsweep()
                WarpScan
                <|| (raking_partial, ref raking_partial)
                <|  !identity
                <|  ref tf.temp_storage.block_aggregate

                ExclusiveDownsweep raking_partial

            __syncthreads()

            output := placement_ptr.[0]

            block_aggregate := tf.temp_storage.block_aggregate


    let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (template:_Template<'T>)
        
        (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()

    let [<ReflectedDefinition>] api (template:_Template<'T>)
         =
            {
                WithAggregate               =   WithAggregate template tf
                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp template tf
            }

module ExclusiveSum =
    open Template
    open Internal

    type API<'T> =
        {
            WithAggregate               : Sig.ExclusiveSum.WithAggregate<'T>
            WithAggregateAndCallbackOp  : Sig.ExclusiveSum.WithAggregateAndCallbackOp<'T>
        }

    let [<ReflectedDefinition>] inline WithAggregate (template:_Template<'T>)
        
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) =
        let c = _Constants.Init template

        let WarpScan            = (WarpScan template).Init(tf.temp_storage.warp_scan, 0, tf.linear_tid).ExclusiveSum.WithAggregate
        let PlacementPtr        = (BlockRakingLayout template).PlacementPtr.Default
        let Upsweep             = (Upsweep.api template tf).Default
        let ExclusiveDownsweep  = (ExclusiveDownsweep.api template tf).Default

    
        if c.WARP_SYNCHRONOUS then
            WarpScan input output block_aggregate
        else
            let placement_ptr = PlacementPtr tf.temp_storage.raking_grid tf.linear_tid
            placement_ptr.[0] <- input

            __syncthreads()

            if tf.linear_tid < c.RAKING_THREADS then
                let raking_partial = Upsweep()
                WarpScan
                <|| (raking_partial, ref raking_partial)
                <|  ref tf.temp_storage.block_aggregate

                ExclusiveDownsweep raking_partial

            __syncthreads()

            output := placement_ptr.[0]

            block_aggregate := tf.temp_storage.block_aggregate
    

    let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (template:_Template<'T>)
        
        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = ()

    let [<ReflectedDefinition>] api (template:_Template<'T>)
         =
            {
                WithAggregate               =   WithAggregate template tf

                WithAggregateAndCallbackOp  =   WithAggregateAndCallbackOp template tf
            }


module BlockScanRaking =
    open Template
    open Internal

    [<Record>]
    type API<'T> =
        {
            ExclusiveSum    : ExclusiveSum.API<'T>
            ExclusiveScan   : ExclusiveScan.API<'T>
        }

        [<ReflectedDefinition>]
        static member Create(block_threads, memoize, scan_op) =
            let template = _TemplateParams<'T>.Init(block_threads, memoize, scan_op)            
            let tf = _ThreadFields<'T>.Init(tp)
            {
                ExclusiveSum    =   ExclusiveSum.api template tf
                ExclusiveScan   =   ExclusiveScan.api template tf
            }



    let [<ReflectedDefinition>] api (template:_Template<'T>)
         =
            {
                ExclusiveSum    =   ExclusiveSum.api template tf
                ExclusiveScan   =   ExclusiveScan.api template tf
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
//    member inline this.Upsweep(scan_op:IScanOp<'T>) =
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
//        fun (temp_storage:_TempStorage<'T>) (linear_tid:int) ->
//            <@ fun (input:'T) (output:Ref<'T>) (identity:Ref<int>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<int>) ->
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
//    member inline this.Upsweep(scan_op:IScanOp<'T>) =
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
//                            fun (raking_ptr:deviceptr<int>) (scan_op:IScanOp<'T>) (raking_partial:int) ->
//                                let mutable raking_partial = raking_partial
//                                if unguarded || (((linear_tid * segment_length) + iteration) < block_threads) then
//                                    let addend = raking_ptr.[iteration]
//                                    raking_partial <- (raking_partial, addend) ||> scan_op
//
//    let upsweep =
//        fun (scan_op:IScanOp<'T>) ->
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