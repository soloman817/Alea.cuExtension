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
    module Host =
        module Params =
            type API = 
                { BLOCK_THREADS : int; MEMOIZE : bool } 
                static member Init(block_threads, memoize) = 
                    { BLOCK_THREADS = block_threads; MEMOIZE = memoize }

        
        module Constants =
            type API =
                {
                    WARPS               : int
                    RAKING_THREADS      : int
                    SEGMENT_LENGTH      : int
                    WARP_SYNCHRONOUS    : bool
                }
            
                static member Init(p:Params.API) =
                    let brl_c = (BlockRakingLayout.HostApi.Init(p.BLOCK_THREADS)).Constants
                    let warps               = (p.BLOCK_THREADS + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS
                    let raking_threads      = brl_c.RAKING_THREADS
                    let segment_length      = brl_c.SEGMENT_LENGTH
                    let warp_synchronous    = (p.BLOCK_THREADS = raking_threads)
                    {
                        WARPS               = warps
                        RAKING_THREADS      = raking_threads
                        SEGMENT_LENGTH      = segment_length
                        WARP_SYNCHRONOUS    = warp_synchronous
                    }

        type API =
            {
                Params                      : Params.API
                Constants                   : Constants.API
                BlockRakingLayoutHostApi    : BlockRakingLayout.HostApi
                WarpScanHostApi             : WarpScan.HostApi
            }

            static member Init(block_threads, memoize) =
                let p = Params.API.Init(block_threads, memoize)
                let c = Constants.API.Init(p)
                let brlhapi = BlockRakingLayout.HostApi.Init(p.BLOCK_THREADS)
                let wshapi = WarpScan.HostApi.Init(1, c.RAKING_THREADS)
                { Params = p; Constants = c; BlockRakingLayoutHostApi = brlhapi; WarpScanHostApi = wshapi }


    module Device =
        module TempStorage =
            [<Record>]
            type API<'T> =
                {
                    mutable warp_scan               :   WarpScan.TempStorage<'T>
                    mutable raking_grid             :   BlockRakingLayout.TempStorage<'T>
                    mutable block_aggregate         :   'T
                }

                [<ReflectedDefinition>]
                static member inline Uninitialized(h:Host.API) =
                    let b_a = __shared__.Variable<'T>()
                    {
                        warp_scan       = WarpScan.TempStorage<'T>.Uninitialized(h.WarpScanHostApi.SharedMemoryLength)
                        raking_grid     = BlockRakingLayout.TempStorage<'T>.Init(h.BlockRakingLayoutHostApi.SharedMemoryLength)
                        block_aggregate = !b_a
                    }

        [<Record>]
        type API<'T> =
            {
                mutable temp_storage    : TempStorage.API<'T>
                mutable linear_tid      : int
                mutable cached_segment  : 'T[]
            }

            [<ReflectedDefinition>]
            static member Init(h:Host.API, temp_storage, linear_tid) =
                let c = h.Constants
                let cs = __local__.Array<'T>(c.SEGMENT_LENGTH)
                {
                    temp_storage    = temp_storage
                    linear_tid      = linear_tid
                    cached_segment  = cs
                }



    type _TemplateParams        = Host.Params.API
    type _Constants             = Host.Constants.API
    type _HostApi               = Host.API
    
    type _TempStorage<'T>       = Device.TempStorage.API<'T>
    type _DeviceApi<'T>         = Device.API<'T>



module GuardedReduce =
    open Template
    ///@TODO Need to do Attribute loop unrolling stuff
    
    let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
        (d:_DeviceApi<'T>)
        (raking_ptr:deviceptr<'T>) (raking_partial:'T) =
        let p = h.Params
        let c = h.Constants
        let brl_c = h.BlockRakingLayoutHostApi.Constants
            
            
        let mutable raking_partial = raking_partial
        
        for i = 0 to (c.SEGMENT_LENGTH - 1) do
            if brl_c.UNGUARDED || (((d.linear_tid * c.SEGMENT_LENGTH) + i) < p.BLOCK_THREADS) then
                let addend = raking_ptr.[i]
                raking_partial <- (raking_partial, addend) ||> scan_op
        raking_partial
        


module Upsweep =
    open Template 

    let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
        (d:_DeviceApi<'T>) =
        let p = h.Params
        let c = h.Constants
        let brl_h = h.BlockRakingLayoutHostApi   

        let smem_raking_ptr =   BlockRakingLayout.API<'T>.Create(brl_h).RakingPtr(brl_h, d.temp_storage.raking_grid.Ptr, d.linear_tid)
        let mutable raking_ptr = __local__.Array(c.SEGMENT_LENGTH) |> __array_to_ptr

        if p.MEMOIZE then 
            for i = 0 to (c.SEGMENT_LENGTH - 1) do d.cached_segment.[i] <- smem_raking_ptr.[i]
            raking_ptr <- d.cached_segment |> __array_to_ptr
        else
            raking_ptr <- smem_raking_ptr

        let raking_partial = raking_ptr.[0]
        GuardedReduce.Default h scan_op d raking_ptr raking_partial
        

 

module ExclusiveDownsweep =
    open Template


    let WithApplyPrefix (h:_HostApi) (scan_op:'T -> 'T -> 'T)
        (d:_DeviceApi<'T>) 
        (raking_partial:'T) (apply_prefix:bool) =
        let p = h.Params
        let c = h.Constants
        let brl_h = h.BlockRakingLayoutHostApi

        let smem_raking_ptr = BlockRakingLayout.API<'T>.Create(brl_h).RakingPtr(brl_h, d.temp_storage.raking_grid.Ptr, d.linear_tid)
                
        let raking_ptr = if p.MEMOIZE then d.cached_segment |> __array_to_ptr else smem_raking_ptr
                
        ThreadScanExclusive.WithApplyPrefix c.SEGMENT_LENGTH scan_op raking_ptr raking_ptr raking_partial apply_prefix
        |> ignore

        if p.MEMOIZE then
            for i = 0 to (c.SEGMENT_LENGTH - 1) do smem_raking_ptr.[i] <- d.cached_segment.[i]            
        

    let Default (h:_HostApi) (scan_op:'T -> 'T -> 'T) 
        (d:_DeviceApi<'T>)
        (raking_partial:'T) =
        WithApplyPrefix h scan_op d raking_partial true
        



module InclusiveDownsweep =
    open Template

    let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
        (d:_DeviceApi<'T>)
        (raking_partial:'T) (apply_prefix:bool) =
        let p = h.Params
        let c = h.Constants    
        let brl_h = h.BlockRakingLayoutHostApi
            
        let smem_raking_ptr = BlockRakingLayout.API<'T>.Create(brl_h).RakingPtr(brl_h, d.temp_storage.raking_grid.Ptr, d.linear_tid)
        let raking_ptr = if p.MEMOIZE then d.cached_segment |> __array_to_ptr else smem_raking_ptr

        ThreadScanInclusive.WithApplyPrefix c.SEGMENT_LENGTH scan_op raking_ptr raking_ptr raking_partial apply_prefix
        


module ExclusiveScan =
    open Template
//
//    let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//        (d:_DeviceApi<'T>)
//        (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) =
//        let p = h.Params
//        let c = h.Constants
//        let ws_h = h.WarpScanHostApi
//        let brl_h = h.BlockRakingLayoutHostApi
//
//        if c.WARP_SYNCHRONOUS then
//            WarpScan.API<'T>.Create(ws_h, d.temp_storage.warp_scan, 0, d.linear_tid).ExclusiveScan(ws_h, scan_op, input, output, !identity, block_aggregate)
//        else
//            let placement_ptr = BlockRakingLayout.API<'T>.Create(brl_h).PlacementPtr(brl_h, d.temp_storage.raking_grid.Ptr, d.linear_tid)
//            placement_ptr.[0] <- input
//
//            __syncthreads()
//                
//            let ts_block_aggregate = __local__.Variable<'T>(d.temp_storage.block_aggregate)
//            if d.linear_tid < c.RAKING_THREADS then
//                let raking_partial = __local__.Variable<'T>(Upsweep.Default h scan_op d)
//                    
//                WarpScan.API<'T>.Create(ws_h, d.temp_storage.warp_scan, 0, d.linear_tid).ExclusiveScan(ws_h, scan_op, !raking_partial, raking_partial, !identity, ts_block_aggregate)
//
//                ExclusiveDownsweep.Default h scan_op d !raking_partial
//
//            __syncthreads()
//
//            output := placement_ptr.[0]
//
//            block_aggregate := !ts_block_aggregate
// 
//
//    let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//        (d:_DeviceApi<'T>)
//        (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callbackop:Ref<'T -> 'T>) = ()       
//
//
//
//    module Identityless =
//        let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:_DeviceApi<'T>)
//            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) =
//            let p = h.Params
//            let c = h.Constants
//            let brl_h = h.BlockRakingLayoutHostApi
//            let ws_h = h.WarpScanHostApi
//
//            if c.WARP_SYNCHRONOUS then
//                WarpScan.API<'T>.Create(ws_h, d.temp_storage.warp_scan, 0, d.linear_tid).ExclusiveScan(ws_h, scan_op, input, output, block_aggregate)
//            else
//                let placement_ptr = BlockRakingLayout.API<'T>.Create(brl_h).PlacementPtr(brl_h, d.temp_storage.raking_grid.Ptr, d.linear_tid)
//                placement_ptr.[0] <- input
//
//                __syncthreads()
//
//                let ts_block_aggregate = __local__.Variable<'T>(d.temp_storage.block_aggregate)
//                if d.linear_tid < c.RAKING_THREADS then
//                    let raking_partial = __local__.Variable<'T>(Upsweep.Default h scan_op d)
//
//                    WarpScan.API<'T>.Create(ws_h, d.temp_storage.warp_scan, 0, d.linear_tid).ExclusiveScan(ws_h, scan_op, !raking_partial, raking_partial, ts_block_aggregate)
//
//                    ExclusiveDownsweep.WithApplyPrefix h scan_op d !raking_partial (d.linear_tid <> 0)
//
//                __syncthreads()
//
//                output := placement_ptr.[0]
//
//                block_aggregate := !ts_block_aggregate
//
//
//
//        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:_DeviceApi<'T>)
//            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callbackop:Ref<'T -> 'T>) = ()



module ExclusiveSum =
    open Template

//    let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//        (d:_DeviceApi<'T>)
//        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) =
//        let p = h.Params
//        let c = h.Constants
//        let ws_h = h.WarpScanHostApi
//        let brl_h = h.BlockRakingLayoutHostApi
//
//        if c.WARP_SYNCHRONOUS then
//            WarpScan.API<'T>.Create(ws_h, d.temp_storage.warp_scan, 0, d.linear_tid).ExclusiveSum(ws_h, scan_op, input, output, block_aggregate)
//        else
//            let placement_ptr = BlockRakingLayout.API<'T>.Create(brl_h).PlacementPtr(brl_h, d.temp_storage.raking_grid.Ptr, d.linear_tid)
//            placement_ptr.[0] <- input
//
//            __syncthreads()
//                
//            let ts_block_aggregate = __local__.Variable<'T>(d.temp_storage.block_aggregate)
//            if d.linear_tid < c.RAKING_THREADS then
//                let raking_partial = __local__.Variable<'T>(Upsweep.Default h scan_op d)
//                    
//                WarpScan.API<'T>.Create(ws_h, d.temp_storage.warp_scan, 0, d.linear_tid).ExclusiveSum(ws_h, scan_op, !raking_partial, raking_partial, ts_block_aggregate)
//
//                ExclusiveDownsweep.Default h scan_op d !raking_partial
//
//            __syncthreads()
//
//            output := placement_ptr.[0]
//
//            block_aggregate := !ts_block_aggregate
// 
//
//    let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//        (d:_DeviceApi<'T>)
//        (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callbackop:Ref<'T -> 'T>) 
//        = ()       


module BlockScanRaking =
    
    type TemplateParams     = Template._TemplateParams
    type Constants          = Template._Constants
    type TempStorage<'T>    = Template._TempStorage<'T>
    
    type HostApi            = Template._HostApi
    type DeviceApi<'T>      = Template._DeviceApi<'T>

    [<Record>]
    type API<'T> =
        {
            mutable DeviceApi : DeviceApi<'T>
        }

        [<ReflectedDefinition>] static member Create(h:HostApi, temp_storage:TempStorage<'T>, linear_tid) = { DeviceApi = DeviceApi<'T>.Init(h, temp_storage, linear_tid) } 

//        /////// EXCLUSIVE SUM ////////////////////////////////////////////////////////////////////
//        [<ReflectedDefinition>] member this.ExclusiveSum(h, scan_op, input, output, block_aggregate)
//            = ExclusiveSum.WithAggregate h scan_op this.DeviceApi input output block_aggregate
//
//        [<ReflectedDefinition>] member this.ExclusiveSum(h, scan_op, input, output, block_aggregate, block_prefix_callback_op)
//            = ExclusiveSum.WithAggregateAndCallbackOp h scan_op this.DeviceApi input output block_aggregate block_prefix_callback_op
//
//        
//        /////// EXCLUSIVE SCAN ////////////////////////////////////////////////////////////////////
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity, block_aggregate)
//            = ExclusiveScan.WithAggregate h scan_op this.DeviceApi input output identity block_aggregate
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity, block_aggregate, block_prefix_callback_op)
//            = ExclusiveScan.WithAggregateAndCallbackOp h scan_op this.DeviceApi input output identity block_aggregate block_prefix_callback_op
//        
//        /////// EXCLUSIVE SCAN No ID ///////////////////////////////////////////////////////////////
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, block_aggregate)
//            = ExclusiveScan.Identityless.WithAggregate h scan_op this.DeviceApi input output block_aggregate
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, block_aggregate, block_prefix_callback_op)
//            = ExclusiveScan.Identityless.WithAggregateAndCallbackOp h scan_op this.DeviceApi input output block_aggregate block_prefix_callback_op
//

//
//    module ExclusiveScan =
//        type FunctionApi<'T> = Template.ExclusiveScan._FunctionApi<'T>
//
//        let [<ReflectedDefinition>] inline template<'T> (block_threads:int) (memoize:bool) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
//            let h = HostApi.Init(block_threads, memoize)
//
//            let! waggr = (h, scan_op) ||> ExclusiveScan.WithAggregate |> Compiler.DefineFunction
//        
//            return h, {
//                WithAggregate = waggr
//            }}


//    let [<ReflectedDefinition>] api (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//         =
//            {
//                ExclusiveSum    =   ExclusiveSum.api template tf
//                ExclusiveScan   =   ExclusiveScan.api template tf
//            }

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
//        fun (temp_storage:_TempStorage<'T>) (linear_tid:int) =
//            (input:'T) (output:Ref<'T>) (identity:Ref<int>) (scan_op:IScanOp<'T>) (block_aggregate:Ref<int>) =
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
//            
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
//                            fun (raking_ptr:deviceptr<int>) (scan_op:IScanOp<'T>) (raking_partial:int) =
//                                let mutable raking_partial = raking_partial
//                                if unguarded || (((linear_tid * segment_length) + iteration) < block_threads) then
//                                    let addend = raking_ptr.[iteration]
//                                    raking_partial <- (raking_partial, addend) ||> scan_op
//
//    let upsweep =
//        fun (scan_op:IScanOp<'T>) =
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