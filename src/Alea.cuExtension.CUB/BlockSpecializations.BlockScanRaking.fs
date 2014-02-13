[<AutoOpen>]
module Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanRaking

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Warp
open Alea.cuExtension.CUB.Block //.RakingLayout
    
//    let BlockRakingLayout = 
//        fun block_threads ->
//            block_threads |> RakingLayout.Constants.Init

//let inline sum_op x y = (^T : (static member (+):^T * ^T -> ^T) (x,y))

let WARPS =
    fun block_threads ->
        (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS

let RAKING_THREADS = 
    fun block_threads ->
        (block_threads |> BlockRakingLayout<'T>.Init).Constants.RAKING_THREADS
            
let SEGMENT_LENGTH = 
    fun block_threads ->
        (block_threads |> BlockRakingLayout<'T>.Init).Constants.RAKING_THREADS

let WARP_SYNCHRONOUS = 
    fun block_threads raking_threads ->
        block_threads = raking_threads


type TemplateParameters =
    {
        BLOCK_THREADS : int
        MEMOIZE : bool        
    }

    static member Init(block_threads, memoize) =
        {
            BLOCK_THREADS = block_threads
            MEMOIZE = memoize
        }


type Constants<'T> =
    {
        WARPS : int
        RAKING_THREADS : int
        SEGMENT_LENGTH : int
        WARP_SYNCHRONOUS : bool
    }
    
    static member Init(block_threads) =
        let BlockRakingLayout = block_threads |> BlockRakingLayout<'T>.Init 
        let raking_threads, segment_length =  BlockRakingLayout.Constants.RAKING_THREADS, BlockRakingLayout.Constants.SEGMENT_LENGTH
        {
            WARPS               = block_threads |> WARPS
            RAKING_THREADS      = raking_threads
            SEGMENT_LENGTH      = segment_length
            WARP_SYNCHRONOUS    = (block_threads, raking_threads) ||> WARP_SYNCHRONOUS
        }


[<Record>]
type TempStorage<'T> =
    {
        warp_scan : Alea.cuExtension.CUB.Warp.Scan.TempStorage<'T> //deviceptr<'T> //Alea.cuExtension.CUB.Warp.Scan.ITempStorage<'T>
        raking_grid : Alea.cuExtension.CUB.Block.RakingLayout.TempStorage<'T> //deviceptr<'T> //Alea.cuExtension.CUB.Block.RakingLayout.ITempStorage<'T>
        block_aggregate : 'T
    }
    
[<Record>]
type ThreadFields<'T> =
    {
        temp_storage : TempStorage<'T>
        linear_tid : int
        cached_segment : 'T[]
    }

    static member Create(c:Constants<'T>, temp_storage:TempStorage<'T>, linear_tid) =
        {
            temp_storage = temp_storage
            linear_tid = linear_tid
            cached_segment = __local__.Array<'T>(c.SEGMENT_LENGTH)
        }
    
        
[<Record>]
type BlockScanRaking<'T> =
    {
        TemplateParameters  : TemplateParameters
        BlockRakingLayout   : BlockRakingLayout<'T>
        Constants           : Constants<'T>
        ThreadScan          : ThreadScan<'T>
        WarpScan            : WarpScan<'T>
        TempStorage         : TempStorage<'T>
        ThreadFields        : ThreadFields<'T>
    }


    member this.Initialize(temp_storage:TempStorage<'T>, linear_tid) =
        this.ThreadFields.temp_storage <- temp_storage
        this.ThreadFields.linear_tid <- linear_tid
        this

    /// Templated reduction
    //template <int ITERATION, typename ScanOp>
    member this.GuardedReduce(raking_ptr:deviceptr<'T>, scan_op:('T -> 'T -> 'T), raking_partial:'T) = //, iteration:bool) =    
        // localize template params & constants
        let UNGUARDED       = this.BlockRakingLayout.Constants.UNGUARDED
        let SEGMENT_LENGTH  = this.BlockRakingLayout.Constants.SEGMENT_LENGTH
        let BLOCK_THREADS = this.TemplateParameters.BLOCK_THREADS
        // localize thread fields
        let linear_tid = this.ThreadFields.linear_tid

        let mutable raking_partial = raking_partial
        for ITERATION = 0 to (SEGMENT_LENGTH - 1) do
            if ((UNGUARDED) || (((linear_tid * SEGMENT_LENGTH) + ITERATION) < BLOCK_THREADS)) then
                let addend = raking_ptr.[ITERATION]
                raking_partial <- (raking_partial, addend) ||> scan_op
        
        raking_partial
    
    
    /// Performs upsweep raking reduction, returning the aggregate
    //template <typename ScanOp>
    member inline this.Upsweep(scan_op:('T -> 'T -> 'T)) =
        // localize template params & constants
        let SEGMENT_LENGTH = this.Constants.SEGMENT_LENGTH
        let MEMOIZE = this.TemplateParameters.MEMOIZE
        // localize thread fields
        let temp_storage = this.ThreadFields.temp_storage
        let linear_tid = this.ThreadFields.linear_tid
        let cached_segment = this.ThreadFields.cached_segment
//
//            let rakingPtr() = 
//                this.BlockRakingLayout.RakingPtr <|| (temp_storage.raking_grid, linear_tid)

        let smem_raking_ptr = this.BlockRakingLayout.RakingPtr <|| (temp_storage.raking_grid, linear_tid) //rakingPtr()
        let mutable raking_ptr = __null()

        if MEMOIZE then
            // Copy data into registers
            //#pragma unroll
            for i = 0 to (SEGMENT_LENGTH - 1) do cached_segment.[i] <- smem_raking_ptr.[i]
            
            raking_ptr <- cached_segment |> __array_to_ptr
        
        else
            raking_ptr <- smem_raking_ptr
        

        let raking_partial = raking_ptr.[0]

        this.GuardedReduce(raking_ptr, scan_op, raking_partial) //, Int2Type<1>())
    


    /// Performs exclusive downsweep raking scan
    //template <typename ScanOp>
    member inline this.ExclusiveDownsweep(scan_op:('T -> 'T -> 'T), raking_partial:'T, ?apply_prefix:bool) =
        let apply_prefix = if apply_prefix.IsSome then apply_prefix.Value else true

        // localize template params & constants
        let MEMOIZE = this.TemplateParameters.MEMOIZE
        let SEGMENT_LENGTH = this.Constants.SEGMENT_LENGTH
        // localize thread fields
        let temp_storage = this.ThreadFields.temp_storage
        let linear_tid = this.ThreadFields.linear_tid
        let cached_segment = this.ThreadFields.cached_segment

        let smem_raking_ptr = this.BlockRakingLayout.RakingPtr <|| (temp_storage.raking_grid, linear_tid)

        let raking_ptr = if (MEMOIZE) then cached_segment |> __array_to_ptr else smem_raking_ptr

        //SEGMENT_LENGTH |> ThreadScanExclusive<'T> <| raking_ptr <| raking_ptr <| scan_op <| raking_partial <| apply_prefix
        this.ThreadScan.Initialize(SEGMENT_LENGTH).Exclusive(raking_ptr, raking_ptr, scan_op, raking_partial, apply_prefix)

        if (MEMOIZE) then
            // Copy data back to smem
            for i = 0 to (SEGMENT_LENGTH - 1) do smem_raking_ptr.[i] <- cached_segment.[i]


    /// Performs inclusive downsweep raking scan
    //template <typename ScanOp>
    member this.InclusiveDownsweep(scan_op:('T -> 'T -> 'T), raking_partial:'T, ?apply_prefix:bool) =
        // localize template params & constants
        let MEMOIZE = this.TemplateParameters.MEMOIZE
        let SEGMENT_LENGTH = this.Constants.SEGMENT_LENGTH
        // localize thread fields
        let temp_storage = this.ThreadFields.temp_storage
        let linear_tid = this.ThreadFields.linear_tid
        let cached_segment = this.ThreadFields.cached_segment
                        
        let smem_raking_ptr = this.BlockRakingLayout.RakingPtr <|| (temp_storage.raking_grid, linear_tid)

        let raking_ptr = if (MEMOIZE) then cached_segment else smem_raking_ptr

        this.ThreadScan.Initialize(SEGMENT_LENGTH).Inclusive(raking_ptr, raking_ptr, scan_op, raking_partial, apply_prefix)

        if (MEMOIZE) then
            // Copy data back to smem
            for i = 0 to (SEGMENT_LENGTH - 1) do smem_raking_ptr.[i] <- cached_segment.[i]


    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    //template <typename ScanOp>
    member this.ExclusiveScan(input:'T, output:Ref<'T>, identity:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) =
        // localize template params & constants
        let RAKING_THREADS = this.Constants.RAKING_THREADS
        let WARP_SYNCHRONOUS = this.Constants.WARP_SYNCHRONOUS
        // localize thread fields
        let temp_storage = this.ThreadFields.temp_storage
        let linear_tid = this.ThreadFields.linear_tid
            

        if (WARP_SYNCHRONOUS) then
            // Short-circuit directly to warp scan
            this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
                input,
                output,
                !identity,
                scan_op,
                block_aggregate)
        else
            // Place thread partial into shared memory raking grid
            let placement_ptr = this.BlockRakingLayout.PlacementPtr <||| (temp_storage.raking_grid, linear_tid, None)
            placement_ptr.[0] <- input

            __syncthreads()

            // Reduce parallelism down to just raking threads
            if (linear_tid < RAKING_THREADS) then
                // Raking upsweep reduction in grid
                let raking_partial = this.Upsweep(scan_op)

                // Exclusive warp synchronous scan
                this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
                    raking_partial,
                    raking_partial |> __obj_to_ref,
                    !identity,
                    scan_op,
                    temp_storage.block_aggregate |> __obj_to_ref)

                // Exclusive raking downsweep scan
                this.ExclusiveDownsweep(scan_op, raking_partial)
            
            __syncthreads()

            // Grab thread prefix from shared memory
            output := placement_ptr.[0]

            // Retrieve block aggregate
            block_aggregate := temp_storage.block_aggregate
        
    
    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    //template <
    //    typename        ScanOp,
    //    typename        BlockPrefixCallbackOp>
    member this.ExclusiveScan(input:'T, output:Ref<'T>, identity:'T, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T>) =
        // localize template params & constants
        let WARP_SYNCHRONOUS = this.Constants.WARP_SYNCHRONOUS
        let RAKING_THREADS = this.Constants.RAKING_THREADS
        // localize thread fields
        let temp_storage = this.ThreadFields.temp_storage
        let linear_tid = this.ThreadFields.linear_tid


        if (WARP_SYNCHRONOUS) then
            // Short-circuit directly to warp scan
            this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
                input,
                output,
                identity,
                scan_op,
                block_aggregate,
                block_prefix_callback_op)        
        else        
            // Place thread partial into shared memory raking grid
            let placement_ptr = this.BlockRakingLayout.PlacementPtr <||| (temp_storage.raking_grid, linear_tid, None)
            placement_ptr.[0] <- input

            __syncthreads()

            // Reduce parallelism down to just raking threads
            if (linear_tid < RAKING_THREADS) then
                // Raking upsweep reduction in grid
                let raking_partial = this.Upsweep(scan_op)

                // Exclusive warp synchronous scan
                this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
                    raking_partial,
                    raking_partial |> __obj_to_ref,
                    identity,
                    scan_op,
                    temp_storage.block_aggregate |> __obj_to_ref,
                    block_prefix_callback_op)

                // Exclusive raking downsweep scan
                this.ExclusiveDownsweep(scan_op, raking_partial)

            __syncthreads()

            // Grab thread prefix from shared memory
            output := placement_ptr.[0]

            // Retrieve block aggregate
            block_aggregate := temp_storage.block_aggregate
        

    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is undefined.
    //template <typename ScanOp>
    member this.ExclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) =
        // localize template params & constants
        let WARP_SYNCHRONOUS = this.Constants.WARP_SYNCHRONOUS
        let RAKING_THREADS = this.Constants.RAKING_THREADS
        // localize thread fields
        let temp_storage = this.ThreadFields.temp_storage
        let linear_tid = this.ThreadFields.linear_tid

        if (WARP_SYNCHRONOUS) then
            // Short-circuit directly to warp scan
            this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
                input,
                output,
                scan_op,
                block_aggregate)
        else
            // Place thread partial into shared memory raking grid
            let placement_ptr = this.BlockRakingLayout.PlacementPtr <||| (temp_storage.raking_grid, linear_tid, None)
            placement_ptr.[0] <- input

            __syncthreads()

            // Reduce parallelism down to just raking threads
            if (linear_tid < RAKING_THREADS) then
                // Raking upsweep reduction in grid
                let raking_partial = this.Upsweep(scan_op)

                // Exclusive warp synchronous scan
                this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
                    raking_partial,
                    raking_partial |> __obj_to_ref,
                    scan_op,
                    temp_storage.block_aggregate |> __obj_to_ref)

                // Exclusive raking downsweep scan
                this.ExclusiveDownsweep(scan_op, raking_partial, (linear_tid <> 0))
            

            __syncthreads()

            // Grab thread prefix from shared memory
            output := placement_ptr.[0]

            // Retrieve block aggregate
            block_aggregate := temp_storage.block_aggregate
        
    
    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    //template <
    //    typename ScanOp,
    //    typename BlockPrefixCallbackOp>
    member this.ExclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T>) =
        // localize template params & constants
        let WARP_SYNCHRONOUS = this.Constants.WARP_SYNCHRONOUS
        let RAKING_THREADS = this.Constants.RAKING_THREADS
        // localize thread fields
        let temp_storage = this.ThreadFields.temp_storage
        let linear_tid = this.ThreadFields.linear_tid
            
        if (WARP_SYNCHRONOUS) then
            // Short-circuit directly to warp scan
            this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
                input,
                output,
                scan_op,
                block_aggregate,
                block_prefix_callback_op)
        else       
            // Place thread partial into shared memory raking grid
            let placement_ptr = this.BlockRakingLayout.PlacementPtr <||| (temp_storage.raking_grid, linear_tid, None)
            placement_ptr.[0] <- input

            __syncthreads()

            // Reduce parallelism down to just raking threads
            if (linear_tid < RAKING_THREADS) then
                // Raking upsweep reduction in grid
                let raking_partial = this.Upsweep(scan_op)

                // Exclusive warp synchronous scan
                this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
                    raking_partial,
                    raking_partial |> __obj_to_ref,
                    scan_op,
                    temp_storage.block_aggregate |> __obj_to_ref,
                    block_prefix_callback_op)

                // Exclusive raking downsweep scan
                this.ExclusiveDownsweep(scan_op, raking_partial)
            

            __syncthreads()

            // Grab thread prefix from shared memory
            output := placement_ptr.[0]

            // Retrieve block aggregate
            block_aggregate := temp_storage.block_aggregate
        
    
    /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    member this.ExclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>) =
        // localize template params & constants
        let WARP_SYNCHRONOUS = this.Constants.WARP_SYNCHRONOUS
        let RAKING_THREADS = this.Constants.RAKING_THREADS
        // localize thread fields
        let temp_storage = this.ThreadFields.temp_storage
        let linear_tid = this.ThreadFields.linear_tid
            
        if (WARP_SYNCHRONOUS) then
            // Short-circuit directly to warp scan
            this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveSum(
                input,
                output,
                block_aggregate)
        
        else        
            // Raking scan
            let inline sum_op x y = (^T : (static member (+):^T * ^T -> ^T) (x,y))

            // Place thread partial into shared memory raking grid
            let placement_ptr = this.BlockRakingLayout.PlacementPtr <||| (temp_storage.raking_grid, linear_tid, None)
            placement_ptr.[0] <- input

            __syncthreads()

            // Reduce parallelism down to just raking threads
            if (linear_tid < RAKING_THREADS) then            
                // Raking upsweep reduction in grid
                let raking_partial = this.Upsweep(sum_op)

                // Exclusive warp synchronous scan
                this.WarpScan.Initialize(temp_storage.warp_scan, 0, linear_tid).ExclusiveSum(
                    raking_partial,
                    raking_partial |> __obj_to_ref,
                    temp_storage.block_aggregate |> __obj_to_ref)

                // Exclusive raking downsweep scan
                this.ExclusiveDownsweep(sum_op, raking_partial)
            
            __syncthreads()

            // Grab thread prefix from shared memory
            output := placement_ptr.[0]

            // Retrieve block aggregate
            block_aggregate := temp_storage.block_aggregate
        
    
    /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    //template <typename BlockPrefixCallbackOp>
    member this.ExclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T>) =
        if (WARP_SYNCHRONOUS) then
            // Short-circuit directly to warp scan
            WarpScan(temp_storage.warp_scan, 0, linear_tid).ExclusiveSum(
                input,
                output,
                block_aggregate,
                block_prefix_callback_op)
        else
            // Raking scan
            let scan_op = (+)

            // Place thread partial into shared memory raking grid
            let placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid)
            placement_ptr.[0] <- input

            __syncthreads()

            // Reduce parallelism down to just raking threads
            if (linear_tid < RAKING_THREADS) then
                // Raking upsweep reduction in grid
                let raking_partial = Upsweep(scan_op)

                // Exclusive warp synchronous scan
                WarpScan(temp_storage.warp_scan, 0, linear_tid).ExclusiveSum(
                    raking_partial,
                    raking_partial,
                    temp_storage.block_aggregate,
                    block_prefix_callback_op)

                // Exclusive raking downsweep scan
                this.ExclusiveDownsweep(scan_op, raking_partial)

            __syncthreads()

            // Grab thread prefix from shared memory
            output := placement_ptr.[0]

            // Retrieve block aggregate
            block_aggregate := !temp_storage.block_aggregate
    

    /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    //template <typename ScanOp>
    member this.InclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) =
        if (WARP_SYNCHRONOUS) then        
            // Short-circuit directly to warp scan
            WarpScan(temp_storage.warp_scan, 0, linear_tid).InclusiveScan(
                input,
                output,
                scan_op,
                block_aggregate)        
        else        
            // Place thread partial into shared memory raking grid
            let placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid)
            placement_ptr.[0] <- input

            __syncthreads()

            // Reduce parallelism down to just raking threads
            if (linear_tid < RAKING_THREADS) then            
                // Raking upsweep reduction in grid
                let raking_partial = Upsweep(scan_op)

                // Exclusive warp synchronous scan
                WarpScan(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
                    raking_partial,
                    raking_partial,
                    scan_op,
                    temp_storage.block_aggregate)

                // Inclusive raking downsweep scan
                this.InclusiveDownsweep(scan_op, raking_partial, (linear_tid != 0))

            __syncthreads()

            // Grab thread prefix from shared memory
            output := placement_ptr.[0]

            // Retrieve block aggregate
            block_aggregate := !temp_storage.block_aggregate
        

    /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    //template <
    //    typename ScanOp,
    //    typename BlockPrefixCallbackOp>
    member this.InclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T>) = 
        if (WARP_SYNCHRONOUS) then
            // Short-circuit directly to warp scan
            WarpScan(temp_storage.warp_scan, 0, linear_tid).InclusiveScan(
                input,
                output,
                scan_op,
                block_aggregate,
                block_prefix_callback_op)
        else        
            // Place thread partial into shared memory raking grid
            let placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid)
            placement_ptr.[0] <- input

            __syncthreads()

            // Reduce parallelism down to just raking threads
            if (linear_tid < RAKING_THREADS) then
                // Raking upsweep reduction in grid
                let raking_partial = Upsweep(scan_op)

                // Warp synchronous scan
                WarpScan(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
                    raking_partial,
                    raking_partial,
                    scan_op,
                    temp_storage.block_aggregate,
                    block_prefix_callback_op)

                // Inclusive raking downsweep scan
                this.InclusiveDownsweep(scan_op, raking_partial)
            
            __syncthreads()

            // Grab thread prefix from shared memory
            output := placement_ptr.[0]

            // Retrieve block aggregate
            block_aggregate := !temp_storage.block_aggregate
        
    
    /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    member this.InclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>) =
        if (WARP_SYNCHRONOUS) then
            // Short-circuit directly to warp scan
            WarpScan(temp_storage.warp_scan, 0, linear_tid).InclusiveSum(
                input,
                output,
                block_aggregate)
        else
            // Raking scan
            let scan_op = (+)

            // Place thread partial into shared memory raking grid
            let placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid)
            placement_ptr.[0] <- input

            __syncthreads()

            // Reduce parallelism down to just raking threads
            if (linear_tid < RAKING_THREADS) then
                // Raking upsweep reduction in grid
                let raking_partial = Upsweep(scan_op)

                // Exclusive warp synchronous scan
                WarpScan(temp_storage.warp_scan, 0, linear_tid).ExclusiveSum(
                    raking_partial,
                    raking_partial,
                    temp_storage.block_aggregate)

                // Inclusive raking downsweep scan
                this.InclusiveDownsweep(scan_op, raking_partial, (linear_tid != 0))
            
            __syncthreads()

            // Grab thread prefix from shared memory
            output := placement_ptr.[0]

            // Retrieve block aggregate
            block_aggregate := !temp_storage.block_aggregate
        
    
    /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    //template <typename BlockPrefixCallbackOp>
    member this.InclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T>) =
        if (WARP_SYNCHRONOUS) then
            // Short-circuit directly to warp scan
            WarpScan(temp_storage.warp_scan, 0, linear_tid).InclusiveSum(
                input,
                output,
                block_aggregate,
                block_prefix_callback_op)        
        else
            // Raking scan
            let scan_op = (+)

            // Place thread partial into shared memory raking grid
            let placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid)
            placement_ptr.[0] <- input

            __syncthreads()

            // Reduce parallelism down to just raking threads
            if (linear_tid < RAKING_THREADS) then
                // Raking upsweep reduction in grid
                let raking_partial = Upsweep(scan_op)

                // Warp synchronous scan
                WarpScan(temp_storage.warp_scan, 0, linear_tid).ExclusiveSum(
                    raking_partial,
                    raking_partial,
                    temp_storage.block_aggregate,
                    block_prefix_callback_op)

                // Inclusive raking downsweep scan
                this.InclusiveDownsweep(scan_op, raking_partial)
            
            __syncthreads()

            // Grab thread prefix from shared memory
            output := placement_ptr.[0]

            // Retrieve block aggregate
            block_aggregate := !temp_storage.block_aggregate
        

    static member Create(block_threads:int, memoize:bool) =
        let tp = (block_threads, memoize) |> TemplateParameters.Init
        let c = block_threads |> Constants.Init
        {
            TemplateParameters = tp
            BlockRakingLayout  = block_threads |> BlockRakingLayout<'T>.Init
            Constants          = c
            ThreadScan         = c.SEGMENT_LENGTH |> ThreadScan<'T>.Create
            WarpScan           = (1, RAKING_THREADS) |> WarpScan<'T>.Create
            TempStorage        = TempStorage<'T>.Init
            ThreadFields       = ThreadFields<'T>.Init      
        }
    

//    let WarpScan = ()
//
//
//    [<Record>]
//    type TempStorage<'T> =
//        
//            warp_scan : deviceptr<'T>
//            raking_grid : deviceptr<'T>
//            block_aggregate : 'T
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
//                            fun (raking_ptr:deviceptr<'T>) (scan_op:('T -> 'T -> 'T)) (raking_partial:'T) ->
//                                let mutable raking_partial = raking_partial
//                                if unguarded || (((linear_tid * segment_length) + iteration) < block_threads) then
//                                    let addend = raking_ptr.[iteration]
//                                    raking_partial <- (raking_partial, addend) ||> scan_op
//
//    let upsweep =
//        fun (scan_op:('T -> 'T -> 'T)) ->
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
//    type ThreadFields<'T> =
//        
//            temp_storage    : TempStorage<'T>
//            linear_tid      : int
//            cached_segment  : deviceptr<'T>
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