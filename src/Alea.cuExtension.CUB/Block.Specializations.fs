module Alea.cuExtension.CUB.Block.Specializations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

open Alea.cuExtension.CUB.Warp

module HistogramAtomic =
    let f() = "histogram atomic"

module HistogramSort =
    let f() = "histogram sort"

module BlockReduceRaking =

    let RAKING_THREADS =
        fun block_threads ->
            (block_threads, 1) ||> RakingLayout.RAKING_THREADS
    
    let SEGMENT_LENGTH =
        fun block_threads ->
            (block_threads, 1) ||> RakingLayout.SEGMENT_LENGTH

    let WARP_SYNCHRONOUS =
        fun block_threads ->
            let RAKING_THREADS = block_threads |> RAKING_THREADS
            (RAKING_THREADS = block_threads)

    let WARP_SYNCHRONOUS_UNGUARDED =
        fun block_threads ->
            let RAKING_THREADS = block_threads |> RAKING_THREADS
            ((RAKING_THREADS &&& (RAKING_THREADS - 1)) = 0)

    let RAKING_UNGUARDED =
        fun block_threads ->
        (block_threads, 1) ||> RakingLayout.UNGUARDED


    [<Record>]
    type TempStorage<'T> =
        {
            warp_scan : deviceptr<'T>
            raking_grid : deviceptr<'T>
        }


    [<Record>]
    type ThreadFields<'T> =
        {
            temp_storage : deviceptr<'T>
            linear_tid : int
        }

        static member Init(temp_storage, linear_tid) =
            {
                temp_storage = temp_storage
                linear_tid = linear_tid
            }

//    
//    let rakingReduction (block_threads:int) = 
//        let RAKING_UNGUARDED = block_threads |> RAKING_UNGUARDED
//
//        fun (full_tile:bool) (iteration:int) ->
//            fun (reductionOp:'T -> 'T -> 'R) (raking_segment:deviceptr<'T>) (partial:'T) (num_valid:int) ->
//                
//                if (full_tile && RAKING_UNGUARDED) || ((linear_tid * SEGMENT_LENGTH) + iteration < num_valid) then
//                    let addend = raking_segment.[iteration]
//                    partial <- (partial, addend) ||> reduction_op


    [<Record>]
    type BlockReduceRaking<'T> =
        {
            BLOCK_THREADS : int
        }


module BlockReduceWarpReduction =
    let f() = "reduce warp reduction"

module BlockScanRaking =
    open RakingLayout


    let BlockRakingLayout = 
        fun block_threads ->
            block_threads |> RakingLayout.Constants.Init

    let WARPS =
        fun block_threads ->
            (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS

    let RAKING_THREADS = 
        fun block_threads ->
            (block_threads |> BlockRakingLayout).RAKING_THREADS
            
    let SEGMENT_LENGTH = 
        fun block_threads ->
            (block_threads |> BlockRakingLayout).RAKING_THREADS

    let WARP_SYNCHRONOUS = 
        fun block_threads raking_threads ->
            block_threads = raking_threads


    type TemplateParameters =
        {
            BLOCK_THREADS : int
            MEMOIZE : bool        
        }

        //static member Foo(block_threads, memoize) =
        //member this.ExclusiveScan(input, output) =
              


    type Constants =
        {
            WARPS : int
            RAKING_THREADS : int
            SEGMENT_LENGTH : int
            WARP_SYNCHRONOUS : bool
        }
    
        static member Init(block_threads) =
            let BlockRakingLayout = block_threads |> RakingLayout.BlockRakingLayout.Init 
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
            warp_scan : Alea.cuExtension.CUB.Warp.Scan.ITempStorage<'T>
            raking_grid : Alea.cuExtension.CUB.Block.RakingLayout.ITempStorage<'T>
            block_aggregate : 'T
        }
    
    [<Record>]
    type ThreadFields<'T> =
        {
            temp_storage : TempStorage<'T>
            linear_tid : int
            cached_segment : 'T[]
        }
    
        
    [<Record>]
    type BlockScanRaking<'T> =
        {
            BlockRakingLayout : BlockRakingLayout
            Constants : Constants
            WarpScan : WarpScan<'T>
            ThreadFields : ThreadFields<'T>
        }

        /// Templated reduction
        //template <int ITERATION, typename ScanOp>
//        member this.GuardedReduce(raking_ptr:deviceptr<'T>, scan_op:('T -> 'T -> 'T), raking_partial:'T, iteration:bool) =    
//            let UNGUARDED       = this.BlockRakingLayout.Constants.UNGUARDED
//            let SEGMENT_LENGTH  = this.BlockRakingLayout.Constants.SEGMENT_LENGTH
//
//            let mutable raking_partial = raking_partial
//            if ((UNGUARDED) || (((linear_tid * SEGMENT_LENGTH) + ITERATION) < BLOCK_THREADS)) then
//                let addend = raking_ptr.[ITERATION]
//                raking_partial <- (raking_partial, addend) ||> scan_op
//        
//            GuardedReduce(raking_ptr, scan_op, raking_partial, Int2Type<ITERATION + 1>())
    
    
        /// Templated reduction (base case)
        //template <typename ScanOp>
        //member this.GuardedReduce(raking_ptr:deviceptr<'T>, scan_op:('T -> 'T -> 'T), raking_partial:'T)// Int2Type<SEGMENT_LENGTH>    iteration)
        //    
        //    return raking_partial
    
    
        /// Performs upsweep raking reduction, returning the aggregate
        //template <typename ScanOp>
        member this.Upsweep(scan_op:('T -> 'T -> 'T)) =
            let SEGMENT_LENGTH = this.Constants.SEGMENT_LENGTH
            let MEMOIZE = this.TemplateParameters.MEMOIZE


            let smem_raking_ptr = this.BlockRakingLayout.RakingPtr(temp_storage.raking_grid, linear_tid)
            let raking_ptr = __null()

            if MEMOIZE then
                // Copy data into registers
                //#pragma unroll
                for i = 0 to (SEGMENT_LENGTH - 1) do
                    cached_segment.[i] <- smem_raking_ptr[i]
            
                raking_ptr.[0] <- cached_segment
        
            else
                raking_ptr.[0] <- smem_raking_ptr
        

            let raking_partial = raking_ptr.[0]

            this.GuardedReduce(raking_ptr, scan_op, raking_partial, Int2Type<1>())
    


        /// Performs exclusive downsweep raking scan
        //template <typename ScanOp>
        member this.ExclusiveDownsweep(scan_op:('T -> 'T -> 'T), raking_partial:'T, ?apply_prefix:bool) =
            let smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid)

            let raking_ptr = if (MEMOIZE) then cached_segment else smem_raking_ptr

            ThreadScanExclusive<SEGMENT_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial, apply_prefix)

            if (MEMOIZE) then
                // Copy data back to smem
                for i = 0 to (SEGMENT_LENGTH - 1) do
                    smem_raking_ptr.[i] <- cached_segment.[i]


        /// Performs inclusive downsweep raking scan
        //template <typename ScanOp>
        member this.InclusiveDownsweep(scan_op:('T -> 'T -> 'T), raking_partial:'T, ?apply_prefix:bool) =
            let smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid)

            let raking_ptr = if (MEMOIZE) then cached_segment else smem_raking_ptr

            ThreadScanInclusive<SEGMENT_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial, apply_prefix)

            if (MEMOIZE) then
                // Copy data back to smem
                //#pragma unroll
                for i = 0 to (SEGMENT_LENGTH - 1) do
                    smem_raking_ptr.[i] <- cached_segment[i]


        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        //template <typename ScanOp>
        member this.ExclusiveScan(input:'T, output:Ref<'T>, identity:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) =
            if (WARP_SYNCHRONOUS) then
                // Short-circuit directly to warp scan
                WarpScan(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
                    input,
                    output,
                    identity,
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
                        identity,
                        scan_op,
                        temp_storage.block_aggregate)

                    // Exclusive raking downsweep scan
                    this.ExclusiveDownsweep(scan_op, raking_partial)
            
                __syncthreads()

                // Grab thread prefix from shared memory
                output := placement_ptr.[0]

                // Retrieve block aggregate
                block_aggregate := !temp_storage.block_aggregate
        
    
        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        //template <
        //    typename        ScanOp,
        //    typename        BlockPrefixCallbackOp>
        member this.ExclusiveScan(input:'T, output:Ref<'T>, identity:'T, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallvbackOp>) =
            if (WARP_SYNCHRONOUS) then
                // Short-circuit directly to warp scan
                WarpScan(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
                    input,
                    output,
                    identity,
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

                    // Exclusive warp synchronous scan
                    WarpScan(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
                        raking_partial,
                        raking_partial,
                        identity,
                        scan_op,
                        temp_storage.block_aggregate,
                        block_prefix_callback_op)

                    // Exclusive raking downsweep scan
                    this.ExclusiveDownsweep(scan_op, raking_partial)

                __syncthreads()

                // Grab thread prefix from shared memory
                output := placement_ptr.[0]

                // Retrieve block aggregate
                block_aggregate := !temp_storage.block_aggregate
        

        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is undefined.
        //template <typename ScanOp>
        member this.ExclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) =
            if (WARP_SYNCHRONOUS) then
                // Short-circuit directly to warp scan
                WarpScan(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
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
                    lelet raking_partial = Upsweep(scan_op)

                    // Exclusive warp synchronous scan
                    WarpScan(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
                        raking_partial,
                        raking_partial,
                        scan_op,
                        temp_storage.block_aggregate)

                    // Exclusive raking downsweep scan
                    this.ExclusiveDownsweep(scan_op, raking_partial, (linear_tid != 0))
            

                __syncthreads()

                // Grab thread prefix from shared memory
                output := placement_ptr.[0]

                // Retrieve block aggregate
                block_aggregate := !temp_storage.block_aggregate
        
    
        /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        //template <
        //    typename ScanOp,
        //    typename BlockPrefixCallbackOp>
        member this.ExclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) =
            if (WARP_SYNCHRONOUS) then
               // Short-circuit directly to warp scan
                WarpScan(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
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
                    lelet raking_partial = Upsweep(scan_op)

                    // Exclusive warp synchronous scan
                    WarpScan(temp_storage.warp_scan, 0, linear_tid).ExclusiveScan(
                        raking_partial,
                        raking_partial,
                        scan_op,
                        temp_storage.block_aggregate,
                        block_prefix_callback_op)

                    // Exclusive raking downsweep scan
                    this.ExclusiveDownsweep(scan_op, raking_partial)
            

                __syncthreads()

                // Grab thread prefix from shared memory
                output := placement_ptr.[0]

                // Retrieve block aggregate
                block_aggregate := !temp_storage.block_aggregate
        
    
        /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        member this.ExclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>) =
            if (WARP_SYNCHRONOUS) then
                // Short-circuit directly to warp scan
                WarpScan(temp_storage.warp_scan, 0, linear_tid).ExclusiveSum(
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

                    // Exclusive raking downsweep scan
                    this.ExclusiveDownsweep(scan_op, raking_partial)
            
                __syncthreads()

                // Grab thread prefix from shared memory
                output := placement_ptr.[0]

                // Retrieve block aggregate
                block_aggregate := !temp_storage.block_aggregate
        
    
        /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
        //template <typename BlockPrefixCallbackOp>
        member this.ExclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) =
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
        member this.InclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = 
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
        member this.InclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) =
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




module BlockScanWarpScans =

    let WARPS =
        fun block_threads ->
            (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS    

    [<Record>]
    type TempStorage<'T> =
        {
            warp_scan : deviceptr<'T>
            warp_aggregates : deviceptr<'T>
            mutable block_prefix : 'T
        }


        static member Create(warp_scan, warp_aggregates, block_prefix) =
            {
                warp_scan = warp_scan
                warp_aggregates = warp_aggregates
                block_prefix = block_prefix
            }



    [<Record>]
    type ThreadFields<'T> =
        {
            temp_storage : TempStorage<'T>
            linear_tid : int
            warp_id : int
            lane_id : int
        }


        static member Create(temp_storage, linear_tid, warp_id, lane_id) =
            {
                temp_storage = temp_storage
                linear_tid = linear_tid
                warp_id = warp_id
                lane_id = lane_id
            }


    let applyWarpAggregates block_threads = 
        let WARPS = block_threads |> WARPS
        fun (partial:Ref<'T>) (scan_op:('T -> 'T -> 'T)) (warp_aggregate:'T) (block_aggregate:Ref<'T>) (lane_valid:bool option) ->
            let lane_valid = if lane_valid.IsSome then lane_valid.Value else true
            fun temp_storage warp_id ->
                temp_storage.warp_aggregates.[warp_id] <- warp_aggregate

                __syncthreads()

                block_aggregate := temp_storage.warp_aggregates.[0]

                for WARP = 1 to WARPS - 1 do
                    if warp_id = WARP then
                        partial := if lane_valid then (!block_aggregate, !partial) ||> scan_op else !block_aggregate
                    block_aggregate := (!block_aggregate, temp_storage.warp_aggregates.[WARP]) ||> scan_op

    
            
     
    [<Record>]
    type BlockScanWarpScans<'T> =
        {
            WarpScan        : WarpScan<'T>
            TempStorage    : TempStorage<'T>
            ThreadFields    : ThreadFields<'T>
        }

        member this.ApplyWarpAggregates(partial:Ref<'T>, scan_op:('T -> 'T -> 'T), warp_aggregate:'T, block_aggregate:Ref<'T>, ?lane_valid:bool) = ()
        
        member this.ExclusiveScan(input:'T, output:Ref<'T>, identity:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) = 
            let warp_aggregate = __null() |> __ptr_to_ref
            let temp_storage = this.TempStorage.warp_scan
            WarpScan<'T>.Create(temp_storage |> __ptr_to_ref, this.warp_id, this.lane_id).ExclusiveScan(input, output, !identity, scan_op, warp_aggregate)
            this.ApplyWarpAggregates(output, scan_op, !warp_aggregate, block_aggregate)

        member this.ExclusiveScan(input:'T, output:Ref<'T>, identity:'T, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = 
            let warp_id = this.ThreadFields.warp_id
            let lane_id = this.ThreadFields.lane_id
            let temp_storage = this.ThreadFields.temp_storage
            let identity = identity |> __obj_to_ref
            
            this.ExclusiveScan(input, output, identity, scan_op, block_aggregate)
            if warp_id = 0 then
                let block_prefix = !block_aggregate |> !block_prefix_callback_op 
                if lane_id = 0 then temp_storage.block_prefix <- block_prefix

            __syncthreads()

            output := (temp_storage.block_prefix, !output) ||> scan_op

        member this.ExclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) = 
            let warp_aggregate = __null() |> __ptr_to_ref
            this.WarpScan
        
        member this.ExclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = ()

        member this.ExclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>) = ()
        member this.ExclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = ()

        member this.InclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) = ()
        member this.InclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = ()

        member this.InclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>) = ()
        member this.InclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'BlockPrefixCallbackOp>) = ()


//        static member Create(temp_storage, linear_tid) =
//            
//                ThreadFields = ThreadFields.Create(
//                                                    temp_storage,
//                                                    linear_tid,
//                                                    )
//            