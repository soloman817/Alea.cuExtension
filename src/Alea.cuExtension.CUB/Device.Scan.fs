[<AutoOpen>]
module Alea.cuExtension.CUB.Device.Scan

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Block
open Alea.cuExtension.CUB.Grid
open Region.ScanTypes
open Region.BlockScanRegion




let ScanInitKernel<'T>() =
    let TILE_STATUS_PADDING = CUB_PTX_WARP_THREADS
    <@ fun (grid_queue:GridQueue) (d_tile_status:deviceptr<LookbackTileDescriptor<'T>>) (num_tiles:int) ->
        // Reset queue descriptor
        if (blockIdx.x = 0) && (threadIdx.x = 0) then grid_queue.FillAndResetDrain(num_tiles) <| Locale.Device

        //Initializae tile status
        let tile_offset = (blockIdx.x * blockDim.x) + threadIdx.x
        
        if tile_offset < num_tiles then d_tile_status.[TILE_STATUS_PADDING + tile_offset].status <- LOOKBACK_TILE_INVALID

        if (blockIdx.x = 0) && (threadIdx.x < TILE_STATUS_PADDING) then d_tile_status.[threadIdx.x].status <- LOOKBACK_TILE_OOB
    @>


let ScanRegionKernel<'T>() =
    let TILE_STATUS_PADDING = CUB_PTX_WARP_THREADS
    fun (blockScanRegionT:BlockScanRegionPolicy) ->
        <@ fun (d_in:InputIterator<'T>) (d_out:OutputIterator<'T>) (d_tile_status:deviceptr<LookbackTileDescriptor<'T>>) (scan_op:('T -> 'T -> 'T)) (identity:'T) (num_items:Offset) (queue:GridQueue) ->
            // Shared memory for BlockScanRegion
            let temp_storage = ()
            ()
            // Process Tiles
//    BlockScanRegionT(temp_storage, d_in, d_out, scan_op, identity).ConsumeRegion(
//        num_items,
//        queue,
//        d_tile_status + TILE_STATUS_PADDING);
        @>






type Policy350<'T> =
    {
        NOMINAL_4B_ITEMS_PER_THREAD : int
        ITEMS_PER_THREAD : int
        ScanRegionPolicy : BlockScanRegionPolicy
    }

    static member Default() =
        let nominal_4B_items_per_thread = 16
        let items_per_thread = CUB_MIN nominal_4B_items_per_thread (CUB_MAX 1 (nominal_4B_items_per_thread * 4 / sizeof<'T>))
        {
            NOMINAL_4B_ITEMS_PER_THREAD = nominal_4B_items_per_thread
            ITEMS_PER_THREAD = items_per_thread
            ScanRegionPolicy = 
                BlockScanRegionPolicy.Create(   
                    128,
                    items_per_thread,
                    BlockLoadAlgorithm.BLOCK_LOAD_DIRECT,
                    false,
                    CacheLoadModifier.LOAD_LDG,
                    BlockStoreAlgorithm.BLOCK_STORE_WARP_TRANSPOSE,
                    true,
                    BlockScanAlgorithm.BLOCK_SCAN_RAKING_MEMOIZE )
        }


type Policy300<'T> =
    {
        NOMINAL_4B_ITEMS_PER_THREAD : int
        ITEMS_PER_THREAD : int
        ScanRegionPolicy : BlockScanRegionPolicy
    }

    static member Default() =
        let nominal_4B_items_per_thread = 9
        let items_per_thread = CUB_MIN nominal_4B_items_per_thread (CUB_MAX 1 (nominal_4B_items_per_thread * 4 / sizeof<'T>))
        {
            NOMINAL_4B_ITEMS_PER_THREAD = nominal_4B_items_per_thread
            ITEMS_PER_THREAD = items_per_thread
            ScanRegionPolicy = 
                BlockScanRegionPolicy.Create(   
                    256,
                    items_per_thread,
                    BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE,
                    false,
                    CacheLoadModifier.LOAD_DEFAULT,
                    BlockStoreAlgorithm.BLOCK_STORE_WARP_TRANSPOSE,
                    false,
                    BlockScanAlgorithm.BLOCK_SCAN_RAKING_MEMOIZE )
        }


type Policy200<'T> =
    {
        NOMINAL_4B_ITEMS_PER_THREAD : int
        ITEMS_PER_THREAD : int
        ScanRegionPolicy : BlockScanRegionPolicy
    }

    static member Default() =
        let nominal_4B_items_per_thread = 15
        let items_per_thread = CUB_MIN nominal_4B_items_per_thread (CUB_MAX 1 (nominal_4B_items_per_thread * 4 / sizeof<'T>))
        {
            NOMINAL_4B_ITEMS_PER_THREAD = nominal_4B_items_per_thread
            ITEMS_PER_THREAD = items_per_thread
            ScanRegionPolicy = 
                BlockScanRegionPolicy.Create(   
                    128,
                    items_per_thread,
                    BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE,
                    false,
                    CacheLoadModifier.LOAD_DEFAULT,
                    BlockStoreAlgorithm.BLOCK_STORE_WARP_TRANSPOSE,
                    false,
                    BlockScanAlgorithm.BLOCK_SCAN_RAKING_MEMOIZE )
        }


type Policy130<'T> =
    {
        NOMINAL_4B_ITEMS_PER_THREAD : int
        ITEMS_PER_THREAD : int
        ScanRegionPolicy : BlockScanRegionPolicy
    }

    static member Default() =
        let nominal_4B_items_per_thread = 19
        let items_per_thread = CUB_MIN nominal_4B_items_per_thread (CUB_MAX 1 (nominal_4B_items_per_thread * 4 / sizeof<'T>))
        {
            NOMINAL_4B_ITEMS_PER_THREAD = nominal_4B_items_per_thread
            ITEMS_PER_THREAD = items_per_thread
            ScanRegionPolicy = 
                BlockScanRegionPolicy.Create(   
                    64,
                    items_per_thread,
                    BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE,
                    true,
                    CacheLoadModifier.LOAD_DEFAULT,
                    BlockStoreAlgorithm.BLOCK_STORE_WARP_TRANSPOSE,
                    true,
                    BlockScanAlgorithm.BLOCK_SCAN_RAKING_MEMOIZE )
        }


type Policy100<'T> =
    {
        NOMINAL_4B_ITEMS_PER_THREAD : int
        ITEMS_PER_THREAD : int
        ScanRegionPolicy : BlockScanRegionPolicy
    }

    static member Default() =
        let nominal_4B_items_per_thread = 19
        let items_per_thread = CUB_MIN nominal_4B_items_per_thread (CUB_MAX 1 (nominal_4B_items_per_thread * 4 / sizeof<'T>))
        {
            NOMINAL_4B_ITEMS_PER_THREAD = nominal_4B_items_per_thread
            ITEMS_PER_THREAD = items_per_thread
            ScanRegionPolicy = 
                BlockScanRegionPolicy.Create(   
                    128,
                    items_per_thread,
                    BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE,
                    true,
                    CacheLoadModifier.LOAD_DEFAULT,
                    BlockStoreAlgorithm.BLOCK_STORE_WARP_TRANSPOSE,
                    true,
                    BlockScanAlgorithm.BLOCK_SCAN_RAKING )
        }


type PtxPolicy<'T> =
    | Policy350 of Policy350<'T>
    | Policy300 of Policy300<'T>
    | Policy200 of Policy200<'T>
    | Policy130 of Policy130<'T>
    | Policy100 of Policy100<'T>


type KernelConfig =
    {
        block_threads : int
        items_per_thread : int
        load_policy : BlockLoadAlgorithm
        store_policy : BlockStoreAlgorithm
        scan_algorithm : BlockScanAlgorithm
    }

    static member Init(blockScanRegionPolicy:BlockScanRegionPolicy) =
        {
            block_threads = blockScanRegionPolicy.BLOCK_THREADS
            items_per_thread = blockScanRegionPolicy.ITEMS_PER_THREAD
            load_policy = blockScanRegionPolicy.LOAD_ALGORITHM
            store_policy = blockScanRegionPolicy.STORE_ALGORITHM
            scan_algorithm = blockScanRegionPolicy.SCAN_ALGORITHM
        }

    member this.Print() = 
        printfn "%A, %A, %A, %A, %A"
            this.block_threads
            this.items_per_thread
            this.load_policy
            this.store_policy
            this.scan_algorithm



let dispatch (d_temp_storage:deviceptr<'T>) (temp_storage_bytes:int) (d_in:InputIterator<'T>) (d_out:OutputIterator<'T>) (scan_op:('T -> 'T -> 'T)) (identity:'T) (num_items:int) (stream:CUstream) (debug_synchronous:bool) = 
    //scan_region_kernel scan_grid_size scan_region_config.block_threads 0 stream
    // -> d_in, d_out, d_tile_status, scan_op, identity, num_items, queue
    ()

let deviceScanDispatch tile_status_padding init_kernel_threads = ()
    

let inline exclusiveSum() =
    fun (d_temp_storage:deviceptr<'T>) (temp_storage_bytes:int) (d_in:InputIterator<'T>) (d_out:OutputIterator<'T>) (num_items:int) (stream:CUstream option) (debug_synchronous:bool option) ->
        let stream = if stream.IsNone then 0n else stream.Value
        let debug_synchronous = if debug_synchronous.IsNone then false else debug_synchronous.Value
        dispatch
        <| d_temp_storage
        <| temp_storage_bytes
        <| d_in
        <| d_out
        <| ( + )
        <| 0G
        <| num_items
        <| stream
        <| debug_synchronous



type DeviceScanDispatch<'T> =
    {
        TILE_STATUS_PADDING : int
        INIT_KERNEL_THREADS : int
        PtxPolicy : PtxPolicy<'T>
        scan_region_config : KernelConfig
    }