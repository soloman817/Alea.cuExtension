[<AutoOpen>]
module Alea.cuExtension.CUB.Device.DeviceRegion.BlockScanRegion

open Alea.CUDA
open Alea.CUDA.Utilities


type BlockScanRegionPolicy =
    {
        BLOCK_THREADS : int
        ITEMS_PER_THREAD : int
        LOAD_ALGORITHM  : BlockLoadAlgorithm
        LOAD_WARP_TIME_SLICING : bool
        LOAD_MODIFIER   : CacheLoadModifier
        STORE_ALGORITHM : BlockStoreAlgorithm
        STORE_WARP_TIME_SLICING : bool           
        SCAN_ALGORITHM  : BlockScanAlgorithm 
    }

    static member Create(   block_threads, 
                            items_per_thread,
                            load_algorithm,
                            load_warp_time_slicing,
                            load_modifier,
                            store_algorithm,
                            store_warp_time_slicing,
                            scan_algorithm  ) =
        {
            BLOCK_THREADS = block_threads
            ITEMS_PER_THREAD = items_per_thread
            LOAD_ALGORITHM = load_algorithm
            LOAD_WARP_TIME_SLICING = load_warp_time_slicing
            LOAD_MODIFIER = load_modifier
            STORE_ALGORITHM = store_algorithm
            STORE_WARP_TIME_SLICING = store_warp_time_slicing
            SCAN_ALGORITHM = scan_algorithm
        }

//
// Block scan utility methods (first tile)
//
     
let scanBlock (scan_op:IScanOp<'T> option) (identity:'T option) (prefixCallback:'PrefixCallback option) =
    match scan_op, identity, prefixCallback with
    | Some scan_op, Some identity, None -> ()
//            fun (items:deviceptr<'T>) (block_aggregate:'T ref) ->
            //BlockScanT(temp_storage.scan).ExclusiveScan(items, items, identity, scan_op, block_aggregate);
    | None, Some identity, None -> ()
    | Some scan_op, None, None -> ()
    | None, None, None -> ()
    | Some scan_op, Some identity, Some prefixCallback -> ()
    | None, Some identity, Some prefixCallback -> ()
    | Some scan_op, None, Some prefixCallback -> ()
    | None, None, Some prefixCallback -> ()