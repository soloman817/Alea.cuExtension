[<AutoOpen>]
module Alea.cuExtension.CUB.Device.Region

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Block

//
//module HistoRegion =
//    let f() = "histo region"
//
//module RadixSortDowsweepRegion =
//    let f() = "radix sort downsweep region"
//
//module RadixSortUpsweepRegion =
//    let f() = "radix sort upsweep region"
//
//module ReduceRegion =
//    let f() = "reduce region"
//
//module BlockScanRegion =
//
//    type BlockScanRegionPolicy =
//        {
//            BLOCK_THREADS : int
//            ITEMS_PER_THREAD : int
//            LOAD_ALGORITHM  : BlockLoadAlgorithm
//            LOAD_WARP_TIME_SLICING : bool
//            LOAD_MODIFIER   : CacheLoadModifier
//            STORE_ALGORITHM : BlockStoreAlgorithm
//            STORE_WARP_TIME_SLICING : bool           
//            SCAN_ALGORITHM  : BlockScanAlgorithm 
//        }
//
//        static member Create(   block_threads, 
//                                items_per_thread,
//                                load_algorithm,
//                                load_warp_time_slicing,
//                                load_modifier,
//                                store_algorithm,
//                                store_warp_time_slicing,
//                                scan_algorithm  ) =
//            {
//                BLOCK_THREADS = block_threads
//                ITEMS_PER_THREAD = items_per_thread
//                LOAD_ALGORITHM = load_algorithm
//                LOAD_WARP_TIME_SLICING = load_warp_time_slicing
//                LOAD_MODIFIER = load_modifier
//                STORE_ALGORITHM = store_algorithm
//                STORE_WARP_TIME_SLICING = store_warp_time_slicing
//                SCAN_ALGORITHM = scan_algorithm
//            }
//
//    //
//    // Block scan utility methods (first tile)
//    //
//     
//    let scanBlock (scan_op:('T -> 'T -> 'T) option) (identity:'T option) (prefixCallback:'PrefixCallback option) =
//        match scan_op, identity, prefixCallback with
//        | Some scan_op, Some identity, None -> ()
////            fun (items:deviceptr<'T>) (block_aggregate:'T ref) ->
//                //BlockScanT(temp_storage.scan).ExclusiveScan(items, items, identity, scan_op, block_aggregate);
//        | None, Some identity, None -> ()
//        | Some scan_op, None, None -> ()
//        | None, None, None -> ()
//        | Some scan_op, Some identity, Some prefixCallback -> ()
//        | None, Some identity, Some prefixCallback -> ()
//        | Some scan_op, None, Some prefixCallback -> ()
//        | None, None, Some prefixCallback -> ()
//
//module SelectRegion =
//    let f() = "select region"
//    
//module ScanTypes =
//    
//    type LookbackTileStatus =
//        | LOOKBACK_TILE_OOB
//        | LOOKBACK_TILE_INVALID
//        | LOOKBACK_TILE_PARTIAL
//        | LOOKBACK_TILE_PREFIX
//
////    let lookbackTileDescriptor<'T> (single_word:bool) =
////        if single_word = ((powerOfTwo sizeof<'T>) && (sizeof<'T> <= 8)) then
////            let SetPrefix =
////                fun ptr prefix ->
////                    let tile_descriptor = (LookbackTileStatus.LOOKBACK_TILE_PREFIX, prefix)
////                    ptr := tile_descriptor
////        else
//
//    [<Record>]
//    type LookbackTileDescriptor<'T> =
//        {
//            mutable status : LookbackTileStatus
//        }
//        static member inline SetPrefix(ptr:deviceptr<LookbackTileDescriptor<'T>>, prefix:'T) = ()
//        static member inline SetPartial(ptr:deviceptr<LookbackTileDescriptor<'T>>, partial:'T) = ()
//        static member inline WaitForValid(ptr:deviceptr<LookbackTileDescriptor<'T>>, status:int, value:'T) = ()
//
//module Specializations =
//    module HistoRegionGAtomic =
//        let f() = "histo region gatomic"
//
//    module HistoRegionSAtomic =
//        let f() = "histo region satomic"
//
//    module HistoRegionSort =
//        let f() = "histo region sort"