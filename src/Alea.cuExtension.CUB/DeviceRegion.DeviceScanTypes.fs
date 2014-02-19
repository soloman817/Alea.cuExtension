[<AutoOpen>]
module Alea.cuExtension.CUB.Device.DeviceRegion.DeviceScanTypes

open Alea.CUDA
open Alea.CUDA.Utilities

type LookbackTileStatus =
    | LOOKBACK_TILE_OOB
    | LOOKBACK_TILE_INVALID
    | LOOKBACK_TILE_PARTIAL
    | LOOKBACK_TILE_PREFIX

//    let lookbackTileDescriptor<int> (single_word:bool) =
//        if single_word = ((powerOfTwo sizeof<int>) && (sizeof<int> <= 8)) then
//            let SetPrefix =
//                fun ptr prefix ->
//                    let tile_descriptor = (LookbackTileStatus.LOOKBACK_TILE_PREFIX, prefix)
//                    ptr := tile_descriptor
//        else

[<Record>]
type LookbackTileDescriptor =
    {
        mutable status : LookbackTileStatus
    }
    static member inline SetPrefix(ptr:deviceptr<LookbackTileDescriptor>, prefix:int) = ()
    static member inline SetPartial(ptr:deviceptr<LookbackTileDescriptor>, partial:int) = ()
    static member inline WaitForValid(ptr:deviceptr<LookbackTileDescriptor>, status:int, value:int) = ()
