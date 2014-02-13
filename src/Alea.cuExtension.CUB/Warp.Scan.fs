﻿[<AutoOpen>]
module Alea.cuExtension.CUB.Warp.Scan

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Common
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Warp.WarpSpecializations




let POW_OF_TWO =
    fun logical_warp_threads ->
        ((logical_warp_threads &&& (logical_warp_threads - 1)) = 0)

let InternalWarpScan =
    fun logical_warps logical_warp_threads ->
        if (CUB_PTX_VERSION >= 300) && ((logical_warps = 1) || (logical_warps |> POW_OF_TWO)) then
            (logical_warps, logical_warp_threads) |> WarpScanShfl.Create
        else
            (logical_warps, logical_warp_threads) |> WarpScanShfl.Create //WarpScanSmem


let warp_id =
    fun logical_warps logical_warp_threads ->
        if logical_warps = 1 then 0 else threadIdx.x / logical_warp_threads

let lane_id =
    fun logical_warps logical_warp_threads ->
        if ((logical_warps = 1) || (logical_warp_threads = CUB_PTX_WARP_THREADS)) then __ptx__.LaneId() else threadIdx.x % logical_warp_threads


let [<ReflectedDefinition>] privateStorage<'T>() = __shared__.Extern<'T>()


type [<RequireQualifiedAccess>] ITempStorage<'T> = abstract temp_storage : deviceptr<'T>
let tempStorage<'T>(grid_elements:int)() = { new ITempStorage<'T> with member this.temp_storage = __shared__.Array<'T>(grid_elements) |> __array_to_ptr }


type WarpScanAPI<'T> =
    abstract POW_OF_TWO : bool
    abstract InternalScan : int
    abstract temp_storage : Ref<'T>
    abstract warp_id : int
    abstract lane_id : int
    abstract PrivateStorage : unit -> Ref<'T>


let api<'T> (logical_warps:int) (logical_warp_threads:int) =
    { new WarpScanAPI<'T> with
        member this.POW_OF_TWO = logical_warp_threads |> POW_OF_TWO
        member this.InternalScan = 0
        member this.temp_storage = privateStorage<'T>() |> __ptr_to_ref
        member this.warp_id = (logical_warps, logical_warp_threads) ||> warp_id
        member this.lane_id = (logical_warps, logical_warp_threads) ||> lane_id
        member this.PrivateStorage() = privateStorage<'T>() |> __ptr_to_ref
    }

type TemplateParameters =
    {
        LOGICAL_WARPS : int
        LOGICAL_WARP_THREADS : int
    }

    static member Init() =
        {
            LOGICAL_WARPS = 1
            LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS        
        }

    static member Init(?logical_warps:int, ?logical_warp_threads:int) =
        match logical_warps, logical_warp_threads with
        | None, None ->                                     { LOGICAL_WARPS = 1;                LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS     }
        | Some logical_warps, None ->                       { LOGICAL_WARPS = logical_warps;    LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS     }
        | None, Some logical_warp_threads ->                { LOGICAL_WARPS = 1;                LOGICAL_WARP_THREADS = logical_warp_threads     }
        | Some logical_warps, Some logical_warp_threads ->  { LOGICAL_WARPS = logical_warps;    LOGICAL_WARP_THREADS    = logical_warp_threads  }


type Constants =
    {
        POW_OF_TWO : bool
    }

    static member Init(logical_warp_threads) =
        {
            POW_OF_TWO = logical_warp_threads |> POW_OF_TWO 
        }


type TempStorage<'T> =
    {
        x : int
    }




[<Record>]
type ThreadFields<'T> =
    {
        mutable temp_storage : deviceptr<'T>
        mutable warp_id : int
        mutable lane_id : int
    }

    member this.Set(temp_storage:deviceptr<'T>, warp_id:int, lane_id:int) =
        this.temp_storage <- temp_storage
        this.warp_id <- warp_id
        this.lane_id <- lane_id

    static member Init(templateParameters:TemplateParameters) =
        let logical_warps = templateParameters.LOGICAL_WARPS
        let logical_warp_threads = templateParameters.LOGICAL_WARP_THREADS
        {
            temp_storage = __null()
            warp_id = (logical_warps, logical_warp_threads) ||> warp_id
            lane_id = (logical_warps, logical_warp_threads) ||> lane_id
        }

    static member Init(templateParameters:TemplateParameters, temp_storage:deviceptr<'T>) =
        let logical_warps = templateParameters.LOGICAL_WARPS
        let logical_warp_threads = templateParameters.LOGICAL_WARP_THREADS
        {
            temp_storage = temp_storage
            warp_id = (logical_warps, logical_warp_threads) ||> warp_id
            lane_id = (logical_warps, logical_warp_threads) ||> lane_id
        }

    static member Init(templateParameters:TemplateParameters, temp_storage:deviceptr<'T>, warp_id:int, lane_id:int) =
        let logical_warps = templateParameters.LOGICAL_WARPS
        let logical_warp_threads = templateParameters.LOGICAL_WARP_THREADS
        {
            temp_storage = temp_storage
            warp_id = warp_id
            lane_id = lane_id
        }

[<Record>]
type WarpScan<'T> =
    {
        TemplateParameters  : TemplateParameters
        Constants           : Constants
        ThreadFields        : ThreadFields<'T>
    }

    member this.InclusiveSum(input:'T, output:Ref<'T>) = ()
    member this.InclusiveSum(input:'T, output:Ref<'T>, warp_aggregate:Ref<'T>) = ()

    member this.InclusiveSum(input:'T, output:Ref<'T>, warp_aggregate:Ref<'T>, warp_prefix_op:Ref<'T>) = ()


    member private this.ExclusiveSum(input:'T, output:Ref<'T>, is_primitive:bool) = ()
    member private this.ExclusiveSum(input:'T, output:Ref<'T>, warp_aggregate:Ref<'T>, is_primitive:bool) = ()
    member private this.ExclusiveSum(input:'T, output:Ref<'T>, warp_aggregate:Ref<'T>, warp_prefix_op:Ref<'T>, is_primitive:bool) = ()


    member this.ExclusiveSum(input:'T, output:Ref<'T>) = ()
    member this.ExclusiveSum(input:'T, output:Ref<'T>, warp_aggregate:Ref<'T>) = ()
    member this.ExclusiveSum(input:'T, output:Ref<'T>, warp_aggregate:Ref<'T>, warp_prefix_op:Ref<'T>) = ()

    member this.InclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T)) = ()
    member this.InclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), warp_aggregate:Ref<'T>) = ()
    member this.InclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), warp_aggregate:Ref<'T>, warp_prefix_op:Ref<'T>) = ()

    member this.ExclusiveScan(input:'T, output:Ref<'T>, identity:'T, scan_op:('T -> 'T -> 'T)) = ()
    member inline this.ExclusiveScan(input:'T, output:Ref<'T>, identity:'T, scan_op:('T -> 'T -> 'T), warp_aggregate:Ref<'T>) = ()
    member this.ExclusiveScan(input:'T, output:Ref<'T>, identity:'T, scan_op:('T -> 'T -> 'T), warp_aggregate:Ref<'T>, warp_prefix_op:Ref<'T>) = ()

    member this.ExclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T)) = ()
    member this.ExclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), warp_aggregate:Ref<'T>) = ()
    member this.ExclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), warp_aggregate:Ref<'T>, warp_prefix_op:Ref<'T>) = ()


    member this.Initialize(temp_storage:deviceptr<'T>, warp_id:int, lane_id:int) = 
        (temp_storage, warp_id, lane_id) |> this.ThreadFields.Set
        this

    static member Create(templateParameters:TemplateParameters, temp_storage:deviceptr<'T>) = 
        let logical_warps = templateParameters.LOGICAL_WARPS
        let logical_warp_threads = templateParameters.LOGICAL_WARP_THREADS
        let constants = logical_warp_threads |> Constants.Init
        let threadFields = (templateParameters, temp_storage) |> ThreadFields.Init
        {
            TemplateParameters = templateParameters
            Constants = constants
            ThreadFields = threadFields
        }

    static member Create(templateParameters:TemplateParameters, constants:Constants, threadFields:ThreadFields<'T>) = 
        {
            TemplateParameters = templateParameters
            Constants = constants
            ThreadFields = threadFields
        }
    
//    static member Create(warp_id:int, lane_id:int) = 
//        {
//            temp_storage = privateStorage<'T>()
//            warp_id = warp_id
//            lane_id = lane_id
//        }

//    static member Create(temp_storage:Ref<'TempStorage>, warp_id:int, lane_id:int) =
//        {
//            temp_storage = temp_storage |> __ref_to_ptr
//            warp_id = warp_id
//            lane_id = lane_id
//        }

    static member Create(logical_warps:int, logical_warp_threads:int) =
        let tp = (logical_warps, logical_warp_threads) |> TemplateParameters.Init
        {
            TemplateParameters = tp
            Constants = logical_warp_threads |> Constants.Init
            ThreadFields = tp |> ThreadFields<'T>.Init
        }

//
//[<Record>]
//type TempStorage<'T> =
//    {
//        temp_storage : deviceptr<'T>
//    }
//
//    static member Create() = 
//        {
//            temp_storage = privateStorage()
//        }
//
//
//[<Record>]
//type ThreadFields<'T> =
//    {
//        temp_storage : TempStorage<'T>
//        warp_id : int
//        lane_id : int
//    }
//
//    static member Create(temp_storage, warp_id, lane_id) =
//        {
//            temp_storage = temp_storage
//            warp_id = warp_id
//            lane_id = lane_id
//        }
//
//    static member Create(logical_warps, logical_warp_threads) =
//        {
//            temp_storage = TempStorage<'T>.Create()
//            warp_id = (logical_warps, logical_warp_threads) ||> warp_id
//            lane_id = (logical_warps, logical_warp_threads) ||> lane_id
//        }
//[<Record>]
//type WarpScan<'T> =
//    {
//        LOGICAL_WARPS : int
//        LOGICAL_WARP_THREADS : int
//        ThreadFields : ThreadFields<'T>
//    }
//
//    static member Create(logical_warps, logical_warp_threads, threadFields) =
//        {
//            LOGICAL_WARPS = logical_warps
//            LOGICAL_WARP_THREADS = logical_warp_threads
//            ThreadFields = threadFields
//        }
//
//    static member Create() =
//        {
//            LOGICAL_WARPS = 1
//            LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS
//            ThreadFields = ThreadFields.Create( privateStorage(),
//                                                
//        }