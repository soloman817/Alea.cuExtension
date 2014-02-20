[<AutoOpen>]
module Alea.cuExtension.CUB.Warp.Scan

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Common
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Warp.WarpSpecializations


module InternalWarpScan =
    
    let x = ()


module InclusiveSum =

    module SingleDatumPerThread =
        let private Default =
            <@ fun (input:int) (output:Ref<int>) -> () @>

        let private WithAggregate =
            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp =
            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>


    module MultipleDataPerThread =
        let private Default items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) -> () @>

        let private WithAggregate items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>

module ExclusiveSum =

    module SingleDatumPerThread =
        let private Default =
            <@ fun (input:int) (output:Ref<int>) -> () @>

        let private WithAggregate =
            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp =
            <@ fun (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>


    module MultipleDataPerThread =
        let private Default items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) -> () @>

        let private WithAggregate items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>


module ExclusiveScan =
    type ScanOp = int -> int -> int

    module SingleDatumPerThread =
        let private Default =
            <@ fun (input:int) (output:Ref<int>) (identity:int) (scan_op:ScanOp) -> () @>

        let private WithAggregate =
            <@ fun (input:int) (output:Ref<int>) (identity:Ref<int>) (scan_op:ScanOp) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp =
            <@ fun (input:int) (output:Ref<int>) (identity:Ref<int>) (scan_op:ScanOp) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>


        module Identityless =
            let private Default =
                <@ fun (input:int) (output:Ref<int>) (scan_op:ScanOp) -> () @>

            let private WithAggregate =
                <@ fun (input:int) (output:Ref<int>) (scan_op:ScanOp) (block_aggregate:Ref<int>) -> () @>

            let private WithAggregateAndCallbackOp =
                <@ fun (input:int) (output:Ref<int>) (scan_op:ScanOp) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>

    module MultipleDataPerThread =
        let private Default items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (identity:Ref<int>) (scan_op:ScanOp) -> () @>

        let private WithAggregate items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (identity:Ref<int>) (scan_op:ScanOp) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (identity:Ref<int>) (scan_op:ScanOp) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>


        module Identityless =
            let private Default items_per_thread =
                <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:ScanOp) -> () @>

            let private WithAggregate items_per_thread =
                <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:ScanOp) (block_aggregate:Ref<int>) -> () @>

            let private WithAggregateAndCallbackOp items_per_thread =
                <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:ScanOp) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>




module InclusiveScan =
    type ScanOp = int -> int -> int

    module SingleDatumPerThread =
        let private Default =
            <@ fun (input:int) (output:Ref<int>) (scan_op:ScanOp) -> () @>

        let private WithAggregate =
            <@ fun (input:int) (output:Ref<int>) (scan_op:ScanOp) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp =
            <@ fun (input:int) (output:Ref<int>) (scan_op:ScanOp) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>


    module MultipleDataPerThread =
        let private Default items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:ScanOp) -> () @>

        let private WithAggregate items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:ScanOp) (block_aggregate:Ref<int>) -> () @>

        let private WithAggregateAndCallbackOp items_per_thread =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:ScanOp) (block_aggregate:Ref<int>) (block_prefix_callback_op:Ref<int>) -> () @>


//let POW_OF_TWO =
//    fun logical_warp_threads ->
//        ((logical_warp_threads &&& (logical_warp_threads - 1)) = 0)
//
//let InternalWarpScan =
//    fun logical_warps logical_warp_threads ->
//        if (CUB_PTX_VERSION >= 300) && ((logical_warps = 1) || (logical_warps |> POW_OF_TWO)) then
//            (logical_warps, logical_warp_threads) |> WarpScanShfl.Create
//        else
//            (logical_warps, logical_warp_threads) |> WarpScanShfl.Create //WarpScanSmem
//
//
//let warp_id =
//    fun logical_warps logical_warp_threads ->
//        if logical_warps = 1 then 0 else threadIdx.x / logical_warp_threads
//
//let lane_id =
//    fun logical_warps logical_warp_threads ->
//        if ((logical_warps = 1) || (logical_warp_threads = CUB_PTX_WARP_THREADS)) then __ptx__.LaneId() else threadIdx.x % logical_warp_threads
//
//
//let [<ReflectedDefinition>] PrivateStorage() = __null()
//
//
//type [<RequireQualifiedAccess>] ITempStorage = abstract temp_storage : deviceptr<int>
//let tempStorage(grid_elements:int)() = { new ITempStorage with member this.temp_storage = __shared__.Array(grid_elements) |> __array_to_ptr }


//type WarpScanAPI =
//    abstract POW_OF_TWO : bool
//    abstract InternalScan : int
//    abstract temp_storage : Ref<int>
//    abstract warp_id : int
//    abstract lane_id : int
//    abstract PrivateStorage : unit -> Ref<int>
//
//
//let api<int> (logical_warps:int) (logical_warp_threads:int) =
//    { new WarpScanAPI<int> with
//        member this.POW_OF_TWO = logical_warp_threads |> POW_OF_TWO
//        member this.InternalScan = 0
//        member this.temp_storage = privateStorage() |> __ptr_to_ref
//        member this.warp_id = (logical_warps, logical_warp_threads) ||> warp_id
//        member this.lane_id = (logical_warps, logical_warp_threads) ||> lane_id
//        member this.PrivateStorage() = privateStorage() |> __ptr_to_ref
//    }

//
//type Constants =
//    {
//        POW_OF_TWO : bool
//    }
//
//    static member Init(logical_warp_threads) =
//        {
//            POW_OF_TWO = logical_warp_threads |> POW_OF_TWO 
//        }
//
//
//[<Record>]
//type InternalWarpScan =
//    {
//        WarpScanShfl : WarpScanShfl
//        WarpScanSmem : WarpScanSmem
//    }
//
//    static member Init(logical_warps, logical_warp_threads) =
//        {
//            WarpScanShfl = (logical_warps, logical_warp_threads) |> WarpScanShfl.Create
//            WarpScanSmem = (logical_warps, logical_warp_threads) |> WarpScanSmem.Create
//        }
//
//
//
//
//[<Record>]
//type ThreadFields =
//    {
//        mutable temp_storage : deviceptr<int>
//        mutable warp_id : int
//        mutable lane_id : int
//    }
//
//    member this.Set(temp_storage:deviceptr<int>, warp_id:int, lane_id:int) =
//        this.temp_storage <- temp_storage
//        this.warp_id <- warp_id
//        this.lane_id <- lane_id
//
//    static member Init(logical_warps, logical_warp_threads) =
//        {
//            temp_storage = __null()
//            warp_id = (logical_warps, logical_warp_threads) ||> warp_id
//            lane_id = (logical_warps, logical_warp_threads) ||> lane_id
//        }
//
//    static member Init(logical_warps, logical_warp_threads, temp_storage:deviceptr<int>) =
//        {
//            temp_storage = temp_storage
//            warp_id = (logical_warps, logical_warp_threads) ||> warp_id
//            lane_id = (logical_warps, logical_warp_threads) ||> lane_id
//        }
//
//    static member Init(logical_warps, logical_warp_threads, temp_storage:deviceptr<int>, warp_id:int, lane_id:int) =
//        {
//            temp_storage = temp_storage
//            warp_id = warp_id
//            lane_id = lane_id
//        }
//
//[<Record>]
//type WarpScan =
//    {
//        LOGICAL_WARPS           : int
//        LOGICAL_WARP_THREADS    : int
//        Constants               : Constants
//        ThreadFields            : ThreadFields
//    }
//
//    member this.InclusiveSum(input:int, output:Ref<int>) = ()
//    member this.InclusiveSum(input:int, output:Ref<int>, warp_aggregate:Ref<int>) = ()
//
//    member this.InclusiveSum(input:int, output:Ref<int>, warp_aggregate:Ref<int>, warp_prefix_op:Ref<int -> int>) = ()
//
//
//    member private this.ExclusiveSum(input:int, output:Ref<int>, is_primitive:bool) = ()
//    member private this.ExclusiveSum(input:int, output:Ref<int>, warp_aggregate:Ref<int>, is_primitive:bool) = ()
//    member private this.ExclusiveSum(input:int, output:Ref<int>, warp_aggregate:Ref<int>, warp_prefix_op:Ref<int -> int>, is_primitive:bool) = ()
//
//
//    member this.ExclusiveSum(input:int, output:Ref<int>) = ()
//    member this.ExclusiveSum(input:int, output:Ref<int>, warp_aggregate:Ref<int>) = ()
//    member this.ExclusiveSum(input:int, output:Ref<int>, warp_aggregate:Ref<int>, warp_prefix_op:Ref<int -> int>) = ()
//
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int)) = ()
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), warp_aggregate:Ref<int>) = ()
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), warp_aggregate:Ref<int>, warp_prefix_op:Ref<int -> int>) = ()
//
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:(int -> int -> int)) = ()
//    member inline this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:(int -> int -> int), warp_aggregate:Ref<int>) = ()
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:(int -> int -> int), warp_aggregate:Ref<int>, warp_prefix_op:Ref<int -> int>) = ()
//
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int)) = ()
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), warp_aggregate:Ref<int>) = ()
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), warp_aggregate:Ref<int>, warp_prefix_op:Ref<int -> int>) = ()
//
//
//    member this.Initialize(temp_storage:deviceptr<int>, warp_id:int, lane_id:int) = 
//        (temp_storage, warp_id, lane_id) |> this.ThreadFields.Set
//        this
//
//    static member Create(?logical_warps:int, ?logical_warp_threads:int) =
//        (logical_warps, logical_warp_threads) |> function
//        | None, None ->
//            let logical_warps = 1
//            let logical_warp_threads = CUB_PTX_WARP_THREADS
//            {   
//                LOGICAL_WARPS           = logical_warps
//                LOGICAL_WARP_THREADS    = logical_warp_threads
//                Constants               = logical_warp_threads |> Constants.Init
//                ThreadFields            = (logical_warps, logical_warp_threads) |> ThreadFields.Init
//            }
//
//        | Some logical_warps, None ->
//            let logical_warp_threads = CUB_PTX_WARP_THREADS
//            { 
//                LOGICAL_WARPS           = logical_warps
//                LOGICAL_WARP_THREADS    = logical_warp_threads
//                Constants               = logical_warp_threads |> Constants.Init
//                ThreadFields            = (logical_warps, logical_warp_threads) |> ThreadFields.Init
//            }
//
//        | None, Some logical_warp_threads ->
//            let logical_warps = 1
//            { 
//                LOGICAL_WARPS           = 1
//                LOGICAL_WARP_THREADS    = logical_warp_threads
//                Constants               = logical_warp_threads |> Constants.Init
//                ThreadFields            = (logical_warps, logical_warp_threads) |> ThreadFields.Init                
//            }
//
//        | Some logical_warps, Some logical_warp_threads ->
//            { 
//                LOGICAL_WARPS           = logical_warps
//                LOGICAL_WARP_THREADS    = logical_warp_threads
//                Constants               = logical_warp_threads |> Constants.Init
//                ThreadFields            = (logical_warps, logical_warp_threads) |> ThreadFields.Init
//            }
//


//
//[<Record>]
//type TempStorage =
//    {
//        temp_storage : deviceptr<int>
//    }
//
//    static member Create() = 
//        {
//            temp_storage = privateStorage()
//        }
//
//
//[<Record>]
//type ThreadFields =
//    {
//        temp_storage : TempStorage<int>
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
//            temp_storage = TempStorage.Create()
//            warp_id = (logical_warps, logical_warp_threads) ||> warp_id
//            lane_id = (logical_warps, logical_warp_threads) ||> lane_id
//        }
//[<Record>]
//type WarpScan =
//    {
//        LOGICAL_WARPS : int
//        LOGICAL_WARP_THREADS : int
//        ThreadFields : ThreadFields<int>
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