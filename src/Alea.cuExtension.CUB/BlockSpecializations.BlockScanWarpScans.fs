[<AutoOpen>]
module Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanWarpScans

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Warp




let WARPS =
    fun block_threads ->
        (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS    


//type internal TemplateParameters =
//    {
//        BLOCK_THREADS : int
//    }
//
//    static member Init(block_threads:int) =
//        {
//            BLOCK_THREADS = block_threads
//        }


type Constants =
    {
        WARPS : int
        WarpScan : WarpScan
    }

    static member Init(block_threads) =
        let warps = block_threads |> WARPS
        {
            WARPS = warps
            WarpScan = WarpScan.Create(warps, CUB_PTX_WARP_THREADS)
        }


[<Record>] [<RequireQualifiedAccess>]
type TempStorage =
    {
        mutable warp_scan : deviceptr<int>
        mutable warp_aggregates : deviceptr<int>
        mutable block_prefix : int
    }


    static member Init(warp_scan, warp_aggregates, block_prefix) =
        {
            warp_scan = warp_scan
            warp_aggregates = warp_aggregates
            block_prefix = block_prefix
        }

    static member Default() =
        {
            warp_scan       = __null()
            warp_aggregates = __null()
            block_prefix    = 0
        }



[<Record>]
type ThreadFields =
    {
        mutable temp_storage : TempStorage
        mutable linear_tid : int
        mutable warp_id : int
        mutable lane_id : int
    }


    static member Init(temp_storage, linear_tid, warp_id, lane_id) =
        {
            temp_storage = temp_storage
            linear_tid = linear_tid
            warp_id = warp_id
            lane_id = lane_id
        }

    static member Init(block_threads, temp_storage, linear_tid) =
        {
            temp_storage = temp_storage
            linear_tid = linear_tid
            warp_id = if block_threads <= CUB_PTX_WARP_THREADS then 0 else linear_tid / CUB_PTX_WARP_THREADS
            lane_id = if block_threads <= CUB_PTX_WARP_THREADS then linear_tid else linear_tid % CUB_PTX_WARP_THREADS
        }


let applyWarpAggregates block_threads = 
    let WARPS = block_threads |> WARPS
    fun (partial:Ref<int>) (scan_op:(int -> int -> int)) (warp_aggregate:int) (block_aggregate:Ref<int>) (lane_valid:bool option) ->
        let lane_valid = if lane_valid.IsSome then lane_valid.Value else true
        fun (temp_storage:TempStorage) (warp_id:int) ->
            temp_storage.warp_aggregates.[warp_id] <- warp_aggregate

            __syncthreads()

            block_aggregate := temp_storage.warp_aggregates.[0]

            for WARP = 1 to WARPS - 1 do
                if warp_id = WARP then
                    partial := if lane_valid then (!block_aggregate, !partial) ||> scan_op else !block_aggregate
                block_aggregate := (!block_aggregate, temp_storage.warp_aggregates.[WARP]) ||> scan_op
            
     
[<Record>]
type BlockScanWarpScans =
    {
        BLOCK_THREADS       : int
        Constants           : Constants
        ThreadFields        : ThreadFields
    }


    member inline this.Initialize(temp_storage:deviceptr<int>, linear_tid) =
        this.ThreadFields.temp_storage.warp_scan <- temp_storage
        this.ThreadFields.linear_tid <- linear_tid
        this

    member this.ApplyWarpAggregates(partial:Ref<int>, scan_op:(int -> int -> int), warp_aggregate:int, block_aggregate:Ref<int>, ?lane_valid:bool) = 
        applyWarpAggregates
        <| this.BLOCK_THREADS
        <| partial <| scan_op <| warp_aggregate <| block_aggregate <| lane_valid
        <| this.ThreadFields.temp_storage <| this.ThreadFields.warp_id
    
        
    member this.ExclusiveScan(input:int, output:Ref<int>, identity:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>) = 
        let temp_storage = this.ThreadFields.temp_storage.warp_scan
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id

        let warp_aggregate = __null() |> __ptr_to_ref
        this.Constants.WarpScan.Initialize(temp_storage, warp_id, lane_id).ExclusiveScan(input, output, !identity, scan_op, warp_aggregate)
        this.ApplyWarpAggregates(output, scan_op, !warp_aggregate, block_aggregate)

    member this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:(int -> int -> int), block_aggregate:Ref<int>, block_prefix_callback_op:Ref<'T -> 'T>) = 
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

    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>) = 
        let temp_storage = this.ThreadFields.temp_storage
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id

        let warp_aggregate = __null() |> __ptr_to_ref
        this.Constants.WarpScan.Initialize(temp_storage.warp_scan, warp_id, lane_id).ExclusiveScan(input, output, scan_op, warp_aggregate)
            
        this.ApplyWarpAggregates(output, scan_op, !warp_aggregate, block_aggregate, lane_id > 0)

    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>, block_prefix_callback_op:Ref<'T -> 'T>) =
        let temp_storage = this.ThreadFields.temp_storage
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id
        let linear_tid = this.ThreadFields.linear_tid

        this.ExclusiveScan(input, output, scan_op, block_aggregate)

        if warp_id = 0 then
            let block_prefix = !block_aggregate |> !block_prefix_callback_op
            if lane_id = 0 then temp_storage.block_prefix <- block_prefix

        __syncthreads()

        output :=   if linear_tid = 0 then temp_storage.block_prefix 
                    else (temp_storage.block_prefix, !output) ||> scan_op



    member inline this.ExclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>) = 
        let temp_storage = this.ThreadFields.temp_storage
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id
            
        let warp_aggregate = __null() |> __ptr_to_ref
        this.Constants.WarpScan.Initialize(temp_storage.warp_scan, warp_id, lane_id).ExclusiveSum(input, output, warp_aggregate)
        let inline sum x y = x + y
        this.ApplyWarpAggregates(output, (+), !warp_aggregate, block_aggregate)

    member this.ExclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>, block_prefix_callback_op:Ref<'T -> 'T>) = 
        let temp_storage = this.ThreadFields.temp_storage
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id

        this.ExclusiveSum(input, output, block_aggregate)

        if warp_id = 0 then
            let block_prefix = !block_aggregate |> !block_prefix_callback_op
            if lane_id = 0 then temp_storage.block_prefix <- block_prefix 

        __syncthreads()

        output := (temp_storage.block_prefix, !output) ||> (+)


    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>) = 
        let temp_storage = this.ThreadFields.temp_storage
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id

        let warp_aggregate = __null() |> __ptr_to_ref
        this.Constants.WarpScan.Initialize(temp_storage.warp_scan, warp_id, lane_id).InclusiveScan(input, output, scan_op, warp_aggregate)

        this.ApplyWarpAggregates(output, scan_op, !warp_aggregate, block_aggregate)


    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>, block_prefix_callback_op:Ref<'T -> 'T>) =
        let temp_storage = this.ThreadFields.temp_storage
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id

        this.InclusiveScan(input, output, scan_op, block_aggregate)

        if warp_id = 0 then
            let block_prefix = !block_aggregate |> !block_prefix_callback_op
            if lane_id = 0 then temp_storage.block_prefix <- block_prefix

        __syncthreads()

        output := (temp_storage.block_prefix, !output) ||> scan_op

    member this.InclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>) = 
        let temp_storage = this.ThreadFields.temp_storage
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id

        let warp_aggregate = __null() |> __ptr_to_ref
        this.Constants.WarpScan.Initialize(temp_storage.warp_scan, warp_id, lane_id).InclusiveSum(input, output, warp_aggregate)

        this.ApplyWarpAggregates(output, (+), !warp_aggregate, block_aggregate)

    member this.InclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) = 
        let temp_storage = this.ThreadFields.temp_storage
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id

        this.InclusiveSum(input, output, block_aggregate)

        if warp_id = 0 then
            let block_prefix = !block_aggregate |> !block_prefix_callback_op
            if lane_id = 0 then temp_storage.block_prefix <- block_prefix

        __syncthreads()

        output := (temp_storage.block_prefix, !output) ||> (+)

   


    static member Create(block_threads:int, linear_tid:int) =
        let c = block_threads |> Constants.Init
        {   
            BLOCK_THREADS = block_threads
            Constants = c
            ThreadFields = ThreadFields.Init(block_threads, TempStorage.Default(), linear_tid)
        }

    static member Create(block_threads:int) =
        let c = block_threads |> Constants.Init
        let tf = (block_threads, TempStorage.Default(), 0) |> ThreadFields.Init
        {
            BLOCK_THREADS = block_threads
            Constants = c
            ThreadFields = tf
        }






let exclusiveScan (block_threads:int) =
    let WARPS = (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS
    let WarpScan = WarpScan.Create(WARPS, CUB_PTX_WARP_THREADS)
    fun (temp_storage:deviceptr<int>) (linear_tid:int) (warp_id:int) (lane_id:int) ->
        fun (input:int) (output:Ref<int>) (identity:Ref<int> option) (scan_op:(int -> int -> int) option) (block_aggregate:Ref<int> option) (block_prefix_callback_op:Ref<int> option) ->
            (identity, scan_op, block_aggregate, block_prefix_callback_op) |> function
            | Some identity, Some scan_op, Some block_aggregate, None ->
                let warp_aggregate = __local__.Variable()
                //WarpScan
                ()
            | Some identity, Some scan_op, Some block_aggregate, Some block_prefix_callback_op -> ()
            | None, Some scan_op, Some block_aggregate, None -> ()
            | None, Some scan_op, Some block_aggregate, Some block_prefix_callback_op -> ()
            | None, None, Some block_aggregate, None -> ()
            | None, None, Some block_aggregate, Some block_prefix_callback_op -> ()
            | _, _, _, _ -> ()
                