[<AutoOpen>]
module Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanWarpScans

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Warp



let WARPS =
    fun block_threads ->
        (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS    


type internal TemplateParameters<'T> =
    {
        BLOCK_THREADS : int
    }

    static member Init(block_threads:int) =
        {
            BLOCK_THREADS = block_threads
        }


type Constants<'T> =
    {
        WARPS : int
        WarpScan : WarpScan<'T>
    }

    static member Init(tp:TemplateParameters<'T>) =
        let warps = tp.BLOCK_THREADS |> WARPS
        {
            WARPS = warps
            WarpScan = WarpScan<'T>.Create(warps, CUB_PTX_WARP_THREADS)
        }


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


    static member Init(temp_storage, linear_tid, warp_id, lane_id) =
        {
            temp_storage = temp_storage
            linear_tid = linear_tid
            warp_id = warp_id
            lane_id = lane_id
        }

    static member Init(tp:TemplateParameters<'T>, temp_storage, linear_tid) =
        {
            temp_storage = temp_storage
            linear_tid = linear_tid
            warp_id = if tp.BLOCK_THREADS <= CUB_PTX_WARP_THREADS then 0 else linear_tid / CUB_PTX_WARP_THREADS
            lane_id = if tp.BLOCK_THREADS <= CUB_PTX_WARP_THREADS then linear_tid else linear_tid % CUB_PTX_WARP_THREADS
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
        TemplateParameters  : TemplateParameters<'T>
        Constants           : Constants<'T>
        ThreadFields        : ThreadFields<'T>
    }

    member this.ApplyWarpAggregates(partial:Ref<'T>, scan_op:('T -> 'T -> 'T), warp_aggregate:'T, block_aggregate:Ref<'T>, ?lane_valid:bool) = 
        applyWarpAggregates
        <| this.TemplateParameters.BLOCK_THREADS
        <| partial <| scan_op <| warp_aggregate <| block_aggregate <| lane_valid
        <| this.ThreadFields.temp_storage <| this.ThreadFields.warp_id
        
    member this.ExclusiveScan(input:'T, output:Ref<'T>, identity:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) = 
        let temp_storage = this.ThreadFields.temp_storage.warp_scan
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id

        let warp_aggregate = __null() |> __ptr_to_ref
        this.Constants.WarpScan.Initialize(temp_storage, warp_id, lane_id).ExclusiveScan(input, output, !identity, scan_op, warp_aggregate)
        this.ApplyWarpAggregates(output, scan_op, !warp_aggregate, block_aggregate)

    member this.ExclusiveScan(input:'T, output:Ref<'T>, identity:'T, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T -> 'T>) = 
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
        let temp_storage = this.ThreadFields.temp_storage
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id

        let warp_aggregate = __null() |> __ptr_to_ref
        this.Constants.WarpScan.Initialize(temp_storage.warp_scan, warp_id, lane_id).ExclusiveScan(input, output, scan_op, warp_aggregate)
            
        this.ApplyWarpAggregates(output, scan_op, !warp_aggregate, block_aggregate, lane_id > 0)

    member this.ExclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T -> 'T>) =
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



    member this.ExclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>) = 
        let temp_storage = this.ThreadFields.temp_storage
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id
            
        let warp_aggregate = __null() |> __ptr_to_ref
        this.Constants.WarpScan.Initialize(temp_storage.warp_scan, warp_id, lane_id).ExclusiveSum(input, output, warp_aggregate)

        this.ApplyWarpAggregates(output, (+), !warp_aggregate, block_aggregate)

    member this.ExclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T -> 'T>) = 
        let temp_storage = this.ThreadFields.temp_storage
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id

        this.ExclusiveSum(input, output, block_aggregate)

        if warp_id = 0 then
            let block_prefix = !block_aggregate |> !block_prefix_callback_op
            if lane_id = 0 then temp_storage.block_prefix <- block_prefix 

        __syncthreads()

        output := (temp_storage.block_prefix, !output) ||> (+)


    member this.InclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>) = 
        let temp_storage = this.ThreadFields.temp_storage
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id

        let warp_aggregate = __null() |> __ptr_to_ref
        this.Constants.WarpScan.Initialize(temp_storage.warp_scan, warp_id, lane_id).InclusiveScan(input, output, scan_op, warp_aggregate)

        this.ApplyWarpAggregates(output, scan_op, warp_aggregate, block_aggregate)


    member this.InclusiveScan(input:'T, output:Ref<'T>, scan_op:('T -> 'T -> 'T), block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T - 'T>) =
        let temp_storage = this.ThreadFields.temp_storage
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id

        this.InclusiveScan(input, output, scan_op, block_aggregate)

        if warp_id = 0 then
            let block_prefix = !block_aggregate |> !block_prefix_callback_op
            if lane_id = 0 then temp_storage.block_prefix <- block_prefix

        __syncthreads()

        output := (temp_storage.block_prefix, !output) ||> scan_op

    member this.InclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>) = 
        let temp_storage = this.ThreadFields.temp_storage
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id

        let warp_aggregate = __null() |> __ptr_to_ref
        this.Constants.WarpScan.Initialize(temp_storage.warp_scan, warp_id, lane_id).InclusiveSum(input, output, warp_aggregate)

        this.ApplyWarpAggregates(output, (+), warp_aggregate, block_aggregate)

    member this.InclusiveSum(input:'T, output:Ref<'T>, block_aggregate:Ref<'T>, block_prefix_callback_op:Ref<'T>) = 
        let temp_storage = this.ThreadFields.temp_storage
        let warp_id = this.ThreadFields.warp_id
        let lane_id = this.ThreadFields.lane_id

        this.InclusiveSum(intput, output, block_aggregate)

        if warp_id = 0 then
            let block_prefix = !block_aggregate |> !block_prefix_callback_op
            if lane_id = 0 then temp_storage.block_prefix <- block_prefix

        __syncthreads()

        output := (temp_storage.block_prefix, !output) ||> (+)

    
    member this.Initialize(temp_storage:deviceptr<'T>, linear_tid) =
        this.ThreadFields.temp_storage <- temp_storage
        this.ThreadFields.linear_tid <- linear_tid
        this


    static member Create(block_threads:int, temp_storage:deviceptr<'T>, linear_tid:int) =
        let tp = block_threads |> TemplateParameters.Init
        let c = block_threads |> Constants.Init
        {   
            TemplateParameters = tp
            Constants = c
            ThreadFields = ThreadFields<'T>.Create(temp_storage, linear_tid)
        }

    static member Create(block_threads:int) =
        let tp = block_threads |> TemplateParameters.Init
        let c = block_threads |> Constants.Init
        let tf = (tp, __null(), 0) |> ThreadFields<'T>.Init
        {
            TemplateParameters = tp
            Constants = c
            ThreadFields = tf
        }

