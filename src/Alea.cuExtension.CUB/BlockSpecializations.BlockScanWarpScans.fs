[<AutoOpen>]
module Alea.cuExtension.CUB.Block.BlockSpecializations.BlockScanWarpScans

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Warp

   

module BlockScanWarpScans =

    type StaticParam =
        { 
            BLOCK_THREADS   : int
            WARPS           : int

            WarpScanParam   : WarpScan.StaticParam
        }
        
        static member inline Init(block_threads) = 
            let warps = (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS
            { 
                BLOCK_THREADS   = block_threads
                WARPS           = warps

                WarpScanParam   = WarpScan.StaticParam.Init(warps, CUB_PTX_WARP_THREADS)
            }


    [<Record>]
    type TempStorage<'T> =
        {
            mutable warp_scan               : WarpScan.TempStorage<'T>
            mutable warp_aggregates         : SharedRecord<'T>
            mutable block_prefix            : Ref<'T>
        }

        [<ReflectedDefinition>]
        static member Init(sp:StaticParam) =
            { 
                warp_scan       = WarpScan.TempStorage<'T>.Init(sp.WarpScanParam.WarpScanSmemParam)
                warp_aggregates = SharedRecord<'T>.Init(sp.WARPS)
                block_prefix    = __shared__.Variable<'T>()
            }

    let [<ReflectedDefinition>] warp_id (block_threads:int) (linear_tid:int) = 
        if (block_threads <= CUB_PTX_WARP_THREADS) then 0 else linear_tid / CUB_PTX_WARP_THREADS;
                
                
    let [<ReflectedDefinition>] lane_id (block_threads:int) (linear_tid:int) = 
        if (block_threads <= CUB_PTX_WARP_THREADS) then linear_tid else linear_tid % CUB_PTX_WARP_THREADS


    [<Record>]
    type InstanceParam<'T> =
        {
            mutable temp_storage    : TempStorage<'T>
            mutable linear_tid      : int
            mutable warp_id         : int
            mutable lane_id         : int
        }
                
        [<ReflectedDefinition>]
        static member Init(sp:StaticParam, temp_storage, linear_tid) = 
            {   temp_storage    = temp_storage; 
                linear_tid      = linear_tid; 
                warp_id         = 
                    if (sp.BLOCK_THREADS <= CUB_PTX_WARP_THREADS) then 0 else linear_tid / CUB_PTX_WARP_THREADS;
                lane_id         = 
                    if (sp.BLOCK_THREADS <= CUB_PTX_WARP_THREADS) then linear_tid else linear_tid % CUB_PTX_WARP_THREADS }


    module ApplyWarpAggregates =

        let [<ReflectedDefinition>] inline WithLaneValidationInt (sp:StaticParam)
            (ip:InstanceParam<int>)
            (partial:Ref<int>) (warp_aggregate:int) (block_aggregate:Ref<int>) (lane_valid:bool) =
            
            

            ip.temp_storage.warp_aggregates.[ip.warp_id] <- warp_aggregate

            __syncthreads()

            block_aggregate := ip.temp_storage.warp_aggregates.[0]

            for WARP = 1 to sp.WARPS - 1 do
                if ip.warp_id = WARP then
                    partial := if lane_valid then !block_aggregate + !partial else !block_aggregate
                block_aggregate := !block_aggregate + ip.temp_storage.warp_aggregates.[WARP]
     
      
        let [<ReflectedDefinition>] inline DefaultInt (sp:StaticParam)
            (ip:InstanceParam<int>)
            (partial:Ref<int>) (warp_aggregate:int) (block_aggregate:Ref<int>) =
            WithLaneValidationInt sp ip partial warp_aggregate block_aggregate true



        let [<ReflectedDefinition>] inline WithLaneValidation (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (partial:Ref<'T>) (warp_aggregate:'T) (block_aggregate:Ref<'T>) (lane_valid:bool) =
            
            
            ip.temp_storage.warp_aggregates.[ip.warp_id] <- warp_aggregate

            __syncthreads()

            block_aggregate := ip.temp_storage.warp_aggregates.[0]

            for WARP = 1 to sp.WARPS - 1 do
                if ip.warp_id = WARP then
                    partial := if lane_valid then (!block_aggregate, !partial) ||> scan_op else !block_aggregate
                block_aggregate := (!block_aggregate, ip.temp_storage.warp_aggregates.[WARP]) ||> scan_op
     
      
        let [<ReflectedDefinition>] inline Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (partial:Ref<'T>) (warp_aggregate:'T) (block_aggregate:Ref<'T>) =
            WithLaneValidation sp scan_op ip partial warp_aggregate block_aggregate true
        



    module ExclusiveSum =
        
       

        let [<ReflectedDefinition>] inline WithAggregateInt (sp:StaticParam)
            (ip:InstanceParam<int>)
            (input:int) (output:Ref<int>) (block_aggregate:Ref<int>) =
            let wsip = WarpScan.InstanceParam<int>.Init(sp.WarpScanParam, ip.temp_storage.warp_scan, ip.warp_id, ip.lane_id)
            let warp_aggregate = __local__.Variable<int>()
            WarpScan.ExclusiveSum.WithAggregateInt sp.WarpScanParam wsip input output warp_aggregate
            ApplyWarpAggregates.WithLaneValidationInt sp ip output !warp_aggregate block_aggregate (ip.lane_id > 0)
        
        let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) =
            let wsip = WarpScan.InstanceParam<'T>.Init(sp.WarpScanParam, ip.temp_storage.warp_scan, ip.warp_id, ip.lane_id)
            let warp_aggregate = __local__.Variable<'T>()
            WarpScan.ExclusiveSum.WithAggregate sp.WarpScanParam wsip input output warp_aggregate
            ApplyWarpAggregates.WithLaneValidation sp (+) ip output !warp_aggregate block_aggregate (ip.lane_id > 0)
        

//        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
//            (ip:InstanceParam<'T>)
//            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
//            WithAggregate sp scan_op ip input output block_aggregate
//    
//            if ip.warp_id = 0 then 
//                let block_prefix = !block_aggregate |> !block_prefix_callback_op
//                if lane_id = 0 then
//                    ip.temp_storage.block_prefix := block_prefix
//    
//            __syncthreads()
//    
//            output := (!ip.temp_storage.block_prefix, !output) ||> scan_op

        

    module ExclusiveScan =
        ()

        let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) =
            let warp_aggregate = __local__.Variable<'T>()
            let wsip = WarpScan.InstanceParam<'T>.Init(sp.WarpScanParam, ip.temp_storage.warp_scan, ip.warp_id, ip.lane_id)
            WarpScan.ExclusiveScan.WithAggregate sp.WarpScanParam scan_op wsip input output !identity warp_aggregate
            ApplyWarpAggregates.Default sp scan_op ip output !warp_aggregate block_aggregate
            
    
        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp  (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (identity:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = 
            () 
            
    
    
        module Identityless =
            let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
                (ip:InstanceParam<'T>)
                (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = 
                let warp_aggregate = __local__.Variable<'T>()
                let wsip = WarpScan.InstanceParam<'T>.Init(sp.WarpScanParam, ip.temp_storage.warp_scan, ip.warp_id, ip.lane_id)
                WarpScan.ExclusiveScan.Identityless.WithAggregate sp.WarpScanParam scan_op wsip input output warp_aggregate
                ApplyWarpAggregates.Default sp scan_op ip output !warp_aggregate block_aggregate
                
    
            let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
                (ip:InstanceParam<'T>)
                (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
                ()
                

    

    module InclusiveSum =
        

        let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = 
            () 
        

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) = 
            () 
        



    module InclusiveScan =
        

        let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) = 
            ()
        

        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) =
            ()



//
//    [<Record>]
//    type IntApi =
//        {
//            mutable temp_storage    : TempStorage<int>
//            mutable linear_tid      : int
//            mutable warp_id         : int
//            mutable lane_id         : int
//        }
//                
//        [<ReflectedDefinition>]
//        static member Init(sp:StaticParam, temp_storage, linear_tid) = 
//            
//            {   temp_storage    = temp_storage; 
//                linear_tid      = linear_tid; 
//                warp_id         = 
//                    if (p.BLOCK_THREADS <= CUB_PTX_WARP_THREADS) then 0 else linear_tid / CUB_PTX_WARP_THREADS;
//                lane_id         = 
//                    if (p.BLOCK_THREADS <= CUB_PTX_WARP_THREADS) then linear_tid else linear_tid % CUB_PTX_WARP_THREADS }
//
//
//
//        [<ReflectedDefinition>] member this.ExclusiveSum(h, input, output, block_aggregate)
//            = ExclusiveSum.WithAggregateInt sp this.temp_storage this.linear_tid this.warp_id this.lane_id input output block_aggregate

//        [<ReflectedDefinition>] member this.ExclusiveSum(h, scan_op, input, output, block_aggregate, block_prefix_callback_op)
//            = ExclusiveSum.WithAggregateAndCallbackOp sp scan_op this.temp_storage this.warp_id this.lane_id input output


//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity, block_aggregate)
//            = ExclusiveScan.WithAggregate sp scan_op this.DeviceApi input output identity block_aggregate
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity, block_aggregate, block_prefix_callback_op)
//            = ExclusiveScan.WithAggregateAndCallbackOp sp scan_op this.DeviceApi input output identity block_aggregate block_prefix_callback_op
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, block_aggregate)
//            = ExclusiveScan.Identityless.WithAggregate sp scan_op this.DeviceApi input output
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, block_aggregate, block_prefix_callback_op)
//            = ExclusiveScan.Identityless.WithAggregateAndCallbackOp sp scan_op this.DeviceApi input output block_aggregate block_prefix_callback_op
//

//    module ExclusiveSum =
//        type FunctionApi<'T> = Template.ExclusiveSum._FunctionApi<'T>
//
//        let [<ReflectedDefinition>] inline template<'T> (block_threads:int) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
//            let sp = HostApi.Init(block_threads)
//            let! waggr      = (h, scan_op) ||> ExclusiveSum.WithAggregate              |> Compiler.DefineFunction
//            
//            return h, {
//                WithAggregate = waggr
//            }}
//
//    module ExclusiveScan =
//        type FunctionApi<'T> = Template.ExclusiveScan._FunctionApi<'T>
//
//        let [<ReflectedDefinition>] inline template<'T> (block_threads:int) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
//            let sp = HostApi.Init(block_threads)
//            let! waggr      = (h, scan_op) ||> ExclusiveScan.WithAggregate              |> Compiler.DefineFunction
//            let! waggr_noid = (h, scan_op) ||> ExclusiveScan.Identityless.WithAggregate |> Compiler.DefineFunction
//
//            return h, {
//                WithAggregate       = waggr
//                WithAggregate_NoID  = waggr_noid
//            }}
//                
//    module InclusiveSum =
//        type FunctionApi<'T> = Template.InclusiveSum._FunctionApi<'T>
//    module InclusiveScan =
//        type FunctionApi<'T> = Template.InclusiveScan._FunctionApi<'T>




//let WARPS =
//    fun block_threads ->
//        (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS    
//
//
////type internal TemplateParameters =
////    {
////        BLOCK_THREADS : int
////    }
////
////    static member Init(block_threads:int) =
////        {
////            BLOCK_THREADS = block_threads
////        }
//
//
//type Constants =
//    {
//        WARPS : int
//        WarpScan : WarpScan
//    }
//
//    static member Init(block_threads) =
//        let warps = block_threads |> WARPS
//        {
//            WARPS = warps
//            WarpScan = WarpScan.Init(warps, CUB_PTX_WARP_THREADS)
//        }
//
//
//[<Record>] [<RequireQualifiedAccess>]
//type TempStorage =
//    {
//        mutable warp_scan : deviceptr<int>
//        mutable warp_aggregates : deviceptr<int>
//        mutable block_prefix : int
//    }
//
//
//    static member Init(warp_scan, warp_aggregates, block_prefix) =
//        {
//            warp_scan = warp_scan
//            warp_aggregates = warp_aggregates
//            block_prefix = block_prefix
//        }
//
//    static member Default() =
//        {
//            warp_scan       = __null()
//            warp_aggregates = __null()
//            block_prefix    = 0
//        }
//
//
//
//[<Record>]
//type ThreadFields =
//    {
//        mutable temp_storage : TempStorage
//        mutable linear_tid : int
//        mutable warp_id : int
//        mutable lane_id : int
//    }
//
//
//    static member Init(temp_storage, linear_tid, warp_id, lane_id) =
//        {
//            temp_storage = temp_storage
//            linear_tid = linear_tid
//            warp_id = warp_id
//            lane_id = lane_id
//        }
//
//    static member Init(block_threads, temp_storage, linear_tid) =
//        {
//            temp_storage = temp_storage
//            linear_tid = linear_tid
//            warp_id = if block_threads <= CUB_PTX_WARP_THREADS then 0 else linear_tid / CUB_PTX_WARP_THREADS
//            lane_id = if block_threads <= CUB_PTX_WARP_THREADS then linear_tid else linear_tid % CUB_PTX_WARP_THREADS
//        }
//
//
//let applyWarpAggregates block_threads = 
//    let WARPS = block_threads |> WARPS
//    fun (partial:Ref<int>) (scan_op:(int -> int -> int)) (warp_aggregate:int) (block_aggregate:Ref<int>) (lane_valid:bool option) =
//        let lane_valid = if lane_valiIsSome then lane_valiValue else true
//        fun (temp_storage:TempStorage) (ip.warp_id:int) =
//            ip.temp_storage.warp_aggregates.[ip.warp_id] <- warp_aggregate
//
//            __syncthreads()
//
//            block_aggregate := ip.temp_storage.warp_aggregates.[0]
//
//            for WARP = 1 to WARPS - 1 do
//                if ip.warp_id = WARP then
//                    partial := if lane_valid then (!block_aggregate, !partial) ||> scan_op else !block_aggregate
//                block_aggregate := (!block_aggregate, ip.temp_storage.warp_aggregates.[WARP]) ||> scan_op
//            
//     
//[<Record>]
//type BlockScanWarpScans =
//    {
//        BLOCK_THREADS       : int
//        Constants           : Constants
//        ThreadFields        : ThreadFields
//    }
//
//
//    member inline this.Initialize(temp_storage:deviceptr<int>, linear_tid) =
//        this.ThreadFields.ip.temp_storage.warp_scan <- temp_storage
//        this.ThreadFields.linear_tid <- linear_tid
//        this
//
//    member this.ApplyWarpAggregates(partial:Ref<int>, scan_op:(int -> int -> int), warp_aggregate:int, block_aggregate:Ref<int>, ?lane_valid:bool) = 
//        applyWarpAggregates
//        <| this.BLOCK_THREADS
//        <| partial <| scan_op <| warp_aggregate <| block_aggregate <| lane_valid
//        <| this.ThreadFields.temp_storage <| this.ThreadFields.warp_id
//    
//        
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>) = 
//        let temp_storage = this.ThreadFields.ip.temp_storage.warp_scan
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//
//        let warp_aggregate = __null() |> __ptr_to_ref
//        this.Constants.WarpScan.Initialize(temp_storage, warp_id, lane_id).ExclusiveScan(input, output, !identity, scan_op, warp_aggregate)
//        this.ApplyWarpAggregates(output, scan_op, !warp_aggregate, block_aggregate)
//
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:(int -> int -> int), block_aggregate:Ref<int>, block_prefix_callback_op:Ref<'T -> 'T>) = 
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//        let temp_storage = this.ThreadFields.temp_storage
//        let identity = identity |> __obj_to_ref
//            
//        this.ExclusiveScan(input, output, identity, scan_op, block_aggregate)
//        if ip.warp_id = 0 then
//            let block_prefix = !block_aggregate |> !block_prefix_callback_op 
//            if lane_id = 0 then ip.temp_storage.block_prefix <- block_prefix
//
//        __syncthreads()
//
//        output := (ip.temp_storage.block_prefix, !output) ||> scan_op
//
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>) = 
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//
//        let warp_aggregate = __null() |> __ptr_to_ref
//        this.Constants.WarpScan.Initialize(ip.temp_storage.warp_scan, warp_id, lane_id).ExclusiveScan(input, output, scan_op, warp_aggregate)
//            
//        this.ApplyWarpAggregates(output, scan_op, !warp_aggregate, block_aggregate, lane_id > 0)
//
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>, block_prefix_callback_op:Ref<'T -> 'T>) =
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//        let linear_tid = this.ThreadFields.linear_tid
//
//        this.ExclusiveScan(input, output, scan_op, block_aggregate)
//
//        if ip.warp_id = 0 then
//            let block_prefix = !block_aggregate |> !block_prefix_callback_op
//            if lane_id = 0 then ip.temp_storage.block_prefix <- block_prefix
//
//        __syncthreads()
//
//        output :=   if linear_tid = 0 then ip.temp_storage.block_prefix 
//                    else (ip.temp_storage.block_prefix, !output) ||> scan_op
//
//
//
//    member inline this.ExclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>) = 
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//            
//        let warp_aggregate = __null() |> __ptr_to_ref
//        this.Constants.WarpScan.Initialize(ip.temp_storage.warp_scan, warp_id, lane_id).ExclusiveSum(input, output, warp_aggregate)
//        let [<ReflectedDefinition>] inline sum x y = x + y
//        this.ApplyWarpAggregates(output, (+), !warp_aggregate, block_aggregate)
//
//    member this.ExclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>, block_prefix_callback_op:Ref<'T -> 'T>) = 
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//
//        this.ExclusiveSum(input, output, block_aggregate)
//
//        if ip.warp_id = 0 then
//            let block_prefix = !block_aggregate |> !block_prefix_callback_op
//            if lane_id = 0 then ip.temp_storage.block_prefix <- block_prefix 
//
//        __syncthreads()
//
//        output := (ip.temp_storage.block_prefix, !output) ||> (+)
//
//
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>) = 
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//
//        let warp_aggregate = __null() |> __ptr_to_ref
//        this.Constants.WarpScan.Initialize(ip.temp_storage.warp_scan, warp_id, lane_id).InclusiveScan(input, output, scan_op, warp_aggregate)
//
//        this.ApplyWarpAggregates(output, scan_op, !warp_aggregate, block_aggregate)
//
//
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), block_aggregate:Ref<int>, block_prefix_callback_op:Ref<'T -> 'T>) =
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//
//        this.InclusiveScan(input, output, scan_op, block_aggregate)
//
//        if ip.warp_id = 0 then
//            let block_prefix = !block_aggregate |> !block_prefix_callback_op
//            if lane_id = 0 then ip.temp_storage.block_prefix <- block_prefix
//
//        __syncthreads()
//
//        output := (ip.temp_storage.block_prefix, !output) ||> scan_op
//
//    member this.InclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>) = 
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//
//        let warp_aggregate = __null() |> __ptr_to_ref
//        this.Constants.WarpScan.Initialize(ip.temp_storage.warp_scan, warp_id, lane_id).InclusiveSum(input, output, warp_aggregate)
//
//        this.ApplyWarpAggregates(output, (+), !warp_aggregate, block_aggregate)
//
//    member this.InclusiveSum(input:int, output:Ref<int>, block_aggregate:Ref<int>, block_prefix_callback_op:Ref<int -> int>) = 
//        let temp_storage = this.ThreadFields.temp_storage
//        let warp_id = this.ThreadFields.warp_id
//        let lane_id = this.ThreadFields.lane_id
//
//        this.InclusiveSum(input, output, block_aggregate)
//
//        if ip.warp_id = 0 then
//            let block_prefix = !block_aggregate |> !block_prefix_callback_op
//            if lane_id = 0 then ip.temp_storage.block_prefix <- block_prefix
//
//        __syncthreads()
//
//        output := (ip.temp_storage.block_prefix, !output) ||> (+)
//
//   
//
//
//    static member Create(block_threads:int, linear_tid:int) =
//        let c = block_threads |> Constants.Init
//        {   
//            BLOCK_THREADS = block_threads
//            Constants = c
//            ThreadFields = ThreadFields.Init(block_threads, TempStorage.Default(), linear_tid)
//        }
//
//    static member Create(block_threads:int) =
//        let c = block_threads |> Constants.Init
//        let tf = (block_threads, TempStorage.Default(), 0) |> ThreadFields.Init
//        {
//            BLOCK_THREADS = block_threads
//            Constants = c
//            ThreadFields = tf
//        }
//
//
//
//
//
//
//let exclusiveScan (block_threads:int) =
//    let WARPS = (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS
//    let WarpScan = WarpScan.Init(WARPS, CUB_PTX_WARP_THREADS)
//    fun (temp_storage:deviceptr<int>) (linear_tid:int) (ip.warp_id:int) (ip.lane_id:int) =
//        fun (input:'T) (output:Ref<'T>) (identity:Ref<int> option) (scan_op:(int -> int -> int) option) (block_aggregate:Ref<int> option) (block_prefix_callback_op:Ref<int> option) =
//            (identity, scan_op, block_aggregate, block_prefix_callback_op) |> function
//            | Some identity, Some scan_op, Some block_aggregate, None ->
//                let warp_aggregate = __local__.Variable()
//                //WarpScan
//                ()
//            | Some identity, Some scan_op, Some block_aggregate, Some block_prefix_callback_op -> ()
//            | None, Some scan_op, Some block_aggregate, None -> ()
//            | None, Some scan_op, Some block_aggregate, Some block_prefix_callback_op -> ()
//            | None, None, Some block_aggregate, None -> ()
//            | None, None, Some block_aggregate, Some block_prefix_callback_op -> ()
//            | _, _, _, _ -> ()
