[<AutoOpen>]
module Alea.cuExtension.CUB.Warp.Scan

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Common

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Warp.WarpSpecializations

//type private InternalWarpScan<'T> =
//    | WarpScanShfl of WarpScanShfl.API<'T>
//    | WarpScanSmem of WarpScanSmem.API<'T>

module Template =
    [<AutoOpen>]
    module Host =
//        type InternalWarpScan<'T> =
//            | WarpScanShfl of WarpScanShfl.API<'T>
//            | WarpScanSmem of WarpScanSmem.API<'T>

        module Params =
            [<Record>]
            type API = 
                { LOGICAL_WARPS : int; LOGICAL_WARP_THREADS : int }
                static member Init(logical_warps, logical_warp_threads) = { LOGICAL_WARPS = logical_warps; LOGICAL_WARP_THREADS = logical_warp_threads }
                static member Init() = { LOGICAL_WARPS = 1; LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS }

        module Constants =            
            [<Record>] 
            type API = 
                { POW_OF_TWO : bool } 
                static member Init(p:Params.API) = { POW_OF_TWO = ((p.LOGICAL_WARP_THREADS &&& (p.LOGICAL_WARP_THREADS - 1)) = 0) }


        ///@TODO
//        let pickScanKind (p:Params.API) =
//            let c = Constants.API.Init(p)
//            ((CUB_PTX_VERSION >= 300) && ((p.LOGICAL_WARPS = 1) || c.POW_OF_TWO))
//
//        let  [<ReflectedDefinition>] inline (|WarpScanShfl|_|) (scan_op:'T -> 'T -> 'T) (p:Params.API) =
//            if pickScanKind p then
//                WarpScanShfl.template<'T> p.LOGICAL_WARPS p.LOGICAL_WARP_THREADS scan_op
//                |>      Some
//            else
//                None 
//
//        let  [<ReflectedDefinition>] inline (|WarpScanSmem|_|) (scan_op:'T -> 'T -> 'T) (p:Params.API) =
//            if pickScanKind p |> not then
//                WarpScanSmem.template<'T> p.LOGICAL_WARPS p.LOGICAL_WARP_THREADS scan_op
//                |>      Some
//            else
//            None
        type ScanKind =
            | Shfl = 0
            | Smem = 1

        type InternalScanHostApi =
            | WarpScanShflHostApi of WarpScanShfl.HostApi
            | WarpScanSmemHostApi of WarpScanSmem.HostApi

        type API =
            {
                ScanKind            : ScanKind
//                InternalScanHostApi : InternalScanHostApi
                WarpScanSmemHostApi : WarpScanSmem.HostApi
                WarpScanShflHostApi : WarpScanShfl.HostApi
                Params              : Params.API
                Constants           : Constants.API
                SharedMemoryLength  : int
            }

            static member Init(logical_warps, logical_warp_threads) =
                let p = Params.API.Init(logical_warps, logical_warp_threads)
                let c = Constants.API.Init(p)
                let wsSmem_h = WarpScanSmem.HostApi.Init(p.LOGICAL_WARPS, p.LOGICAL_WARP_THREADS)
                let wsShfl_h = WarpScanShfl.HostApi.Init(p.LOGICAL_WARPS, p.LOGICAL_WARP_THREADS)
                let kind = if (CUB_PTX_VERSION >= 300) && ((p.LOGICAL_WARPS = 1) || c.POW_OF_TWO) then ScanKind.Shfl else ScanKind.Smem
//                let internalHostApi = 
//                    kind |> function
//                    | ScanKind.Shfl -> WarpScanShflHostApi(wsShfl_h)
//                    | ScanKind.Smem -> WarpScanSmemHostApi(wsSmem_h)
//                    | _             -> WarpScanSmemHostApi(wsSmem_h)
//                { ScanKind = kind; InternalScanHostApi = internalHostApi; Params = p; Constants = c; SharedMemoryLength = wsSmem_h.SharedMemoryLength }
                { ScanKind = kind; WarpScanSmemHostApi = wsSmem_h; WarpScanShflHostApi = wsShfl_h; Params = p; Constants = c; SharedMemoryLength = wsSmem_h.SharedMemoryLength }

            static member Init() = API.Init(1, CUB_PTX_WARP_THREADS)

    module Device =
        module TempStorage =
            type [<Record>] API<'T> = WarpScanSmem.TempStorage<'T>
            
        //let [<ReflectedDefinition>] inline PrivateStorage<'T>(h:Host.API) = TempStorage.API<'T>.Uninitialized(h.SharedMemoryLength)

        [<Record>]
        type API<'T> = 
            { mutable temp_storage : TempStorage.API<'T>; mutable warp_id : int; mutable lane_id : int }
            
            [<ReflectedDefinition>] 
            static member Init(h:Host.API) =
                let p = h.Params
                {
                    temp_storage    = TempStorage.API<'T>.Uninitialized(h.SharedMemoryLength) //PrivateStorage<'T>(h)
                    warp_id 
                        = if p.LOGICAL_WARPS = 1 then 0 else threadIdx.x / p.LOGICAL_WARP_THREADS
                    lane_id 
                        = if ((p.LOGICAL_WARPS = 1) || (p.LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS)) then __ptx__.LaneId() else threadIdx.x % p.LOGICAL_WARP_THREADS
                }

            [<ReflectedDefinition>]
            static member Init(h:Host.API, temp_storage) =
                let p = h.Params
                {
                    temp_storage    = temp_storage
                    warp_id 
                        = if p.LOGICAL_WARPS = 1 then 0 else threadIdx.x / p.LOGICAL_WARP_THREADS
                    lane_id 
                        = if ((p.LOGICAL_WARPS = 1) || (p.LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS)) then __ptx__.LaneId() else threadIdx.x % p.LOGICAL_WARP_THREADS
                }

            [<ReflectedDefinition>]
            static member Init(h:Host.API, warp_id, lane_id) =
                { temp_storage = TempStorage.API<'T>.Uninitialized(h.SharedMemoryLength); warp_id = warp_id; lane_id = lane_id }                

            [<ReflectedDefinition>]
            static member Init(h:Host.API, temp_storage, warp_id, lane_id) =
                { temp_storage = temp_storage; warp_id = warp_id; lane_id = lane_id } 


    type _TemplateParams    = Host.Params.API
    type _Constants         = Host.Constants.API
    type _HostApi           = Host.API

    type _TempStorage<'T>   = Device.TempStorage.API<'T>
    type _DeviceApi<'T>     = Device.API<'T>

//    module InclusiveSum =
//        type _FunctionApi<'T> =
//            {
//                Default         : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> unit>
//                WithAggregate   : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> Ref<'T> -> unit>
//            }
//
//    module InclusiveScan =
//        type _FunctionApi<'T> =
//            {
//                Default         : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> unit>
//                WithAggregate   : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> Ref<'T> -> unit>                
//            }
//            
//    module ExclusiveSum =
//        type _FunctionApi<'T> =
//            {
//                Default         : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> unit>
//                WithAggregate   : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> Ref<'T> -> unit>                
//            }
//
//
//    module ExclusiveScan =
//        type _FunctionApi<'T> =
//            {
//                Default             : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> 'T -> unit>
//                Default_NoID        : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> unit>
//                WithAggregate       : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> 'T -> Ref<'T> -> unit>
//                WithAggregate_NoID  : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> Ref<'T> -> unit>
//            }


        
module InternalWarpScan =
    open Template

    ///@TODO
//    type InternalWarpScan<'T> =
//        | IWarpScanShfl of Template<WarpScanShfl.HostApi*WarpScanShfl.FunctionApi<'T>> //WarpScanShfl.API<'T>
//        | IWarpScanSmem of Template<WarpScanSmem.HostApi*WarpScanSmem.FunctionApi<'T>> //WarpScanSmem.API<'T>
//
//    module InclusiveSum =
//        let  [<ReflectedDefinition>] inline spec (h:_HostApi) (scan_op:'T -> 'T -> 'T) =
//            let p = h.Params 
//            p |> function
//            | WarpScanShfl scan_op wsShfl -> IWarpScanShfl(wsShfl)
//            | WarpScanSmem scan_op wsSmem -> IWarpScanSmem(wsSmem)
//            | _ -> failwith "Invalid Template Parameters"
    
    module InclusiveSum = 
        let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
            (d:_DeviceApi<'T>)
            (input:'T) (output:Ref<'T>) = 
//            h.ScanKind |> function
//            | ScanKind.Shfl ->
                let wsShfl_d = WarpScanShfl.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
                WarpScanShfl.InclusiveSum.Default h.WarpScanShflHostApi wsShfl_d input output
//            | _ ->
//                let wsSmem_d = WarpScanSmem.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                WarpScanSmem.InclusiveSum.Default h.WarpScanSmemHostApi scan_op wsSmem_d input output
//

        let [<ReflectedDefinition>] inline DefaultInt (h:_HostApi)
            (d:_DeviceApi<int>)
            (input:int) (output:Ref<int>) = 
//            h.ScanKind |> function
//            | ScanKind.Shfl ->
                let wsShfl_d = WarpScanShfl.DeviceApi<int>.Init(d.temp_storage, d.warp_id, d.lane_id)
                WarpScanShfl.InclusiveSum.Default h.WarpScanShflHostApi wsShfl_d input output
//            | _ ->
//                let wsSmem_d = WarpScanSmem.DeviceApi<int>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                WarpScanSmem.InclusiveSum.DefaultInt h.WarpScanSmemHostApi wsSmem_d input output


        let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
            (d:_DeviceApi<'T>)
            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//            h.ScanKind |> function
//            | ScanKind.Shfl ->
                let wsShfl_d = WarpScanShfl.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
                WarpScanShfl.InclusiveSum.Generic h.WarpScanShflHostApi wsShfl_d input output warp_aggregate
//            | _ ->
//                let wsSmem_d = WarpScanSmem.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                WarpScanSmem.InclusiveSum.WithAggregate h.WarpScanSmemHostApi scan_op wsSmem_d input output warp_aggregate
//      
        let [<ReflectedDefinition>] inline WithAggregateInt (h:_HostApi)
            (d:_DeviceApi<int>)
            (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) =
//            h.ScanKind |> function
//            | ScanKind.Shfl ->
//                let wsShfl_d = WarpScanShfl.DeviceApi<int>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                WarpScanShfl.InclusiveSum.Generic h.WarpScanShflHostApi wsShfl_d input output warp_aggregate
//            | _ ->
                let wsSmem_d = WarpScanSmem.DeviceApi<int>.Init(d.temp_storage, d.warp_id, d.lane_id)
                WarpScanSmem.InclusiveSum.WithAggregateInt h.WarpScanSmemHostApi wsSmem_d input output warp_aggregate
//      


//    module InclusiveScan =
//        let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:_DeviceApi<'T>) 
//            (input:'T) (output:Ref<'T>) =
//            h.ScanKind |> function
//            | ScanKind.Shfl ->
//                let wsShfl_d = WarpScanShfl.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                WarpScanShfl.InclusiveScan.Default h.WarpScanShflHostApi scan_op wsShfl_d input output
//            | _ ->
//                let wsSmem_d = WarpScanSmem.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                WarpScanSmem.InclusiveScan.Default h.WarpScanSmemHostApi scan_op wsSmem_d input output
//
//        let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:_DeviceApi<'T>)
//            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//            h.ScanKind |> function
//            | ScanKind.Shfl ->
//                let wsShfl_d = WarpScanShfl.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                WarpScanShfl.InclusiveScan.WithAggregate h.WarpScanShflHostApi scan_op wsShfl_d input output
//            | _ ->
//                let wsSmem_d = WarpScanSmem.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                WarpScanSmem.InclusiveScan.WithAggregate h.WarpScanSmemHostApi scan_op wsSmem_d input output
//
//
//    module ExclusiveScan =
//        let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:_DeviceApi<'T>)
//            (input:'T) (output:Ref<'T>) (identity:'T) =
//            h.ScanKind |> function
//            | ScanKind.Shfl ->
//                let wsShfl_d = WarpScanShfl.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                WarpScanShfl.ExclusiveScan.Default h.WarpScanShflHostApi scan_op wsShfl_d input output identity
//            | _ ->
//                let wsSmem_d = WarpScanSmem.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                WarpScanSmem.ExclusiveScan.Default h.WarpScanSmemHostApi scan_op wsSmem_d input output identity
//
//        let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:_DeviceApi<'T>)
//            (input:'T) (output:Ref<'T>) (identity:'T) (warp_aggregate:Ref<'T>) =
//            h.ScanKind |> function
//            | ScanKind.Shfl ->
//                let wsShfl_d = WarpScanShfl.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                WarpScanShfl.ExclusiveScan.WithAggregate h.WarpScanShflHostApi scan_op wsShfl_d input output identity warp_aggregate
//            | _ ->
//                let wsSmem_d = WarpScanSmem.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                WarpScanSmem.ExclusiveScan.WithAggregate h.WarpScanSmemHostApi scan_op wsSmem_d input output identity warp_aggregate
//
//
//        module Identityless =
//            let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//                (d:_DeviceApi<'T>)
//                (input:'T) (output:Ref<'T>) =
//                h.ScanKind |> function
//                | ScanKind.Shfl ->
//                    let wsShfl_d = WarpScanShfl.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                    WarpScanShfl.ExclusiveScan.Identityless.Default h.WarpScanShflHostApi scan_op wsShfl_d input output
//                | _ ->
//                    let wsSmem_d = WarpScanSmem.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                    WarpScanSmem.ExclusiveScan.Identityless.Default h.WarpScanSmemHostApi scan_op wsSmem_d input output
//
//            let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//                (d:_DeviceApi<'T>)
//                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//                h.ScanKind |> function
//                | ScanKind.Shfl ->
//                    let wsShfl_d = WarpScanShfl.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                    WarpScanShfl.ExclusiveScan.Identityless.WithAggregate h.WarpScanShflHostApi scan_op wsShfl_d input output warp_aggregate
//                | _ ->
//                    let wsSmem_d = WarpScanSmem.DeviceApi<'T>.Init(d.temp_storage, d.warp_id, d.lane_id)
//                    WarpScanSmem.ExclusiveScan.Identityless.WithAggregate h.WarpScanSmemHostApi scan_op wsSmem_d input output warp_aggregate
//            
//
//
module InclusiveSum =
    open Template

    let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
        (d:_DeviceApi<'T>)
        (input:'T) (output:Ref<'T>) =         
        InternalWarpScan.InclusiveSum.Default h scan_op d input output

    let [<ReflectedDefinition>] inline DefaultInt (h:_HostApi)
        (d:_DeviceApi<int>)
        (input:int) (output:Ref<int>) =         
        InternalWarpScan.InclusiveSum.DefaultInt h d input output


    let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
        (d:_DeviceApi<'T>) 
        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =         
        InternalWarpScan.InclusiveSum.WithAggregate h scan_op d input output warp_aggregate

    let [<ReflectedDefinition>] inline WithAggregateInt (h:_HostApi)
        (d:_DeviceApi<int>) 
        (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) =         
        InternalWarpScan.InclusiveSum.WithAggregateInt h d input output warp_aggregate

module private PrivateExclusiveSum =
    open Template

    let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T  -> 'T)
        (d:_DeviceApi<'T>)
        (input:'T) (output:Ref<'T>) =
        let inclusive = __local__.Variable<'T>()
        InclusiveSum.Default h scan_op d input output
        output := scan_op !inclusive input
        

    let [<ReflectedDefinition>] inline DefaultInt (h:_HostApi)
        (d:_DeviceApi<int>)
        (input:int) (output:Ref<int>) =
        let inclusive = __local__.Variable<int>()
        InclusiveSum.DefaultInt h d input output
        output := !inclusive + input

    let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
        (d:_DeviceApi<'T>)
        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
        let inclusive = __local__.Variable<'T>()
        InclusiveSum.WithAggregate h scan_op d input inclusive warp_aggregate
        output := scan_op !inclusive input
        

    let [<ReflectedDefinition>] inline WithAggregateInt (h:_HostApi)
        (d:_DeviceApi<int>)
        (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) =
        let inclusive = __local__.Variable<int>()
        InclusiveSum.WithAggregateInt h d input inclusive warp_aggregate
        output := !inclusive + input


module ExclusiveSum =
    open Template

//    let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//        (d:_DeviceApi<'T>)
//        (input:'T) (output:Ref<'T>) = 
//        PrivateExclusiveSum.Default h scan_op d input output
  
    let [<ReflectedDefinition>] inline DefaultInt (h:_HostApi)
        (d:_DeviceApi<int>)
        (input:int) (output:Ref<int>) = 
        PrivateExclusiveSum.DefaultInt h d input output
  
    
//    let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//        (d:_DeviceApi<'T>)
//        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//        PrivateExclusiveSum.WithAggregate h scan_op d input output warp_aggregate
//    
    let [<ReflectedDefinition>] inline WithAggregateInt (h:_HostApi)
        (d:_DeviceApi<int>)
        (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) =
        PrivateExclusiveSum.WithAggregateInt h d input output warp_aggregate

//module InclusiveScan =
//    open Template
//
//    let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//        (d:_DeviceApi<'T>)
//        (input:'T) (output:Ref<'T>) = 
//        InternalWarpScan.InclusiveScan.Default h scan_op d input output
//
//    let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//        (d:_DeviceApi<'T>)
//        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) = 
//        InternalWarpScan.InclusiveScan.WithAggregate h scan_op d input output warp_aggregate
//
//
//
//module ExclusiveScan =
//    open Template
//
//    let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//        (d:_DeviceApi<'T>)
//        (input:'T) (output:Ref<'T>) (identity:'T) = 
//        InternalWarpScan.ExclusiveScan.Default h scan_op d input output identity
//
//    let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//        (d:_DeviceApi<'T>)
//        (input:'T) (output:Ref<'T>) (identity:'T) (warp_aggregate:Ref<'T>) = 
//        InternalWarpScan.ExclusiveScan.WithAggregate h scan_op d input output identity warp_aggregate
//
////    let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:_HostApi) (scan_op:'T -> 'T  -> 'T) =
////        <@ fun (d:_DeviceApi<'T>) (input:'T) (output:Ref<'T>) (identity:'T) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) -> () 
//
//    module Identityless =
//        let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:_DeviceApi<'T>)
//            (input:'T) (output:Ref<'T>) = 
//            InternalWarpScan.ExclusiveScan.Identityless.Default h scan_op d input output
//
//        let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:_DeviceApi<'T>)
//            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) = 
//            InternalWarpScan.ExclusiveScan.Identityless.WithAggregate h scan_op d input output warp_aggregate
//
////        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (h:_HostApi) (scan_op:'T -> 'T  -> 'T) =
////            <@ fun (d:_DeviceApi<'T>) (input:'T) (output:Ref<'T>) (block_aggregate:Ref<'T>) (block_prefix_callback_op:Ref<'T -> 'T>) -> () 
//



[<Record>]
module WarpScan =
    
    type TemplateParams     = Template._TemplateParams
    type Constants          = Template._Constants
    type TempStorage<'T>    = Template._TempStorage<'T>

    type HostApi            = Template._HostApi
    type DeviceApi<'T>      = Template._DeviceApi<'T>

    //let [<ReflectedDefinition>] inline PrivateStorage<'T>() = TempStorage<'T>.Uninitialized()

    [<Record>]
    type API<'T> =
        {
            mutable DeviceApi : DeviceApi<'T>
        }

        [<ReflectedDefinition>] static member Create(h) 
            : API<'T> = { DeviceApi = DeviceApi<'T>.Init(h) }

        [<ReflectedDefinition>] static member Create(h, temp_storage)
            : API<'T> = { DeviceApi = DeviceApi<'T>.Init(h, temp_storage) }

        [<ReflectedDefinition>] static member Create(h, warp_id, lane_id)
            : API<'T> = { DeviceApi = DeviceApi<'T>.Init(h, warp_id, lane_id) }

        [<ReflectedDefinition>] static member Create(h, temp_storage, warp_id, lane_id)
            : API<'T> = { DeviceApi = DeviceApi<'T>.Init(h, temp_storage, warp_id, lane_id) }

        //^T when ^T : (static member (+): ^T * ^T -> ^T)
        [<ReflectedDefinition>] member this.InclusiveSum(h, scan_op, input, output)
            = InternalWarpScan.InclusiveSum.Default h scan_op this.DeviceApi input output

        [<ReflectedDefinition>] member this.InclusiveSum(h, scan_op, input, output, warp_aggregate)
            = InternalWarpScan.InclusiveSum.WithAggregate h scan_op this.DeviceApi input output warp_aggregate

//        [<ReflectedDefinition>] member this.InclusiveScan(h, scan_op, input, output)
//            = InternalWarpScan.InclusiveScan.Default h scan_op this.DeviceApi input output
//    
//        [<ReflectedDefinition>] member this.InclusiveScan(h, scan_op, input, output, warp_aggregate)
//            = InternalWarpScan.InclusiveScan.WithAggregate h scan_op this.DeviceApi input output warp_aggregate
//
//        [<ReflectedDefinition>] member this.ExclusiveSum(h, scan_op, input, output)
//            = ExclusiveSum.Default h scan_op (DeviceApi<int>.Init(h)) input output


        [<ReflectedDefinition>] member this.ExclusiveSumInt(h, input, output)
            = ExclusiveSum.DefaultInt h (DeviceApi<int>.Init(h)) input output

        [<ReflectedDefinition>] member this.ExclusiveSumInt(h, input, output, warp_aggregate)
            = ExclusiveSum.WithAggregateInt h (DeviceApi<int>.Init(h)) input output warp_aggregate
    
//        [<ReflectedDefinition>] member this.ExclusiveSum(h, scan_op, input, output, warp_aggregate)
//            = ExclusiveSum.WithAggregate h scan_op this.DeviceApi input output warp_aggregate
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity)
//            = InternalWarpScan.ExclusiveScan.Default h scan_op this.DeviceApi input output identity
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity, warp_aggregate)
//            = InternalWarpScan.ExclusiveScan.WithAggregate h scan_op this.DeviceApi input output identity warp_aggregate
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output)
//            = InternalWarpScan.ExclusiveScan.Identityless.Default h scan_op this.DeviceApi input output
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, warp_aggregate)
//            = InternalWarpScan.ExclusiveScan.Identityless.WithAggregate h scan_op this.DeviceApi input output warp_aggregate

//    module InclusiveSum =
//        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:DeviceApi<'T>)
//            (input:'T) (output:Ref<'T>) =
//            InclusiveSum.Default h scan_op d input output
//
//        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:DeviceApi<'T>)
//            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//            InclusiveSum.WithAggregate h scan_op d input output warp_aggregate

//    module InclusiveScan =
//        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:DeviceApi<'T>)
//            (input:'T) (output:Ref<'T>) =
//            InclusiveScan.Default h scan_op d input output
//
//        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:DeviceApi<'T>)
//            (input:'T) (output:Ref<'T>) =
//            InclusiveScan.WithAggregate h scan_op d input output

    module ExclusiveSum =
//        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:DeviceApi<'T>)
//            (input:'T) (output:Ref<'T>) = 
//            ExclusiveSum.Default h scan_op d input output
//
//        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:DeviceApi<'T>)
//            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//            ExclusiveSum.WithAggregate h scan_op d input output

        let [<ReflectedDefinition>] inline WithAggregateInt (h:HostApi)
            (d:DeviceApi<int>)
            (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) =
            ExclusiveSum.WithAggregateInt h d input output warp_aggregate

//    module ExclusiveScan =
////        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (d:DeviceApi<'T>)
////            (input:'T) (output:Ref<'T>) (identity:'T) =
////            ExclusiveScan.Default h scan_op d input output identity
////
////        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (d:DeviceApi<'T>)
////            (input:'T) (output:Ref<'T>) (identity:'T) (warp_aggregate:Ref<'T>) =
////            ExclusiveScan.WithAggregate h scan_op d input output identity warp_aggregate
//        
//        module Identityless =
//            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
//                (d:DeviceApi<'T>)
//                (input:'T) (output:Ref<'T>) =
//                ExclusiveScan.Identityless.Default h scan_op d input output
//
//            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
//                (d:DeviceApi<'T>)
//                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//                ExclusiveScan.Identityless.WithAggregate h scan_op d input output warp_aggregate

//    module InclusiveSum =
//        type FunctionApi<'T> = Template.InclusiveSum._FunctionApi<'T>
//
//        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) = InclusiveSum.Default h scan_op
//        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) = InclusiveSum.WithAggregate h scan_op
//        
//        let  [<ReflectedDefinition>] inline template<'T> (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
//            let h = HostApi.Init(logical_warps, logical_warp_threads)
//            
//            let! dfault = (h, scan_op) ||> InclusiveSum.Default |> Compiler.DefineFunction
//            let! waggr  = (h, scan_op) ||> InclusiveSum.WithAggregate |> Compiler.DefineFunction
//
//            return h, {
//                Default         = dfault
//                WithAggregate   = waggr
//            }}
//
//    module InclusiveScan =
//        type FunctionApi<'T> = Template.InclusiveScan._FunctionApi<'T>
//
//        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) = InclusiveScan.Default h scan_op
//        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) = InclusiveScan.WithAggregate h scan_op
//
//        let  [<ReflectedDefinition>] inline template<'T> (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
//            let h = HostApi.Init(logical_warps, logical_warp_threads)
//            
//            let! dfault = (h, scan_op) ||> InclusiveScan.Default |> Compiler.DefineFunction
//            let! waggr  = (h, scan_op) ||> InclusiveScan.WithAggregate |> Compiler.DefineFunction
//
//            return h, {
//                Default         = dfault
//                WithAggregate   = waggr
//            }}
//
//    module ExclusiveSum =
//        type FunctionApi<'T> = Template.ExclusiveSum._FunctionApi<'T>
//
//        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) = ExclusiveSum.Default h scan_op
//        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) = ExclusiveSum.WithAggregate h scan_op       
//
//        let  [<ReflectedDefinition>] inline template<'T> (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
//            let h = HostApi.Init(logical_warps, logical_warp_threads)
//            
//            let! dfault = (h, scan_op) ||> ExclusiveSum.Default |> Compiler.DefineFunction
//            let! waggr  = (h, scan_op) ||> ExclusiveSum.WithAggregate |> Compiler.DefineFunction
//
//            return h, {
//                Default         = dfault
//                WithAggregate   = waggr
//            }}
//
//    module ExclusiveScan =
//        type FunctionApi<'T> = Template.ExclusiveScan._FunctionApi<'T>
//
//        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) = ExclusiveScan.Default h scan_op
//        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) = ExclusiveScan.WithAggregate h scan_op       
//
//
//        let  [<ReflectedDefinition>] inline template<'T> (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
//            let h = HostApi.Init(logical_warps, logical_warp_threads)
//        
//            let! dfault = (h, scan_op) ||> ExclusiveScan.Default                    |> Compiler.DefineFunction
//            let! dfaultnoid = (h, scan_op) ||> ExclusiveScan.Identityless.Default   |> Compiler.DefineFunction
//            let! waggr = (h, scan_op) ||> ExclusiveScan.WithAggregate |> Compiler.DefineFunction
//            let! waggrnoid = (h, scan_op) ||> ExclusiveScan.Identityless.WithAggregate |> Compiler.DefineFunction
//
//            return h, {
//                Default             = dfault
//                Default_NoID        = dfaultnoid
//                WithAggregate       = waggr
//                WithAggregate_NoID  = waggrnoid
//            }}
//        module Identityless =
//            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T) = ExclusiveScan.Identityless.Default h scan_op
//            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T) = ExclusiveScan.Identityless.WithAggregate h scan_op       
//
//    type FunctionApi<'T> =
//        {
//            InclusiveSum    : InclusiveSum.FunctionApi<'T>
//            InclusiveScan   : InclusiveScan.FunctionApi<'T>
//            ExclusiveSum    : ExclusiveSum.FunctionApi<'T>
//            ExclusiveScan   : ExclusiveScan.FunctionApi<'T>        
//        }
//
//    let  [<ReflectedDefinition>] inline template<'T> (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
//        let! h, inclusiveSum = InclusiveSum.template<'T> logical_warps logical_warp_threads scan_op
//        let! _, inclusiveScan = InclusiveScan.template<'T> logical_warps logical_warp_threads scan_op
//        let!  _, exclusiveSum = ExclusiveSum.template<'T> logical_warps logical_warp_threads scan_op
//        let!  _, exclusiveScan = ExclusiveScan.template<'T> logical_warps logical_warp_threads scan_op
//        
//        return h, {
//            InclusiveSum    = inclusiveSum
//            InclusiveScan   = inclusiveScan
//            ExclusiveSum    = exclusiveSum
//            ExclusiveScan   = exclusiveScan
//        }}


//    module InclusiveSum = 
//        let [<ReflectedDefinition>] inline Default (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
//            let template = _Template<'T>.Init(logical_warps, logical_warp_threads)
//            InclusiveSum.Default template scan_op
//
//        let [<ReflectedDefinition>] inline WithAggregate (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
//            let template = _Template<'T>.Init(logical_warps, logical_warp_threads)
//            InclusiveSum.WithAggregate template scan_op
//
//        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
//            let template = _Template<'T>.Init(logical_warps, logical_warp_threads)
//            InclusiveSum.WithAggregateAndCallbackOp template scan_op
//
////        type API<'T> =
////            {
////                template : _Template<'T>
////            }
////            [<ReflectedDefinition>] member this.Default = 
//
//    module InclusiveScan =
//        let [<ReflectedDefinition>] inline Default (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
//            let template = _Template<'T>.Init(logical_warps, logical_warp_threads)
//            InclusiveScan.Default template scan_op
//
//        let [<ReflectedDefinition>] inline WithAggregate (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
//            let template = _Template<'T>.Init(logical_warps, logical_warp_threads)
//            InclusiveScan.WithAggregate template scan_op
//
////        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) =  
////            let template = _Template<'T>.Init(logical_warps, logical_warp_threads)
////            InclusiveScan.WithAggregateAndCallbackOp template scan_op
//
//    module ExclusiveSum =
//        let [<ReflectedDefinition>] inline Default (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
//            let template = _Template<'T>.Init(logical_warps, logical_warp_threads)
//            ExclusiveSum.Default template scan_op
//
//        let [<ReflectedDefinition>] inline WithAggregate (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
//            let template = _Template<'T>.Init(logical_warps, logical_warp_threads)
//            ExclusiveSum.WithAggregate template scan_op
//
//        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) =  
//            let template = _Template<'T>.Init(logical_warps, logical_warp_threads)
//            ExclusiveSum.WithAggregateAndCallbackOp template scan_op
//
//    module ExclusiveScan =
//        let [<ReflectedDefinition>] inline Default (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
//            let template = _Template<'T>.Init(logical_warps, logical_warp_threads)
//            ExclusiveScan.Default template scan_op
//
//        let [<ReflectedDefinition>] inline WithAggregate (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) = 
//            let template = _Template<'T>.Init(logical_warps, logical_warp_threads)
//            ExclusiveScan.WithAggregate template scan_op
//
//        let [<ReflectedDefinition>] inline WithAggregateAndCallbackOp (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) =  
//            let template = _Template<'T>.Init(logical_warps, logical_warp_threads)
//            ExclusiveScan.WithAggregateAndCallbackOp template scan_op


//    [<Record>]
//    type API<'T> =
//        {
//            Constants       : _Constants
//            ThreadFields    : _ThreadFields<'T>
//            InclusiveSum    : InclusiveSum.API<'T>
//            InclusiveScan   : InclusiveScan.API<'T>
//            ExclusiveSum    : ExclusiveSum.API<'T>
//            ExclusiveScan   : ExclusiveScan.API<'T>
//        }

//        [<ReflectedDefinition>]
//        member this.Init(temp_storage) = 
//            this.ThreadFields.temp_storage <- temp_storage
//            this
//                    
//        [<ReflectedDefinition>]
//        member this.Init(warp_id, lane_id) = 
//            this.ThreadFields.warp_id <- warp_id
//            this.ThreadFields.lane_id <- lane_id
//            this
//
//        [<ReflectedDefinition>]
//        member this.Init(temp_storage, warp_id, lane_id) = 
//            this.ThreadFields.temp_storage  <- temp_storage
//            this.ThreadFields.warp_id       <- warp_id
//            this.ThreadFields.lane_id       <- lane_id
//            this
//
//        [<ReflectedDefinition>]
//        static member Create(logical_warps, logical_warp_threads, scan_op) =
//            let template = _TemplateParams<'T>.Init(logical_warps, logical_warp_threads)
//            let tf = _ThreadFields<'T>.Init(tp)
//            {
//                Constants       =   _Constants.Init template
//                ThreadFields    =   tf
//                InclusiveSum    =   InclusiveSum.api template scan_op                                    
//                InclusiveScan   =   InclusiveScan.api template scan_op
//                ExclusiveSum    =   ExclusiveSum.api template scan_op
//                ExclusiveScan   =   ExclusiveScan.api template scan_op
//            }
//
//        [<ReflectedDefinition>]
//        static member Create(template:_Template<'T>) = API<'T>.Create(tp.LOGICAL_WARPS, LOGICAL_WARP_THREADS)
//
//
//
//    let [<ReflectedDefinition>] api (template:_Template<'T>) =
//        let tf = _ThreadFields<'T>.Init(tp)
//        {
//            Constants       =   _Constants.Init template
//            ThreadFields    =   tf
//            InclusiveSum    =   InclusiveSum.api template scan_op                                    
//            InclusiveScan   =   InclusiveScan.api template scan_op
//            ExclusiveSum    =   ExclusiveSum.api template scan_op
//            ExclusiveScan   =   ExclusiveScan.api template scan_op
//        }