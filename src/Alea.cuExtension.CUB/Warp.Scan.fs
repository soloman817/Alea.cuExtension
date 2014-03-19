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



module WarpScan =
    
    type Params = 
        { LOGICAL_WARPS : int; LOGICAL_WARP_THREADS : int }
            static member Init(logical_warps, logical_warp_threads) = 
                { LOGICAL_WARPS = logical_warps; LOGICAL_WARP_THREADS = logical_warp_threads }
            
            static member Init() = 
                { LOGICAL_WARPS = 1; LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS }

    type Constants = 
        { POW_OF_TWO : bool } 
        static member Init(p:Params) = 
            { POW_OF_TWO = ((p.LOGICAL_WARP_THREADS &&& (p.LOGICAL_WARP_THREADS - 1)) = 0) }


    type ScanKind =
        | Shfl = 0
        | Smem = 1

    type InternalScanHostApi =
        | WarpScanShflHostApi of WarpScanShfl.HostApi
        | WarpScanSmemHostApi of WarpScanSmem.HostApi

    type HostApi =
        {
            ScanKind            : ScanKind
            WarpScanSmemHostApi : WarpScanSmem.HostApi
            WarpScanShflHostApi : WarpScanShfl.HostApi
            Params              : Params
            Constants           : Constants
            SharedMemoryLength  : int
        }

        static member Init(logical_warps, logical_warp_threads) =
            let p = Params.Init(logical_warps, logical_warp_threads)
            let c = Constants.Init(p)
            let wsSmem_h = WarpScanSmem.HostApi.Init(p.LOGICAL_WARPS, p.LOGICAL_WARP_THREADS)
            let wsShfl_h = WarpScanShfl.HostApi.Init(p.LOGICAL_WARPS, p.LOGICAL_WARP_THREADS)
            let kind = if (CUB_PTX_VERSION >= 300) && ((p.LOGICAL_WARPS = 1) || c.POW_OF_TWO) then ScanKind.Shfl else ScanKind.Smem
            { ScanKind = kind; WarpScanSmemHostApi = wsSmem_h; WarpScanShflHostApi = wsShfl_h; Params = p; Constants = c; SharedMemoryLength = wsSmem_h.SharedMemoryLength }

        static member Init() = HostApi.Init(1, CUB_PTX_WARP_THREADS)
        static member Init(logical_warps) = HostApi.Init(logical_warps, CUB_PTX_WARP_THREADS)

    
    type TempStorage<'T> = WarpScanSmem.TempStorage<'T>

    module InternalWarpScan =
    
        module InclusiveSum = 

            let [<ReflectedDefinition>] inline DefaultInt (h:HostApi)
                (temp_storage:TempStorage<int>) (warp_id:uint32) (lane_id:uint32)
                (input:int) (output:Ref<int>) =
                WarpScanSmem.InclusiveSum.DefaultInt h.WarpScanSmemHostApi temp_storage warp_id lane_id input output

            let [<ReflectedDefinition>] inline WithAggregateInt (h:HostApi)
                (temp_storage:TempStorage<int>) (warp_id:uint32) (lane_id:uint32)
                (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) =
                WarpScanSmem.InclusiveSum.WithAggregateInt h.WarpScanSmemHostApi temp_storage warp_id lane_id input output warp_aggregate                

            let [<ReflectedDefinition>] inline _Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
                (input:'T) (output:Ref<'T>) = 
                WarpScanSmem.InclusiveSum.Default h.WarpScanSmemHostApi scan_op temp_storage (warp_id |> uint32) (lane_id |> uint32) input output


            let [<ReflectedDefinition>] inline _WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
                WarpScanSmem.InclusiveSum.WithAggregate h.WarpScanSmemHostApi scan_op temp_storage warp_id lane_id input output warp_aggregate


            let [<ReflectedDefinition>] inline Default (h:HostApi)
                (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
                (input:'T) (output:Ref<'T>) =
                _Default h (+) temp_storage warp_id lane_id input output

            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi)
                (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
                _WithAggregate h (+) temp_storage warp_id lane_id input output warp_aggregate


        module InclusiveScan =
            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32) 
                (input:'T) (output:Ref<'T>) =
                WarpScanSmem.InclusiveScan.Default h.WarpScanSmemHostApi scan_op temp_storage warp_id lane_id input output
    
            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
                WarpScanSmem.InclusiveScan.WithAggregate h.WarpScanSmemHostApi scan_op temp_storage warp_id lane_id input output
    
    
        module ExclusiveScan =
            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
                (input:'T) (output:Ref<'T>) (identity:'T) =
                WarpScanSmem.ExclusiveScan.Default h.WarpScanSmemHostApi scan_op temp_storage warp_id lane_id input output identity
    
            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
                (input:'T) (output:Ref<'T>) (identity:'T) (warp_aggregate:Ref<'T>) =
                WarpScanSmem.ExclusiveScan.WithAggregate h.WarpScanSmemHostApi scan_op temp_storage warp_id lane_id input output identity warp_aggregate
    
    
            module Identityless =
                let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
                    (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
                    (input:'T) (output:Ref<'T>) =
                    WarpScanSmem.ExclusiveScan.Identityless.Default h.WarpScanSmemHostApi scan_op temp_storage warp_id lane_id input output
    
                let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
                    (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
                    (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
                    WarpScanSmem.ExclusiveScan.Identityless.WithAggregate h.WarpScanSmemHostApi scan_op temp_storage warp_id lane_id input output warp_aggregate
                
    
 

    
    module PrivateExclusiveSum =
        open InternalWarpScan

        let [<ReflectedDefinition>] inline Default (h:HostApi)
            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
            (input:'T) (output:Ref<'T>) =
            let inclusive = __local__.Variable<'T>()
            InclusiveSum.Default h temp_storage warp_id lane_id input output
            output := !inclusive + input
        
        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi)
            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
            let inclusive = __local__.Variable<'T>()
            InclusiveSum.WithAggregate h temp_storage warp_id lane_id input inclusive warp_aggregate
            output := !inclusive + input

//        let [<ReflectedDefinition>] inline Default (h:HostApi)
//            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
//            (input:'T) (output:Ref<'T>) =
//            _Default h (+) temp_storage warp_id lane_id input output
//        
//        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi)
//            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
//            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//            _WithAggregate h (+) temp_storage warp_id lane_id input output warp_aggregate

        let [<ReflectedDefinition>] inline DefaultInt (h:HostApi)
            (temp_storage:TempStorage<int>) (warp_id:uint32) (lane_id:uint32)
            (input:int) (output:Ref<int>) =
            let inclusive = __local__.Variable<int>()
            InclusiveSum.DefaultInt h temp_storage warp_id lane_id input output
            output := !inclusive + input
                    

        let [<ReflectedDefinition>] inline WithAggregateInt (h:HostApi)
            (temp_storage:TempStorage<int>) (warp_id:uint32) (lane_id:uint32)
            (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) =
            let inclusive = __local__.Variable<int>()
            InclusiveSum.WithAggregateInt h temp_storage warp_id lane_id input inclusive warp_aggregate
            output := !inclusive + input


    module ExclusiveSum =

        let [<ReflectedDefinition>] inline Default (h:HostApi)
            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
            (input:'T) (output:Ref<'T>) = 
            PrivateExclusiveSum.Default h temp_storage warp_id lane_id input output

        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi)
            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
            PrivateExclusiveSum.WithAggregate h temp_storage warp_id lane_id input output warp_aggregate
          

        let [<ReflectedDefinition>] inline DefaultInt (h:HostApi)
            (temp_storage:TempStorage<int>) (warp_id:uint32) (lane_id:uint32)
            (input:int) (output:Ref<int>) = 
            PrivateExclusiveSum.DefaultInt h temp_storage warp_id lane_id input output


        let [<ReflectedDefinition>] inline WithAggregateInt (h:HostApi)
            (temp_storage:TempStorage<int>) (warp_id:uint32) (lane_id:uint32)
            (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) =
            PrivateExclusiveSum.WithAggregateInt h temp_storage warp_id lane_id input output warp_aggregate
  
    


    module InclusiveScan =
        open Template
    
        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
            (input:'T) (output:Ref<'T>) = 
            InternalWarpScan.InclusiveScan.Default h scan_op temp_storage warp_id lane_id input output
    
        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) = 
            InternalWarpScan.InclusiveScan.WithAggregate h scan_op temp_storage warp_id lane_id input output warp_aggregate
    
    
    
    module ExclusiveScan =
        open Template
    
        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
            (input:'T) (output:Ref<'T>) (identity:'T) = 
            InternalWarpScan.ExclusiveScan.Default h scan_op temp_storage warp_id lane_id input output identity
    
        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
            (input:'T) (output:Ref<'T>) (identity:'T) (warp_aggregate:Ref<'T>) = 
            InternalWarpScan.ExclusiveScan.WithAggregate h scan_op temp_storage warp_id lane_id input output identity warp_aggregate
    
    
        module Identityless =
            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
                (input:'T) (output:Ref<'T>) = 
                InternalWarpScan.ExclusiveScan.Identityless.Default h scan_op temp_storage warp_id lane_id input output
    
            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) = 
                InternalWarpScan.ExclusiveScan.Identityless.WithAggregate h scan_op temp_storage warp_id lane_id input output warp_aggregate
    
    
    
    
    let [<ReflectedDefinition>] inline PrivateStorage<'T>(h:HostApi) = TempStorage<'T>.Uninitialized(h.WarpScanSmemHostApi)


    [<Record>]
    type IntApi = 
        { mutable temp_storage : TempStorage<int>; mutable warp_id : int; mutable lane_id : int }
            
        [<ReflectedDefinition>] 
        static member Init(h:HostApi) =
            let p = h.Params
            {
                temp_storage    = PrivateStorage<int>(h)
                warp_id 
                    = if p.LOGICAL_WARPS = 1 then 0 else threadIdx.x / p.LOGICAL_WARP_THREADS
                lane_id 
                    = if ((p.LOGICAL_WARPS = 1) || (p.LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS)) then __ptx__.LaneId() |> int else threadIdx.x % p.LOGICAL_WARP_THREADS
            }

        [<ReflectedDefinition>]
        static member Init(h:HostApi, temp_storage) =
            let p = h.Params
            {
                temp_storage    = temp_storage
                warp_id 
                    = if p.LOGICAL_WARPS = 1 then 0 else threadIdx.x / p.LOGICAL_WARP_THREADS
                lane_id 
                    = if ((p.LOGICAL_WARPS = 1) || (p.LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS)) then __ptx__.LaneId() |> int else threadIdx.x % p.LOGICAL_WARP_THREADS
            }

        [<ReflectedDefinition>]
        static member Init(h:HostApi, warp_id, lane_id) =
            { temp_storage = PrivateStorage<int>(h); warp_id = warp_id; lane_id = lane_id }                

        [<ReflectedDefinition>]
        static member Init(h:HostApi, temp_storage, warp_id, lane_id) =
            { temp_storage = temp_storage; warp_id = warp_id; lane_id = lane_id } 


        [<ReflectedDefinition>] member this.ExclusiveSum(h, input, output)
            = ExclusiveSum.DefaultInt h this.temp_storage (this.warp_id |> uint32) (this.lane_id |> uint32) input output

        [<ReflectedDefinition>] member this.ExclusiveSum(h, input, output, warp_aggregate)
            = ExclusiveSum.WithAggregateInt h this.temp_storage (this.warp_id |> uint32) (this.lane_id |> uint32) input output warp_aggregate
    


//    type API<'T when 'T : static member 'T -> get_zero> = 
    [<Record>]
    type API<'T> = 
        { mutable temp_storage : TempStorage<'T>; mutable warp_id : int; mutable lane_id : int }
            
        [<ReflectedDefinition>] 
        static member Init(h:HostApi) =
            let p = h.Params
            {
                temp_storage    = PrivateStorage<'T>(h)
                warp_id 
                    = if p.LOGICAL_WARPS = 1 then 0 else threadIdx.x / p.LOGICAL_WARP_THREADS
                lane_id 
                    = if ((p.LOGICAL_WARPS = 1) || (p.LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS)) then __ptx__.LaneId() |> int else threadIdx.x % p.LOGICAL_WARP_THREADS
            }

        [<ReflectedDefinition>]
        static member Init(h:HostApi, temp_storage) =
            let p = h.Params
            {
                temp_storage    = temp_storage
                warp_id 
                    = if p.LOGICAL_WARPS = 1 then 0 else threadIdx.x / p.LOGICAL_WARP_THREADS
                lane_id 
                    = if ((p.LOGICAL_WARPS = 1) || (p.LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS)) then __ptx__.LaneId() |> int else threadIdx.x % p.LOGICAL_WARP_THREADS
            }

        [<ReflectedDefinition>]
        static member Init(h:HostApi, warp_id, lane_id) =
            { temp_storage = PrivateStorage<'T>(h); warp_id = warp_id; lane_id = lane_id }                

        [<ReflectedDefinition>]
        static member Init(h:HostApi, temp_storage, warp_id, lane_id) =
            { temp_storage = temp_storage; warp_id = warp_id; lane_id = lane_id } 

    
//        ^T when ^T : (static member (+): ^T * ^T -> ^T)
//        [<ReflectedDefinition>] member this.InclusiveSum(h, input, output)
//            = InternalWarpScan.InclusiveSum.Default h this.temp_storage (this.warp_id |> uint32) (this.lane_id |> uint32) input output

//        [<ReflectedDefinition>] member this.InclusiveSum(h, input, output, warp_aggregate)
//            = InternalWarpScan.InclusiveSum.WithAggregate h this.temp_storage (this.warp_id |> uint32) (this.lane_id |> uint32) input output warp_aggregate
//
//        [<ReflectedDefinition>] member this.InclusiveScan(h, scan_op, input, output)
//            = InternalWarpScan.InclusiveScan.Default h scan_op this.temp_storage  (this.warp_id |> uint32) (this.lane_id |> uint32) input output
//    
//        [<ReflectedDefinition>] member this.InclusiveScan(h, scan_op, input, output, warp_aggregate)
//            = InternalWarpScan.InclusiveScan.WithAggregate h scan_op this.temp_storage  (this.warp_id |> uint32) (this.lane_id |> uint32) input output warp_aggregate
//
//        [<ReflectedDefinition>] member this.ExclusiveSum(h, scan_op, input, output)
//            = ExclusiveSum.Default h scan_op this.temp_storage (this.warp_id |> uint32) (this.lane_id |> uint32) input output
//
//
        [<ReflectedDefinition>] member this.ExclusiveSumInt(h, temp_storage, input, output)
            = ExclusiveSum.DefaultInt h temp_storage (this.warp_id |> uint32) (this.lane_id |> uint32) input output

        [<ReflectedDefinition>] member this.ExclusiveSumInt(h, temp_storage, input, output, warp_aggregate)
            = ExclusiveSum.WithAggregateInt h temp_storage (this.warp_id |> uint32) (this.lane_id |> uint32) input output warp_aggregate
               
//        [<ReflectedDefinition>] member this.ExclusiveSum(h, input, output, warp_aggregate)
//            = ExclusiveSum.WithAggregate h this.temp_storage  (this.warp_id |> uint32) (this.lane_id |> uint32) input output warp_aggregate

//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity)
//            = InternalWarpScan.ExclusiveScan.Default h scan_op this.temp_storage  (this.warp_id |> uint32) (this.lane_id |> uint32) input output identity
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity, warp_aggregate)
//            = InternalWarpScan.ExclusiveScan.WithAggregate h scan_op this.temp_storage  (this.warp_id |> uint32) (this.lane_id |> uint32) input output identity warp_aggregate
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output)
//            = InternalWarpScan.ExclusiveScan.Identityless.Default h scan_op this.temp_storage  (this.warp_id |> uint32) (this.lane_id |> uint32) input output
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, warp_aggregate)
//            = InternalWarpScan.ExclusiveScan.Identityless.WithAggregate h scan_op this.temp_storage  (this.warp_id |> uint32) (this.lane_id |> uint32) input output warp_aggregate
