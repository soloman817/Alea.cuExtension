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
    type ScanKind =
        | Shfl = 0
        | Smem = 1
           
             
    type StaticParam = 
        {
            LOGICAL_WARPS           : int
            LOGICAL_WARP_THREADS    : int
            POW_OF_TWO              : bool

            ScanKind                : ScanKind
            WarpScanShflParam       : WarpScanShfl.StaticParam
            WarpScanSmemParam       : WarpScanSmem.StaticParam
            SharedMemoryLength      : int
        }

        static member Init(logical_warps, logical_warp_threads) = 
            let logical_warps = logical_warps
            let logical_warp_threads = CUB_PTX_WARP_THREADS
            let pow_of_two = ((logical_warp_threads &&& (logical_warp_threads - 1)) = 0)
            let wsSmem_h = WarpScanSmem.StaticParam.Init(logical_warps, logical_warp_threads)
            let wsShfl_h = WarpScanShfl.StaticParam.Init(logical_warps, logical_warp_threads)
            let kind = if (CUB_PTX_VERSION >= 300) && ((logical_warps = 1) || pow_of_two) then ScanKind.Shfl else ScanKind.Smem
            { 
                LOGICAL_WARPS           = logical_warps
                LOGICAL_WARP_THREADS    = logical_warp_threads
                POW_OF_TWO              = ((logical_warp_threads &&& (logical_warp_threads - 1)) = 0)

                ScanKind                = kind
                WarpScanShflParam       = wsShfl_h
                WarpScanSmemParam       = wsSmem_h
                SharedMemoryLength      = 0 //todo
            }

        static member Init() = StaticParam.Init(1, CUB_PTX_WARP_THREADS)
        static member Init(logical_warps) = StaticParam.Init(logical_warps, CUB_PTX_WARP_THREADS)
            

    type TempStorage<'T> = WarpScanSmem.TempStorage<'T>

    let [<ReflectedDefinition>] inline PrivateStorage<'T>(sp:StaticParam) = TempStorage<'T>.Init(sp.WarpScanSmemParam)


    [<Record>]
    type InstanceParam<'T> = 
        {
            mutable temp_storage    : TempStorage<'T>
            mutable warp_id         : int
            mutable lane_id         : int
        }
            
        [<ReflectedDefinition>] 
        static member Init(sp:StaticParam) =
            {
                temp_storage    = PrivateStorage<'T>(sp)
                warp_id 
                    = if sp.LOGICAL_WARPS = 1 then 0 else threadIdx.x / sp.LOGICAL_WARP_THREADS
                lane_id 
                    = if ((sp.LOGICAL_WARPS = 1) || (sp.LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS)) then __ptx__.LaneId() |> int else threadIdx.x % sp.LOGICAL_WARP_THREADS
            }

        [<ReflectedDefinition>]
        static member Init(sp:StaticParam, temp_storage) =
            {
                temp_storage    = temp_storage
                warp_id 
                    = if sp.LOGICAL_WARPS = 1 then 0 else threadIdx.x / sp.LOGICAL_WARP_THREADS
                lane_id 
                    = if ((sp.LOGICAL_WARPS = 1) || (sp.LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS)) then __ptx__.LaneId() |> int else threadIdx.x % sp.LOGICAL_WARP_THREADS
            }

        [<ReflectedDefinition>]
        static member Init(sp:StaticParam, warp_id, lane_id) =
            {
                temp_storage    = PrivateStorage<'T>(sp)
                warp_id         = warp_id
                lane_id         = lane_id
            }                

        [<ReflectedDefinition>]
        static member Init(sp:StaticParam, temp_storage, warp_id, lane_id) =
            {
                temp_storage    = temp_storage
                warp_id         = warp_id
                lane_id         = lane_id
            } 
    

    module InternalWarpScan =
    
        module InclusiveSum = 

            let [<ReflectedDefinition>] inline DefaultInt (sp:StaticParam)
                (ip:InstanceParam<int>)
                (input:int) (output:Ref<int>) =
                let wssip = WarpScanSmem.InstanceParam<int>.Init(ip.temp_storage, ip.warp_id |> uint32, ip.lane_id |> uint32)
                WarpScanSmem.InclusiveSum.DefaultInt sp.WarpScanSmemParam wssip input output

            let [<ReflectedDefinition>] inline WithAggregateInt (sp:StaticParam)
                (ip:InstanceParam<int>)
                (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) =
                let wssip = WarpScanSmem.InstanceParam<int>.Init(ip.temp_storage, ip.warp_id |> uint32, ip.lane_id |> uint32)
                WarpScanSmem.InclusiveSum.WithAggregateInt sp.WarpScanSmemParam wssip input output warp_aggregate                

            let [<ReflectedDefinition>] inline _Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
                (ip:InstanceParam<'T>)
                (input:'T) (output:Ref<'T>) = 
                let wssip = WarpScanSmem.InstanceParam<'T>.Init(ip.temp_storage, ip.warp_id |> uint32, ip.lane_id |> uint32)
                WarpScanSmem.InclusiveSum.Default sp.WarpScanSmemParam scan_op wssip input output


            let [<ReflectedDefinition>] inline _WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
                (ip:InstanceParam<'T>)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
                let wssip = WarpScanSmem.InstanceParam<'T>.Init(ip.temp_storage, ip.warp_id |> uint32, ip.lane_id |> uint32)
                WarpScanSmem.InclusiveSum.WithAggregate sp.WarpScanSmemParam scan_op wssip input output warp_aggregate


            let [<ReflectedDefinition>] inline Default (sp:StaticParam)
                (ip:InstanceParam<'T>)
                (input:'T) (output:Ref<'T>) =
                _Default sp (+) ip input output

            let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam)
                (ip:InstanceParam<'T>)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
                _WithAggregate sp (+) ip input output warp_aggregate


        module InclusiveScan =
            let [<ReflectedDefinition>] inline Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
                (ip:InstanceParam<'T>) 
                (input:'T) (output:Ref<'T>) =
                let wssip = WarpScanSmem.InstanceParam<'T>.Init(ip.temp_storage, ip.warp_id |> uint32, ip.lane_id |> uint32)
                WarpScanSmem.InclusiveScan.Default sp.WarpScanSmemParam scan_op wssip input output
    
            let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
                (ip:InstanceParam<'T>)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
                let wssip = WarpScanSmem.InstanceParam<'T>.Init(ip.temp_storage, ip.warp_id |> uint32, ip.lane_id |> uint32)
                WarpScanSmem.InclusiveScan.WithAggregate sp.WarpScanSmemParam scan_op wssip input output
    
    
        module ExclusiveScan =
            let [<ReflectedDefinition>] inline Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
                (ip:InstanceParam<'T>)
                (input:'T) (output:Ref<'T>) (identity:'T) =
                let wssip = WarpScanSmem.InstanceParam<'T>.Init(ip.temp_storage, ip.warp_id |> uint32, ip.lane_id |> uint32)
                WarpScanSmem.ExclusiveScan.Default sp.WarpScanSmemParam scan_op wssip input output identity
    
            let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
                (ip:InstanceParam<'T>)
                (input:'T) (output:Ref<'T>) (identity:'T) (warp_aggregate:Ref<'T>) =
                let wssip = WarpScanSmem.InstanceParam<'T>.Init(ip.temp_storage, ip.warp_id |> uint32, ip.lane_id |> uint32)
                WarpScanSmem.ExclusiveScan.WithAggregate sp.WarpScanSmemParam scan_op wssip input output identity warp_aggregate
    
    
            module Identityless =
                let [<ReflectedDefinition>] inline Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
                    (ip:InstanceParam<'T>)
                    (input:'T) (output:Ref<'T>) =
                    let wssip = WarpScanSmem.InstanceParam<'T>.Init(ip.temp_storage, ip.warp_id |> uint32, ip.lane_id |> uint32)
                    WarpScanSmem.ExclusiveScan.Identityless.Default sp.WarpScanSmemParam scan_op wssip input output
    
                let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
                    (ip:InstanceParam<'T>)
                    (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
                    let wssip = WarpScanSmem.InstanceParam<'T>.Init(ip.temp_storage, ip.warp_id |> uint32, ip.lane_id |> uint32)
                    WarpScanSmem.ExclusiveScan.Identityless.WithAggregate sp.WarpScanSmemParam scan_op wssip input output warp_aggregate
                
    
 

    
    module PrivateExclusiveSum =
        open InternalWarpScan

        let [<ReflectedDefinition>] inline Default (sp:StaticParam)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) =
            let inclusive = __local__.Variable<'T>()
            InclusiveSum.Default sp ip input output
            output := !inclusive + input
        
        let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
            let inclusive = __local__.Variable<'T>()
            InclusiveSum.WithAggregate sp ip input inclusive warp_aggregate
            output := !inclusive + input

//        let [<ReflectedDefinition>] inline Default (sp:StaticParam)
//            (ip:InstanceParam<'T>)
//            (input:'T) (output:Ref<'T>) =
//            _Default sp (+) ip input output
//        
//        let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam)
//            (ip:InstanceParam<'T>)
//            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//            _WithAggregate sp (+) ip input output warp_aggregate

        let [<ReflectedDefinition>] inline DefaultInt (sp:StaticParam)
            (ip:InstanceParam<int>)
            (input:int) (output:Ref<int>) =
            let inclusive = __local__.Variable<int>()
            InclusiveSum.DefaultInt sp ip input output
            output := !inclusive + input
                    

        let [<ReflectedDefinition>] inline WithAggregateInt (sp:StaticParam)
            (ip:InstanceParam<int>)
            (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) =
            let inclusive = __local__.Variable<int>()
            InclusiveSum.WithAggregateInt sp ip input inclusive warp_aggregate
            output := !inclusive + input


    module ExclusiveSum =

        let [<ReflectedDefinition>] inline Default (sp:StaticParam)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) = 
            PrivateExclusiveSum.Default sp ip input output

        let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
            PrivateExclusiveSum.WithAggregate sp ip input output warp_aggregate
          

        let [<ReflectedDefinition>] inline DefaultInt (sp:StaticParam)
            (ip:InstanceParam<int>)
            (input:int) (output:Ref<int>) = 
            PrivateExclusiveSum.DefaultInt sp ip input output


        let [<ReflectedDefinition>] inline WithAggregateInt (sp:StaticParam)
            (ip:InstanceParam<int>)
            (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) =
            PrivateExclusiveSum.WithAggregateInt sp ip input output warp_aggregate
  
    


    module InclusiveScan =
        
    
        let [<ReflectedDefinition>] inline Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) = 
            InternalWarpScan.InclusiveScan.Default sp scan_op ip input output
    
        let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) = 
            InternalWarpScan.InclusiveScan.WithAggregate sp scan_op ip input output warp_aggregate
    
    
    
    module ExclusiveScan =
        
    
        let [<ReflectedDefinition>] inline Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (identity:'T) = 
            InternalWarpScan.ExclusiveScan.Default sp scan_op ip input output identity
    
        let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (identity:'T) (warp_aggregate:Ref<'T>) = 
            InternalWarpScan.ExclusiveScan.WithAggregate sp scan_op ip input output identity warp_aggregate
    
    
        module Identityless =
            let [<ReflectedDefinition>] inline Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
                (ip:InstanceParam<'T>)
                (input:'T) (output:Ref<'T>) = 
                InternalWarpScan.ExclusiveScan.Identityless.Default sp scan_op ip input output
    
            let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
                (ip:InstanceParam<'T>)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) = 
                InternalWarpScan.ExclusiveScan.Identityless.WithAggregate sp scan_op ip input output warp_aggregate
    
    
    
//    
//  
//
//    [<Record>]
//    type IntApi = 
//        { mutable temp_storage : TempStorage<int>; mutable warp_id : int; mutable lane_id : int }
//            
//        [<ReflectedDefinition>] 
//        static member Init(sp:StaticParam) =
//            let p = h.Params
//            {
//                temp_storage    = PrivateStorage<int>(h)
//                warp_id 
//                    = if p.LOGICAL_WARPS = 1 then 0 else threadIdx.x / p.LOGICAL_WARP_THREADS
//                lane_id 
//                    = if ((p.LOGICAL_WARPS = 1) || (p.LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS)) then __ptx__.LaneId() |> int else threadIdx.x % p.LOGICAL_WARP_THREADS
//            }
//
//        [<ReflectedDefinition>]
//        static member Init(sp:StaticParam, temp_storage) =
//            let p = h.Params
//            {
//                temp_storage    = temp_storage
//                warp_id 
//                    = if p.LOGICAL_WARPS = 1 then 0 else threadIdx.x / p.LOGICAL_WARP_THREADS
//                lane_id 
//                    = if ((p.LOGICAL_WARPS = 1) || (p.LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS)) then __ptx__.LaneId() |> int else threadIdx.x % p.LOGICAL_WARP_THREADS
//            }
//
//        [<ReflectedDefinition>]
//        static member Init(sp:StaticParam, warp_id, lane_id) =
//            { temp_storage = PrivateStorage<int>(h); warp_id = warp_id; lane_id = lane_id }                
//
//        [<ReflectedDefinition>]
//        static member Init(sp:StaticParam, temp_storage, warp_id, lane_id) =
//            { temp_storage = temp_storage; warp_id = warp_id; lane_id = lane_id } 
//
//
//        [<ReflectedDefinition>] member this.ExclusiveSum(h, input, output)
//            = ExclusiveSum.DefaultInt sp this.temp_storage (this.warp_id |> uint32) (this.lane_id |> uint32) input output
//
//        [<ReflectedDefinition>] member this.ExclusiveSum(h, input, output, warp_aggregate)
//            = ExclusiveSum.WithAggregateInt sp this.temp_storage (this.warp_id |> uint32) (this.lane_id |> uint32) input output warp_aggregate
//    


//    type API<'T when 'T : static member 'T -> get_zero> = 


    
//        ^T when ^T : (static member (+): ^T * ^T -> ^T)
//        [<ReflectedDefinition>] member this.InclusiveSum(h, input, output)
//            = InternalWarpScan.InclusiveSum.Default sp this.temp_storage (this.warp_id |> uint32) (this.lane_id |> uint32) input output

//        [<ReflectedDefinition>] member this.InclusiveSum(h, input, output, warp_aggregate)
//            = InternalWarpScan.InclusiveSum.WithAggregate sp this.temp_storage (this.warp_id |> uint32) (this.lane_id |> uint32) input output warp_aggregate
//
//        [<ReflectedDefinition>] member this.InclusiveScan(h, scan_op, input, output)
//            = InternalWarpScan.InclusiveScan.Default sp scan_op this.temp_storage  (this.warp_id |> uint32) (this.lane_id |> uint32) input output
//    
//        [<ReflectedDefinition>] member this.InclusiveScan(h, scan_op, input, output, warp_aggregate)
//            = InternalWarpScan.InclusiveScan.WithAggregate sp scan_op this.temp_storage  (this.warp_id |> uint32) (this.lane_id |> uint32) input output warp_aggregate
//
//        [<ReflectedDefinition>] member this.ExclusiveSum(h, scan_op, input, output)
//            = ExclusiveSum.Default sp scan_op this.temp_storage (this.warp_id |> uint32) (this.lane_id |> uint32) input output
////
////
//        [<ReflectedDefinition>] member this.ExclusiveSumInt(h, temp_storage, input, output)
//            = ExclusiveSum.DefaultInt sp temp_storage (this.warp_id |> uint32) (this.lane_id |> uint32) input output
//
//        [<ReflectedDefinition>] member this.ExclusiveSumInt(h, temp_storage, input, output, warp_aggregate)
//            = ExclusiveSum.WithAggregateInt sp temp_storage (this.warp_id |> uint32) (this.lane_id |> uint32) input output warp_aggregate
//               
//        [<ReflectedDefinition>] member this.ExclusiveSum(h, input, output, warp_aggregate)
//            = ExclusiveSum.WithAggregate sp this.temp_storage  (this.warp_id |> uint32) (this.lane_id |> uint32) input output warp_aggregate

//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity)
//            = InternalWarpScan.ExclusiveScan.Default sp scan_op this.temp_storage  (this.warp_id |> uint32) (this.lane_id |> uint32) input output identity
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity, warp_aggregate)
//            = InternalWarpScan.ExclusiveScan.WithAggregate sp scan_op this.temp_storage  (this.warp_id |> uint32) (this.lane_id |> uint32) input output identity warp_aggregate
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output)
//            = InternalWarpScan.ExclusiveScan.Identityless.Default sp scan_op this.temp_storage  (this.warp_id |> uint32) (this.lane_id |> uint32) input output
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, warp_aggregate)
//            = InternalWarpScan.ExclusiveScan.Identityless.WithAggregate sp scan_op this.temp_storage  (this.warp_id |> uint32) (this.lane_id |> uint32) input output warp_aggregate
