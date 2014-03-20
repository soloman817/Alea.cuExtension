[<AutoOpen>]
module Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Utilities.NumericLiteralG

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread



module WarpScanSmem =
    
    type StaticParam = 
        { 
            LOGICAL_WARPS           : int
            LOGICAL_WARP_THREADS    : int
            STEPS                   : int
            HALF_WARP_THREADS       : int
            WARP_SMEM_ELEMENTS      : int
        }

        static member Init(logical_warps:int, logical_warp_threads:int) =
            let steps               = logical_warp_threads |> log2
            let half_warp_threads   = 1 <<< (steps - 1)
            let warp_smem_elements  = logical_warp_threads + half_warp_threads
            {
                LOGICAL_WARPS           = logical_warps
                LOGICAL_WARP_THREADS    = logical_warp_threads
                STEPS                   = steps
                HALF_WARP_THREADS       = half_warp_threads
                WARP_SMEM_ELEMENTS      = warp_smem_elements
            }
        
        static member Init(logical_warps) = StaticParam.Init(logical_warps, CUB_PTX_WARP_THREADS)
                    

    [<Record>]
    type TempStorage<'T> =
        { 
            mutable Ptr : deviceptr<'T> 
            Rows : int
            Cols : int
        }

        member this.Item
            with    [<ReflectedDefinition>] get (idx:int) = this.Ptr.[idx]
            and     [<ReflectedDefinition>] set (idx:int) (v:'T) = this.Ptr.[idx] <- v

        member this.Get
            with    [<ReflectedDefinition>] get (i:int, j:int) = this.Ptr.[j + i * this.Cols]
            and     [<ReflectedDefinition>] set (i:int, j:int) (v:'T) = this.Ptr.[j + i * this.Cols] <- v
        

        [<ReflectedDefinition>] member this.GetPtr(i:int, j:int) : deviceptr<'T> = this.Ptr + (j + i * this.Cols)
        [<ReflectedDefinition>] member this.GetPtr(i:uint32, j:uint32) : deviceptr<'T> = this.Ptr + ((j |> int) + (i |> int) * this.Cols)

        [<ReflectedDefinition>]
        static member Init(sp:StaticParam) =
            let rows = sp.LOGICAL_WARPS
            let cols = sp.WARP_SMEM_ELEMENTS
            let s = __shared__.Array<'T>(rows * cols)
            let ptr = s |> __array_to_ptr
            { Ptr = ptr; Rows = rows; Cols = cols }



    [<Record>]
    type InstanceParam<'T> =
        {
            mutable temp_storage    : TempStorage<'T>
            mutable warp_id         : uint32
            mutable lane_id         : uint32
        }
        
        [<ReflectedDefinition>]
        static member Init(temp_storage:TempStorage<'T>, warp_id:uint32, lane_id:uint32) = 
            { temp_storage = temp_storage; warp_id = warp_id; lane_id = lane_id }


    let [<ReflectedDefinition>] inline Broadcast (ip:InstanceParam<'T>) (input:'T) (src_lane:uint32) =
        if ip.lane_id = src_lane then ThreadStore.STORE_VOLATILE (ip.temp_storage.Ptr + (ip.warp_id |> int)) (input)
            
        ThreadLoad.LOAD_VOLATILE (ip.temp_storage.Ptr + (ip.warp_id |> int))


    ///@TODO
    module InitIdentity =    
        let [<ReflectedDefinition>] inline True (ip:InstanceParam<'T>) =
            ThreadStore.STORE_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, ip.lane_id)) 0G
            
            
        let [<ReflectedDefinition>] inline False (ip:InstanceParam<'T>) = ()
            

        let [<ReflectedDefinition>] inline api (has_identity:bool) 
            (ip:InstanceParam<'T>) = 
            if has_identity then True ip else False ip


    let [<ReflectedDefinition>] inline _ScanStep (sp:StaticParam) (has_identity:bool) (scan_op:'T -> 'T -> 'T) 
        (ip:InstanceParam<'T>)
        (partial:Ref<'T>) (_STEP:int) =
        
        let OFFSET = 1u <<< _STEP
        ThreadStore.STORE_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, (sp.HALF_WARP_THREADS |> uint32) + ip.lane_id)) !partial

        if has_identity || (ip.lane_id >= OFFSET) then
            let addend =    ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, (sp.HALF_WARP_THREADS |> uint32) + ip.lane_id - OFFSET))
            partial := (addend, !partial) ||> scan_op
        

    let [<ReflectedDefinition>] inline ScanStep (sp:StaticParam) (has_identity:bool) (scan_op:'T -> 'T -> 'T) 
        (ip:InstanceParam<'T>)
        (partial:Ref<'T>) =
        
        _ScanStep sp has_identity scan_op ip partial 0
        for STEP = 1 to sp.STEPS - 1 do
            _ScanStep sp has_identity scan_op ip partial STEP


    let [<ReflectedDefinition>] inline _ScanStepInt (sp:StaticParam) (has_identity:bool)
        (ip:InstanceParam<int>)
        (partial:Ref<int>) (_STEP:int) =
        
        let OFFSET = 1u <<< _STEP
        ThreadStore.STORE_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, (sp.HALF_WARP_THREADS |> uint32) + ip.lane_id - OFFSET)) (!partial)

        if has_identity || (ip.lane_id >= OFFSET) then
            let addend = ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, (sp.HALF_WARP_THREADS |> uint32) + ip.lane_id - OFFSET))
            partial := addend + !partial
        
    
    let [<ReflectedDefinition>] inline ScanStepInt (sp:StaticParam) (has_identity:bool)
        (ip:InstanceParam<int>)
        (partial:Ref<int>) =
        
        _ScanStepInt sp has_identity ip partial 0
        for STEP = 1 to sp.STEPS - 1 do
            _ScanStepInt sp has_identity ip partial STEP
            


    let [<ReflectedDefinition>] inline BasicScan (sp:StaticParam) (has_identity:bool) (share_final:bool) (scan_op:'T -> 'T -> 'T)
        (ip:InstanceParam<'T>) 
        (partial:'T) =
        
        let partial = __local__.Variable<'T>(partial)
        ScanStep sp has_identity scan_op ip partial
        if share_final then 
            ThreadStore.STORE_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, (sp.HALF_WARP_THREADS |> uint32) + ip.lane_id)) (!partial)
        !partial
        

    let [<ReflectedDefinition>] inline BasicScanInt (sp:StaticParam) (has_identity:bool) (share_final:bool)
        (ip:InstanceParam<int>) (partial:int) =
        
        //let mutable partial = partial
        ScanStepInt sp has_identity ip (partial |> __obj_to_ref)
        if share_final then 
            ThreadStore.STORE_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, (sp.HALF_WARP_THREADS |> uint32) + ip.lane_id)) (partial)
        partial




    module InclusiveSum =   

        let [<ReflectedDefinition>] inline Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) =
            let has_identity = true //PRIMITIVE()
            let share_final = false
            InitIdentity.api has_identity ip
            output := BasicScan sp has_identity share_final scan_op ip input
   
        let [<ReflectedDefinition>] inline DefaultInt (sp:StaticParam)
            (ip:InstanceParam<int>)
            (input:int) (output:Ref<int>) =
            let has_identity = true //PRIMITIVE()
            let share_final = false
            InitIdentity.api has_identity ip
            output := BasicScanInt sp has_identity share_final ip input  
      
    
        let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
            let has_identity = true //PRIMITIVE()
            let share_final = true
            let p = sp
            
            
            InitIdentity.api has_identity ip
            
            output := BasicScan sp has_identity share_final scan_op ip input            
        
            warp_aggregate := ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id |> int, sp.WARP_SMEM_ELEMENTS - 1))


        let [<ReflectedDefinition>] inline WithAggregateInt (sp:StaticParam)
            (ip:InstanceParam<int>)
            (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) =
            let has_identity = true //PRIMITIVE()
            let share_final = true
            let p = sp
            
            
            InitIdentity.api has_identity ip
            
            output := BasicScanInt sp has_identity share_final ip input            
            warp_aggregate := ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id |> int, sp.WARP_SMEM_ELEMENTS - 1))



    module InclusiveScan =
        

        let [<ReflectedDefinition>] inline Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) =
            output := BasicScan sp false false scan_op ip input
        

        let [<ReflectedDefinition>] WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
            
        
            output := BasicScan sp true false scan_op ip input
            warp_aggregate := ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id |> int, sp.WARP_SMEM_ELEMENTS - 1))

    module ExclusiveSum = ()
    
    module ExclusiveScan =
        

        let [<ReflectedDefinition>] inline Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (identity:'T) =
            
            ThreadStore.STORE_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, ip.lane_id)) (identity)

            let inclusive = BasicScan sp true true scan_op ip input
            output := ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, (sp.HALF_WARP_THREADS |> uint32) + ip.lane_id - 1u))
        

        let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
            (ip:InstanceParam<'T>)
            (input:'T) (output:Ref<'T>) (identity:'T) (warp_aggregate:Ref<'T>) =
            
        
            Default sp scan_op ip input output identity
            warp_aggregate := ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, (sp.HALF_WARP_THREADS |> uint32) + ip.lane_id - 1u))
        

        module Identityless =
            let [<ReflectedDefinition>] inline Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
                (ip:InstanceParam<'T>) 
                (input:'T) (output:Ref<'T>) =
                

                let inclusive = BasicScan sp false true scan_op ip input
                output := ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, (sp.HALF_WARP_THREADS |> uint32) + ip.lane_id - 1u))
            
            let [<ReflectedDefinition>] inline DefaultInt (sp:StaticParam)
                (ip:InstanceParam<int>) 
                (input:int) (output:Ref<int>) =
                

                let inclusive = BasicScanInt sp false true ip input
                output := ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, (sp.HALF_WARP_THREADS |> uint32) + ip.lane_id - 1u))
            


            let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
                (ip:InstanceParam<'T>)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
                
            
                Default sp scan_op ip input output

    [<Record>]
    type IntApi =
        {
            mutable InstanceParam : InstanceParam<int>
        }
        
        [<ReflectedDefinition>] static member Init(temp_storage, warp_id, lane_id) = { InstanceParam = InstanceParam<int>.Init(temp_storage, warp_id, lane_id) }

        [<ReflectedDefinition>] 
        member this.InclusiveSum(sp, input:int, output:Ref<int>) = 
            InclusiveSum.DefaultInt sp this.InstanceParam input output

        [<ReflectedDefinition>]
        member this.InclusiveSum(sp, input:int, output:Ref<int>, warp_aggregate) =
            InclusiveSum.WithAggregateInt sp this.InstanceParam input output warp_aggregate
        
        [<ReflectedDefinition>] member this.ExclusiveScan(sp, input, output) 
            = ExclusiveScan.Identityless.DefaultInt sp this.InstanceParam input output



//    [<Record>]
//    type API<'T> =
//        {
//            mutable temp_storage    : TempStorage<'T>
//            mutable warp_id         : uint32
//            mutable lane_id         : uint32
//        }
//        
//        [<ReflectedDefinition>] static member Init(ip.temp_storage, warp_id, ip.lane_id) = { temp_storage = temp_storage; warp_id = warp_id; lane_id = lane_id }
//
//        [<ReflectedDefinition>] 
//        member this.InclusiveSum(h, input:int, output:Ref<int>) = 
//            InclusiveSum.DefaultInt sp this.temp_storage this.warp_id this.lane_id input output

//        [<ReflectedDefinition>] member this.InclusiveSum(h, scan_op, input, output) 
//            = InclusiveSum.Default sp scan_op this.temp_storage this.warp_id this.lane_id input output
////        
//        [<ReflectedDefinition>] member this.InclusiveSum(h, scan_op, input, output, warp_aggregate) 
//            = InclusiveSum.WithAggregate sp scan_op this.temp_storage this.warp_id this.lane_id input output warp_aggregate
//
//        [<ReflectedDefinition>] member this.InclusiveScan(h, scan_op, input, output) 
//            = InclusiveScan.Default sp scan_op this.temp_storage this.warp_id this.lane_id input output
//        
//        [<ReflectedDefinition>] member this.InclusiveScan(h, scan_op, input, output, warp_aggregate) 
//            = InclusiveScan.WithAggregate sp scan_op this.temp_storage this.warp_id this.lane_id input output warp_aggregate  
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity) 
//            = ExclusiveScan.Default sp scan_op this.temp_storage this.warp_id this.lane_id input output identity
//        
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity, warp_aggregate) 
//            = ExclusiveScan.WithAggregate sp scan_op this.temp_storage this.warp_id this.lane_id input output identity warp_aggregate
//        
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output) 
//            = ExclusiveScan.Identityless.Default sp scan_op this.temp_storage this.warp_id this.lane_id input output
//        
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, input, output) 
//            = ExclusiveScan.Identityless.Default sp this.temp_storage this.warp_id this.lane_id input output
        
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, warp_aggregate) 
//            = ExclusiveScan.Identityless.WithAggregate sp scan_op this.temp_storage this.warp_id this.lane_id input output warp_aggregate


















////    type _DeviceApi<'T>     = Device.API<'T>
//
////    module InclusiveSum =
////        type _FunctionApi<'T> =
////            {
////                Default         : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> unit>
////                WithAggregate   : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> Ref<'T> -> unit>
////            }
////
////    module InclusiveScan =
////        type _FunctionApi<'T> =
////            {
////                Default         : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> unit>
////                WithAggregate   : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> Ref<'T> -> unit>                
////            }
////            
////    module ExclusiveScan =
////        type _FunctionApi<'T> =
////            {
////                Default             : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> 'T -> unit>
////                Default_NoID        : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> unit>
////                WithAggregate       : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> 'T -> Ref<'T> -> unit>
////                WithAggregate_NoID  : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> Ref<'T> -> unit>
////            }
//
////        
//module InclusiveSum =
//    
//    
//
//    let [<ReflectedDefinition>] inline Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
//        (ip:InstanceParam<'T>)
//        (input:'T) (output:Ref<'T>) =
//        let has_identity = true //PRIMITIVE()
//        let share_final = false
//            //(%InitIdentity) d
//        output := BasicScan sp has_identity share_final scan_op ip input
//   
//    let [<ReflectedDefinition>] inline DefaultInt (sp:StaticParam)
//        (ip:InstanceParam<int>)
//        (input:int) (output:Ref<int>) =
//        let has_identity = true //PRIMITIVE()
//        let share_final = false
//            //(%InitIdentity) d
//        output := BasicScanInt sp has_identity share_final ip input  
//      
//    
//    let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
//        (ip:InstanceParam<'T>)
//        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//        let has_identity = true //PRIMITIVE()
//        let share_final = true
//        let p = sp
//        
//            
//        //(%InitIdentity) d
//            
//        output := BasicScan sp has_identity share_final scan_op ip input            
//        
//        warp_aggregate := ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, sp.WARP_SMEM_ELEMENTS - 1))
//
//
//    let [<ReflectedDefinition>] inline WithAggregateInt (sp:StaticParam)
//        (ip:InstanceParam<int>)
//        (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) =
//        let has_identity = true //PRIMITIVE()
//        let share_final = true
//        let p = sp
//        
//            
//        //(%InitIdentity) d
//            
//        output := BasicScanInt sp has_identity share_final ip input            
//        warp_aggregate := ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, sp.WARP_SMEM_ELEMENTS - 1))
////
////module InclusiveScan =
////    
////
////    let [<ReflectedDefinition>] inline Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
////        (ip:InstanceParam<'T>)
////        (input:'T) (output:Ref<'T>) =
////        output := BasicScan sp false false scan_op ip input
////        
////
////    let [<ReflectedDefinition>] WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
////        (ip:InstanceParam<'T>)
////        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
////        
////        
////        output := BasicScan sp true false scan_op ip input
////        warp_aggregate := ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, sp.WARP_SMEM_ELEMENTS - 1))
////
//
////module ExclusiveScan =
////    
////
////    let [<ReflectedDefinition>] inline Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
////        (ip:InstanceParam<'T>)
////        (input:'T) (output:Ref<'T>) (identity:'T) =
////        
////        ThreadStore.STORE_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, ip.lane_id)) (identity)
////
////        let inclusive = BasicScan sp true true scan_op ip input
////        output := ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, sp.HALF_WARP_THREADS + ip.lane_id - 1))
////        
////
////    let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
////        (ip:InstanceParam<'T>)
////        (input:'T) (output:Ref<'T>) (identity:'T) (warp_aggregate:Ref<'T>) =
////        
////        
////        Default sp scan_op ip input output identity
////        warp_aggregate := ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, sp.HALF_WARP_THREADS + ip.lane_id - 1))
////        
////
////    module Identityless =
////        let [<ReflectedDefinition>] inline Default (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
////            (ip:InstanceParam<'T>) 
////            (input:'T) (output:Ref<'T>) =
////            
////
////            let inclusive = BasicScan sp false true scan_op ip input
////            output := ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, sp.HALF_WARP_THREADS + ip.lane_id - 1))
////            
////
////        let [<ReflectedDefinition>] inline WithAggregate (sp:StaticParam) (scan_op:'T -> 'T -> 'T)
////            (ip:InstanceParam<'T>)
////            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
////            
////            
////            Default sp scan_op ip input output
////            warp_aggregate := ThreadLoad.LOAD_VOLATILE (ip.temp_storage.GetPtr(ip.warp_id, sp.HALF_WARP_THREADS + ip.lane_id - 1))
//    
////    [<Record>]
////    type API<'T> =
////        {
////            mutable DeviceApi : _DeviceApi<'T>
////        }
////
////        [<ReflectedDefinition>] static member Create(ip.temp_storage, warp_id, ip.lane_id) = { DeviceApi = _DeviceApi<'T>.Init(ip.temp_storage, warp_id, ip.lane_id)}
////
////        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity) 
////            = Default sp scan_op this.temp_storage this.warp_id this.lane_id input output identity
////        
////        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity, warp_aggregate) 
////            = WithAggregate sp scan_op this.temp_storage this.warp_id this.lane_id input output identity warp_aggregate
////        
////        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output) 
////            = Identityless.Default sp scan_op this.temp_storage this.warp_id this.lane_id input output
////        
////        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, warp_aggregate) 
////            = Identityless.WithAggregate sp scan_op this.temp_storage this.warp_id this.lane_id input output warp_aggregate
//    module InclusiveSum =
//        type FunctionApi<'T> = Template.InclusiveSum._FunctionApi<'T>
//
//        let inline api (sp:StaticParam) (scan_op:'T -> 'T -> 'T) = InclusiveSum.api sp scan_op
//        
//        let inline template<'T> (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
//            let sp = HostApi.Init(logical_warps, logical_warp_threads)
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
//        let inline api (sp:StaticParam) (scan_op:'T -> 'T -> 'T) = InclusiveScan.api sp scan_op
//
//        let inline template<'T> (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
//            let sp = HostApi.Init(logical_warps, logical_warp_threads)
//            
//            let! dfault = (h, scan_op) ||> InclusiveScan.Default |> Compiler.DefineFunction
//            let! waggr  = (h, scan_op) ||> InclusiveScan.WithAggregate |> Compiler.DefineFunction
//
//            return h, {
//                Default         = dfault
//                WithAggregate   = waggr
//            }}
//
//    module ExclusiveScan =
//        type FunctionApi<'T> = Template.ExclusiveScan._FunctionApi<'T>
//
//        let inline api (sp:StaticParam) (scan_op:'T -> 'T -> 'T) = ExclusiveScan.api sp scan_op
//
//        let inline template<'T> (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
//            let sp = HostApi.Init(logical_warps, logical_warp_threads)
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
//
//    type FunctionApi<'T> =
//        {
//            InclusiveSum : InclusiveSum.FunctionApi<'T>
//            InclusiveScan : InclusiveScan.FunctionApi<'T>
//            ExclusiveScan : ExclusiveScan.FunctionApi<'T>        
//        }
//
//    let inline template<'T> (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
//        let! h, inclusiveSum = InclusiveSum.template<'T> logical_warps logical_warp_threads scan_op
//        let! _, inclusiveScan = InclusiveScan.template<'T> logical_warps logical_warp_threads scan_op
//        let!  _, exclusiveScan = ExclusiveScan.template<'T> logical_warps logical_warp_threads scan_op
//        
//        return h, {
//            InclusiveSum = inclusiveSum
//            InclusiveScan = inclusiveScan
//            ExclusiveScan = exclusiveScan
//        }}
//

//let inline _TempStorage() =
//    fun logical_warps warp_smem_elements ->
//        __shared__.Array2D(logical_warps, warp_smem_elements)
//
//
//let initIdentity logical_warps logical_warp_threads =
//    let WARP_SMEM_ELEMENTS = logical_warp_threads |> WARP_SMEM_ELEMENTS
//    fun (has_identity:bool) ->
//        let temp_storage = (logical_warps, WARP_SMEM_ELEMENTS) ||> _TempStorage()
//        let store = STORE_VOLATILE |> threadStore()
//        fun warp_id lane_id ->
//            match has_identity with
//            | true ->
//                let identity = ZeroInitialize() |> __ptr_to_obj
//                (ip.temp_storage.[warp_id, ip.lane_id] |> __array_to_ptr, identity) ||> store
//            | false ->
//                ()
//
//let scanStep logical_warps logical_warp_threads =
//    let HALF_WARP_THREADS = logical_warp_threads |> HALF_WARP_THREADS
//    let STEPS = logical_warp_threads |> STEPS
//    fun (ip.temp_storage:int[,]) warp_id lane_id ->
//        let load = LOAD_VOLATILE |> threadLoad()
//        let store = STORE_VOLATILE |> threadStore()
//        fun has_identity step ->
//            let step = ref 0
//            fun (partial:Ref<int>) (scan_op:(int -> int -> int)) ->
//                while !step < STEPS do
//                    let OFFSET = 1 <<< !step
//                    
//                    //(ip.temp_storage |> __array_to_ptr, !partial) ||> store
//
//                    if has_identity || (ip.lane_id >= OFFSET) then
//                        let addend = (ip.temp_storage.[warp_id, (HALF_WARP_THREADS + ip.lane_id)] |> __obj_to_ptr, Some(partial |> __ref_to_ptr)) ||> load
//                        partial := (addenValue, !partial) ||> scan_op
//                        
//                    step := !step + 1
//
//
//let broadcast =
//    fun (ip.temp_storage:TempStorage<int>) warp_id lane_id ->
//        let load = LOAD_VOLATILE |> threadLoad()
//        let store = STORE_VOLATILE |> threadStore()
//        fun input src_lane ->
//            if ip.lane_id = src_lane then (ip.temp_storage.[warp_id] |> __obj_to_ptr, input) ||> store
//            (ip.temp_storage.[warp_id] |> __obj_to_ptr, None) ||> load
//            |> Option.get
//
//
//let inline basicScan logical_warps logical_warp_threads = 
//    let HALF_WARP_THREADS = logical_warp_threads |> HALF_WARP_THREADS
//    let scanStep = (logical_warps, logical_warp_threads) ||> scanStep
//    fun has_identity share_final ->
//        fun (ip.temp_storage:int[,]) warp_id lane_id ->
//            let store = STORE_VOLATILE |> threadStore()
//            fun (partial:int) (scan_op:(int -> int -> int)) ->
//                let partial = partial |> __obj_to_ref
//                scanStep
//                <|||    (ip.temp_storage, warp_id, ip.lane_id)
//                <||     (has_identity, 0)
//                <||     (partial, scan_op)
//                if share_final then (ip.temp_storage.[warp_id, (HALF_WARP_THREADS + ip.lane_id)] |> __obj_to_ptr, !partial) ||> store
//                !partial
//
//
//let inline inclusiveSum logical_warps logical_warp_threads =
//    let WARP_SMEM_ELEMENTS = logical_warp_threads |> WARP_SMEM_ELEMENTS
//    let initIdentity = (logical_warps, logical_warp_threads) ||> initIdentity
//    let basicScan = (logical_warps, logical_warp_threads) ||> basicScan
//
//    fun (ip.temp_storage:int[,]) warp_id lane_id ->
//        let load = LOAD_VOLATILE |> threadLoad()
//            
//        fun (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<int> option) ->
//            match warp_aggregate with
//            | None ->
//                let HAS_IDENTITY = true // Traits<int>::PRIMITIVE
//                initIdentity
//                <|  HAS_IDENTITY 
//                <|| (ip.warp_id, ip.lane_id)
//                
//                output :=
//                    basicScan
//                    <||     (HAS_IDENTITY, false)
//                    <|||    (ip.temp_storage, warp_id, ip.lane_id) 
//                    <||     (input, ( + ))
//
//            | Some warp_aggregate ->
//                let HAS_IDENTITY = true // Traits<int>::PRIMITIVE
//                initIdentity
//                <|  HAS_IDENTITY
//                <|| (ip.warp_id, ip.lane_id)
//
//                output :=
//                    basicScan
//                    <||     (HAS_IDENTITY, true)
//                    <|||    (ip.temp_storage, warp_id, ip.lane_id)
//                    <||     (input, ( + ))
//
//                warp_aggregate :=
//                    (ip.temp_storage.[warp_id, (ip.warp_SMEM_ELEMENTS - 1)] |> __obj_to_ptr, None) 
//                    ||> load
//                    |> Option.get
//
//
//let inline inclusiveScan logical_warps logical_warp_threads =
//    let WARP_SMEM_ELEMENTS = logical_warp_threads |> WARP_SMEM_ELEMENTS
//    let basicScan = (logical_warps, logical_warp_threads) ||> basicScan
//
//    fun (ip.temp_storage:int[,]) warp_id lane_id ->
//        let load = LOAD_VOLATILE |> threadLoad()
//
//        fun (input:'T) (output:Ref<'T>) (scan_op:(int -> int -> int)) (warp_aggregate:Ref<int> option) ->
//            match warp_aggregate with
//            | None ->
//                output :=
//                    basicScan
//                    <||     (false, false)
//                    <|||    (ip.temp_storage, warp_id, ip.lane_id)
//                    <||     (input, scan_op)
//
//            | Some warp_aggregate ->
//                output :=
//                    basicScan
//                    <||     (false, true)
//                    <|||    (ip.temp_storage, warp_id, ip.lane_id)
//                    <||     (input, scan_op)
//
//                warp_aggregate :=
//                    (ip.temp_storage.[warp_id, (ip.warp_SMEM_ELEMENTS - 1)] |> __obj_to_ptr, None) 
//                    ||> load
//                    |> Option.get 
//
//    
//let inline exclusiveScan logical_warps logical_warp_threads =
//    let HALF_WARP_THREADS = logical_warp_threads |> HALF_WARP_THREADS
//    let WARP_SMEM_ELEMENTS = logical_warp_threads |> WARP_SMEM_ELEMENTS
//    let basicScan = (logical_warps, logical_warp_threads) ||> basicScan
//
//    fun (ip.temp_storage:int[,]) warp_id lane_id ->
//        let load = LOAD_VOLATILE |> threadLoad()
//        let store = STORE_VOLATILE |> threadStore()
//
//        fun (input:'T) (output:Ref<'T>) (scan_op:(int -> int -> int)) (identity:int option) (warp_aggregate:Ref<int> option) ->
//            match identity, warp_aggregate with
//            | Some identity, None ->
//                (ip.temp_storage.[warp_id, ip.lane_id] |> __obj_to_ptr, identity) ||> store
//                let inclusive =
//                    basicScan
//                    <||     (true, true)
//                    <|||    (ip.temp_storage, warp_id, ip.lane_id)
//                    <||     (input, scan_op)
//
//                output :=
//                    (ip.temp_storage.[warp_id, (HALF_WARP_THREADS + ip.lane_id - 1)] |> __obj_to_ptr, None) 
//                    ||> load
//                    |> Option.get
//
//            | Some identity, Some warp_aggregate ->
//                (ip.temp_storage.[warp_id, ip.lane_id] |> __obj_to_ptr, identity) ||> store
//                let inclusive =
//                    basicScan
//                    <||     (true, true)
//                    <|||    (ip.temp_storage, warp_id, ip.lane_id)
//                    <||     (input, scan_op)
//
//                output :=
//                    (ip.temp_storage.[warp_id, (HALF_WARP_THREADS + ip.lane_id - 1)] |> __obj_to_ptr, None) 
//                    ||> load
//                    |> Option.get
//
//                warp_aggregate :=
//                    (ip.temp_storage.[warp_id, (ip.warp_SMEM_ELEMENTS - 1)] |> __obj_to_ptr, None)
//                    ||> load
//                    |> Option.get
//
//            | None, None ->
//                let inclusive =
//                    basicScan
//                    <||     (false, true)
//                    <|||    (ip.temp_storage, warp_id, ip.lane_id)
//                    <||     (input, scan_op)
//
//                output :=
//                    (ip.temp_storage.[warp_id, (HALF_WARP_THREADS + ip.lane_id - 1)] |> __obj_to_ptr, None)
//                    ||> load
//                    |> Option.get
//
//            | None, Some warp_aggregate ->
//                let inclusive =
//                    basicScan
//                    <||     (false, true)
//                    <|||    (ip.temp_storage, warp_id, ip.lane_id)
//                    <||     (input, scan_op)
//
//                output :=
//                    (ip.temp_storage.[warp_id, (HALF_WARP_THREADS + ip.lane_id - 1)] |> __obj_to_ptr, None)
//                    ||> load
//                    |> Option.get
//
//                warp_aggregate :=
//                    (ip.temp_storage.[warp_id, (ip.warp_SMEM_ELEMENTS - 1)] |> __obj_to_ptr, None)
//                    ||> load
//                    |> Option.get
//
//
//type Constants =
//    {
//        STEPS : int
//        HALF_WARP_THREADS : int
//        WARP_SMEM_ELEMENTS : int
//    }
//
//    static member Init(logical_warp_threads:int) =
//        {
//            STEPS               = logical_warp_threads |> STEPS
//            HALF_WARP_THREADS   = logical_warp_threads |> HALF_WARP_THREADS
//            WARP_SMEM_ELEMENTS  = logical_warp_threads |> WARP_SMEM_ELEMENTS
//        }
//                    
//type TempStorage = int[,]
//
//[<Record>]
//type internal ThreadFields =
//    {
//        temp_storage : TempStorage
//        warp_id : uint32
//        lane_id : uint32
//    }
//
//    static member Init(ip.temp_storage:int[,], warp_id:uint32, lane_id:uint32) =
//        {
//            temp_storage = temp_storage
//            warp_id = warp_id
//            lane_id = lane_id
//        }
//            
//[<Record>]
//type WarpScanSmem =
//    {
//        // template parameters
//        LOGICAL_WARPS : int
//        LOGICAL_WARP_THREADS : int
//        // constants / enum
//        Constants : Constants
//        //TempStorage : int[,]
//        //ThreadFields : ThreadFields<int>
//
//    }
//
//        
//    member this.ScanStep(partial:Ref<int>) =
//        fun has_identity step -> ()
//
//    member this.ScanStep(partial:Ref<int>, scan_op:(int -> int -> int), step:bool) = 
//        fun has_identity step -> ()
//
//    member this.Broadcast(input:int, src_lane:uint32) = ()
//        
//    member this.BasicScan(partial:int, scan_op:(int -> int -> int)) = ()
//
//    member this.InclusiveSum(input:int, output:Ref<int>) = ()
//    member this.InclusiveSum(input:int, output:Ref<int>, warp_aggregate:Ref<int>) = ()
//
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), warp_aggregate:Ref<int>) = ()
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int)) = ()
//
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:(int -> int -> int)) = ()
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:(int -> int -> int), warp_aggregate:Ref<int>) = ()
//
//
//    static member Create(logical_warps, logical_warp_threads) =
//        let c = logical_warp_threads |> Constants.Init
//        //let temp_storage = Array2zeroCreate logical_warps sp.WARP_SMEM_ELEMENTS
//        let temp_storage = __shared__.Array2D(logical_warps)
//        {
//            LOGICAL_WARPS           = logical_warps
//            LOGICAL_WARP_THREADS    = logical_warp_threads
//            Constants               = c
//            //TempStorage             = temp_storage
//            //ThreadFields            = (ip.temp_storage, 0u, 0u) |> ThreadFields.Init
//        }