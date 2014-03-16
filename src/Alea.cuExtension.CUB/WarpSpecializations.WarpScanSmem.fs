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
    
    type Params = { LOGICAL_WARPS : int;    LOGICAL_WARP_THREADS : int }

                    static member inline Init(logical_warps, logical_warp_threads) =
                        { LOGICAL_WARPS = logical_warps; LOGICAL_WARP_THREADS = logical_warp_threads }

    type Constants =
        {
            POW_OF_TWO          : bool
            STEPS               : int
            HALF_WARP_THREADS   : int
            WARP_SMEM_ELEMENTS  : int
        }

        static member Init(p:Params) =                    
            let pow_of_two          = ((p.LOGICAL_WARP_THREADS &&& (p.LOGICAL_WARP_THREADS - 1)) = 0)
            let steps               = p.LOGICAL_WARP_THREADS |> log2
            let half_warp_threads   = 1 <<< (steps - 1)
            let warp_smem_elements  = p.LOGICAL_WARP_THREADS + half_warp_threads
            {
                POW_OF_TWO          = pow_of_two
                STEPS               = steps
                HALF_WARP_THREADS   = half_warp_threads
                WARP_SMEM_ELEMENTS  = warp_smem_elements
            }
                    
    type HostApi =
        { Params : Params; Constants : Constants; SharedMemoryLength : int }
            
        static member Init(logical_warps, logical_warp_threads) =
            let p = Params.Init(logical_warps, logical_warp_threads)
            let c = Constants.Init(p)
            { Params = p; Constants = c; SharedMemoryLength = p.LOGICAL_WARPS * c.WARP_SMEM_ELEMENTS }

        static member Init(logical_warps) = HostApi.Init(logical_warps, CUB_PTX_WARP_THREADS)



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
        static member Uninitialized(h:HostApi) =
            let rows = h.Params.LOGICAL_WARPS
            let cols = h.Constants.WARP_SMEM_ELEMENTS
            let s = __shared__.Array<'T>(rows * cols)
            let ptr = s |> __array_to_ptr
            { Ptr = ptr; Rows = rows; Cols = cols }









    let [<ReflectedDefinition>] inline Broadcast (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32) (input:'T) (src_lane:uint32) =
        if lane_id = src_lane then ThreadStore.STORE_VOLATILE (temp_storage.Ptr + (warp_id |> int)) (input)
            
        ThreadLoad.LOAD_VOLATILE (temp_storage.Ptr + (warp_id |> int))


    ///@TODO
    module InitIdentity =    
        let [<ReflectedDefinition>] inline True (h:HostApi) (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32) =
            let c = h.Constants
            ThreadStore.STORE_VOLATILE (temp_storage.GetPtr(warp_id, lane_id)) 0G
            
            
        let [<ReflectedDefinition>] inline False (h:HostApi) (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32) = ()
            

        let inline api (has_identity:bool) = if has_identity then True else False


//    module ScanStep =
//        type private ScanStepAttribute(modifier:string) =
//            inherit Attribute()
//
//            interface ICustomCallBuilder with
//                member this.Build(ctx, irObject, info, irParams) =
//                    match irObject, irParams with
//                    | None, irMax :: irPtr :: irVals :: [] ->
//                        let max = irMax.HasObject |> function
//                            | true -> irMax.Object :?> int
//                            | false -> failwith "max must be constant"
//
//                        // if we do the loop here, it is unrolled by compiler, not kernel runtime
//                        // think of this job as the C++ template expanding job, same thing!
//                        for i = 0 to max - 1 do
//                            let irIndex = IRCommonInstructionBuilder.Instance.BuildConstant(ctx, i)
//                            let irVal = IRCommonInstructionBuilder.Instance.BuildGEP(ctx, irPtr, irIndex :: [])
//                            let irPtr = ()
//                    
//                            let irPtr = IRCommonInstructionBuilder.Instance.BuildGEP(ctx, irVals, irIndex :: [])
//                            IRCommonInstructionBuilder.Instance.BuildStore(ctx, irPtr, irVal) 
//
//                        IRCommonInstructionBuilder.Instance.BuildNop(ctx) |> Some
//
//                    | _ -> None
//        
//        let [<ScanStep>] scanStep ()

    let [<ReflectedDefinition>] inline ScanStep (h:HostApi) (has_identity:bool) (scan_op:'T -> 'T -> 'T) 
        (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
        (partial:Ref<'T>) =
        let c = h.Constants
        for STEP = 0 to c.STEPS - 1 do
            let OFFSET = 1u <<< STEP
            ThreadStore.STORE_VOLATILE (temp_storage.GetPtr(warp_id, (c.HALF_WARP_THREADS |> uint32) + lane_id)) !partial

            if has_identity || (lane_id >= OFFSET) then
                let addend =    ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id, (c.HALF_WARP_THREADS |> uint32) + lane_id - OFFSET))
                partial := (addend, !partial) ||> scan_op
        

    let [<ReflectedDefinition>] inline ScanStepInt (h:HostApi) (has_identity:bool)
        (temp_storage:TempStorage<int>) (warp_id:uint32) (lane_id:uint32)
        (partial:Ref<int>) =
        let c = h.Constants
        for STEP = 0 to c.STEPS - 1 do
            let OFFSET = 1u <<< STEP
            ThreadStore.STORE_VOLATILE (temp_storage.GetPtr(warp_id, (c.HALF_WARP_THREADS |> uint32) + lane_id - OFFSET)) (!partial)

            if has_identity || (lane_id >= OFFSET) then
                let addend = ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id, (c.HALF_WARP_THREADS |> uint32) + lane_id - OFFSET))
                partial := addend + !partial


    let [<ReflectedDefinition>] inline BasicScan (h:HostApi) (has_identity:bool) (share_final:bool) (scan_op:'T -> 'T -> 'T)
        (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32) (partial:'T) =
        let c = h.Constants
        let partial = __local__.Variable<'T>(partial)
        ScanStep h has_identity scan_op temp_storage warp_id lane_id partial
        if share_final then 
            ThreadStore.STORE_VOLATILE (temp_storage.GetPtr(warp_id, (c.HALF_WARP_THREADS |> uint32) + lane_id)) (!partial)
        !partial
        

    let [<ReflectedDefinition>] inline BasicScanInt (h:HostApi) (has_identity:bool) (share_final:bool)
        (temp_storage:TempStorage<int>) (warp_id:uint32) (lane_id:uint32) (partial:int) =
        let c = h.Constants
        let partial = __local__.Variable<int>(partial)
        ScanStepInt h has_identity temp_storage warp_id lane_id partial
        if share_final then 
            ThreadStore.STORE_VOLATILE (temp_storage.GetPtr(warp_id, (c.HALF_WARP_THREADS |> uint32) + lane_id)) (!partial)
        !partial




    module InclusiveSum =
        open Template
    

        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
            (input:'T) (output:Ref<'T>) =
            let has_identity = true //PRIMITIVE()
            let share_final = false
                //(%InitIdentity) d
            output := BasicScan h has_identity share_final scan_op temp_storage warp_id lane_id input
   
        let [<ReflectedDefinition>] inline DefaultInt (h:HostApi)
            (temp_storage:TempStorage<int>) (warp_id:uint32) (lane_id:uint32)
            (input:int) (output:Ref<int>) =
            let has_identity = true //PRIMITIVE()
            let share_final = false
                //(%InitIdentity) d
            output := BasicScanInt h has_identity share_final temp_storage warp_id lane_id input  
      
    
        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
            let has_identity = true //PRIMITIVE()
            let share_final = true
            let p = h.Params
            let c = h.Constants
            
            //(%InitIdentity) d
            
            output := BasicScan h has_identity share_final scan_op temp_storage warp_id lane_id input            
        
            warp_aggregate := ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id |> int, c.WARP_SMEM_ELEMENTS - 1))


        let [<ReflectedDefinition>] inline WithAggregateInt (h:HostApi)
            (temp_storage:TempStorage<int>) (warp_id:uint32) (lane_id:uint32)
            (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) =
            let has_identity = true //PRIMITIVE()
            let share_final = true
            let p = h.Params
            let c = h.Constants
            
            //(%InitIdentity) d
            
            output := BasicScanInt h has_identity share_final temp_storage warp_id lane_id input            
            warp_aggregate := ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id |> int, c.WARP_SMEM_ELEMENTS - 1))


    module InclusiveScan =
        open Template

        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
            (input:'T) (output:Ref<'T>) =
            output := BasicScan h false false scan_op temp_storage warp_id lane_id input
        

        let [<ReflectedDefinition>] WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
            let c = h.Constants
        
            output := BasicScan h true false scan_op temp_storage warp_id lane_id input
            warp_aggregate := ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id |> int, c.WARP_SMEM_ELEMENTS - 1))

    module ExclusiveSum = ()
    
    module ExclusiveScan =
        open Template

        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
            (input:'T) (output:Ref<'T>) (identity:'T) =
            let c = h.Constants
            ThreadStore.STORE_VOLATILE (temp_storage.GetPtr(warp_id, lane_id)) (identity)

            let inclusive = BasicScan h true true scan_op temp_storage warp_id lane_id input
            output := ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id, (c.HALF_WARP_THREADS |> uint32) + lane_id - 1u))
        

        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
            (input:'T) (output:Ref<'T>) (identity:'T) (warp_aggregate:Ref<'T>) =
            let c = h.Constants
        
            Default h scan_op temp_storage warp_id lane_id input output identity
            warp_aggregate := ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id, (c.HALF_WARP_THREADS |> uint32) + lane_id - 1u))
        

        module Identityless =
            let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32) 
                (input:'T) (output:Ref<'T>) =
                let c = h.Constants

                let inclusive = BasicScan h false true scan_op temp_storage warp_id lane_id input
                output := ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id, (c.HALF_WARP_THREADS |> uint32) + lane_id - 1u))
            
            let [<ReflectedDefinition>] inline DefaultInt (h:HostApi)
                (temp_storage:TempStorage<int>) (warp_id:uint32) (lane_id:uint32) 
                (input:int) (output:Ref<int>) =
                let c = h.Constants

                let inclusive = BasicScanInt h false true temp_storage warp_id lane_id input
                output := ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id, (c.HALF_WARP_THREADS |> uint32) + lane_id - 1u))
            


            let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
                (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
                (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
                let c = h.Constants
            
                Default h scan_op temp_storage warp_id lane_id input output

    [<Record>]
    type IntApi =
        {
            mutable temp_storage    : TempStorage<int>
            mutable warp_id         : uint32
            mutable lane_id         : uint32
        }
        
        [<ReflectedDefinition>] static member Init(temp_storage, warp_id, lane_id) = { temp_storage = temp_storage; warp_id = warp_id; lane_id = lane_id }

        [<ReflectedDefinition>] 
        member this.InclusiveSum(h, input:int, output:Ref<int>) = 
            InclusiveSum.DefaultInt h this.temp_storage this.warp_id this.lane_id input output

        [<ReflectedDefinition>]
        member this.InclusiveSum(h, input:int, output:Ref<int>, warp_aggregate) =
            InclusiveSum.WithAggregateInt h this.temp_storage this.warp_id this.lane_id input output warp_aggregate
        
        [<ReflectedDefinition>] member this.ExclusiveScan(h, input, output) 
            = ExclusiveScan.Identityless.DefaultInt h this.temp_storage this.warp_id this.lane_id input output



//    [<Record>]
//    type API<'T> =
//        {
//            mutable temp_storage    : TempStorage<'T>
//            mutable warp_id         : uint32
//            mutable lane_id         : uint32
//        }
//        
//        [<ReflectedDefinition>] static member Init(temp_storage, warp_id, lane_id) = { temp_storage = temp_storage; warp_id = warp_id; lane_id = lane_id }
//
//        [<ReflectedDefinition>] 
//        member this.InclusiveSum(h, input:int, output:Ref<int>) = 
//            InclusiveSum.DefaultInt h this.temp_storage this.warp_id this.lane_id input output

//        [<ReflectedDefinition>] member this.InclusiveSum(h, scan_op, input, output) 
//            = InclusiveSum.Default h scan_op this.temp_storage this.warp_id this.lane_id input output
////        
//        [<ReflectedDefinition>] member this.InclusiveSum(h, scan_op, input, output, warp_aggregate) 
//            = InclusiveSum.WithAggregate h scan_op this.temp_storage this.warp_id this.lane_id input output warp_aggregate
//
//        [<ReflectedDefinition>] member this.InclusiveScan(h, scan_op, input, output) 
//            = InclusiveScan.Default h scan_op this.temp_storage this.warp_id this.lane_id input output
//        
//        [<ReflectedDefinition>] member this.InclusiveScan(h, scan_op, input, output, warp_aggregate) 
//            = InclusiveScan.WithAggregate h scan_op this.temp_storage this.warp_id this.lane_id input output warp_aggregate  
//
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity) 
//            = ExclusiveScan.Default h scan_op this.temp_storage this.warp_id this.lane_id input output identity
//        
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity, warp_aggregate) 
//            = ExclusiveScan.WithAggregate h scan_op this.temp_storage this.warp_id this.lane_id input output identity warp_aggregate
//        
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output) 
//            = ExclusiveScan.Identityless.Default h scan_op this.temp_storage this.warp_id this.lane_id input output
//        
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, input, output) 
//            = ExclusiveScan.Identityless.Default h this.temp_storage this.warp_id this.lane_id input output
        
//        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, warp_aggregate) 
//            = ExclusiveScan.Identityless.WithAggregate h scan_op this.temp_storage this.warp_id this.lane_id input output warp_aggregate


















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
//    open Template
//    
//
//    let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
//        (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
//        (input:'T) (output:Ref<'T>) =
//        let has_identity = true //PRIMITIVE()
//        let share_final = false
//            //(%InitIdentity) d
//        output := BasicScan h has_identity share_final scan_op temp_storage warp_id lane_id input
//   
//    let [<ReflectedDefinition>] inline DefaultInt (h:HostApi)
//        (temp_storage:TempStorage<int>) (warp_id:uint32) (lane_id:uint32)
//        (input:int) (output:Ref<int>) =
//        let has_identity = true //PRIMITIVE()
//        let share_final = false
//            //(%InitIdentity) d
//        output := BasicScanInt h has_identity share_final temp_storage warp_id lane_id input  
//      
//    
//    let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
//        (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
//        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//        let has_identity = true //PRIMITIVE()
//        let share_final = true
//        let p = h.Params
//        let c = h.Constants
//            
//        //(%InitIdentity) d
//            
//        output := BasicScan h has_identity share_final scan_op temp_storage warp_id lane_id input            
//        
//        warp_aggregate := ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id, c.WARP_SMEM_ELEMENTS - 1))
//
//
//    let [<ReflectedDefinition>] inline WithAggregateInt (h:HostApi)
//        (temp_storage:TempStorage<int>) (warp_id:uint32) (lane_id:uint32)
//        (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) =
//        let has_identity = true //PRIMITIVE()
//        let share_final = true
//        let p = h.Params
//        let c = h.Constants
//            
//        //(%InitIdentity) d
//            
//        output := BasicScanInt h has_identity share_final temp_storage warp_id lane_id input            
//        warp_aggregate := ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id, c.WARP_SMEM_ELEMENTS - 1))
////
////module InclusiveScan =
////    open Template
////
////    let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
////        (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
////        (input:'T) (output:Ref<'T>) =
////        output := BasicScan h false false scan_op temp_storage warp_id lane_id input
////        
////
////    let [<ReflectedDefinition>] WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
////        (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
////        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
////        let c = h.Constants
////        
////        output := BasicScan h true false scan_op temp_storage warp_id lane_id input
////        warp_aggregate := ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id, c.WARP_SMEM_ELEMENTS - 1))
////
//
////module ExclusiveScan =
////    open Template
////
////    let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
////        (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
////        (input:'T) (output:Ref<'T>) (identity:'T) =
////        let c = h.Constants
////        ThreadStore.STORE_VOLATILE (temp_storage.GetPtr(warp_id, lane_id)) (identity)
////
////        let inclusive = BasicScan h true true scan_op temp_storage warp_id lane_id input
////        output := ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id, c.HALF_WARP_THREADS + lane_id - 1))
////        
////
////    let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
////        (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
////        (input:'T) (output:Ref<'T>) (identity:'T) (warp_aggregate:Ref<'T>) =
////        let c = h.Constants
////        
////        Default h scan_op temp_storage warp_id lane_id input output identity
////        warp_aggregate := ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id, c.HALF_WARP_THREADS + lane_id - 1))
////        
////
////    module Identityless =
////        let [<ReflectedDefinition>] inline Default (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32) 
////            (input:'T) (output:Ref<'T>) =
////            let c = h.Constants
////
////            let inclusive = BasicScan h false true scan_op temp_storage warp_id lane_id input
////            output := ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id, c.HALF_WARP_THREADS + lane_id - 1))
////            
////
////        let [<ReflectedDefinition>] inline WithAggregate (h:HostApi) (scan_op:'T -> 'T -> 'T)
////            (temp_storage:TempStorage<'T>) (warp_id:uint32) (lane_id:uint32)
////            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
////            let c = h.Constants
////            
////            Default h scan_op temp_storage warp_id lane_id input output
////            warp_aggregate := ThreadLoad.LOAD_VOLATILE (temp_storage.GetPtr(warp_id, c.HALF_WARP_THREADS + lane_id - 1))
//    
////    [<Record>]
////    type API<'T> =
////        {
////            mutable DeviceApi : _DeviceApi<'T>
////        }
////
////        [<ReflectedDefinition>] static member Create(temp_storage, warp_id, lane_id) = { DeviceApi = _DeviceApi<'T>.Init(temp_storage, warp_id, lane_id)}
////
////        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity) 
////            = Default h scan_op this.temp_storage this.warp_id this.lane_id input output identity
////        
////        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity, warp_aggregate) 
////            = WithAggregate h scan_op this.temp_storage this.warp_id this.lane_id input output identity warp_aggregate
////        
////        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output) 
////            = Identityless.Default h scan_op this.temp_storage this.warp_id this.lane_id input output
////        
////        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, warp_aggregate) 
////            = Identityless.WithAggregate h scan_op this.temp_storage this.warp_id this.lane_id input output warp_aggregate
//    module InclusiveSum =
//        type FunctionApi<'T> = Template.InclusiveSum._FunctionApi<'T>
//
//        let inline api (h:HostApi) (scan_op:'T -> 'T -> 'T) = InclusiveSum.api h scan_op
//        
//        let inline template<'T> (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
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
//        let inline api (h:HostApi) (scan_op:'T -> 'T -> 'T) = InclusiveScan.api h scan_op
//
//        let inline template<'T> (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
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
//    module ExclusiveScan =
//        type FunctionApi<'T> = Template.ExclusiveScan._FunctionApi<'T>
//
//        let inline api (h:HostApi) (scan_op:'T -> 'T -> 'T) = ExclusiveScan.api h scan_op
//
//        let inline template<'T> (logical_warps:int) (logical_warp_threads:int) (scan_op:'T -> 'T -> 'T) : Template<HostApi*FunctionApi<'T>> = cuda {
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
//                (temp_storage.[warp_id, lane_id] |> __array_to_ptr, identity) ||> store
//            | false ->
//                ()
//
//let scanStep logical_warps logical_warp_threads =
//    let HALF_WARP_THREADS = logical_warp_threads |> HALF_WARP_THREADS
//    let STEPS = logical_warp_threads |> STEPS
//    fun (temp_storage:int[,]) warp_id lane_id ->
//        let load = LOAD_VOLATILE |> threadLoad()
//        let store = STORE_VOLATILE |> threadStore()
//        fun has_identity step ->
//            let step = ref 0
//            fun (partial:Ref<int>) (scan_op:(int -> int -> int)) ->
//                while !step < STEPS do
//                    let OFFSET = 1 <<< !step
//                    
//                    //(temp_storage |> __array_to_ptr, !partial) ||> store
//
//                    if has_identity || (lane_id >= OFFSET) then
//                        let addend = (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id)] |> __obj_to_ptr, Some(partial |> __ref_to_ptr)) ||> load
//                        partial := (addenValue, !partial) ||> scan_op
//                        
//                    step := !step + 1
//
//
//let broadcast =
//    fun (temp_storage:TempStorage<int>) warp_id lane_id ->
//        let load = LOAD_VOLATILE |> threadLoad()
//        let store = STORE_VOLATILE |> threadStore()
//        fun input src_lane ->
//            if lane_id = src_lane then (temp_storage.[warp_id] |> __obj_to_ptr, input) ||> store
//            (temp_storage.[warp_id] |> __obj_to_ptr, None) ||> load
//            |> Option.get
//
//
//let inline basicScan logical_warps logical_warp_threads = 
//    let HALF_WARP_THREADS = logical_warp_threads |> HALF_WARP_THREADS
//    let scanStep = (logical_warps, logical_warp_threads) ||> scanStep
//    fun has_identity share_final ->
//        fun (temp_storage:int[,]) warp_id lane_id ->
//            let store = STORE_VOLATILE |> threadStore()
//            fun (partial:int) (scan_op:(int -> int -> int)) ->
//                let partial = partial |> __obj_to_ref
//                scanStep
//                <|||    (temp_storage, warp_id, lane_id)
//                <||     (has_identity, 0)
//                <||     (partial, scan_op)
//                if share_final then (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id)] |> __obj_to_ptr, !partial) ||> store
//                !partial
//
//
//let inline inclusiveSum logical_warps logical_warp_threads =
//    let WARP_SMEM_ELEMENTS = logical_warp_threads |> WARP_SMEM_ELEMENTS
//    let initIdentity = (logical_warps, logical_warp_threads) ||> initIdentity
//    let basicScan = (logical_warps, logical_warp_threads) ||> basicScan
//
//    fun (temp_storage:int[,]) warp_id lane_id ->
//        let load = LOAD_VOLATILE |> threadLoad()
//            
//        fun (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<int> option) ->
//            match warp_aggregate with
//            | None ->
//                let HAS_IDENTITY = true // Traits<int>::PRIMITIVE
//                initIdentity
//                <|  HAS_IDENTITY 
//                <|| (warp_id, lane_id)
//                
//                output :=
//                    basicScan
//                    <||     (HAS_IDENTITY, false)
//                    <|||    (temp_storage, warp_id, lane_id) 
//                    <||     (input, ( + ))
//
//            | Some warp_aggregate ->
//                let HAS_IDENTITY = true // Traits<int>::PRIMITIVE
//                initIdentity
//                <|  HAS_IDENTITY
//                <|| (warp_id, lane_id)
//
//                output :=
//                    basicScan
//                    <||     (HAS_IDENTITY, true)
//                    <|||    (temp_storage, warp_id, lane_id)
//                    <||     (input, ( + ))
//
//                warp_aggregate :=
//                    (temp_storage.[warp_id, (WARP_SMEM_ELEMENTS - 1)] |> __obj_to_ptr, None) 
//                    ||> load
//                    |> Option.get
//
//
//let inline inclusiveScan logical_warps logical_warp_threads =
//    let WARP_SMEM_ELEMENTS = logical_warp_threads |> WARP_SMEM_ELEMENTS
//    let basicScan = (logical_warps, logical_warp_threads) ||> basicScan
//
//    fun (temp_storage:int[,]) warp_id lane_id ->
//        let load = LOAD_VOLATILE |> threadLoad()
//
//        fun (input:'T) (output:Ref<'T>) (scan_op:(int -> int -> int)) (warp_aggregate:Ref<int> option) ->
//            match warp_aggregate with
//            | None ->
//                output :=
//                    basicScan
//                    <||     (false, false)
//                    <|||    (temp_storage, warp_id, lane_id)
//                    <||     (input, scan_op)
//
//            | Some warp_aggregate ->
//                output :=
//                    basicScan
//                    <||     (false, true)
//                    <|||    (temp_storage, warp_id, lane_id)
//                    <||     (input, scan_op)
//
//                warp_aggregate :=
//                    (temp_storage.[warp_id, (WARP_SMEM_ELEMENTS - 1)] |> __obj_to_ptr, None) 
//                    ||> load
//                    |> Option.get 
//
//    
//let inline exclusiveScan logical_warps logical_warp_threads =
//    let HALF_WARP_THREADS = logical_warp_threads |> HALF_WARP_THREADS
//    let WARP_SMEM_ELEMENTS = logical_warp_threads |> WARP_SMEM_ELEMENTS
//    let basicScan = (logical_warps, logical_warp_threads) ||> basicScan
//
//    fun (temp_storage:int[,]) warp_id lane_id ->
//        let load = LOAD_VOLATILE |> threadLoad()
//        let store = STORE_VOLATILE |> threadStore()
//
//        fun (input:'T) (output:Ref<'T>) (scan_op:(int -> int -> int)) (identity:int option) (warp_aggregate:Ref<int> option) ->
//            match identity, warp_aggregate with
//            | Some identity, None ->
//                (temp_storage.[warp_id, lane_id] |> __obj_to_ptr, identity) ||> store
//                let inclusive =
//                    basicScan
//                    <||     (true, true)
//                    <|||    (temp_storage, warp_id, lane_id)
//                    <||     (input, scan_op)
//
//                output :=
//                    (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id - 1)] |> __obj_to_ptr, None) 
//                    ||> load
//                    |> Option.get
//
//            | Some identity, Some warp_aggregate ->
//                (temp_storage.[warp_id, lane_id] |> __obj_to_ptr, identity) ||> store
//                let inclusive =
//                    basicScan
//                    <||     (true, true)
//                    <|||    (temp_storage, warp_id, lane_id)
//                    <||     (input, scan_op)
//
//                output :=
//                    (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id - 1)] |> __obj_to_ptr, None) 
//                    ||> load
//                    |> Option.get
//
//                warp_aggregate :=
//                    (temp_storage.[warp_id, (WARP_SMEM_ELEMENTS - 1)] |> __obj_to_ptr, None)
//                    ||> load
//                    |> Option.get
//
//            | None, None ->
//                let inclusive =
//                    basicScan
//                    <||     (false, true)
//                    <|||    (temp_storage, warp_id, lane_id)
//                    <||     (input, scan_op)
//
//                output :=
//                    (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id - 1)] |> __obj_to_ptr, None)
//                    ||> load
//                    |> Option.get
//
//            | None, Some warp_aggregate ->
//                let inclusive =
//                    basicScan
//                    <||     (false, true)
//                    <|||    (temp_storage, warp_id, lane_id)
//                    <||     (input, scan_op)
//
//                output :=
//                    (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id - 1)] |> __obj_to_ptr, None)
//                    ||> load
//                    |> Option.get
//
//                warp_aggregate :=
//                    (temp_storage.[warp_id, (WARP_SMEM_ELEMENTS - 1)] |> __obj_to_ptr, None)
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
//    static member Init(temp_storage:int[,], warp_id:uint32, lane_id:uint32) =
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
//        //let temp_storage = Array2zeroCreate logical_warps c.WARP_SMEM_ELEMENTS
//        let temp_storage = __shared__.Array2D(logical_warps)
//        {
//            LOGICAL_WARPS           = logical_warps
//            LOGICAL_WARP_THREADS    = logical_warp_threads
//            Constants               = c
//            //TempStorage             = temp_storage
//            //ThreadFields            = (temp_storage, 0u, 0u) |> ThreadFields.Init
//        }