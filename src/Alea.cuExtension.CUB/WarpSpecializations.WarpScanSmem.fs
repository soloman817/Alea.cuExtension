[<AutoOpen>]
module Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread


let private Broadcast (temp_storage:deviceptr<int>) (warp_id:int) (lane_id:int) =
    let threadStore = STORE_VOLATILE |> ThreadStore
    let threadLoad  = ThreadLoad() //LOAD_VOLATILE |> ThreadLoad<int>
    let lane_id = lane_id |> uint32
    <@ fun (input:int) (src_lane:int) ->
        let src_lane = src_lane |> uint32
        if lane_id = src_lane then (temp_storage + warp_id, input) ||> %threadStore
        (temp_storage + warp_id) |> %threadLoad
    @>


module private Internal =
    module Constants =
        
        let POW_OF_TWO =
            fun logical_warp_threads -> ((logical_warp_threads &&& (logical_warp_threads - 1)) = 0)

        /// The number of warp scan steps
        let STEPS =
            fun logical_warp_threads ->
                logical_warp_threads |> log2

        /// The number of threads in half a warp
        let HALF_WARP_THREADS =
            fun logical_warp_threads ->
                let STEPS = logical_warp_threads |> STEPS
                1 <<< (STEPS - 1)

        /// The number of shared memory elements per warp
        let WARP_SMEM_ELEMENTS =
            fun logical_warp_threads ->
                logical_warp_threads + (logical_warp_threads |> HALF_WARP_THREADS)

        
    module Sig =
        module InclusiveSum =
            type DefaultExpr = Expr<int -> Ref<int> -> unit>
            type WithAggregateExpr = Expr<int -> Ref<int> -> Ref<int> -> unit>
                
        module InclusiveScan =
            type DefaultExpr = Expr<int -> Ref<int> -> unit>
            type WithAggregateExpr = Expr<int -> Ref<int> -> Ref<int> -> unit>

        module ExclusiveScan =
            type DefaultExpr = Expr<int -> Ref<int> -> int -> unit> 
            type WithAggregateExpr = Expr<int -> Ref<int> -> int -> Ref<int> -> unit>

            module Identityless =
                type DefaultExpr = Expr<int -> Ref<int> -> unit>
                type WithAggregateExpr = Expr<int -> Ref<int> -> Ref<int> -> unit>
                
        type BroadcastExpr = Expr<int -> int -> int>

    module InitIdentity =
    
        let private True (temp_storage:int[,]) (warp_id:int) (lane_id:int) =
            let threadStore = STORE_VOLATILE |> ThreadStore
            <@ fun () ->
                let identity = ZeroInitialize()
                ()
                //(temp_storage.[warp_id, lane_id]  |> __array2d_to_ptr, identity) ||> threadStore
            @>

        let private False (temp_storage:int[,]) (warp_id:int) (lane_id:int) =
            <@ fun () -> () @>

        let api has_identity = if has_identity then True else False


    module ScanStep =
        type private ScanStepAttribute(modifier:string) =
            inherit Attribute()

            interface ICustomCallBuilder with
                member this.Build(ctx, irObject, info, irParams) =
                    match irObject, irParams with
                    | None, irMax :: irPtr :: irVals :: [] ->
                        let max = irMax.HasObject |> function
                            | true -> irMax.Object :?> int
                            | false -> failwith "max must be constant"

                        // if we do the loop here, it is unrolled by compiler, not kernel runtime
                        // think of this job as the C++ template expanding job, same thing!
                        for i = 0 to max - 1 do
                            let irIndex = IRCommonInstructionBuilder.Instance.BuildConstant(ctx, i)
                            let irVal = IRCommonInstructionBuilder.Instance.BuildGEP(ctx, irPtr, irIndex :: [])
                            let irPtr = ()
                    
                            let irPtr = IRCommonInstructionBuilder.Instance.BuildGEP(ctx, irVals, irIndex :: [])
                            IRCommonInstructionBuilder.Instance.BuildStore(ctx, irPtr, irVal) |> ignore

                        IRCommonInstructionBuilder.Instance.BuildNop(ctx) |> Some

                    | _ -> None



    let BasicScan has_identity share_final (scan_op:IScanOp) =
        let scan_op = scan_op.op
        <@ fun (partial:int) -> 
        
            __null() |> __ptr_to_obj 
        @>


module InclusiveSum =
    open Internal

    type API =
        {
            Default         : Sig.InclusiveSum.DefaultExpr
            WithAggregate   : Sig.InclusiveSum.WithAggregateExpr
        }


    let private Default _ _ (scan_op:IScanOp) =
        fun (temp_storage:int[,]) (warp_id:int) (lane_id:int)  ->
            let has_identity = true //PRIMITIVE()
            let initIdentity = 
                InitIdentity.api 
                <|  has_identity
                <|  temp_storage
                <|| (warp_id, lane_id)

            let basicScan = (has_identity, false, scan_op) |||> BasicScan
            let scan_op = scan_op.op        
            <@ fun (input:int) (output:Ref<int>) ->
                (%initIdentity)()
                output := (input, %scan_op) ||> %basicScan
            @>

    let private WithAggregate logical_warps logical_warp_threads (scan_op:IScanOp) =
        let warp_smem_elements = logical_warp_threads |> Constants.WARP_SMEM_ELEMENTS
        fun (temp_storage:int[,]) (warp_id:int) (lane_id:int) ->
            let has_identity = true //PRIMITIVE()
            let initIdentity = 
                InitIdentity.api 
                <|  has_identity
                <|  temp_storage
                <|| (warp_id, lane_id)


            let basicScan = (has_identity, true, scan_op) |||> BasicScan
            let threadLoad = LOAD_VOLATILE |> ThreadLoad
        
            <@ fun (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) ->
                (%initIdentity)()
                output := input |> %basicScan
                //@TODO
                let w_a : 'T = temp_storage.GetValue((warp_smem_elements - 1) + warp_id * logical_warps) |> __obj_reinterpret
                warp_aggregate := w_a //|> __obj_reinterpret |> %threadLoad //(w_a |> __unbox) // |> %threadLoad  
            @>

    let api logical_warps logical_warp_threads (scan_op:IScanOp) = 
        fun (temp_storage:int[,]) (warp_id:int) (lane_id:int) ->
            {
                Default         =   Default
                                    <|||    (logical_warps, logical_warp_threads, scan_op)
                                    <|||    (temp_storage, warp_id, lane_id)

                WithAggregate   =   WithAggregate
                                    <|||    (logical_warps, logical_warp_threads, scan_op)
                                    <|||    (temp_storage, warp_id, lane_id)
            }


module InclusiveScan =
    open Internal

    type API =
        {
            Default         : Sig.InclusiveScan.DefaultExpr
            WithAggregate   : Sig.InclusiveScan.WithAggregateExpr
        }    

    let private Default (scan_op:IScanOp) =
        fun (temp_storage:int[,]) (warp_id:int) (lane_id:int)  ->
            let scan_op = scan_op.op
            <@ fun (input:int) (output:Ref<int>) -> () @>

    let private WithAggregate (scan_op:IScanOp) =
        fun (temp_storage:int[,]) (warp_id:int) (lane_id:int) ->
            let scan_op = scan_op.op
            <@ fun (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) -> () @>

    let api _ _ scan_op =
        fun (temp_storage:int[,]) (warp_id:int) (lane_id:int) ->
            {
                Default         =   Default
                                    <|      scan_op
                                    <|||    (temp_storage, warp_id, lane_id)

                WithAggregate   =   WithAggregate
                                    <|      scan_op
                                    <|||    (temp_storage, warp_id, lane_id)
            }


module ExclusiveScan =
    open Internal

    type API =
        {
            Default             : Sig.ExclusiveScan.DefaultExpr
            Default_NoID        : Sig.ExclusiveScan.Identityless.DefaultExpr
            WithAggregate       : Sig.ExclusiveScan.WithAggregateExpr
            WithAggregate_NoID  : Sig.ExclusiveScan.Identityless.WithAggregateExpr
        }

    let private Default (scan_op:IScanOp) =
        fun (temp_storage:int[,]) (warp_id:int) (lane_id:int) ->
            let scan_op = scan_op.op
            <@ fun (input:int) (output:Ref<int>) (identity:int) -> () @>

    let private WithAggregate (scan_op:IScanOp) =
        fun (temp_storage:int[,]) (warp_id:int) (lane_id:int) ->
            let scan_op = scan_op.op
            <@ fun (input:int) (output:Ref<int>) (identity:int) (warp_aggregate:Ref<int>) -> () @>

    module private Identityless =

        let Default (scan_op:IScanOp) =
            fun (temp_storage:int[,]) (warp_id:int) (lane_id:int) ->
                let scan_op = scan_op.op
                <@ fun (input:int) (output:Ref<int>) -> () @>

        let WithAggregate (scan_op:IScanOp) =
            fun (temp_storage:int[,]) (warp_id:int) (lane_id:int) ->
                let scan_op = scan_op.op
                <@ fun (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) -> () @>        

    let api _ _ scan_op =
        fun (temp_storage:int[,]) (warp_id:int) (lane_id:int) ->
            {
                Default         =   Default
                                    <|      scan_op
                                    <|||    (temp_storage, warp_id, lane_id)

                Default_NoID    =   Identityless.Default
                                    <|      scan_op
                                    <|||    (temp_storage, warp_id, lane_id)

                WithAggregate   =   WithAggregate
                                    <|      scan_op
                                    <|||    (temp_storage, warp_id, lane_id)

                WithAggregate_NoID =    Identityless.WithAggregate
                                        <|      scan_op
                                        <|||    (temp_storage, warp_id, lane_id)
            }


module WarpScanSmem =
    type API =
        {
            InclusiveScan   : InclusiveScan.API
            InclusiveSum    : InclusiveSum.API
            ExclusiveScan   : ExclusiveScan.API
            Broadcast       : Internal.Sig.BroadcastExpr
        }

    let api logical_warps logical_warp_threads scan_op =
        fun (temp_storage:int[,]) (warp_id:int) (lane_id:int) ->
            {
                InclusiveScan   =   InclusiveScan.api
                                    <|||    (logical_warps, logical_warp_threads, scan_op)
                                    <|||    (temp_storage, warp_id, lane_id)
                                    
                InclusiveSum    =   InclusiveSum.api
                                    <|||    (logical_warps, logical_warp_threads, scan_op)
                                    <|||    (temp_storage, warp_id, lane_id)

                ExclusiveScan   =   ExclusiveScan.api
                                    <|||    (logical_warps, logical_warp_threads, scan_op)
                                    <|||    (temp_storage, warp_id, lane_id)

                Broadcast       =   Broadcast
                                    // temp_storage |> __array_to_ptr
                                    <|||    (__null(), warp_id, lane_id)
            }





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
//                        partial := (addend.Value, !partial) ||> scan_op
//                        
//                    step := !step + 1
//
//
//let broadcast =
//    fun (temp_storage:deviceptr<int>) warp_id lane_id ->
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
//        fun (input:int) (output:Ref<int>) (warp_aggregate:Ref<int> option) ->
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
//        fun (input:int) (output:Ref<int>) (scan_op:(int -> int -> int)) (warp_aggregate:Ref<int> option) ->
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
//        fun (input:int) (output:Ref<int>) (scan_op:(int -> int -> int)) (identity:int option) (warp_aggregate:Ref<int> option) ->
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
//        //let temp_storage = Array2D.zeroCreate logical_warps c.WARP_SMEM_ELEMENTS
//        let temp_storage = __shared__.Array2D(logical_warps, c.WARP_SMEM_ELEMENTS)
//        {
//            LOGICAL_WARPS           = logical_warps
//            LOGICAL_WARP_THREADS    = logical_warp_threads
//            Constants               = c
//            //TempStorage             = temp_storage
//            //ThreadFields            = (temp_storage, 0u, 0u) |> ThreadFields.Init
//        }