[<AutoOpen>]
module Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanShfl

open System    
open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Warp

let private Broadcast logical_warp_threads =
    let shuffleBroadcast = logical_warp_threads |> __ptx__.ShuffleBroadcast
    <@ fun (input:int) (src_lane:int) ->
        (input, src_lane) ||> %shuffleBroadcast
    @>

module private Internal =
    module Enum =
        let STEPS = 
            fun logical_warp_threads ->
                logical_warp_threads |> log2

        let SHFL_C =
            fun logical_warp_threads ->
                let STEPS = logical_warp_threads |> STEPS
                ((-1 <<< STEPS) &&& 31) <<< 8


    module Sig =
        module InclusiveScan =
            type DefaultExpr        = Expr<int -> Ref<int> -> unit>
            type WithAggregateExpr  = Expr<int -> Ref<int> -> Ref<int> -> unit>

        module InclusiveSum =
            type DefaultExpr = Expr<int -> Ref<int> -> unit>

            module WithAggregate =
                type GenericExpr    = Expr<int -> Ref<int> -> Ref<int> -> unit>
                type ULongLongExpr  = Expr<ulonglong -> Ref<ulonglong> -> Ref<ulonglong> -> unit>
                type Float32Expr    = Expr<float32 -> Ref<float32> -> Ref<float32> -> unit>
                type MultiShfl      = Expr<int -> Ref<int> -> Ref<int> -> unit>
                type SingleShfl     = Expr<int -> Ref<int> -> Ref<int> -> unit>

        module ExclusiveScan =
            type DefaultExpr        = Expr<int -> Ref<int> -> int -> unit>
            type WithAggregateExpr  = Expr<int -> Ref<int> -> int -> Ref<int> -> unit>
            
            module Identityless =
                type DefaultExpr        = Expr<int -> Ref<int> -> unit>
                type WithAggregateExpr  = Expr<int -> Ref<int> -> Ref<int> -> unit>

        type BroadcastExpr = Expr<int -> int -> int>
                




module InclusiveScan =
    open Internal

    type API =
        {
            Default         :   Sig.InclusiveScan.DefaultExpr
            WithAggregate   :   Sig.InclusiveScan.WithAggregateExpr            
        }

    let private WithAggregate logical_warp_threads (scan_op:IScanOp) =
        let STEPS = logical_warp_threads |> Enum.STEPS
        let broadcast = logical_warp_threads |> Broadcast
        let scan_op = scan_op.op
        <@ fun (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) -> 
            output := input

            for STEP = 0 to (STEPS - 1) do
                let OFFSET = 1 <<< STEP
                let temp = (!output, OFFSET) |> __ptx__.ShuffleUp
                ()
//                if lane_id >= OFFSET then output := (temp |> __obj_reinterpret, !output) ||> %scan_op

            warp_aggregate := (!output, logical_warp_threads - 1) ||> %broadcast
        @>

    let private Default logical_warp_threads (scan_op:IScanOp) =
        let inclusiveScan = (logical_warp_threads, scan_op) ||> WithAggregate
        let scan_op = scan_op.op
        <@ fun (input:int) (output:Ref<int>) ->
            let warp_aggregate = __local__.Variable()
            %inclusiveScan
            <|| (input,output)
            <|  warp_aggregate
        @>

    let api logical_warp_threads scan_op =
        {
            Default         = (logical_warp_threads, scan_op) ||> Default
            WithAggregate   = (logical_warp_threads, scan_op) ||> WithAggregate
        }

//let x = InclusiveScan.api 0 (scan_op ADD 0)

module InclusiveSum =
    open Internal

    type API =
        {
            Default                 : Sig.InclusiveSum.DefaultExpr
            Generic                 : Sig.InclusiveSum.WithAggregate.GenericExpr
            ULongLongSpecialized    : Sig.InclusiveSum.WithAggregate.ULongLongExpr
            Float32Specialized      : Sig.InclusiveSum.WithAggregate.Float32Expr
            MultiShfl               : Sig.InclusiveSum.WithAggregate.MultiShfl
            SingleShfl              : Sig.InclusiveSum.WithAggregate.SingleShfl
        }


    [<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
    type private InclusiveSumPtxAttribute() =
        inherit Attribute()

        interface ICustomCallBuilder with
            member this.Build(ctx, irObject, info, irParams) =
                match irObject, irParams with
                | None, temp :: shlStep :: shfl_c :: [] ->
                    let clrType = info.GetGenericArguments().[0]
                    let irType = IRTypeBuilder.Instance.Build(ctx, clrType)
                    let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<uint32 -> int -> int-> uint32>)
                    let irFunctionType = IRTypeBuilder.Instance. BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
                    IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
                        "{
                            .reg .u32 r0;
                            .reg .pred p;
                            shfl.up.b32 r0|p, $1, $2, $3;
                            @p add.u32 r0, r0, %4;
                            mov.u32 %0, r0;
                        }", "=r,r,r,r,r", temp :: shlStep :: shfl_c :: []) |> Some
                | _ -> None

    let [<InclusiveSumPtx>] private inclusiveSumPtx (temp:uint32) (shlStep:int) (shfl_c:int) : uint32 = failwith ""


    let private SingleShfl logical_warp_threads = 
        let STEPS = logical_warp_threads |> Enum.STEPS
        let SHFL_C = logical_warp_threads |> Enum.SHFL_C
        let broadcast = logical_warp_threads |> Broadcast
        <@ fun (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) ->
            let temp : Ref<uint32> = input |> __obj_to_ref |> __ref_reinterpret
            for STEP = 0 to (STEPS - 1) do
                temp := (!temp |> uint32, (1 <<< STEP), SHFL_C) |||> inclusiveSumPtx
            output := !temp |> __obj_reinterpret
            warp_aggregate := (!output, logical_warp_threads - 1) ||> %broadcast
        @>

    let private MultiShfl _ =
        <@ fun (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) -> () @>
        

    [<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
    type private InclusiveSumPtx_Float32Attribute() =
        inherit Attribute()

        interface ICustomCallBuilder with
            member this.Build(ctx, irObject, info, irParams) =
                match irObject, irParams with
                | None, temp :: shlStep :: shfl_c :: [] ->
                    let clrType = info.GetGenericArguments().[0]
                    let irType = IRTypeBuilder.Instance.Build(ctx, clrType)
                    let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<float32 -> int -> int-> float32>)
                    let irFunctionType = IRTypeBuilder.Instance. BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
                    IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
                        "{
                            .reg .f32 r0;
                            .reg .pred p;
                            shfl.up.b32 r0|p, $1, $2, $3;
                            @p add.f32 r0, r0, $4;
                            mov.f32 $0, r0;
                        }", "=f,f,r,r,f", temp :: shlStep :: shfl_c :: []) |> Some
                | _ -> None

    let [<InclusiveSumPtx_Float32>] private inclusiveSumPtx_Float32 (output:float32) (shlStep:int) (shfl_c:int) : float32 = failwith ""

    let private Float32Specialized logical_warp_threads =
        let STEPS = logical_warp_threads |> Enum.STEPS
        let SHFL_C = logical_warp_threads |> Enum.SHFL_C
        let broadcast = logical_warp_threads |> Broadcast
        <@ fun (input:float32) (output:Ref<float32>) (warp_aggregate:Ref<float32>) ->
            output := input
            for STEP = 0 to (STEPS - 1) do
                output := (!output, (1 <<< STEP), SHFL_C) |||> inclusiveSumPtx_Float32

            warp_aggregate := (!output |> int, logical_warp_threads - 1) ||> %broadcast
        @>

    [<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
    type private InclusiveSumPtx_ULongLongAttribute() =
        inherit Attribute()

        interface ICustomCallBuilder with
            member this.Build(ctx, irObject, info, irParams) =
                match irObject, irParams with
                | None, temp :: shlStep :: shfl_c :: [] ->
                    let clrType = info.GetGenericArguments().[0]
                    let irType = IRTypeBuilder.Instance.Build(ctx, clrType)
                    let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<ulonglong -> int -> int-> ulonglong>)
                    let irFunctionType = IRTypeBuilder.Instance. BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
                    IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
                        "{
                            .reg .u32 r0;
                            .reg .u32 r1;
                            .reg .u32 lo;
                            .reg .u32 hi;
                            .reg .pred p;
                            mov.b64 {lo, hi}, $1;
                            shfl.up.b32 r0|p, lo, $2, $3;
                            shfl.up.b32 r1|p, hi, $2, $3;
                            @p add.cc.u32 r0, r0, lo;
                            @p addc.u32 r1, r1, hi;
                            mov.b64 $0, {r0, r1};
                        }", "=l,l,r,r,l", temp :: shlStep :: shfl_c :: []) |> Some
                | _ -> None

    let [<InclusiveSumPtx_Float32>] private inclusiveSumPtx_ULongLong (output:ulonglong) (shlStep:int) (shfl_c:int) : ulonglong = failwith ""

    let private ULongLongSpecialized logical_warp_threads =
        let STEPS = logical_warp_threads |> Enum.STEPS
        let SHFL_C = logical_warp_threads |> Enum.SHFL_C
        let broadcast = logical_warp_threads |> Broadcast
        <@ fun (input:ulonglong) (output:Ref<ulonglong>) (warp_aggregate:Ref<ulonglong>) ->
            output := input
            for STEP = 0 to (STEPS - 1) do
                output := (!output, (1 <<< STEP), SHFL_C) |||> inclusiveSumPtx_ULongLong
            
            warp_aggregate := (!output |> int, logical_warp_threads - 1) ||> %broadcast
        @>

    let private Generic logical_warp_threads =
        let inclusiveSum = if sizeof<int> <= sizeof<uint32> then logical_warp_threads |> SingleShfl else MultiShfl()
        <@ fun (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) -> 
            (input, output, warp_aggregate) |||> %inclusiveSum
        @>

    let private Default logical_warp_threads =
        let inclusiveSum = logical_warp_threads |> Generic
        <@ fun (input:int) (output:Ref<int>) ->
            let warp_aggregate = __local__.Variable()
            (input, output, warp_aggregate) |||> %inclusiveSum
        @>

    let api logical_warp_threads _ =
        {
            Default                 = logical_warp_threads |> Default
            Generic                 = logical_warp_threads |> Generic
            ULongLongSpecialized    = logical_warp_threads |> ULongLongSpecialized
            Float32Specialized      = logical_warp_threads |> Float32Specialized
            MultiShfl               = logical_warp_threads |> MultiShfl
            SingleShfl              = logical_warp_threads |> SingleShfl
        }

//let y = InclusiveSum.api 0 (scan_op ADD 0)

module ExclusiveScan =
    open Internal

    type API =
        {
            Default             : Sig.ExclusiveScan.DefaultExpr
            Default_NoID        : Sig.ExclusiveScan.Identityless.DefaultExpr
            WithAggregate       : Sig.ExclusiveScan.WithAggregateExpr
            WithAggregate_NoID  : Sig.ExclusiveScan.Identityless.WithAggregateExpr    
        }


    let private WithAggregate logical_warp_threads (scan_op:IScanOp) =
        let inclusiveScan = ((logical_warp_threads, scan_op) ||> InclusiveScan.api).WithAggregate
        <@ fun (input:int) (output:Ref<int>) (identity:int) (warp_aggregate:Ref<int>) ->
            let inclusive = __local__.Variable()
            %inclusiveScan
            <|| (input, inclusive)
            <|  warp_aggregate

            let exclusive = (!inclusive, 1) |> __ptx__.ShuffleUp

//            output := if lane_id = 0 then identity else exclusive |> __obj_reinterpret
            ()
        @>

    let private Default logical_warp_threads (scan_op:IScanOp) =
        let exclusiveScan = (logical_warp_threads, scan_op) ||> WithAggregate
        <@ fun (input:int) (output:Ref<int>) (identity:int) -> 
            let warp_aggregate = __local__.Variable()
            %exclusiveScan
            <|| (input, output)
            <|  identity
            <|  warp_aggregate
        @>


    module private Identityless =
        let WithAggregate logical_warp_threads (scan_op:IScanOp)  =
            let inclusiveScan = ((logical_warp_threads, scan_op) ||> InclusiveScan.api).WithAggregate
            <@ fun (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) ->
                let inclusive = __local__.Variable()
                %inclusiveScan
                <|| (input, inclusive)
                <|  warp_aggregate

                output := (!inclusive, 1) |> __ptx__.ShuffleUp |> __obj_reinterpret
            @>

        let Default logical_warp_threads (scan_op:IScanOp) =
            let exclusiveScan = (logical_warp_threads, scan_op) ||> WithAggregate
            <@ fun (input:int) (output:Ref<int>) ->
                let warp_aggregate = __local__.Variable()
                %exclusiveScan
                <|| (input, output)
                <|  warp_aggregate
            @>

    let api logical_warp_threads scan_op =
        {
            Default             = (logical_warp_threads, scan_op) ||> Default
            Default_NoID        = (logical_warp_threads, scan_op) ||> Identityless.Default
            WithAggregate       = (logical_warp_threads, scan_op) ||> WithAggregate
            WithAggregate_NoID  = (logical_warp_threads, scan_op) ||> Identityless.WithAggregate
        }

//let z = ExclusiveScan.api 0 (scan_op ADD 0)

module WarpScanShfl =
    type API =
        {
            InclusiveScan : InclusiveScan.API
            InclusiveSum  : InclusiveSum.API
            ExclusiveScan : ExclusiveScan.API
            Broadcast     : Internal.Sig.BroadcastExpr
        }

    let api _ logical_warp_threads scan_op =
        fun _ (warp_id:int) (lane_id:int) ->
            {
                InclusiveScan   = (logical_warp_threads, scan_op) ||> InclusiveScan.api
                InclusiveSum    = (logical_warp_threads, scan_op) ||> InclusiveSum.api
                ExclusiveScan   = (logical_warp_threads, scan_op) ||> ExclusiveScan.api
                Broadcast       = logical_warp_threads |> Broadcast
            }


//
//let foo = WarpScanShfl.api 12 (scan_op ADD 0)
//
//let x = foo.InclusiveScan.Default

//let STEPS = 
//    fun logical_warp_threads ->
//        logical_warp_threads |> log2
//
//let SHFL_C =
//    fun logical_warp_threads ->
//        let STEPS = logical_warp_threads |> STEPS
//        ((-1 <<< STEPS) &&& 31) <<< 8
//
//let broadcast =
//    fun logical_warp_threads ->
//        fun input src_lane ->
//            (input, src_lane, logical_warp_threads) |||> ShuffleBroadcast
//
//    
//let inclusiveScan w x y z = ()
//
//[<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
//type InclusiveSumPtxAttribute() =
//    inherit Attribute()
//
//    interface ICustomCallBuilder with
//        member this.Build(ctx, irObject, info, irParams) =
//            match irObject, irParams with
//            | None, temp :: shlStep :: shfl_c :: [] ->
//                let clrType = info.GetGenericArguments().[0]
//                let irType = IRTypeBuilder.Instance.Build(ctx, clrType)
//                let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<uint32 -> int -> int-> uint32>)
//                let irFunctionType = IRTypeBuilder.Instance. BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
//                IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
//                    "{
//                        .reg .u32 r0;
//                        .reg .pred p;
//                        shfl.up.b32 r0|p, $1, $2, $3;
//                        @p add.u32 r0, r0, %4;
//                        mov.u32 %0, r0;
//                    }", "=r,r,r,r,r", temp :: shlStep :: shfl_c :: []) |> Some
//            | _ -> None
//
//let [<InclusiveSumPtx>] inclusiveSumPtx (temp:uint32) (shlStep:int) (shfl_c:int) : uint32 = failwith ""
//let inclusiveSum logical_warps logical_warp_threads =
//    let STEPS = logical_warp_threads |> STEPS
//    let SHFL_C = logical_warp_threads |> SHFL_C
//    let broadcast = logical_warp_threads |> broadcast
//
//    fun (input:int) (output:Ref<int>) (warp_aggregate:Ref<int>) (single_shfl:bool option) ->
//        match single_shfl with
//        | Some single_shfl ->
//            if single_shfl then
//                let temp = ref input
//                for STEP = 0 to (STEPS - 1) do
//                    temp := (!temp |> uint32, (1 <<< STEP), SHFL_C) |||> inclusiveSumPtx |> int
//                output := !temp
//
//                warp_aggregate := (!output, (logical_warp_threads - 1)) ||> broadcast
////                else
////                    (input, output, ( + ), warp_aggregate) |||> inclusiveScan
//            
////type TemplateParameters =
////    {
////        LOGICAL_WARPS           : int
////        LOGICAL_WARP_THREADS    : int
////    }
//
//type Constants =
//    {
//        STEPS   : int
//        SHFL_C  : int
//    }
//
//    static member Init(logical_warp_threads:int) =
//        let steps = logical_warp_threads |> log2
//        {
//            STEPS = steps
//            SHFL_C = ((-1 <<< steps) &&& 31) <<< 8
//        }
//
//
//[<Record>]
//type ThreadFields =
//    {
//        mutable warp_id : int
//        mutable lane_id : int
//    }
//
//    static member Init(warp_id, lane_id) =
//        {
//            warp_id = warp_id
//            lane_id = lane_id
//        }
//
//
//[<Record>]
//type WarpScanShfl =
//    {
//        LOGICAL_WARPS        : int
//        LOGICAL_WARP_THREADS : int 
//        Constants           : Constants
//        ThreadFields        : ThreadFields
//    }
//
//    member this.Initialize(temp_storage, warp_id, lane_id) =
//        this.ThreadFields.warp_id <- warp_id
//        this.ThreadFields.lane_id <- lane_id
//        this
//
//    /// Broadcast
//    member this.Broadcast(input:int,src_lane:int) = 
//        (input, src_lane, this.LOGICAL_WARP_THREADS) |||> ShuffleBroadcast
//
//
//    //---------------------------------------------------------------------
//    // Inclusive operations
//    //---------------------------------------------------------------------
//
//    /// Inclusive prefix sum with aggregate (single-SHFL)
//    member this.InclusiveSum(input:int, output:Ref<int>, warp_aggregate:Ref<int>, ?single_shfl:bool) =
//        let LOGICAL_WARP_THREADS = this.LOGICAL_WARP_THREADS
//        let STEPS = this.Constants.STEPS
//        let temp : uint32 = input |> __obj_reinterpret
//
//        // Iterate scan steps
//        for STEP = 0 to STEPS - 1 do ()
//            // Use predicate set from SHFL to guard against invalid peers
////            asm(
////                "{"
////                "  .reg .u32 r0;"
////                "  .reg .pred p;"
////                "  shfl.up.b32 r0|p, %1, %2, %3;"
////                "  @p add.u32 r0, r0, %4;"
////                "  mov.u32 %0, r0;"
////                "}"
////                : "=r"(temp) : "r"(temp), "r"(1 << STEP), "r"(SHFL_C), "r"(temp));
//        
//        let temp = temp |> __obj_reinterpret
//        output := temp
//
//        // Grab aggregate from last warp lane
//        warp_aggregate := this.Broadcast(!output, LOGICAL_WARP_THREADS - 1)
//
//
//    /// Inclusive prefix sum with aggregate (multi-SHFL)
//    member  inline this.InclusiveSum(input:int, output:Ref<int>, warp_aggregate:Ref<int>) = //, single_shfl)        ///< [in] Marker type indicating whether only one SHFL instruction is required
//        let LOGICAL_WARP_THREADS = this.LOGICAL_WARP_THREADS
//        let STEPS = this.Constants.STEPS
//        typeof<int> |> function
//        | ty when ty = typeof<float32> ->
//            output := input
//
//            // Iterate scan steps
//            for STEP = 0 to STEPS - 1 do ()
//    //            // Use predicate set from SHFL to guard against invalid peers
//    //            asm(
//    //                "{"
//    //                "  .reg .f32 r0;"
//    //                "  .reg .pred p;"
//    //                "  shfl.up.b32 r0|p, %1, %2, %3;"
//    //                "  @p add.f32 r0, r0, %4;"
//    //                "  mov.f32 %0, r0;"
//    //                "}"
//    //                : "=f"(output) : "f"(output), "r"(1 << STEP), "r"(SHFL_C), "f"(output));
//
//            // Grab aggregate from last warp lane
//            warp_aggregate := this.Broadcast(!output, LOGICAL_WARP_THREADS - 1)
//        | ty when ty = typeof<ulonglong> ->
//            output := input
//
//            // Iterate scan steps
//            for STEP = 0 to STEPS - 1 do ()
//                // Use predicate set from SHFL to guard against invalid peers
//    //            asm(
//    //                "{"
//    //                "  .reg .u32 r0;"
//    //                "  .reg .u32 r1;"
//    //                "  .reg .u32 lo;"
//    //                "  .reg .u32 hi;"
//    //                "  .reg .pred p;"
//    //                "  mov.b64 {lo, hi}, %1;"
//    //                "  shfl.up.b32 r0|p, lo, %2, %3;"
//    //                "  shfl.up.b32 r1|p, hi, %2, %3;"
//    //                "  @p add.cc.u32 r0, r0, lo;"
//    //                "  @p addc.u32 r1, r1, hi;"
//    //                "  mov.b64 %0, {r0, r1};"
//    //                "}"
//    //                : "=l"(output) : "l"(output), "r"(1 << STEP), "r"(SHFL_C));
//
//
//            // Grab aggregate from last warp lane
//            warp_aggregate := this.Broadcast(!output, LOGICAL_WARP_THREADS - 1)
//
//
//        // Delegate to generic scan
//        | _ ->
//            this.InclusiveScan(input, output, (+), warp_aggregate)
//
//
////    /// Inclusive prefix sum with aggregate (specialized for float)
////    member this.InclusiveSum(input:float32, output:Ref<float32>, warp_aggregate:Ref<float32>) =
////        let LOGICAL_WARP_THREADS = this.LOGICAL_WARP_THREADS
////        let STEPS = this.Constants.STEPS
////        output := input
////
////        // Iterate scan steps
////        for STEP = 0 to STEPS - 1 do ()
//////            // Use predicate set from SHFL to guard against invalid peers
//////            asm(
//////                "{"
//////                "  .reg .f32 r0;"
//////                "  .reg .pred p;"
//////                "  shfl.up.b32 r0|p, %1, %2, %3;"
//////                "  @p add.f32 r0, r0, %4;"
//////                "  mov.f32 %0, r0;"
//////                "}"
//////                : "=f"(output) : "f"(output), "r"(1 << STEP), "r"(SHFL_C), "f"(output));
////
////        // Grab aggregate from last warp lane
////        warp_aggregate := this.Broadcast(!output, LOGICAL_WARP_THREADS - 1)
//
////
////    /// Inclusive prefix sum with aggregate (specialized for unsigned long long)
////    member this.InclusiveSum(input:ulonglong, output:Ref<ulonglong>, warp_aggregate:Ref<ulonglong>) =
////        let LOGICAL_WARP_THREADS = this.LOGICAL_WARP_THREADS
////        let STEPS = this.Constants.STEPS
////
////        output := input
////
////        // Iterate scan steps
////        for STEP = 0 to STEPS - 1 do ()
////            // Use predicate set from SHFL to guard against invalid peers
//////            asm(
//////                "{"
//////                "  .reg .u32 r0;"
//////                "  .reg .u32 r1;"
//////                "  .reg .u32 lo;"
//////                "  .reg .u32 hi;"
//////                "  .reg .pred p;"
//////                "  mov.b64 {lo, hi}, %1;"
//////                "  shfl.up.b32 r0|p, lo, %2, %3;"
//////                "  shfl.up.b32 r1|p, hi, %2, %3;"
//////                "  @p add.cc.u32 r0, r0, lo;"
//////                "  @p addc.u32 r1, r1, hi;"
//////                "  mov.b64 %0, {r0, r1};"
//////                "}"
//////                : "=l"(output) : "l"(output), "r"(1 << STEP), "r"(SHFL_C));
////
////
////        // Grab aggregate from last warp lane
////        warp_aggregate := this.Broadcast(!output, LOGICAL_WARP_THREADS - 1)
//
//
//    /// Inclusive prefix sum with aggregate (generic)
//    //template <typename _T>
////    member this.InclusiveSum(input:int, output:Ref<int>, warp_aggregate:Ref<int>) =
////        // Whether sharing can be done with a single SHFL instruction (vs multiple SFHL instructions)
////        //Int2Type<(Traits<_T>::PRIMITIVE) && (sizeof(_T) <= sizeof(unsigned int))> single_shfl;
////        let single_shfl = true
////        this.InclusiveSum(input, output, warp_aggregate, single_shfl)
//
//
//    /// Inclusive prefix sum
//    member this.InclusiveSum(input:int, output:Ref<int>) =
//        let warp_aggregate = __local__.Variable<int>() //__null() |> __ptr_to_ref
//        ()
//        //this.InclusiveSum(input, output, warp_aggregate)
//
//
//    /// Inclusive scan with aggregate
//    //template <typename ScanOp>
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), warp_aggregate:Ref<int>) =
//        let LOGICAL_WARP_THREADS = this.LOGICAL_WARP_THREADS
//        let STEPS = this.Constants.STEPS
//        let lane_id = this.ThreadFields.lane_id
//
//        output := input
//
//        // Iterate scan steps
//        for STEP = 0 to STEPS - 1 do
//            // Grab addend from peer
//            let OFFSET = 1 <<< STEP
//            let temp = __ptx__.ShuffleUp(!output, OFFSET)
//
//            // Perform scan op if from a valid peer
//            if (lane_id >= OFFSET) then output := (temp, !output) ||> scan_op
//
//        // Grab aggregate from last warp lane
//        warp_aggregate := this.Broadcast(!output, LOGICAL_WARP_THREADS - 1)
//
//
//    /// Inclusive scan
//    //template <typename ScanOp>
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int)) =
//        let warp_aggregate = __null() |> __ptr_to_ref
//        this.InclusiveScan(input, output, scan_op, warp_aggregate)
//
//
//    //---------------------------------------------------------------------
//    // Exclusive operations
//    //---------------------------------------------------------------------
//
//    /// Exclusive scan with aggregate
//    // template <typename ScanOp>
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:(int -> int -> int), warp_aggregate:Ref<int>) =
//        let lane_id = this.ThreadFields.lane_id
//        // Compute inclusive scan
//        let inclusive = __null() |> __ptr_to_ref
//        this.InclusiveScan(input, inclusive, scan_op, warp_aggregate);
//
//        // Grab result from predecessor
//        let exclusive = __ptx__.ShuffleUp(!inclusive, 1)
//
//        output := if lane_id = 0 then identity else exclusive
//
//
//    /// Exclusive scan
//    /// template <typename ScanOp>
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:(int -> int -> int)) =
//        let warp_aggregate = __local__.Variable()
//        this.ExclusiveScan(input, output, identity, scan_op, warp_aggregate);
//
//
//    /// Exclusive scan with aggregate, without identity
//    /// template <typename ScanOp>
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), warp_aggregate:Ref<int>) =
//        // Compute inclusive scan
//        let inclusive = __local__.Variable()
//        this.InclusiveScan(input, inclusive, scan_op, warp_aggregate)
//
//        // Grab result from predecessor
//        output := __ptx__.ShuffleUp(!inclusive, 1)
//
//
//    /// Exclusive scan without identity
//    /// template <typename ScanOp>
//    member this.ExclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int)) =
//        let warp_aggregate = __local__.Variable()
//        this.ExclusiveScan(input, output, scan_op, warp_aggregate)
//
//
//    static member Create(logical_warps, logical_warp_threads) : WarpScanShfl = //temp_storage, warp_id, lane_id) =
//        let c = logical_warp_threads |> Constants.Init
//        {
//            LOGICAL_WARPS           = logical_warps
//            LOGICAL_WARP_THREADS    = logical_warp_threads
//            Constants = c
//            ThreadFields = (0,0) |> ThreadFields.Init
//        }

