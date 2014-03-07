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


let [<ReflectedDefinition>] inline Broadcast<'T> logical_warp_threads (input:'T) (src_lane:int) =
    (logical_warp_threads |> __ptx__.ShuffleBroadcast)
    <|| (input, src_lane)
    

module Template =
    [<AutoOpen>]
    module Params =
        [<Record>]
        type API =
            {
                LOGICAL_WARPS           : int
                LOGICAL_WARP_THREADS    : int                
            }

            [<ReflectedDefinition>]
            member this.Get() = (this.LOGICAL_WARPS, this.LOGICAL_WARP_THREADS)

            [<ReflectedDefinition>]
            static member Init(logical_warps, logical_warp_threads) =
                {
                    LOGICAL_WARPS           = logical_warps
                    LOGICAL_WARP_THREADS    = logical_warp_threads
                }

    [<AutoOpen>]
    module Constants =
        [<Record>]
        type API =
            {
                STEPS   : int
                SHFL_C  : int
            }

            [<ReflectedDefinition>]
            static member Init(logical_warp_threads:int) =
                let steps = logical_warp_threads |> log2
                {
                    STEPS   = steps
                    SHFL_C  = ((-1 <<< steps) &&& 31) <<< 8 
                }

            [<ReflectedDefinition>]
            static member Init(tp:Params.API) = API.Init(tp.LOGICAL_WARP_THREADS)


    [<AutoOpen>]
    module TempStorage =
        [<Record>]
        type API<'T> =
            {
                mutable Ptr                 : deviceptr<'T>
            }

            member this.Item
                with    [<ReflectedDefinition>] get (idx:int) = this.Ptr.[idx]
                and     [<ReflectedDefinition>] set (idx:int) (v:'T) = this.Ptr.[idx] <- v

            [<ReflectedDefinition>]
            static member inline Uninitialized<'T>() = { Ptr = __null<'T>() }
        
    
    [<AutoOpen>]
    module ThreadFields =
        [<Record>]
        type API<'T> =
            {
                mutable temp_storage    : TempStorage.API<'T>
                mutable warp_id         : int
                mutable lane_id         : int
            }

            [<ReflectedDefinition>]
            member this.Get() = (this.temp_storage, this.warp_id, this.lane_id)

            [<ReflectedDefinition>] 
            static member inline Init(temp_storage, warp_id, lane_id) =
                {
                    temp_storage    = temp_storage
                    warp_id         = warp_id
                    lane_id         = lane_id
                }

            [<ReflectedDefinition>]
            static member inline Init<'T>(warp_id, lane_id) =
                API<'T>.Init(TempStorage.API<'T>.Uninitialized(), warp_id, lane_id)

            [<ReflectedDefinition>]
            static member inline Uninitialized<'T>() =
                API<'T>.Init(TempStorage.API<'T>.Uninitialized(), 0, 0)


    type _TemplateParams    = Params.API
    type _TempStorage<'T>   = TempStorage.API<'T>
    type _ThreadFields<'T>  = ThreadFields.API<'T>


    [<Record>]
    type API<'T> =
        {
            mutable Params          : Params.API
            mutable Constants       : Constants.API
            mutable ThreadFields    : ThreadFields.API<'T>
        }

        [<ReflectedDefinition>]
        member this.Get() = (this.Params, this.Constants, this.ThreadFields)

        [<ReflectedDefinition>]
        static member Init(logical_warps, logical_warp_threads) =
            let tp = Params.API.Init(logical_warps, logical_warp_threads)
            {
                Params          = tp
                Constants       = Constants.API.Init(tp)
                ThreadFields    = ThreadFields.API<'T>.Uninitialized()
            }


type _Template<'T> = Template.API<'T>
               

module InclusiveScan =
    open Template


    [<ReflectedDefinition>]
    let private WithAggregate (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
        let STEPS = template.Constants.STEPS
        let LOGICAL_WARP_THREADS = template.Params.LOGICAL_WARP_THREADS
        let broadcast = LOGICAL_WARP_THREADS |> Broadcast
        output := input

        for STEP = 0 to (STEPS - 1) do
            let OFFSET = 1 <<< STEP
            let temp = (!output, OFFSET) |> __ptx__.ShuffleUp
            ()
//                if lane_id >= OFFSET then output := (temp |> __obj_reinterpret, !output) ||> %scan_op

        warp_aggregate := (!output, LOGICAL_WARP_THREADS - 1) ||> broadcast
        

    [<ReflectedDefinition>]
    let inline Default (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) =
        let InclusiveScan = WithAggregate template scan_op
        let warp_aggregate = __local__.Variable()
        InclusiveScan input output warp_aggregate
    

    [<Record>]
    type API<'T> =
        {
            mutable template : _Template<'T>
        }

        [<ReflectedDefinition>] member this.WithAggregate = WithAggregate this.template
        [<ReflectedDefinition>] member this.Default = Default this.template
        
        [<ReflectedDefinition>]
        static member Init(template:_Template<'T>) = {template = template}


module InclusiveSum =
    open Template


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

    
    let [<ReflectedDefinition>] inline SingleShfl (template:_Template<'T>)
        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) = 
        
        let STEPS = template.Constants.STEPS
        let SHFL_C = template.Constants.SHFL_C
        let LOGICAL_WARP_THREADS = template.Params.LOGICAL_WARP_THREADS
        let broadcast = LOGICAL_WARP_THREADS |> Broadcast
        
        let temp : Ref<uint32> = input |> __obj_to_ref |> __ref_reinterpret
        for STEP = 0 to (STEPS - 1) do
            temp := (!temp |> uint32, (1 <<< STEP), SHFL_C) |||> inclusiveSumPtx
        output := !temp |> __obj_reinterpret
        warp_aggregate := (!output, LOGICAL_WARP_THREADS - 1) ||> broadcast
        

    let [<ReflectedDefinition>] inline MultiShfl (template:_Template<'T>)
        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
        ()
        

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
    
    let [<ReflectedDefinition>] private Float32Specialized (template:_Template<'T>)
        (input:float32) (output:Ref<float32>) (warp_aggregate:Ref<float32>) =
        
        let STEPS = template.Constants.STEPS
        let SHFL_C = template.Constants.SHFL_C
        let LOGICAL_WARP_THREADS = template.Params.LOGICAL_WARP_THREADS
        let broadcast = LOGICAL_WARP_THREADS |> Broadcast<float32>
        
        output := input
        for STEP = 0 to (STEPS - 1) do
            output := (!output, (1 <<< STEP), SHFL_C) |||> inclusiveSumPtx_Float32

        warp_aggregate := (!output, LOGICAL_WARP_THREADS - 1) ||> broadcast
        

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
    
    let [<ReflectedDefinition>] inline ULongLongSpecialized (template:_Template<'T>)
        (input:ulonglong) (output:Ref<ulonglong>) (warp_aggregate:Ref<ulonglong>) =
        
        let STEPS = template.Constants.STEPS
        let SHFL_C = template.Constants.SHFL_C
        let LOGICAL_WARP_THREADS = template.Params.LOGICAL_WARP_THREADS
        let broadcast = LOGICAL_WARP_THREADS |> Broadcast
    
        output := input
        for STEP = 0 to (STEPS - 1) do
            output := (!output, (1 <<< STEP), SHFL_C) |||> inclusiveSumPtx_ULongLong
            
        warp_aggregate := (!output, LOGICAL_WARP_THREADS - 1) ||> broadcast

    let [<ReflectedDefinition>] inline Generic (template:_Template<'T>)
        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
        let inclusiveSum = if sizeof<'T> <= sizeof<uint32> then SingleShfl template else MultiShfl template
        
        (input, output, warp_aggregate) |||> inclusiveSum
        

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (input:'T) (output:Ref<'T>) =
        let inclusiveSum = Generic template
        
        let warp_aggregate = __local__.Variable()
        (input, output, warp_aggregate) |||> inclusiveSum
    
    [<Record>]
    type API<'T> =
        {
            template : _Template<'T>
        }

        [<ReflectedDefinition>] member this.Default                 = Default this.template
        [<ReflectedDefinition>] member this.Generic                 = Generic this.template
        [<ReflectedDefinition>] member this.ULongLongSpecialized    = ULongLongSpecialized this.template
        [<ReflectedDefinition>] member this.Float32Specialized      = Float32Specialized this.template
        [<ReflectedDefinition>] member this.MultiShfl               = MultiShfl this.template
        [<ReflectedDefinition>] member this.ShingleShfl             = SingleShfl this.template

        [<ReflectedDefinition>]
        static member Init(template:_Template<'T>) = { template = template }



module ExclusiveScan =
    open Template

    let [<ReflectedDefinition>] inline WithAggregate (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (identity:'T) (warp_aggregate:Ref<'T>) =
    
        let inclusive = __local__.Variable()
        InclusiveScan.API<'T>.Init(template).WithAggregate scan_op input inclusive warp_aggregate

        let exclusive = (!inclusive, 1) |> __ptx__.ShuffleUp

//            output := if lane_id = 0 then identity else exclusive |> __obj_reinterpret
        ()

    
    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (scan_op:'T -> 'T -> 'T)
        (input:'T) (output:Ref<'T>) (identity:'T) =
    
        let warp_aggregate = __local__.Variable()
        WithAggregate template scan_op input output identity warp_aggregate
    


    module private Identityless =
        let [<ReflectedDefinition>] inline WithAggregate (template:_Template<'T>)
            (scan_op:'T -> 'T -> 'T)
            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
            
            let inclusive = __local__.Variable()
            InclusiveScan.API<'T>.Init(template).WithAggregate scan_op input inclusive warp_aggregate

            output := (!inclusive, 1) |> __ptx__.ShuffleUp |> __obj_reinterpret
        

        let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
            (scan_op:'T -> 'T -> 'T)
            (input:'T) (output:Ref<'T>) =
                    
            let warp_aggregate = __local__.Variable()
            WithAggregate template scan_op input output warp_aggregate
        
    [<Record>]
    type API<'T> =
        {
            template : _Template<'T>
        }

        [<ReflectedDefinition>] member this.Default              = Default this.template
        [<ReflectedDefinition>] member this.Default_NoID         = Identityless.Default this.template
        [<ReflectedDefinition>] member this.WithAggregate        = WithAggregate this.template
        [<ReflectedDefinition>] member this.WithAggregate_NoID   = Identityless.WithAggregate this.template

        [<ReflectedDefinition>]
        static member Init(template:_Template<'T>) = { template = template }


module WarpScanShfl =
    open Template


    [<Record>]
    type API<'T> =
        {
            template : _Template<'T>
        }

        [<ReflectedDefinition>] member this.InclusiveScan   = InclusiveScan.API<'T>.Init(this.template)
        [<ReflectedDefinition>] member this.InclusiveSum    = InclusiveSum.API<'T>.Init(this.template)
        [<ReflectedDefinition>] member this.ExclusiveScan   = ExclusiveScan.API<'T>.Init(this.template)
        [<ReflectedDefinition>] member this.Broadcast       = Broadcast<'T> this.template.Params.LOGICAL_WARP_THREADS

        [<ReflectedDefinition>]
        static member Init(template:_Template<'T>) = { template = template }



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
//    fun (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) (single_shfl:bool option) ->
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

