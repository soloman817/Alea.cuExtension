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

   

module Template =
    module Host =
        module Params =
            type API =
                { LOGICAL_WARPS : int; LOGICAL_WARP_THREADS : int }

                static member Init(logical_warps, logical_warp_threads) =
                    { LOGICAL_WARPS = logical_warps; LOGICAL_WARP_THREADS = logical_warp_threads }
        
        module Constants =
            type API =
                { STEPS : int; SHFL_C : int }

                static member Init(p:Params.API) =
                    let steps = p.LOGICAL_WARP_THREADS |> log2
                    { STEPS = steps; SHFL_C  = ((-1 <<< steps) &&& 31) <<< 8 }

        type API =
            { Params : Params.API; Constants : Constants.API; SharedMemoryLength : int }
            
            static member Init(logical_warps, logical_warp_threads) =
                let p = Params.API.Init(logical_warps, logical_warp_threads)
                let c = Constants.API.Init(p)
                { Params = p; Constants = c; SharedMemoryLength = 0 }

    module Device =
        module TempStorage =
            type [<Record>] API<'T> = deviceptr<'T>

        [<Record>]
        type API<'T> =
            {
                mutable temp_storage    : TempStorage.API<'T>
                mutable warp_id         : int
                mutable lane_id         : int
            }

            [<ReflectedDefinition>]
            static member Init(temp_storage, warp_id, lane_id) = 
                { temp_storage = __null(); warp_id = warp_id; lane_id = lane_id }



    let [<ReflectedDefinition>] inline Broadcast<'T> logical_warp_threads (input:'T) (src_lane:int) =
        (logical_warp_threads |> __ptx__.ShuffleBroadcast)
            (input)
            (src_lane)


    type _TemplateParams    = Host.Params.API
    type _Constants         = Host.Constants.API
    type _HostApi           = Host.API

    type _TempStorage<'T>   = Device.TempStorage.API<'T>
    type _DeviceApi<'T>     = Device.API<'T>

           

module InclusiveScan =
    open Template

    let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
        (d:_DeviceApi<'T>) 
        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
        let p = h.Params
        let c = h.Constants

        let STEPS = c.STEPS
        let LOGICAL_WARP_THREADS = p.LOGICAL_WARP_THREADS
        let broadcast = LOGICAL_WARP_THREADS |> Broadcast
        output := input

        for STEP = 0 to (STEPS - 1) do
            let OFFSET = 1 <<< STEP
            let temp = (!output, OFFSET) |> __ptx__.ShuffleUp
                
            if d.lane_id >= OFFSET then output := (temp |> __obj_reinterpret, !output) ||> scan_op

        warp_aggregate := (!output, LOGICAL_WARP_THREADS - 1) ||> broadcast
        
        

    
    let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
        (d:_DeviceApi<'T>) 
        (input:'T) (output:Ref<'T>) =
        let warp_aggregate = __local__.Variable()
        WithAggregate h scan_op d input output warp_aggregate
    


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

    let [<InclusiveSumPtx>] inline inclusiveSumPtx (temp:uint32) (shlStep:int) (shfl_c:int) : uint32 = failwith ""

    
    let [<ReflectedDefinition>] inline SingleShfl (h:_HostApi)
        (d:_DeviceApi<'T>)
        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
        let p = h.Params
        let c = h.Constants
            
        let STEPS = c.STEPS
        let SHFL_C = c.SHFL_C
        let LOGICAL_WARP_THREADS = p.LOGICAL_WARP_THREADS
        let broadcast = LOGICAL_WARP_THREADS |> Broadcast
        
        let temp : Ref<uint32> = input |> __obj_to_ref |> __ref_reinterpret
        for STEP = 0 to (STEPS - 1) do
            temp := (!temp |> uint32, (1 <<< STEP), SHFL_C) |||> inclusiveSumPtx
        output := !temp |> __obj_reinterpret
        warp_aggregate := (!output, LOGICAL_WARP_THREADS - 1) ||> broadcast
        

    let [<ReflectedDefinition>] inline MultiShfl (h:_HostApi)
        (d:_DeviceApi<'T>)
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

    let [<InclusiveSumPtx_Float32>] inclusiveSumPtx_Float32 (output:float32) (shlStep:int) (shfl_c:int) : float32 = failwith ""
    
    let [<ReflectedDefinition>] inline Float32Specialized (h:_HostApi)
        (d:_DeviceApi<float32>)
        (input:float32) (output:Ref<float32>) (warp_aggregate:Ref<float32>) =
        let p = h.Params
        let c = h.Constants

        let STEPS = c.STEPS
        let SHFL_C = c.SHFL_C
        let LOGICAL_WARP_THREADS = p.LOGICAL_WARP_THREADS
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

    let [<InclusiveSumPtx_Float32>] inclusiveSumPtx_ULongLong (output:ulonglong) (shlStep:int) (shfl_c:int) : ulonglong = failwith ""
    
    let [<ReflectedDefinition>] inline ULongLongSpecialized (h:_HostApi)
        (d:_DeviceApi<ulonglong>)
        (input:ulonglong) (output:Ref<ulonglong>) (warp_aggregate:Ref<ulonglong>) =
        let p = h.Params
        let c = h.Constants

        let STEPS = c.STEPS
        let SHFL_C = c.SHFL_C
        let LOGICAL_WARP_THREADS = p.LOGICAL_WARP_THREADS
        let broadcast = LOGICAL_WARP_THREADS |> Broadcast
    
        output := input
        for STEP = 0 to (STEPS - 1) do
            output := (!output, (1 <<< STEP), SHFL_C) |||> inclusiveSumPtx_ULongLong
            
        warp_aggregate := (!output, LOGICAL_WARP_THREADS - 1) ||> broadcast
        

    let [<ReflectedDefinition>] inline Generic (h:_HostApi)
        (d:_DeviceApi<'T>)
        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
        let InclusiveSum = if sizeof<'T> <= sizeof<uint32> then SingleShfl else MultiShfl
        
        InclusiveSum h d input output warp_aggregate
        

    let [<ReflectedDefinition>] inline Default (h:_HostApi)
        (d:_DeviceApi<'T>)
        (input:'T) (output:Ref<'T>) =
        let warp_aggregate = __local__.Variable()
        Generic h d input output warp_aggregate
 


module ExclusiveScan =
    open Template

    let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
        (d:_DeviceApi<'T>) 
        (input:'T) (output:Ref<'T>) (identity:'T) (warp_aggregate:Ref<'T>) =
        
        let inclusive = __local__.Variable()
        InclusiveScan.WithAggregate h scan_op d input inclusive warp_aggregate

        let exclusive = (!inclusive, 1) |> __ptx__.ShuffleUp

        output := if d.lane_id = 0 then identity else exclusive |> __obj_reinterpret
        

    
    let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
        (d:_DeviceApi<'T>)
        (input:'T) (output:Ref<'T>) (identity:'T) =
        let warp_aggregate = __local__.Variable()
        WithAggregate h scan_op d input output identity warp_aggregate
            


    module Identityless =
        let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
            (d:_DeviceApi<'T>)
            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
            let inclusive = __local__.Variable()
            InclusiveScan.WithAggregate h scan_op d input inclusive warp_aggregate
            output := (!inclusive, 1) |> __ptx__.ShuffleUp |> __obj_reinterpret
            

        let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
            (d:_DeviceApi<'T>)
            (input:'T) (output:Ref<'T>) =
            let warp_aggregate = __local__.Variable()
            WithAggregate h scan_op d input output warp_aggregate
            


module WarpScanShfl =
    open Template

    type TemplateParams     = Template._TemplateParams
    type Constants          = Template._Constants
    type TempStorage<'T>    = Template._TempStorage<'T>
    
    type HostApi            = Template._HostApi
    type DeviceApi<'T>      = Template._DeviceApi<'T>

    
    [<Record>]
    type API<'T> =
        {
            mutable DeviceApi      : DeviceApi<'T>
        }
        
        [<ReflectedDefinition>] static member Create(temp_storage, warp_id, lane_id) = { DeviceApi = DeviceApi<'T>.Init(temp_storage, warp_id, lane_id)}
        
        [<ReflectedDefinition>] member this.InclusiveSum(h, d:DeviceApi<float32>, input:float32, output:Ref<float32>, warp_aggregate:Ref<float32>) 
            = InclusiveSum.Float32Specialized h d input output warp_aggregate

        [<ReflectedDefinition>] member this.InclusiveSum(h, d:DeviceApi<ulonglong>, input:ulonglong, output:Ref<ulonglong>, warp_aggregate:Ref<ulonglong>) 
            = InclusiveSum.ULongLongSpecialized h d input output warp_aggregate

        [<ReflectedDefinition>] member this.InclusiveSum(h, input, output, warp_aggregate) 
            = InclusiveSum.Generic h this.DeviceApi input output warp_aggregate

        [<ReflectedDefinition>] member this.InclusiveSum(h, input, output) 
            = InclusiveSum.Default h this.DeviceApi input output

        [<ReflectedDefinition>] member this.InclusiveScan(h, scan_op, input, output) 
            = InclusiveScan.Default h scan_op this.DeviceApi input output
        
        [<ReflectedDefinition>] member this.InclusiveScan(h, scan_op, input, output, warp_aggregate) 
            = InclusiveScan.WithAggregate h scan_op this.DeviceApi input output warp_aggregate  

        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity) 
            = ExclusiveScan.Default h scan_op this.DeviceApi input output identity
        
        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, identity, warp_aggregate) 
            = ExclusiveScan.WithAggregate h scan_op this.DeviceApi input output identity warp_aggregate
        
        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output) 
            = ExclusiveScan.Identityless.Default h scan_op this.DeviceApi input output
        
        [<ReflectedDefinition>] member this.ExclusiveScan(h, scan_op, input, output, warp_aggregate) 
            = ExclusiveScan.Identityless.WithAggregate h scan_op this.DeviceApi input output warp_aggregate






//    module InclusiveSum =
//        type _FunctionApi<'T> =
//            {
//                SingleShfl              : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> Ref<'T> -> unit>
//                MultiShfl               : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> Ref<'T> -> unit>
//                Float32Specialized      : Function<_DeviceApi<float32> -> float32 -> Ref<float32> -> Ref<float32> -> unit>
//                ULongLongSpecialized    : Function<_DeviceApi<ulonglong> -> ulonglong -> Ref<ulonglong> -> Ref<ulonglong> -> unit>
//                Generic                 : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> Ref<'T> -> unit>
//                Default                 : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> unit>      
//            }
//
//    module InclusiveScan =
//        type _FunctionApi<'T> =
//            {
//                Default         : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> unit>
//                WithAggregate   : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> Ref<'T> -> unit>                
//            }
//            
//    module ExclusiveScan =
//        type _FunctionApi<'T> =
//            {
//                Default             : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> 'T -> unit>
//                Default_NoID        : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> unit>
//                WithAggregate       : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> 'T -> Ref<'T> -> unit>
//                WithAggregate_NoID  : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> Ref<'T> -> unit>
//            }    
//    type FunctionApi<'T> =
//        {
//            SingleShfl              : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> Ref<'T> -> unit>
//            MultiShfl               : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> Ref<'T> -> unit>
//            Float32Specialized      : Function<_DeviceApi<float32> -> float32 -> Ref<float32> -> Ref<float32> -> unit>
//            ULongLongSpecialized    : Function<_DeviceApi<ulonglong> -> ulonglong -> Ref<ulonglong> -> Ref<ulonglong> -> unit>
//            Generic                 : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> Ref<'T> -> unit>
//            Default                 : Function<_DeviceApi<'T> -> 'T -> Ref<'T> -> unit>      
//        }
//
//    let [<ReflectedDefinition>] inline api<'T> (h:_HostApi) : Template<FunctionApi<'T>> = cuda {
//        let! singleshfl     = h |> SingleShfl           |> Compiler.DefineFunction
//        let! multishfl      = h |> MultiShfl            |> Compiler.DefineFunction
//        let! float32spec    = h |> Float32Specialized   |> Compiler.DefineFunction
//        let! ulonglongspec  = h |> ULongLongSpecialized |> Compiler.DefineFunction
//        let! generic        = h |> Generic              |> Compiler.DefineFunction
//        let! dfault         = h |> Default              |> Compiler.DefineFunction
//
//        return
//            {
//                SingleShfl              = singleshfl
//                MultiShfl               = multishfl
//                Float32Specialized      = float32spec
//                ULongLongSpecialized    = ulonglongspec
//                Generic                 = generic
//                Default                 = dfault
//            }
//        }

//module InclusiveScan =
//    open Template
//
//    let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T) 
//        (d:_DeviceApi<'T>) (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//        let p = h.Params
//        let c = h.Constants
//        
//
//        let STEPS = c.STEPS
//        let LOGICAL_WARP_THREADS = p.LOGICAL_WARP_THREADS
//        let broadcast = LOGICAL_WARP_THREADS |> Broadcast
//        output := input
//
//        for STEP = 0 to (STEPS - 1) do
//            let OFFSET = 1 <<< STEP
//            let temp = (!output, OFFSET) |> __ptx__.ShuffleUp
//                
//            if d.lane_id >= OFFSET then output := (temp |> __obj_reinterpret, !output) ||> scan_op
//
//        warp_aggregate := (!output, LOGICAL_WARP_THREADS - 1) ||> broadcast
//        
//
//    
//    let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T) 
//        (d:_DeviceApi<'T>) (input:'T) (output:Ref<'T>) =
//        let warp_aggregate = __local__.Variable()
//        WithAggregate h scan_op d input output warp_aggregate
//
//    
//    let [<ReflectedDefinition>] inline api (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//        (
//            (h, scan_op) ||> Default,
//            (h, scan_op) ||> WithAggregate
//        )
//
//
//module InclusiveSum =
//    open Template
//
//    [<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
//    type private InclusiveSumPtxAttribute() =
//        inherit Attribute()
//
//        interface ICustomCallBuilder with
//            member this.Build(ctx, irObject, info, irParams) =
//                match irObject, irParams with
//                | None, temp :: shlStep :: shfl_c :: [] ->
//                    let clrType = info.GetGenericArguments().[0]
//                    let irType = IRTypeBuilder.Instance.Build(ctx, clrType)
//                    let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<uint32 -> int -> int-> uint32>)
//                    let irFunctionType = IRTypeBuilder.Instance. BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
//                    IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
//                        "{
//                            .reg .u32 r0;
//                            .reg .pred p;
//                            shfl.up.b32 r0|p, $1, $2, $3;
//                            @p add.u32 r0, r0, %4;
//                            mov.u32 %0, r0;
//                        }", "=r,r,r,r,r", temp :: shlStep :: shfl_c :: []) |> Some
//                | _ -> None
//
//    let [<InclusiveSumPtx>] inline inclusiveSumPtx (temp:uint32) (shlStep:int) (shfl_c:int) : uint32 = failwith ""
//
//    
//    let [<ReflectedDefinition>] inline SingleShfl (h:_HostApi)
//        (d:_DeviceApi<'T>) (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//        let p = h.Params
//        let c = h.Constants
//            
//        let STEPS = c.STEPS
//        let SHFL_C = c.SHFL_C
//        let LOGICAL_WARP_THREADS = p.LOGICAL_WARP_THREADS
//        let broadcast = LOGICAL_WARP_THREADS |> Broadcast
//        
//        let temp : Ref<uint32> = input |> __obj_to_ref |> __ref_reinterpret
//        for STEP = 0 to (STEPS - 1) do
//            temp := (!temp |> uint32, (1 <<< STEP), SHFL_C) |||> inclusiveSumPtx
//        output := !temp |> __obj_reinterpret
//        warp_aggregate := (!output, LOGICAL_WARP_THREADS - 1) ||> broadcast
//        
//
//    let [<ReflectedDefinition>] inline MultiShfl (h:_HostApi) 
//        (d:_DeviceApi<'T>)
//        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//        ()
//        
//        
//
//    [<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
//    type private InclusiveSumPtx_Float32Attribute() =
//        inherit Attribute()
//
//        interface ICustomCallBuilder with
//            member this.Build(ctx, irObject, info, irParams) =
//                match irObject, irParams with
//                | None, temp :: shlStep :: shfl_c :: [] ->
//                    let clrType = info.GetGenericArguments().[0]
//                    let irType = IRTypeBuilder.Instance.Build(ctx, clrType)
//                    let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<float32 -> int -> int-> float32>)
//                    let irFunctionType = IRTypeBuilder.Instance. BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
//                    IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
//                        "{
//                            .reg .f32 r0;
//                            .reg .pred p;
//                            shfl.up.b32 r0|p, $1, $2, $3;
//                            @p add.f32 r0, r0, $4;
//                            mov.f32 $0, r0;
//                        }", "=f,f,r,r,f", temp :: shlStep :: shfl_c :: []) |> Some
//                | _ -> None
//
//    let [<InclusiveSumPtx_Float32>] private inclusiveSumPtx_Float32 (output:float32) (shlStep:int) (shfl_c:int) : float32 = failwith ""
//    
//    let [<ReflectedDefinition>] inline Float32Specialized (h:_HostApi)
//        (d:_DeviceApi<float32>) (input:float32) (output:Ref<float32>) (warp_aggregate:Ref<float32>) =
//        let p = h.Params
//        let c = h.Constants
//
//        let STEPS = c.STEPS
//        let SHFL_C = c.SHFL_C
//        let LOGICAL_WARP_THREADS = p.LOGICAL_WARP_THREADS
//        let broadcast = LOGICAL_WARP_THREADS |> Broadcast<float32>
//        
//        output := input
//        for STEP = 0 to (STEPS - 1) do
//            output := (!output, (1 <<< STEP), SHFL_C) |||> inclusiveSumPtx_Float32
//
//        warp_aggregate := (!output, LOGICAL_WARP_THREADS - 1) ||> broadcast
//        
//
//    [<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
//    type private InclusiveSumPtx_ULongLongAttribute() =
//        inherit Attribute()
//
//        interface ICustomCallBuilder with
//            member this.Build(ctx, irObject, info, irParams) =
//                match irObject, irParams with
//                | None, temp :: shlStep :: shfl_c :: [] ->
//                    let clrType = info.GetGenericArguments().[0]
//                    let irType = IRTypeBuilder.Instance.Build(ctx, clrType)
//                    let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<ulonglong -> int -> int-> ulonglong>)
//                    let irFunctionType = IRTypeBuilder.Instance. BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
//                    IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
//                        "{
//                            .reg .u32 r0;
//                            .reg .u32 r1;
//                            .reg .u32 lo;
//                            .reg .u32 hi;
//                            .reg .pred p;
//                            mov.b64 {lo, hi}, $1;
//                            shfl.up.b32 r0|p, lo, $2, $3;
//                            shfl.up.b32 r1|p, hi, $2, $3;
//                            @p add.cc.u32 r0, r0, lo;
//                            @p addc.u32 r1, r1, hi;
//                            mov.b64 $0, {r0, r1};
//                        }", "=l,l,r,r,l", temp :: shlStep :: shfl_c :: []) |> Some
//                | _ -> None
//
//    let [<InclusiveSumPtx_Float32>] inclusiveSumPtx_ULongLong (output:ulonglong) (shlStep:int) (shfl_c:int) : ulonglong = failwith ""
//    
//    let [<ReflectedDefinition>] inline ULongLongSpecialized (h:_HostApi)
//        (d:_DeviceApi<'T>) (input:ulonglong) (output:Ref<ulonglong>) (warp_aggregate:Ref<ulonglong>) =
//        let p = h.Params
//        let c = h.Constants
//
//        let STEPS = c.STEPS
//        let SHFL_C = c.SHFL_C
//        let LOGICAL_WARP_THREADS = p.LOGICAL_WARP_THREADS
//        let broadcast = LOGICAL_WARP_THREADS |> Broadcast
//    
//        output := input
//        for STEP = 0 to (STEPS - 1) do
//            output := (!output, (1 <<< STEP), SHFL_C) |||> inclusiveSumPtx_ULongLong
//            
//        warp_aggregate := (!output, LOGICAL_WARP_THREADS - 1) ||> broadcast
//    
//
//    let [<ReflectedDefinition>] inline Generic (h:_HostApi)
//        (d:_DeviceApi<'T>) (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//        let InclusiveSum = if sizeof<'T> <= sizeof<uint32> then SingleShfl else MultiShfl
//        InclusiveSum h d input output warp_aggregate
//        
//
//    let [<ReflectedDefinition>] inline Default (h:_HostApi)
//        (d:_DeviceApi<'T>) (input:'T) (output:Ref<'T>) =
//        let warp_aggregate = __local__.Variable()
//        Generic h d input output warp_aggregate
//    
//
//module ExclusiveScan =
//    open Template
//
//    let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//        (d:_DeviceApi<'T>) (input:'T) (output:Ref<'T>) (identity:'T) (warp_aggregate:Ref<'T>) =
//        
//
//        let inclusive = __local__.Variable()
//        InclusiveScan.WithAggregate h scan_op d input inclusive warp_aggregate
//
//        let exclusive = (!inclusive, 1) |> __ptx__.ShuffleUp
//
//        output := if d.lane_id = 0 then identity else exclusive |> __obj_reinterpret
//
//
//    
//    let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//        (d:_DeviceApi<'T>) (input:'T) (output:Ref<'T>) (identity:'T) =
//        let warp_aggregate = __local__.Variable()
//        WithAggregate h scan_op d input output identity warp_aggregate
//    
//
//
//    module Identityless =
//        let [<ReflectedDefinition>] inline WithAggregate (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:_DeviceApi<'T>) (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
//            let inclusive = __local__.Variable()
//            InclusiveScan.WithAggregate h scan_op d input inclusive warp_aggregate
//            output := (!inclusive, 1) |> __ptx__.ShuffleUp |> __obj_reinterpret
//        
//
//        let [<ReflectedDefinition>] inline Default (h:_HostApi) (scan_op:'T -> 'T -> 'T)
//            (d:_DeviceApi<'T>) (input:'T) (output:Ref<'T>) =
//            let warp_aggregate = __local__.Variable()
//            WithAggregate h scan_op d input output warp_aggregate
//
//               
//
//module WarpScanShfl =
//    open Template
//
//    type TemplateParams     = Template._TemplateParams
//    type Constants          = Template._Constants
//    type TempStorage<'T>    = Template._TempStorage<'T>
//    type ThreadFields<'T>   = Template._ThreadFields<'T>
//    
//    type HostApi            = Template._HostApi
//    type DeviceApi<'T>      = Template._DeviceApi<'T>
//    type FunctionApi<'T>    = Template._FunctionApi<'T>
//
//    
//    [<Record>]
//    type API<'T> =
//        {
//            host                : HostApi
//            mutable device      : DeviceApi<'T>
//        }
//        
//        [<ReflectedDefinition>]
//        member this.Init(temp_storage:TempStorage<'T>, warp_id:int, lane_id:int) =
//            let mutable f = this.device.ThreadFields
//            d.warp_id <- warp_id
//            d.lane_id <- lane_id
//
//        [<ReflectedDefinition>] 
//        member this.InclusiveSum(input:'T, output:Ref<'T>, warp_aggregate:Ref<'T>, single_shfl:bool) = 
//            if single_shfl then InclusiveSum.SingleShfl this.host this.device input output warp_aggregate
//            else InclusiveScan.WithAggregate this.host (+) this.device input output warp_aggregate
//
//        [<ReflectedDefinition>] 
//        static member Create(h:HostApi) = { host = h; device = DeviceApi<'T>.Init() }
    







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
//    fun (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) (single_shfl:bool option) =
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

