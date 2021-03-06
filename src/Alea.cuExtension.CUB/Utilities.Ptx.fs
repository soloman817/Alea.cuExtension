﻿[<AutoOpen>]
module Alea.cuExtension.CUB.Utilities.Ptx

open System
open Alea.CUDA

open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Utilities

open UnitWord


[<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
type SHR_ADDAttribute() =
    inherit Attribute()

    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, x :: shift :: addend :: [] ->
                 let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<uint32 -> uint32 -> uint32 -> uint32>)
                 let irFunctionType = IRTypeBuilder.Instance.BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
                 IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
                    "vshr.u32.u32.u32.clamp.add \t$0, $1, $2, $3;", "=r,r,r,r", x :: shift :: addend :: []) |> Some
             | _ -> None


[<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
type BFIAttribute() =
    inherit Attribute()

    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, x :: y :: bit :: numBits :: [] ->
                 let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<int -> int -> int -> int -> int>)
                 let irFunctionType = IRTypeBuilder.Instance.BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
                 IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
                    "bfi.b32 \t$0, $2, $1, $3, $4;", "=r,r,r,r,r", x :: y :: bit :: numBits :: []) |> Some
             | _ -> None


[<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
type LaneIDAttribute() =
    inherit Attribute()

    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, [] ->
                 let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<unit -> uint32>)
                 let irFunctionType = IRTypeBuilder.Instance.BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
                 IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
                    "mov.u32 \t$0, %laneid;", "=r", []) |> Some
             | _ -> None



[<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
type ShuffleBroadcastAttribute() =
    inherit Attribute()

    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, shuffle_word :: src_lane :: logical_warp_threads :: [] ->
                let clrType = info.GetGenericArguments().[0]
                let irType = IRTypeBuilder.Instance.Build(ctx, clrType)
                let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<uint32 -> int -> int-> uint32>)
                let irFunctionType = IRTypeBuilder.Instance. BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
                IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
                    "shfl.idx.b32 \t$0, $1, $2, $3;", "=r", shuffle_word :: src_lane :: logical_warp_threads :: []) |> Some
            | _ -> None

let [<ShuffleBroadcast>] private ShuffleBroadcast (shuffle_word:uint32) (src_lane:int) (logical_warp_threads:int) : uint32 = failwith ""


[<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
type ShuffleUpAttribute() =
    inherit Attribute()

    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, shuffle_word :: src_offset :: shfl_c :: [] ->
                 let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<int -> int -> int -> int -> int>)
                 let irFunctionType = IRTypeBuilder.Instance. BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
                 IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
                    "shfl.up.b32 \t$0, $1, $2, $3;", "=r,r,r,r", shuffle_word :: src_offset :: shfl_c :: []) |> Some
             | _ -> None
let [<ShuffleUp>] private ShuffleUp (shuffle_word:uint32) (src_offset:int) (shfl_c:int) : uint32 = failwith ""




[<Sealed>]
type __ptx__ private () =
    
    [<BFI>]     static member BFI(x,y,bit,numBits)                          : int       = failwith "device only"
    [<LaneID>]  static member LaneId()                                      : uint32    = failwith "device only"
    [<SHR_ADD>] static member SHR_ADD(x:uint32,shift:uint32,addend:uint32)  : uint32    = failwith "device only"
    [<ReflectedDefinition>]  static member SHR_ADD(x:int, shift:int, addend:int) = __ptx__.SHR_ADD(uint32 x, uint32 shift, uint32 addend) |> int

//    //[<ShuffleUp>] 
//    static member ShuffleUp(input:'T, src_offset:int) =
//        let SHFL_C = 0
//         
//        let WORDS = (__sizeof<'T>() + __sizeof<ShuffleWord>() - 1) / __sizeof<ShuffleWord>()
//        let output = __local__.Variable()
//        let output_alias : deviceptr<ShuffleWord> = output |> __ref_to_ptr
//        let input_alias : deviceptr<ShuffleWord> = input |> __obj_reinterpret
//
//        for WORD = 0 to WORDS - 1 do
//            let mutable shuffle_word : uint32 = input_alias.[WORD] |> __obj_reinterpret
//            shuffle_word <- (shuffle_word, src_offset, SHFL_C) |||> ShuffleUp
//            output_alias.[WORD] <- shuffle_word |> __obj_reinterpret
//
//        !output
//
//    //[<ShuffleBroadcast>]
//    [<ReflectedDefinition>]
//    static member ShuffleBroadcast logical_warp_threads (input:'T) (src_lane:int) =
//        let WORDS = (__sizeof<int>() + __sizeof<ShuffleWord>() - 1) / __sizeof<ShuffleWord>()
//    
//        let output : 'T ref = __local__.Variable()
//
//        let input_alias : deviceptr<ShuffleWord> = input |> __obj_to_ref |> __ref_reinterpret |> __ref_to_ptr
//        let output_alias : deviceptr<ShuffleWord> = output |> __ref_reinterpret |> __ref_to_ptr
//
//        for WORD = 0 to (WORDS - 1) do
//            let shuffle_word = (input_alias.[WORD] |> __obj_reinterpret, src_lane, (logical_warp_threads - 1)) |||> ShuffleBroadcast
//        
//            output_alias.[WORD] <- shuffle_word |> __obj_reinterpret
//
//        output |> __ref_to_obj
//        
//

    //[<SRegField("laneid")>] static member LaneId = laneId


