[<AutoOpen>]
module Alea.cuExtension.CUB.Utilities.Ptx

open System
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Utilities

open UnitWord


//let [<ReflectedDefinition>] SHR_ADD (x:uint32) (shift:int) (addend:uint32) = (x >>> shift) + addend

[<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
type SHR_ADDAttribute() =
    inherit Attribute()

    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, x :: shift :: addend :: [] ->
                 let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<uint32 -> uint32 -> uint32 -> uint32>)
                 let irFunctionType = IRTypeBuilder.Instance. BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
                 IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
                    "vshr.u32.u32.u32.clamp.add \t$0, $1, $2, $3;", "=r,r,r,r,r", x :: shift :: addend :: []) |> Some
             | _ -> None
let [<SHR_ADD>] shr_add (x:uint32) (shift:uint32) (addend:uint32) : uint32 = failwith "" 



[<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
type BFIAttribute() =
    inherit Attribute()

    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, x :: y :: bit :: numBits :: [] ->
                 let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<int -> int -> int -> int -> int>)
                 let irFunctionType = IRTypeBuilder.Instance. BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
                 IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
                    "bfi.b32 \t$0, $2, $1, $3, $4;", "=r,r,r,r,r", x :: y :: bit :: numBits :: []) |> Some
             | _ -> None
let [<BFI>] bfi (x:int) (y:int) (bit:int) (numBits:int) : int = failwith ""




[<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
type LaneIDAttribute() =
    inherit Attribute()

    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, [] ->
                 let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<unit -> int>)
                 let irFunctionType = IRTypeBuilder.Instance. BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
                 IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
                    "mov.u32 \t$0, $laneid;", "=r", []) |> Some
             | _ -> None
let [<LaneID>] laneId () : int = failwith ""


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

let [<ShuffleBroadcast>] shuffleBroadcast (shuffle_word:uint32) (src_lane:int) (logical_warp_threads:int) : uint32 = failwith ""
let inline ShuffleBroadcast (input:int) (src_lane:int) (logical_warp_threads:int) =
    let WORDS = (sizeof<int> + sizeof<ShuffleWord> - 1) / sizeof<ShuffleWord>
    
    let output = __local__.Variable()

    let input_alias : deviceptr<ShuffleWord> = input |> __obj_to_ref |> __ref_reinterpret |> __ref_to_ptr
    let output_alias : deviceptr<ShuffleWord> = output |> __ref_reinterpret |> __ref_to_ptr

    for WORD = 0 to (WORDS - 1) do
        let shuffle_word = (input_alias.[WORD] |> uint32, src_lane, (logical_warp_threads - 1)) |||> shuffleBroadcast
        
        output_alias.[WORD] <- shuffle_word |> __obj_reinterpret

    output |> __ref_to_obj
//
//let inline ShuffleBroadcast() =
//    let ShuffleWord = Type.UnitWord.ShuffleWord

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
let [<ShuffleUp>] shuffleUp (shuffle_word:uint32) (src_offset:int) (shfl_c:int) : uint32 = failwith ""




[<Sealed>]
type __ptx__ private () =
    
    [<BFI>]     static member BFI(x,y,bit,numBits) = bfi x y bit numBits
    [<ShuffleUp>] 
    static member ShuffleUp(input:int, src_offset:int) =
        let SHFL_C = 0
         
        let WORDS = (sizeof<int> + sizeof<ShuffleWord> - 1) / sizeof<ShuffleWord>
        let output = __local__.Variable()
        let output_alias : deviceptr<ShuffleWord> = output |> __ref_to_ptr
        let input_alias : deviceptr<ShuffleWord> = input |> __obj_reinterpret

        for WORD = 0 to WORDS - 1 do
            let mutable shuffle_word : uint32 = input_alias.[WORD] |> __obj_reinterpret
            shuffle_word <- shuffleUp shuffle_word src_offset SHFL_C
            output_alias.[WORD] <- shuffle_word |> __obj_reinterpret

        !output

    [<SHR_ADD>] static member SHR_ADD(x,shift,addend) = shr_add x shift addend
    [<LaneID>]  static member LaneId() = laneId()


