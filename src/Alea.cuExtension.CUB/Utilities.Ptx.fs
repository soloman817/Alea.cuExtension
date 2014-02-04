[<AutoOpen>]
module Alea.cuExtension.CUB.Utilities.Ptx

open System
open Alea.CUDA
open Alea.CUDA.Utilities


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
            | None, [] ->
                 let irLambdaType = IRTypeBuilder.Instance.Build(ctx, typeof<uint32 -> int -> int-> uint32>)
                 let irFunctionType = IRTypeBuilder.Instance. BuildDeviceFunctionTypeFromLambdaType(ctx, irLambdaType)
                 IRCommonInstructionBuilder.Instance.BuildInlineAsm(ctx, irFunctionType, 
                    "shfl.idx.b32 \t$0, $1, $2, $3;", "=r", []) |> Some
             | _ -> None

let [<ShuffleBroadcast>] shuffleBroadcast (shuffle_word:uint32) (src_lane:int) (logcal_warp_threads_minus1:int) : uint32 = failwith ""

let inline ShuffleBroadcast<'T>() =
    let ShuffleWord = 






[<Sealed>]
type __ptx__ private () =
    
    [<BFI>]     static member BFI(x,y,bit,numBits) = bfi x y bit numBits
    [<SHR_ADD>] static member SHR_ADD(x,shift,addend) = shr_add x shift addend
    [<LaneID>]  static member LaneId() = laneId()


