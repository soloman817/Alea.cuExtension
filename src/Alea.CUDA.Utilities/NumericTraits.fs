[<AutoOpen>]
module Alea.CUDA.Utilities.NumericTraits

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA

type RealTraitsAttribute() =
    inherit Attribute()

    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, info.Name, irParams with
            | Some(_), "Of", irValue :: [] ->
                let clrParamType = info.GetParameters().[0].ParameterType
                let irParamType = IRTypeBuilder.Instance.Build(ctx, clrParamType)
                let clrRealType = info.ReturnType
                let irRealType = IRTypeBuilder.Instance.Build(ctx, clrRealType)
                irRealType |> function
                | irRealType when irRealType.IsFloatingPoint -> irRealType.FloatingPoint.Kind |> function
                    | FloatingPointKind.Double -> irParamType |> function
                        | irParamType when irParamType.IsInteger -> (irParamType.Integer.Bits, irParamType.Integer.Signed) |> function
                            |  8,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : int8) @@>) |> Some
                            |  8, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : uint8) @@>) |> Some
                            | 16,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : int16) @@>) |> Some
                            | 16, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : uint16) @@>) |> Some
                            | 32,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : int) @@>) |> Some
                            | 32, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : uint32) @@>) |> Some
                            | 64,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : int64) @@>) |> Some
                            | 64, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : uint64) @@>) |> Some
                            | _ -> None
                        | irParamType when irParamType.IsFloatingPoint -> irParamType.FloatingPoint.Kind |> function
                            | FloatingPointKind.Double -> irValue |> Some
                            | FloatingPointKind.Single -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : float32) @@>) |> Some
                        | _ -> None
                    | FloatingPointKind.Single -> irParamType |> function
                        | irParamType when irParamType.IsInteger -> (irParamType.Integer.Bits, irParamType.Integer.Signed) |> function
                            |  8,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : int8) @@>) |> Some
                            |  8, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : uint8) @@>) |> Some
                            | 16,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : int16) @@>) |> Some
                            | 16, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : uint16) @@>) |> Some
                            | 32,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : int) @@>) |> Some
                            | 32, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : uint32) @@>) |> Some
                            | 64,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : int64) @@>) |> Some
                            | 64, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : uint64) @@>) |> Some
                            | _ -> None
                        | irParamType when irParamType.IsFloatingPoint -> irParamType.FloatingPoint.Kind |> function
                            | FloatingPointKind.Double -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : float) @@>) |> Some
                            | FloatingPointKind.Single -> irValue |> Some
                        | _ -> None
                | _ -> None
            | _ -> None

[<RealTraits>]
type RealTraits<'T> =
    abstract Of : int8 -> 'T
    abstract Of : uint8 -> 'T
    abstract Of : int16 -> 'T
    abstract Of : uint16 -> 'T
    abstract Of : int -> 'T
    abstract Of : uint32 -> 'T
    abstract Of : int64 -> 'T
    abstract Of : uint64 -> 'T
    abstract Of : float32 -> 'T
    abstract Of : float -> 'T

type Real32Traits() =
    interface RealTraits<float32> with
        member this.Of(x:int8) = x |> float32
        member this.Of(x:uint8) = x |> float32
        member this.Of(x:int16) = x |> float32
        member this.Of(x:uint16) = x |> float32
        member this.Of(x:int) = x |> float32
        member this.Of(x:uint32) = x |> float32
        member this.Of(x:int64) = x |> float32
        member this.Of(x:uint64) = x |> float32
        member this.Of(x:float32) = x
        member this.Of(x:float) = x |> float32

type Real64Traits() =
    interface RealTraits<float> with
        member this.Of(x:int8) = x |> float
        member this.Of(x:uint8) = x |> float
        member this.Of(x:int16) = x |> float
        member this.Of(x:uint16) = x |> float
        member this.Of(x:int) = x |> float
        member this.Of(x:uint32) = x |> float
        member this.Of(x:int64) = x |> float
        member this.Of(x:uint64) = x |> float
        member this.Of(x:float32) = x |> float
        member this.Of(x:float) = x

type RealTraits =
    static member Real32 = Real32Traits() :> RealTraits<float32>
    static member Real64 = Real64Traits() :> RealTraits<float>

