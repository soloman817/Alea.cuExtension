[<AutoOpen>]
module Alea.CUDA.Utilities.LibDeviceEx

open System
open Alea.CUDA

type NanAttribute() =
    inherit Attribute()
    
    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, [] -> 
                let clrType = info.GetGenericArguments().[0]
                let irType = IRTypeBuilder.Instance.Build(ctx, clrType)
                match irType with
                | irType when irType.IsFloatingPoint -> irType.FloatingPoint.Kind |> function
                    | FloatingPointKind.Double -> IRInstructionBuilder.Instance.Build(ctx, <@@ nan @@>) |> Some
                    | FloatingPointKind.Single -> IRInstructionBuilder.Instance.Build(ctx, <@@ nanf @@>) |> Some
                | _ -> None
            | _ -> None

let [<Nan>] __nan() : 'T = failwith "device only __nan"

type IsNanAttribute() =
    inherit Attribute()

    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, irValue :: [] -> irValue.Type |> function
                | irType when irType.IsFloatingPoint -> irType.FloatingPoint.Kind |> function
                    | FloatingPointKind.Double -> IRInstructionBuilder.Instance.Build(ctx, <@@ (__nv_isnand %%irValue.Expr) <> 0 @@>) |> Some
                    | FloatingPointKind.Single -> IRInstructionBuilder.Instance.Build(ctx, <@@ (__nv_isnanf %%irValue.Expr) <> 0 @@>) |> Some
                | _ -> None
            | _ -> None                

let [<IsNan>] __isnan(x:'T) : bool = failwith "device only __isnan"

type SinCosAttribute() =
    inherit Attribute()

    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, x :: sref :: cref :: [] -> x.Type |> function
                | irType when irType.IsFloatingPoint -> irType.FloatingPoint.Kind |> function
                    | FloatingPointKind.Double -> IRInstructionBuilder.Instance.Build(ctx, <@@ __nv_sincos %%x.Expr %%sref.Expr %%cref.Expr @@>) |> Some
                    | FloatingPointKind.Single -> IRInstructionBuilder.Instance.Build(ctx, <@@ __nv_sincosf %%x.Expr %%sref.Expr %%cref.Expr @@>) |> Some
                | _ -> None
            | _ -> None

let [<SinCos>] __sincos (x:'T) (sref:'T ref) (cref:'T ref) : unit = failwith "device only __sincos"

