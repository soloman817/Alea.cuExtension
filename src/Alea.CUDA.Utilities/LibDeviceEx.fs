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

type NanHost = NanHost with
    static member ($) (NanHost, _:float  ) = fun () -> nan
    static member ($) (NanHost, _:float32) = fun () -> nanf

let [<Nan>] inline __nan() : ^T = (NanHost $ Unchecked.defaultof< ^T>) ()

type IsNanAttribute() =
    inherit Attribute()

    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, irValue :: [] when irValue.Type.IsFloatingPoint -> irValue.Type.FloatingPoint.Kind |> function
                | FloatingPointKind.Double -> IRInstructionBuilder.Instance.Build(ctx, <@@ (__nv_isnand %%irValue.Expr) <> 0 @@>) |> Some
                | FloatingPointKind.Single -> IRInstructionBuilder.Instance.Build(ctx, <@@ (__nv_isnanf %%irValue.Expr) <> 0 @@>) |> Some
            | _ -> None                

type IsNanHost = IsNanHost with
    static member ($) (IsNanHost, _:float  ) = System.Double.IsNaN
    static member ($) (IsNanHost, _:float32) = System.Single.IsNaN

let [<IsNan>] inline __isnan(x:^T) : bool = (IsNanHost $ Unchecked.defaultof< ^T>) x

type SinCosAttribute() =
    inherit Attribute()

    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, x :: sref :: cref :: [] when x.Type.IsFloatingPoint -> x.Type.FloatingPoint.Kind |> function
                | FloatingPointKind.Double -> IRInstructionBuilder.Instance.Build(ctx, <@@ __nv_sincos %%x.Expr %%sref.Expr %%cref.Expr @@>) |> Some
                | FloatingPointKind.Single -> IRInstructionBuilder.Instance.Build(ctx, <@@ __nv_sincosf %%x.Expr %%sref.Expr %%cref.Expr @@>) |> Some
            | _ -> None

type SinCosHost = SinCosHost with
    static member ($) (SinCosHost, _:float  ) = __nv_sincos
    static member ($) (SinCosHost, _:float32) = __nv_sincosf

let [<SinCos>] inline __sincos (x:^T) (sref:^T ref) (cref:^T ref) : unit = (SinCosHost $ Unchecked.defaultof< ^T>) x sref cref

type PowAttribute() =
    inherit Attribute()
    
    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, x :: y :: [] when x.Type.IsFloatingPoint -> x.Type.FloatingPoint.Kind |> function
                | FloatingPointKind.Double -> IRInstructionBuilder.Instance.Build(ctx, <@@ __nv_pow %%x.Expr %%y.Expr @@>) |> Some
                | FloatingPointKind.Single -> IRInstructionBuilder.Instance.Build(ctx, <@@ __nv_powf %%x.Expr %%y.Expr @@>) |> Some
            | _ -> None

type PowHost = PowHost with
    static member ($) (PowHost, _:float  ) = __nv_pow
    static member ($) (PowHost, _:float32) = __nv_powf

let [<Pow>] inline __pow (x:^T) (y:^T) : ^T = (PowHost $ Unchecked.defaultof< ^T>) x y

type RSqrtAttribute() =
    inherit Attribute()
    
    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, x :: [] when x.Type.IsFloatingPoint -> x.Type.FloatingPoint.Kind |> function
                | FloatingPointKind.Double -> IRInstructionBuilder.Instance.Build(ctx, <@@ __nv_rsqrt %%x.Expr @@>) |> Some
                | FloatingPointKind.Single -> IRInstructionBuilder.Instance.Build(ctx, <@@ __nv_rsqrtf %%x.Expr @@>) |> Some
            | _ -> None

type RSqrtHost = RSqrtHost with
    static member ($) (RSqrtHost, _:float  ) = __nv_rsqrt
    static member ($) (RSqrtHost, _:float32) = __nv_rsqrtf

let [<RSqrt>] inline __rsqrt (x:^T) : ^T = (RSqrtHost $ Unchecked.defaultof< ^T>) x


