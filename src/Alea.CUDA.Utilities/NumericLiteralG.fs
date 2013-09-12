module Alea.CUDA.Utilities.NumericLiteralG

open System
open Alea.CUDA

let [<ReflectedDefinition>] inline FromZero() = LanguagePrimitives.GenericZero
let [<ReflectedDefinition>] inline FromOne() = LanguagePrimitives.GenericOne

type GenericNumberFromInt32 = GenericNumberFromInt32 with
    static member ($) (GenericNumberFromInt32, _:sbyte     ) = fun (x:int) -> sbyte      x
    static member ($) (GenericNumberFromInt32, _:int16     ) = fun (x:int) -> int16      x
    static member ($) (GenericNumberFromInt32, _:int32     ) = id
    static member ($) (GenericNumberFromInt32, _:float     ) = fun (x:int) -> float      x
    static member ($) (GenericNumberFromInt32, _:float32   ) = fun (x:int) -> float32    x
    static member ($) (GenericNumberFromInt32, _:int64     ) = fun (x:int) -> int64      x
    static member ($) (GenericNumberFromInt32, _:nativeint ) = fun (x:int) -> nativeint  x
    static member ($) (GenericNumberFromInt32, _:byte      ) = fun (x:int) -> byte       x
    static member ($) (GenericNumberFromInt32, _:uint16    ) = fun (x:int) -> uint16     x
    static member ($) (GenericNumberFromInt32, _:char      ) = fun (x:int) -> char       x
    static member ($) (GenericNumberFromInt32, _:uint32    ) = fun (x:int) -> uint32     x
    static member ($) (GenericNumberFromInt32, _:uint64    ) = fun (x:int) -> uint64     x
    static member ($) (GenericNumberFromInt32, _:unativeint) = fun (x:int) -> unativeint x
    static member ($) (GenericNumberFromInt32, _:bigint    ) = fun (x:int) -> bigint     x
    static member ($) (GenericNumberFromInt32, _:decimal   ) = fun (x:int) -> decimal    x

type GenericNumberFromInt32Attribute() =
    inherit Attribute()

    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | None, irSrcValue :: [] ->
                let clrDstType = info.GetGenericArguments().[0]
                let irDstType = IRTypeBuilder.Instance.Build(ctx, clrDstType)
                let expr = irDstType |> function
                    | irDstType when irDstType.IsInteger -> (irDstType.Integer.Bits, irDstType.Integer.Signed) |> function
                        |  8,  true -> <@@ %%irSrcValue.Expr |> int8 @@>     |> Some
                        |  8, false -> <@@ %%irSrcValue.Expr |> uint8 @@>    |> Some
                        | 16,  true -> <@@ %%irSrcValue.Expr |> int16 @@>    |> Some
                        | 16, false -> <@@ %%irSrcValue.Expr |> uint16 @@>   |> Some
                        | 32,  true -> <@@ %%irSrcValue.Expr |> int @@>      |> Some
                        | 32, false -> <@@ %%irSrcValue.Expr |> uint32 @@>   |> Some
                        | 64,  true -> <@@ %%irSrcValue.Expr |> int64 @@>    |> Some
                        | 64, false -> <@@ %%irSrcValue.Expr |> uint64 @@>   |> Some
                        | _ -> None
                    | irDstType when irDstType.IsFloatingPoint -> irDstType.FloatingPoint.Kind |> function
                        | FloatingPointKind.Double -> <@@ %%irSrcValue.Expr |> float @@>     |> Some
                        | FloatingPointKind.Single -> <@@ %%irSrcValue.Expr |> float32 @@>   |> Some
                    | _ -> None
                expr |> Option.map (fun expr -> IRInstructionBuilder.Instance.Build(ctx, expr))
            | _ -> None

[<GenericNumberFromInt32>]
let inline FromInt32(i:int) : ^t = (GenericNumberFromInt32 $ Unchecked.defaultof< ^t>) i
