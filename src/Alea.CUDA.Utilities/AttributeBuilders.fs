[<AutoOpen>]
module Alea.CUDA.Utilities.AttributeBuilders

open System
open Alea.CUDA

[<AttributeUsage(AttributeTargets.Field, AllowMultiple = false)>]
type EmbeddedArrayField(length:int) =
    inherit Attribute()

    interface ICustomStructOrUnionFieldTypeBuilder with
        member this.Build(ctx, info) =
            let irElementType = IRTypeBuilder.Instance.Build(ctx, info.FieldType)
            let irArrayType = IRArrayType.Create(ctx.IRContext, irElementType, length)
            irArrayType |> Some

[<AttributeUsage(AttributeTargets.Property, AllowMultiple = false)>]
type EmbeddedArrayProperty(fieldName:string) =
    inherit Attribute()

    interface ICustomPropertyGetBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | Some(irStruct), irIndex :: [] ->
                let irStructPointer = irStruct.Reference.Pointer
                let irStructType = irStruct.Type
                let irFieldInfo = irStructType.Struct.FieldInfoByCLR(fieldName)
                let irFieldPointer = IRCommonInstructionBuilder.Instance.BuildStructGEP(ctx, irStructPointer, irFieldInfo)
                let irIndex0 = IRCommonInstructionBuilder.Instance.BuildConstant(ctx, 0L)
                let irElementPointer = IRCommonInstructionBuilder.Instance.BuildGEP(ctx, irFieldPointer, irIndex0 :: irIndex :: [])
                IRCommonInstructionBuilder.Instance.BuildReference(ctx, irElementPointer) |> Some
            | _ -> None

    interface ICustomPropertySetBuilder with
        member this.Build(ctx, irObject, info, irParams, irValue) =
            match irObject, irParams with
            | Some(irStruct), irIndex :: [] ->
                let irStructPointer = irStruct.Reference.Pointer
                let irStructType = irStruct.Type
                let irFieldInfo = irStructType.Struct.FieldInfoByCLR(fieldName)
                let irFieldPointer = IRCommonInstructionBuilder.Instance.BuildStructGEP(ctx, irStructPointer, irFieldInfo)
                let irIndex0 = IRCommonInstructionBuilder.Instance.BuildConstant(ctx, 0L)
                let irElementPointer = IRCommonInstructionBuilder.Instance.BuildGEP(ctx, irFieldPointer, irIndex0 :: irIndex :: [])
                IRCommonInstructionBuilder.Instance.BuildStore(ctx, irElementPointer, irValue) |> Some
            | _ -> None
