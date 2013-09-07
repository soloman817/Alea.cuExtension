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

[<AttributeUsage(AttributeTargets.Property, AllowMultiple = false)>]
type RefClassFieldAttribute() =
    inherit Attribute()

    interface ICustomPropertyGetBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | Some(irStructPointer), [] ->
                let irFieldInfo = irStructPointer.Type.Pointer.PointeeType.Struct.FieldInfoByCLR(info.Name)
                let irFieldPointer = IRCommonInstructionBuilder.Instance.BuildStructGEP(ctx, irStructPointer, irFieldInfo)
                IRCommonInstructionBuilder.Instance.BuildReference(ctx, irFieldPointer) |> Some
            | _ -> None

    interface ICustomPropertySetBuilder with
        member this.Build(ctx, irObject, info, irParams, irValue) =
            match irObject, irParams with
            | Some(irStructPointer), [] ->
                let irFieldInfo = irStructPointer.Type.Pointer.PointeeType.Struct.FieldInfoByCLR(info.Name)
                let irFieldPointer = IRCommonInstructionBuilder.Instance.BuildStructGEP(ctx, irStructPointer, irFieldInfo)
                IRCommonInstructionBuilder.Instance.BuildStore(ctx, irFieldPointer, irValue) |> Some
            | _ -> None

[<AttributeUsage(AttributeTargets.Property, AllowMultiple = false)>]
type RefClassArrayFieldAttribute(length:int) =
    inherit RefClassFieldAttribute()

    member this.Length = length

    interface ICustomPropertyGetBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, irParams with
            | Some(irStructPointer), [] ->
                let irFieldInfo = irStructPointer.Type.Pointer.PointeeType.Struct.FieldInfoByCLR(info.Name)
                let irFieldPointer = IRCommonInstructionBuilder.Instance.BuildStructGEP(ctx, irStructPointer, irFieldInfo)
                irFieldPointer |> Some // array is a pointer, not reference
            | _ -> None

[<AttributeUsage(AttributeTargets.Class, AllowMultiple = false)>]
type RefClassAttribute() =
    inherit Attribute()

    interface ICustomTypeBuilder with
        member this.Build(ctx, clrType) =
            let clrFields =
                let clrProperties = clrType.GetProperties()
                if clrProperties = null then Array.empty else clrProperties
            let clrFields =
                clrFields
                |> Array.sortBy (fun info -> info.MetadataToken)
                |> Array.choose (fun info ->
                    let attrs = info.GetCustomAttributes(typeof<RefClassFieldAttribute>, true)
                    if attrs.Length = 1 then 
                        let attr = attrs.[0] :?> RefClassFieldAttribute
                        (info, attr) |> Some
                    else None)
            let irStructFields =
                clrFields
                |> Array.map (fun (info, attr) ->
                    let irFieldName = info.Name
                    let irFieldType = attr |> function
                        | :? RefClassArrayFieldAttribute as attr ->
                            if not info.PropertyType.IsArray then failwithf "%s is not an array property" info.Name
                            let length = attr.Length
                            let clrElementType = info.PropertyType.GetElementType()
                            let irElementType = IRTypeBuilder.Instance.Build(ctx, clrElementType)
                            let irArrayType = IRArrayType.Create(ctx.IRContext, irElementType, length)
                            irArrayType
                        | _ -> IRTypeBuilder.Instance.Build(ctx, info.PropertyType)
                    irFieldName, irFieldType)
            let irStructFields = IRStructFields.Named(irStructFields)
            let irStructType = IRStructType.Create(ctx.IRContext, irStructFields)
            let irStructPointerType = IRPointerType.Create(ctx.IRContext, irStructType, TypeQualifier.NotSpecified, AddressSpace.Generic, ctx.CompileOptions.AddressSize, None)
            irStructPointerType |> Some

    interface ICustomNewObjectBuilder with
        member this.Build(ctx, info, irParams) =
            match irParams with
            | [] ->
                let irStructPointerType = IRTypeBuilder.Instance.Build(ctx, info.ReflectedType)
                let irStructType = irStructPointerType.Pointer.PointeeType
                let irStructPointer = IRCommonInstructionBuilder.Instance.BuildAlloca(ctx, irStructType)
                irStructPointer |> Some
            | _ -> None

