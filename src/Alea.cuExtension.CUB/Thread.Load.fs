[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Load

open System

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common

[<AutoOpen>]
module private InternalThreadLoad =
    let buildThreadLoad (modifier:string) (ctx:IRModuleBuildingContext) (irPointer:IRValue) =
        let irPointerType = irPointer.Type
        let irPointeeType = irPointerType.Pointer.PointeeType
        
        // ptx inline cannot accept pointer, must convert to integer
        // I got this by print out the link result and the error of nvvm compiler told me that
        // and here we also need to handle the size of the integer, 32 or 64
        let irPointerInt, ptrstr = ctx.CompileOptions.AddressSize |> function
            | AddressSize.Address32 -> IRCommonInstructionBuilder.Instance.BuildPtrToInt(ctx, irPointer, IRTypeBuilder.Instance.Build(ctx, typeof<uint32>)), "r"
            | AddressSize.Address64 -> IRCommonInstructionBuilder.Instance.BuildPtrToInt(ctx, irPointer, IRTypeBuilder.Instance.Build(ctx, typeof<uint64>)), "l"

        let llvmFunctionType = LLVMFunctionTypeEx(irPointeeType.LLVM, [| irPointerInt.Type.LLVM |], 0)

        let isUInt (bits:int) (irType:IRType) = 
            irType.IsInteger &&
            irType.Integer.Bits = bits &&
            not irType.Integer.Signed

        let isUIntVector (bits:int) (dims:int) (irType:IRType) =
            irType.IsValStruct &&
            irType.Struct.FieldInfos.Length = dims &&
            Array.forall (fun (info:IRStructFieldInfo) -> isUInt bits info.FieldType) irType.Struct.FieldInfos

        let modifier = modifier |> function
            | "ldg" -> ctx.CompileOptions.MinimalArch.Number >= 350 |> function
                | true -> "global.nc"
                | false -> "global"
            | _ -> modifier

        let cmdstr, argstr = irPointeeType |> function
            | irPointeeType when isUInt  8 irPointeeType -> sprintf "ld.%s.u8 $0, [$1];" modifier, sprintf "=c,%s" ptrstr
            | irPointeeType when isUInt 16 irPointeeType -> sprintf "ld.%s.u16 $0, [$1];" modifier, sprintf "=h,%s" ptrstr
            | irPointeeType when isUInt 32 irPointeeType -> sprintf "ld.%s.u32 $0, [$1];" modifier, sprintf "=r,%s" ptrstr
            | irPointeeType when isUInt 64 irPointeeType -> sprintf "ld.%s.u64 $0, [$1];" modifier, sprintf "=l,%s" ptrstr
            | irPointeeType when isUIntVector 32 4 irPointeeType -> sprintf "ld.%s.v4.u32 {$0, $1, $2, $3}, [$4];" modifier, sprintf "=r,=r,=r,=r,%s" ptrstr
            | irPointeeType when isUIntVector 64 2 irPointeeType -> sprintf "ld.%s.v2.u64 {$0, $1}, [$2];" modifier, sprintf "=l,=l,%s" ptrstr
            | irPointeeType when isUIntVector 16 4 irPointeeType -> sprintf "ld.%s.v4.u16 {$0, $1, $2, $3}, [$4];" modifier, sprintf "=h,=h,=h,=h,%s" ptrstr
            | irPointeeType when isUIntVector 32 2 irPointeeType -> sprintf "ld.%s.v2.u32 {$0, $1}, [$2];" modifier, sprintf "=r,=r,%s" ptrstr
            | _ -> failwithf "CUBLOAD: %A doesn't support." irPointeeType



        let llvmFunction = LLVMConstInlineAsm(llvmFunctionType, cmdstr, argstr, 0, 0)
        let llvmCall = LLVMBuildCallEx(ctx.IRBuilder.LLVM, llvmFunction, [| irPointerInt.LLVM |], "")

        IRValue(llvmCall, irPointeeType)

    [<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
    type ThreadLoadAttribute(modifier:string) =
        inherit Attribute()

        interface ICustomCallBuilder with
            member this.Build(ctx, irObject, info, irParams) =
                match irObject, irParams with
                | None, irPointer :: [] -> buildThreadLoad modifier ctx irPointer |> Some
                | _ -> None

    let [<ThreadLoad("ca")>] inline ThreadLoad_CA (ptr:deviceptr<'T>) : 'T = failwith ""
    let [<ThreadLoad("cg")>] inline ThreadLoad_CG (ptr:deviceptr<'T>) : 'T = failwith ""
    let [<ThreadLoad("cs")>] inline ThreadLoad_CS (ptr:deviceptr<'T>) : 'T = failwith ""
    let [<ThreadLoad("cv")>] inline ThreadLoad_CV (ptr:deviceptr<'T>) : 'T = failwith ""
    let [<ThreadLoad("ldg")>] inline ThreadLoad_LDG (ptr:deviceptr<'T>) : 'T = failwith ""

    type IterateThreadLoadAttribute(modifier:string) =
        inherit Attribute()

        interface ICustomCallBuilder with
            member this.Build(ctx, irObject, info, irParams) =
                match irObject, irParams with
                | None, irMax :: irPtr :: irVals :: [] ->
                    let max = irMax.HasObject |> function
                        | true -> irMax.Object :?> int
                        | false -> failwith "max must be constant"

                    // if we do the loop here, it is unrolled by compiler, not kernel runtime
                    // think of this job as the C++ template expanding job, same thing!
                    for i = 0 to max - 1 do
                        let irIndex = IRCommonInstructionBuilder.Instance.BuildConstant(ctx, i)
                        let irPtr = IRCommonInstructionBuilder.Instance.BuildGEP(ctx, irPtr, irIndex :: [])
                        let irVal = buildThreadLoad modifier ctx irPtr
                        let irPtr = IRCommonInstructionBuilder.Instance.BuildGEP(ctx, irVals, irIndex :: [])
                        IRCommonInstructionBuilder.Instance.BuildStore(ctx, irPtr, irVal) |> ignore

                    IRCommonInstructionBuilder.Instance.BuildNop(ctx) |> Some

                | _ -> None

    let [<IterateThreadLoad("ca")>] inline IterateThreadLoad_CA (max:int) (ptr:deviceptr<'T>) (vals:deviceptr<'T>) : unit = failwith ""
    let [<IterateThreadLoad("cg")>] inline IterateThreadLoad_CG (max:int) (ptr:deviceptr<'T>) (vals:deviceptr<'T>) : unit = failwith ""
    let [<IterateThreadLoad("cs")>] inline IterateThreadLoad_CS (max:int) (ptr:deviceptr<'T>) (vals:deviceptr<'T>) : unit = failwith ""
    let [<IterateThreadLoad("cv")>] inline IterateThreadLoad_CV (max:int) (ptr:deviceptr<'T>) (vals:deviceptr<'T>) : unit = failwith ""
    let [<IterateThreadLoad("ldg")>] inline IterateThreadLoad_LDG (max:int) (ptr:deviceptr<'T>) (vals:deviceptr<'T>) : unit = failwith ""

    type IterateThreadDereferenceAttribute() =
        inherit Attribute()

        interface ICustomCallBuilder with
            member this.Build(ctx, irObject, info, irParams) =
                match irObject, irParams with
                | None, irMax :: irPtr :: irVals :: [] ->
                    let max = irMax.HasObject |> function
                        | true -> irMax.Object :?> int
                        | false -> failwith "max must be constant"

                    for i = 0 to max - 1 do
                        let irIndex = IRCommonInstructionBuilder.Instance.BuildConstant(ctx, i)
                        let irPtr = IRCommonInstructionBuilder.Instance.BuildGEP(ctx, irPtr, irIndex :: [])
                        let irVal = IRCommonInstructionBuilder.Instance.BuildLoad(ctx, irPtr)
                        let irPtr = IRCommonInstructionBuilder.Instance.BuildGEP(ctx, irVals, irIndex :: [])
                        IRCommonInstructionBuilder.Instance.BuildStore(ctx, irPtr, irVal) |> ignore

                    IRCommonInstructionBuilder.Instance.BuildNop(ctx) |> Some

                | _ -> None


type CacheLoadModifier =
    | LOAD_DEFAULT  = 0
    | LOAD_CA       = 1
    | LOAD_CG       = 2
    | LOAD_CS       = 3
    | LOAD_CV       = 4
    | LOAD_LDG      = 5
    | LOAD_VOLATILE = 6


[<Sealed>]
type ThreadLoad private () =
    
    [<ReflectedDefinition>] static member inline LOAD_DEFAULT   (ptr:deviceptr<'T>) : 'T = ptr.[0]
    [<ReflectedDefinition>] static member inline LOAD_VOLATILE  (ptr:deviceptr<'T>) : 'T = ptr.[0]
    [<ThreadLoad("ca")>]    static member inline LOAD_CA        (ptr:deviceptr<'T>) : 'T = failwith ""     
    [<ThreadLoad("cg")>]    static member inline LOAD_CG        (ptr:deviceptr<'T>) : 'T = failwith ""
    [<ThreadLoad("cs")>]    static member inline LOAD_CS        (ptr:deviceptr<'T>) : 'T = failwith ""
    [<ThreadLoad("cv")>]    static member inline LOAD_CV        (ptr:deviceptr<'T>) : 'T = failwith ""
    [<ThreadLoad("ldg")>]   static member inline LOAD_LDG       (ptr:deviceptr<'T>) : 'T = failwith ""
