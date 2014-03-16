[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Store

open System

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common

[<AutoOpen>]
module private InternalThreadStore =
    let buildThreadStore (modifier:string) (ctx:IRModuleBuildingContext) (irPointer:IRValue) (irVal:IRValue) =
        let irPointerType = irPointer.Type
        let irPointeeType = irPointerType.Pointer.PointeeType

        let irValType = irVal.Type

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


        let cmdstr, argstr = irPointeeType |> function
            | irPointeeType when isUInt  8 irPointeeType -> sprintf "st.%s.u8 [$0], $1;" modifier, sprintf "%s,c" ptrstr
            | irPointeeType when isUInt 16 irPointeeType -> sprintf "st.%s.u16 [$0], $1;" modifier, sprintf "%s,h" ptrstr
            | irPointeeType when isUInt 32 irPointeeType -> sprintf "st.%s.u32 [$0], $1;" modifier, sprintf "%s,r" ptrstr
            | irPointeeType when isUInt 64 irPointeeType -> sprintf "st.%s.u64 [$0], $1;" modifier, sprintf "%s,l" ptrstr
            | irPointeeType when isUIntVector 32 4 irPointeeType -> sprintf "st.%s.v4.u32 [$0], {$1, $2, $3, $4};" modifier, sprintf "%s,r,r,r,r" ptrstr
            | irPointeeType when isUIntVector 64 2 irPointeeType -> sprintf "st.%s.v2.u64 [$0], {$1, $2};" modifier, sprintf "%s,l,l" ptrstr
            | irPointeeType when isUIntVector 16 4 irPointeeType -> sprintf "st.%s.v4.u16 [$0], {$1, $2, $3, $4};" modifier, sprintf "%s,h,h,h,h" ptrstr
            | irPointeeType when isUIntVector 32 2 irPointeeType -> sprintf "st.%s.v2.u32 [$0], {$1, $2};" modifier, sprintf "%s,r,r" ptrstr
            | _ -> failwithf "CUBLOAD: %A doesn't support." irPointeeType

        let llvmFunction = LLVMConstInlineAsm(llvmFunctionType, cmdstr, argstr, 0, 0)
        let llvmCall = LLVMBuildCallEx(ctx.IRBuilder.LLVM, llvmFunction, [| irPointerInt.LLVM |], "")
        //IRValue()
        IRValue(llvmCall, irPointerType)

    [<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
    type ThreadStoreAttribute(modifier:string) =
        inherit Attribute()

        interface ICustomCallBuilder with
            member this.Build(ctx, irObject, info, irParams) =
                match irObject, irParams with
                | None, irPointer :: irVal :: [] -> buildThreadStore modifier ctx irPointer irVal |> Some
                | _ -> None

    let [<ThreadStore("wb")>] inline ThreadStore_WB (ptr:deviceptr<'T>) (value:'T) : unit = failwith ""
    let [<ThreadStore("cg")>] inline ThreadStore_CG (ptr:deviceptr<'T>) (value:'T) : unit = failwith ""
    let [<ThreadStore("cs")>] inline ThreadStore_CS (ptr:deviceptr<'T>) (value:'T) : unit = failwith ""
    let [<ThreadStore("wt")>] inline ThreadStore_WT (ptr:deviceptr<'T>) (value:'T) : unit = failwith ""


    type IterateThreadStoreAttribute(modifier:string) =
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

                        let irVal = IRCommonInstructionBuilder.Instance.BuildGEP(ctx, irPtr, irIndex :: [])
                        let irPtr = buildThreadStore modifier ctx irVal
                    
                        let irPtr = IRCommonInstructionBuilder.Instance.BuildGEP(ctx, irVals, irIndex :: [])
                        IRCommonInstructionBuilder.Instance.BuildStore(ctx, irPtr, irVal) 

                    IRCommonInstructionBuilder.Instance.BuildNop(ctx) |> Some

                | _ -> None

    let [<IterateThreadStore("wb")>] inline IterateThreadStore_WB (max:int) (ptr:deviceptr<'T>) (vals:deviceptr<'T>) : unit = failwith ""
    let [<IterateThreadStore("cg")>] inline IterateThreadStore_CG (max:int) (ptr:deviceptr<'T>) (vals:deviceptr<'T>) : unit = failwith ""
    let [<IterateThreadStore("cs")>] inline IterateThreadStore_CS (max:int) (ptr:deviceptr<'T>) (vals:deviceptr<'T>) : unit = failwith ""
    let [<IterateThreadStore("wt")>] inline IterateThreadStore_WT (max:int) (ptr:deviceptr<'T>) (vals:deviceptr<'T>) : unit = failwith ""


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
                        IRCommonInstructionBuilder.Instance.BuildStore(ctx, irPtr, irVal) 

                    IRCommonInstructionBuilder.Instance.BuildNop(ctx) |> Some

                | _ -> None


type CacheStoreModifier =
    | STORE_DEFAULT     = 0
    | STORE_WB          = 1
    | STORE_CG          = 2
    | STORE_CS          = 3
    | STORE_WT          = 4
    | STORE_VOLATILE    = 5


[<Sealed>]
type ThreadStore private () =

    [<ReflectedDefinition>] static member inline STORE_DEFAULT  (ptr:deviceptr<'T>) (value:'T) : unit = ptr.[0] <- value
    [<ReflectedDefinition>] static member inline STORE_VOLATILE (ptr:deviceptr<'T>) (value:'T) : unit = ptr.[0] <- value
    [<ThreadStore("wb")>]   static member inline STORE_WB       (ptr:deviceptr<'T>) (value:'T) : unit = failwith ""
    [<ThreadStore("cg")>]   static member inline STORE_CG       (ptr:deviceptr<'T>) (value:'T) : unit = failwith ""
    [<ThreadStore("cs")>]   static member inline STORE_CS       (ptr:deviceptr<'T>) (value:'T) : unit = failwith ""
    [<ThreadStore("wt")>]   static member inline STORE_WT       (ptr:deviceptr<'T>) (value:'T) : unit = failwith ""
             