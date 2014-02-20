[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Load

open System

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common




module internal ThreadLoad =
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

    let [<ThreadLoad("ca")>] ThreadLoad_CA (ptr:deviceptr<'T>) : 'T = failwith ""
    let [<ThreadLoad("cg")>] ThreadLoad_CG (ptr:deviceptr<'T>) : 'T = failwith ""
    let [<ThreadLoad("cs")>] ThreadLoad_CS (ptr:deviceptr<'T>) : 'T = failwith ""
    let [<ThreadLoad("cv")>] ThreadLoad_CV (ptr:deviceptr<'T>) : 'T = failwith ""
    let [<ThreadLoad("ldg")>] ThreadLoad_LDG (ptr:deviceptr<'T>) : 'T = failwith ""

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

    let [<IterateThreadLoad("ca")>] IterateThreadLoad_CA (max:int) (ptr:deviceptr<'T>) (vals:deviceptr<'T>) : unit = failwith ""
    let [<IterateThreadLoad("cg")>] IterateThreadLoad_CG (max:int) (ptr:deviceptr<'T>) (vals:deviceptr<'T>) : unit = failwith ""
    let [<IterateThreadLoad("cs")>] IterateThreadLoad_CS (max:int) (ptr:deviceptr<'T>) (vals:deviceptr<'T>) : unit = failwith ""
    let [<IterateThreadLoad("cv")>] IterateThreadLoad_CV (max:int) (ptr:deviceptr<'T>) (vals:deviceptr<'T>) : unit = failwith ""
    let [<IterateThreadLoad("ldg")>] IterateThreadLoad_LDG (max:int) (ptr:deviceptr<'T>) (vals:deviceptr<'T>) : unit = failwith ""

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
    | LOAD_DEFAULT
    | LOAD_CA
    | LOAD_CG
    | LOAD_CS
    | LOAD_CV
    | LOAD_LDG
    | LOAD_VOLATILE

[<ReflectedDefinition>]
let DefaultLoad (ptr:deviceptr<'T>) = ptr.[0]

let inline ThreadLoad<'T> modifier : Expr<deviceptr<'T> -> 'T> =
    let load =
        modifier |> function
        | LOAD_DEFAULT ->   DefaultLoad
        | LOAD_CA ->        ThreadLoad.ThreadLoad_CA
        | LOAD_CG ->        ThreadLoad.ThreadLoad_CG
        | LOAD_CS ->        ThreadLoad.ThreadLoad_CS
        | LOAD_CV ->        ThreadLoad.ThreadLoad_CV
        | LOAD_LDG ->       ThreadLoad.ThreadLoad_LDG
        | LOAD_VOLATILE ->  DefaultLoad
    <@ load @>

//let dereference (ptr:deviceptr<int>) = ptr |> __ptr_to_obj
//
//let inline threadLoad() =
//    fun modifier ->
//        fun (ptr:deviceptr<int>) (vals:deviceptr<int> option) ->
//            match vals with
//            | Some vals ->
//                match modifier with
//                | LOAD_DEFAULT -> 
//                    ptr.[0] <- vals.[0]
//                    None
//                | LOAD_CA -> 
//                    ptr.[0] <- vals.[0]
//                    None
//                | LOAD_CS -> 
//                    ptr.[0] <- vals.[0]
//                    None
//                | LOAD_CV -> 
//                    ptr.[0] <- vals.[0]
//                    None
//                | LOAD_LDG -> 
//                    ptr.[0] <- vals.[0]
//                    None
//                | LOAD_VOLATILE -> 
//                    ptr.[0] <- vals.[0]
//                    None
//            | None ->
//                ptr |> __ptr_to_obj |> Some
//
//
//
//let inline iterateThreadLoad count max =
//    fun modifier ->
//        let load = modifier |> threadLoad()
//        fun (ptr:deviceptr<int>) (vals:deviceptr<int>) ->
//            for i = count to (max - 1) do
//                (ptr + i, (vals + i) |> Some) 
//                ||> load 
//                |> ignore





///**
// * ThreadLoad definition for LOAD_DEFAULT modifier on iterator types
// */
//template <typename InputIterator>
//__device__ __forceinline__ typename std::iterator_traits<InputIterator>::value_type ThreadLoad(
//    InputIterator           itr,
//    Int2Type<LOAD_DEFAULT>  modifier,
//    Int2Type<false>         is_pointer)
//{
//    return *itr;
//}
//
//
///**
// * ThreadLoad definition for LOAD_DEFAULT modifier on pointer types
// */
//template <typename T>
//__device__ __forceinline__ T ThreadLoad(
//    T                       *ptr,
//    Int2Type<LOAD_DEFAULT>  modifier,
//    Int2Type<true>          is_pointer)
//{
//    return *ptr;
//}
//
//
///**
// * ThreadLoad definition for LOAD_VOLATILE modifier on primitive pointer types
// */
//template <typename T>
//__device__ __forceinline__ T ThreadLoadVolatilePointer(
//    T                       *ptr,
//    Int2Type<true>          is_primitive)
//{
//    T retval = *reinterpret_cast<volatile T*>(ptr);
//
//#if (CUB_PTX_VERSION <= 130)
//    if (sizeof(T) == 1) __threadfence_block();
//#endif
//
//    return retval;
//}
//
//
///**
// * ThreadLoad definition for LOAD_VOLATILE modifier on non-primitive pointer types
// */
//template <typename T>
//__device__ __forceinline__ T ThreadLoadVolatilePointer(
//    T                       *ptr,
//    Int2Type<false>          is_primitive)
//{
//
//#if CUB_PTX_VERSION <= 130
//
//    T retval = *ptr;
//    __threadfence_block();
//    return retval;
//
//#else
//
//    typedef typename UnitWord<T>::VolatileWord VolatileWord;   // Word type for memcopying
//
//    const int VOLATILE_MULTIPLE = sizeof(T) / sizeof(VolatileWord);
//
//    VolatileWord words[VOLATILE_MULTIPLE];
//
//    IterateThreadLoad<0, VOLATILE_MULTIPLE>::Dereference(
//        reinterpret_cast<volatile VolatileWord*>(ptr),
//        words);
//
//    return *reinterpret_cast<T*>(words);
//
//#endif  // CUB_PTX_VERSION <= 130
//}
//
//
///**
// * ThreadLoad definition for LOAD_VOLATILE modifier on pointer types
// */
//template <typename T>
//__device__ __forceinline__ T ThreadLoad(
//    T                       *ptr,
//    Int2Type<LOAD_VOLATILE> modifier,
//    Int2Type<true>          is_pointer)
//{
//    // Apply tags for partial-specialization
//    return ThreadLoadVolatilePointer(ptr, Int2Type<Traits<T>::PRIMITIVE>());
//}
//
//
///**
// * ThreadLoad definition for generic modifiers on pointer types
// */
//template <typename T, int MODIFIER>
//__device__ __forceinline__ T ThreadLoad(
//    T                       *ptr,
//    Int2Type<MODIFIER>      modifier,
//    Int2Type<true>          is_pointer)
//{
//    typedef typename UnitWord<T>::DeviceWord DeviceWord;
//
//    const int DEVICE_MULTIPLE = sizeof(T) / sizeof(DeviceWord);
//
//    DeviceWord words[DEVICE_MULTIPLE];
//
//    IterateThreadLoad<0, DEVICE_MULTIPLE>::template Load<CacheLoadModifier(MODIFIER)>(
//        reinterpret_cast<DeviceWord*>(ptr),
//        words);
//
//    return *reinterpret_cast<T*>(words);
//}
//
//
///**
// * ThreadLoad definition for generic modifiers
// */
//template <
//    CacheLoadModifier MODIFIER,
//    typename InputIterator>
//__device__ __forceinline__ typename std::iterator_traits<InputIterator>::value_type ThreadLoad(InputIterator itr)
//{
//    // Apply tags for partial-specialization
//    return ThreadLoad(
//        itr,
//        Int2Type<MODIFIER>(),
//        Int2Type<IsPointer<InputIterator>::VALUE>());
//}
