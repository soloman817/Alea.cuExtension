[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Store

open System

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common

module internal ThreadStore =
    let buildThreadStore (modifier:string) (ctx:IRModuleBuildingContext) (irPointer:IRValue) (irVal:IRValue) =
        let irPointerType = irVal.Type
        let irPointeeType = irPointer.Type

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
            | irPointeeType when isUInt  8 irPointeeType -> sprintf "st.%s.u8 [$0], $1;" modifier, sprintf "c,%s" ptrstr
            | irPointeeType when isUInt 16 irPointeeType -> sprintf "st.%s.u16 [$0], $1;" modifier, sprintf "h,%s" ptrstr
            | irPointeeType when isUInt 32 irPointeeType -> sprintf "st.%s.u32 [$0], $1;" modifier, sprintf "r,%s" ptrstr
            | irPointeeType when isUInt 64 irPointeeType -> sprintf "st.%s.u64 [$0], $1;" modifier, sprintf "l,%s" ptrstr
            | irPointeeType when isUIntVector 32 4 irPointeeType -> sprintf "st.%s.v4.u32 [$0], {$1, $2, $3, $4};" modifier, sprintf "r,r,r,r,%s" ptrstr
            | irPointeeType when isUIntVector 64 2 irPointeeType -> sprintf "st.%s.v2.u64 [$0], {$1, $2};" modifier, sprintf "l,l,%s" ptrstr
            | irPointeeType when isUIntVector 16 4 irPointeeType -> sprintf "st.%s.v4.u16 [$0], {$1, $2, $3, $4};" modifier, sprintf "h,h,h,h,%s" ptrstr
            | irPointeeType when isUIntVector 32 2 irPointeeType -> sprintf "st.%s.v2.u32 [$0], {$1, $2};" modifier, sprintf "r,r,%s" ptrstr
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
                        IRCommonInstructionBuilder.Instance.BuildStore(ctx, irPtr, irVal) |> ignore

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
                        IRCommonInstructionBuilder.Instance.BuildStore(ctx, irPtr, irVal) |> ignore

                    IRCommonInstructionBuilder.Instance.BuildNop(ctx) |> Some

                | _ -> None




type CacheStoreModifier =
    | STORE_DEFAULT
    | STORE_WB
    | STORE_CG
    | STORE_CS
    | STORE_WT
    | STORE_VOLATILE


[<ReflectedDefinition>]
let inline DefaultStore (ptr:deviceptr<'T>) (value:'T) = ptr.[0] <- value

let inline ThreadStore<'T> modifier : deviceptr<'T> -> 'T -> unit =
    modifier |> function
    | STORE_DEFAULT ->      DefaultStore
    | STORE_WB ->           ThreadStore.ThreadStore_WB
    | STORE_CG ->           ThreadStore.ThreadStore_CG
    | STORE_CS ->           ThreadStore.ThreadStore_CS
    | STORE_WT ->           ThreadStore.ThreadStore_WT
    | STORE_VOLATILE ->     DefaultStore
    



//let inline threadStore() =
//    fun modifier ->
//        match modifier with
//        | STORE_DEFAULT -> fun (ptr:deviceptr<int>) (value:int) ->  ptr.[0] <- value
//        | STORE_WB -> fun (ptr:deviceptr<int>) (value:int) -> ptr.[0] <- value
//        | STORE_CG -> fun (ptr:deviceptr<int>) (value:int) -> ptr.[0] <- value
//        | STORE_CS -> fun (ptr:deviceptr<int>) (value:int) -> ptr.[0] <- value
//        | STORE_WT -> fun (ptr:deviceptr<int>) (value:int) -> ptr.[0] <- value
//        | STORE_VOLATILE -> fun (ptr:deviceptr<int>) (value:int) -> ptr.[0] <- value
//
//let iterateThreadStore count max =
//    fun modifier ->
//        let store = modifier |> threadStore()
//        fun (ptr:deviceptr<int>) (value:int) ->
//            for i = count to (max - 1) do
//                (ptr + i, value) ||> store


///**
// * ThreadStore definition for STORE_DEFAULT modifier on iterator types
// */
//template <typename OutputIterator, typename T>
//__device__ __forceinline__ void ThreadStore(
//    OutputIterator              itr,
//    T                           val,
//    Int2Type<STORE_DEFAULT>     modifier,
//    Int2Type<false>             is_pointer)
//{
//    *itr = val;
//}
//
//
///**
// * ThreadStore definition for STORE_DEFAULT modifier on pointer types
// */
//template <typename T>
//__device__ __forceinline__ void ThreadStore(
//    T                           *ptr,
//    T                           val,
//    Int2Type<STORE_DEFAULT>     modifier,
//    Int2Type<true>              is_pointer)
//{
//    *ptr = val;
//}
//
//
///**
// * ThreadStore definition for STORE_VOLATILE modifier on primitive pointer types
// */
//template <typename T>
//__device__ __forceinline__ void ThreadStoreVolatilePtr(
//    T                           *ptr,
//    T                           val,
//    Int2Type<true>              is_primitive)
//{
//    *reinterpret_cast<volatile T*>(ptr) = val;
//}
//
//
///**
// * ThreadStore definition for STORE_VOLATILE modifier on non-primitive pointer types
// */
//template <typename T>
//__device__ __forceinline__ void ThreadStoreVolatilePtr(
//    T                           *ptr,
//    T                           val,
//    Int2Type<false>             is_primitive)
//{
//#if CUB_PTX_VERSION <= 130
//
//    *ptr = val;
//    __threadfence_block();
//
//#else
//
//    typedef typename UnitWord<T>::VolatileWord VolatileWord;   // Word type for memcopying
//
//    const int VOLATILE_MULTIPLE = sizeof(T) / sizeof(VolatileWord);
//
//    VolatileWord words[VOLATILE_MULTIPLE];
//    *reinterpret_cast<T*>(words) = val;
//
//    IterateThreadStore<0, VOLATILE_MULTIPLE>::template Dereference(
//        reinterpret_cast<volatile VolatileWord*>(ptr),
//        words);
//
//#endif  // CUB_PTX_VERSION <= 130
//
//}
//
//
///**
// * ThreadStore definition for STORE_VOLATILE modifier on pointer types
// */
//template <typename T>
//__device__ __forceinline__ void ThreadStore(
//    T                           *ptr,
//    T                           val,
//    Int2Type<STORE_VOLATILE>    modifier,
//    Int2Type<true>              is_pointer)
//{
//    ThreadStoreVolatilePtr(ptr, val, Int2Type<Traits<T>::PRIMITIVE>());
//}
//
//
///**
// * ThreadStore definition for generic modifiers on pointer types
// */
//template <typename T, int MODIFIER>
//__device__ __forceinline__ void ThreadStore(
//    T                           *ptr,
//    T                           val,
//    Int2Type<MODIFIER>          modifier,
//    Int2Type<true>              is_pointer)
//{
//    typedef typename UnitWord<T>::DeviceWord DeviceWord;   // Word type for memcopying
//
//    const int DEVICE_MULTIPLE = sizeof(T) / sizeof(DeviceWord);
//
//    DeviceWord words[DEVICE_MULTIPLE];
//
//    *reinterpret_cast<T*>(words) = val;
//
//    IterateThreadStore<0, DEVICE_MULTIPLE>::template Store<CacheStoreModifier(MODIFIER)>(
//        reinterpret_cast<DeviceWord*>(ptr),
//        words);
//}
//
//
///**
// * ThreadStore definition for generic modifiers
// */
//template <CacheStoreModifier MODIFIER, typename OutputIterator, typename T>
//__device__ __forceinline__ void ThreadStore(OutputIterator itr, T val)
//{
//    ThreadStore(
//        itr,
//        val,
//        Int2Type<MODIFIER>(),
//        Int2Type<IsPointer<OutputIterator>::VALUE>());
//}
//
//
//[<Record>]
//type ThreadStore =
//    {
//        COUNT : int
//        MAX : int
//    }
//
//    [<ReflectedDefinition>]
//    static member DefaultStore(length:int) = (0, length) ||> iterateThreadStore <| CacheStoreModifier.STORE_DEFAULT
//
//    [<ReflectedDefinition>]
//    member this.Store() = (this.COUNT, this.MAX) ||> iterateThreadStore <| CacheStoreModifier.STORE_DEFAULT
//
//    [<ReflectedDefinition>]
//    member this.Store(modifier) = (this.COUNT, this.MAX) ||> iterateThreadStore <| modifier
//
//    [<ReflectedDefinition>]
//    static member Create(count,max) =
//        {
//            COUNT = count
//            MAX = max
//        }



//        let Store =
//            fun (ptr:deviceptr<int>) (vals:deivceptr<int>) ->
//                