[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Load

open System
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common


type CacheLoadModifier =
    | LOAD_DEFAULT
    | LOAD_CA
    | LOAD_CS
    | LOAD_CV
    | LOAD_LDG
    | LOAD_VOLATILE

let dereference (ptr:deviceptr<int>) = ptr |> __ptr_to_obj

let inline threadLoad() =
    fun modifier ->
        fun (ptr:deviceptr<int>) (vals:deviceptr<int> option) ->
            match vals with
            | Some vals ->
                match modifier with
                | LOAD_DEFAULT -> 
                    ptr.[0] <- vals.[0]
                    None
                | LOAD_CA -> 
                    ptr.[0] <- vals.[0]
                    None
                | LOAD_CS -> 
                    ptr.[0] <- vals.[0]
                    None
                | LOAD_CV -> 
                    ptr.[0] <- vals.[0]
                    None
                | LOAD_LDG -> 
                    ptr.[0] <- vals.[0]
                    None
                | LOAD_VOLATILE -> 
                    ptr.[0] <- vals.[0]
                    None
            | None ->
                ptr |> __ptr_to_obj |> Some



let inline iterateThreadLoad count max =
    fun modifier ->
        let load = modifier |> threadLoad()
        fun (ptr:deviceptr<int>) (vals:deviceptr<int>) ->
            for i = count to (max - 1) do
                (ptr + i, (vals + i) |> Some) 
                ||> load 
                |> ignore





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
