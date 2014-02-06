[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Store

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common

type CacheStoreModifier =
    | STORE_DEFAULT
    | STORE_WB
    | STORE_CG
    | STORE_CS
    | STORE_WT
    | STORE_VOLATILE


let inline threadStore<'T>() =
    fun modifier ->
        match modifier with
        | STORE_DEFAULT -> fun (ptr:deviceptr<'T>) (value:'T) ->  ptr.[0] <- value
        | STORE_WB -> fun (ptr:deviceptr<'T>) (value:'T) -> ptr.[0] <- value
        | STORE_CG -> fun (ptr:deviceptr<'T>) (value:'T) -> ptr.[0] <- value
        | STORE_CS -> fun (ptr:deviceptr<'T>) (value:'T) -> ptr.[0] <- value
        | STORE_WT -> fun (ptr:deviceptr<'T>) (value:'T) -> ptr.[0] <- value
        | STORE_VOLATILE -> fun (ptr:deviceptr<'T>) (value:'T) -> ptr.[0] <- value

let iterateThreadStore count max =
    fun modifier ->
        let store = modifier |> threadStore<'T>()
        fun (ptr:deviceptr<'T>) (value:'T) ->
            for i = count to (max - 1) do
                (ptr + i, value) ||> store


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


[<Record>]
type ThreadStore<'T> =
    {
        COUNT : int
        MAX : int
    }

    [<ReflectedDefinition>]
    static member DefaultStore(length:int) = (0, length) ||> iterateThreadStore <| CacheStoreModifier.STORE_DEFAULT

    [<ReflectedDefinition>]
    member this.Store() = (this.COUNT, this.MAX) ||> iterateThreadStore <| CacheStoreModifier.STORE_DEFAULT

    [<ReflectedDefinition>]
    member this.Store(modifier) = (this.COUNT, this.MAX) ||> iterateThreadStore <| modifier

    [<ReflectedDefinition>]
    static member Create(count,max) =
        {
            COUNT = count
            MAX = max
        }



//        let Store =
//            fun (ptr:deviceptr<'T>) (vals:deivceptr<'T>) ->
//                