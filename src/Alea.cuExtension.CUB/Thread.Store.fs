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


//let f (x:'T) (y:'T option) (z:'T option) (w:'T option) =

//let inline storeType<'T>() =
//    match typeof<'T> with
//    | ty when ty = typeof<uint4> ->
//        fun (ptr:deviceptr<uint4>) (vals:uint4) ->
//             ptr.[0].x <- vals.x
//             ptr.[0].y <- vals.y
//             ptr.[0].z <- vals.z
//             ptr.[0].w <- vals.w
//    | ty when ty = typeof<ulonglong2> -> fun (ptr:deviceptr<ulonglong2>) (vals:ulonglong2) -> ()
//    | ty when ty = typeof<ushort4> -> ()
//    | ty when ty = typeof<uint2> -> ()
//    | ty when ty = typeof<ulonglong> -> ()
//    | ty when ty = typeof<uint32> -> ()
//    | ty when ty = typeof<byte> -> ()
//    | _ -> ()

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