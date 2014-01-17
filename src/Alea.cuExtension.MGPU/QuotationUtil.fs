module Alea.cuExtension.MGPU.QuotationUtil
//
//// This module is for some helper of quotation, no mgpu mapping
//
//#nowarn "9"
//#nowarn "51"
//
//open System.Runtime.InteropServices
//open Microsoft.FSharp.Collections
//open Alea.CUDA
//open Alea.cuExtension
////open Alea.cuExtension.Util
//
//[<Struct;StructLayout(LayoutKind.Sequential, Size=16);Align(16)>]
//type Extent16 =
//    val dummy : byte
//
//[<Struct;StructLayout(LayoutKind.Sequential, Size=8);Align(8)>]
//type Extent8 =
//    val dummy : byte
//
//[<Struct;StructLayout(LayoutKind.Sequential, Size=4);Align(4)>]
//type Extent4 =
//    val dummy : byte
//
//[<Struct;StructLayout(LayoutKind.Sequential, Size=2);Align(2)>]
//type Extent2 =
//    val dummy : byte
//
//[<Struct;StructLayout(LayoutKind.Sequential, Size=1);Align(1)>]
//type Extent1 =
//    val dummy : byte
//
//// we don't have union in Alea.cuBase now, so if we want to make a shared
//// union with specified alignment and size, we have to dynmaically create
//// a shared array with correct alignment, then reinterpret it to byte, so
//// we can use it later by reinterpret it to different types.
//// please google "wikipedia alignment" for more detail on struct alignment.
//// in short word, there is only 5 cases of alignment: 1, 2, 4, 8, 16. Cause
//// cuda in device is acturally 64-bit, and align16 is 64 bit (16 * 8)
//let createSharedExpr (align:int) (size:int) =
//    let length = divup size align
//    match align with
//    | 16 -> <@ __shared__<Extent16>(length) |> __array_to_ptr.Reinterpret<byte>() @>
//    | 8  -> <@ __shared__<Extent8>(length) |> __array_to_ptr.Reinterpret<byte>() @>
//    | 4  -> <@ __shared__<Extent4>(length) |> __array_to_ptr.Reinterpret<byte>() @>
//    | 2  -> <@ __shared__<Extent2>(length) |> __array_to_ptr.Reinterpret<byte>() @>
//    | 1  -> <@ __shared__<Extent1>(length) |> __array_to_ptr.Reinterpret<byte>() @>
//    | _ -> failwithf "wrong align of %d" align
