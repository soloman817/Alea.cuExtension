[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Reduce

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Thread

//let threadReduce length reduction_op =
//    let reduction_op = reduction_op.op
//    <@ fun (input:deviceptr<int>) (prefix:int)  ->
//        let mutable prefix = prefix
//                
//        for i = 0 to length - 1 do
//            let addend = input.[0]
//            prefix <- (prefix, addend) ||> %reduction_op
//
//        prefix
//    @>
//
//let ThreadReduce length reduction_op prefix =
//    let reduction_op = reduction_op.op
//    let threadReduce = (length, reduction_op) ||> threadReduce
//    <@ fun (input:deviceptr<int>) ->
//        prefix |> function
//        | Some prefix ->
//            (%threadReduce)
//            <|      None
//            <||    (input, prefix)
//
//        | None ->
//            let prefix = input.[0]
//            threadReduce
//            <|      Some(length - 1)
//            <|||    (input, reduction_op, prefix)
//    @> 
