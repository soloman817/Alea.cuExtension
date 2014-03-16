[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Reduce

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Thread


module ThreadReduce =
    
    let [<ReflectedDefinition>] inline WithPrefix (length:int) (reduction_op:'T -> 'T -> 'T)
        (input:deviceptr<'T>) (prefix:'T) =
        let mutable addend = input.[0]
        let mutable prefix = (prefix, addend) ||> reduction_op

        for i = 1 to length - 1 do
            addend <- input.[i]
            prefix <- (prefix, addend) ||> reduction_op

        prefix
    

    let [<ReflectedDefinition>] inline Default (length:int) (reduction_op:'T -> 'T -> 'T)
        (input:deviceptr<'T>) =
        WithPrefix length reduction_op input input.[0]


    let [<ReflectedDefinition>] inline WithPrefixInt (length:int)
        (input:deviceptr<int>) (prefix:int) =
        let mutable addend = input.[0]
        let mutable prefix = prefix + addend

        for i = 1 to length - 1 do
            addend <- input.[i]
            prefix <- prefix + addend

        prefix
    

    let [<ReflectedDefinition>] inline DefaultInt (length:int)
        (input:deviceptr<int>) =
        WithPrefixInt length input input.[0]
                   
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
