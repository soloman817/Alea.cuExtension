[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Reduce

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common


let threadReduce length =
    fun (_length:int option) ->
        fun (input:deviceptr<int>) (reduction_op:(int -> int -> int)) (prefix:int)  ->
            let mutable prefix = prefix
            let length = if _length.IsSome then _length.Value else length
        
            for i = 0 to length - 1 do
                let addend = input.[0]
                prefix <- (prefix, addend) ||> reduction_op

            prefix


let ThreadReduce length =
    let threadReduce = length |> threadReduce
    fun (input:deviceptr<int>) (reduction_op:(int -> int -> int)) (prefix:int option) ->
        prefix |> function
        | Some prefix ->
            threadReduce
            <|      None
            <|||    (input, reduction_op, prefix)

        | None ->
            let prefix = input.[0]
            threadReduce
            <|      Some(length - 1)
            <|||    (input, reduction_op, prefix)
        
