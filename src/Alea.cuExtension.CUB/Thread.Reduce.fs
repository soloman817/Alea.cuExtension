﻿[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Reduce

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Thread

module Template =
    [<AutoOpen>]
    module Params =
        [<Record>]
        type API<'T> =
            {
                LENGTH          : int
                reduction_op    : IReductionOp<'T>
            }

            [<ReflectedDefinition>]
            static member Init(length, reduction_op) =
                {
                    LENGTH          = length
                    reduction_op    = reduction_op
                }

    type _TemplateParams<'T> = Params.API

module ThreadReduce =
    open Template

    [<Record>]
    type API<'T> =
        {
            Default : deviceptr<'T> -> 'T
            WithPrefix : deviceptr<'T> -> 'T -> 'T
        }

    let [<ReflectedDefinition>] inline WithPrefix (template:_Template<'T>)
        (input:deviceptr<'T>) (prefix:'T) =
        let reduction_op = template.reduction_op.op
        let mutable addend = input.[0]
        let mutable prefix = (prefix, addend) ||> reduction_op

        for i = 1 to template.LENGTH - 1 do
            addend <- input.[i]
            prefix <- (prefix, addend) ||> reduction_op

        prefix

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (input:deviceptr<'T>) =
        WithPrefixtemplateinput input.[0]

    let [<ReflectedDefinition>] api (length:int) (reduction_op:IReductionOp<'T>) =
        lettemplate= _TemplateParams<'T>.Init(length, reduction_op)
        {
            Default     = Default template
            WithPrefix  = WithPrefix template
        }
            
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
