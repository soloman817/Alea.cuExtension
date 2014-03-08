[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Reduce

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Thread

module Template =
    [<AutoOpen>]
    module Params =
        [<Record>]
        type API =
            {
                LENGTH          : int
            }

            [<ReflectedDefinition>]
            static member Init(length) =
                {
                    LENGTH          = length
                }

        

    type _TemplateParams = Params.API

type _Template = Template._TemplateParams


module ThreadReduce =
    open Template

    let [<ReflectedDefinition>] inline WithPrefix (template:_Template)
        (reduction_op:'T -> 'T -> 'T)
        (input:deviceptr<'T>) (prefix:'T) =
        let mutable addend = input.[0]
        let mutable prefix = (prefix, addend) ||> reduction_op

        for i = 1 to template.LENGTH - 1 do
            addend <- input.[i]
            prefix <- (prefix, addend) ||> reduction_op

        prefix

    let [<ReflectedDefinition>] inline Default (template:_Template)
        (reduction_op:'T -> 'T -> 'T)
        (input:deviceptr<'T>) =
        WithPrefix template reduction_op input input.[0]


    [<Record>]
    type API<'T> =
        {
            template : _Template
        }

        [<ReflectedDefinition>] static member Init(length) = { template = _Template.Init(length) }

        [<ReflectedDefinition>] member this.Default     = Default this.template
        [<ReflectedDefinition>] member this.WithPrefix  = WithPrefix this.template
         
                    
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
