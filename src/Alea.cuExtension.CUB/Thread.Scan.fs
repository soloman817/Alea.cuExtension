﻿[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Scan

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common


module ThreadScanExclusive =
    let [<ReflectedDefinition>] inline DefaultInt (length:int)
        (inclusive:int) (exclusive:int) (input:deviceptr<int>) (output:deviceptr<int>) =
        let mutable addend = input.[0]
        let mutable inclusive = addend + exclusive
        output.[0] <- exclusive
        let mutable exclusive = inclusive

        for i = 1 to (length - 1) do
            addend <- input.[i]
            inclusive <- exclusive + addend
            output.[i] <- exclusive
            exclusive <- inclusive

        inclusive
    

    let [<ReflectedDefinition>] inline WithApplyPrefixDefaultInt (length:int)
        (input:deviceptr<int>) (output:deviceptr<int>) (prefix:int) =
        let apply_prefix = true
        let mutable inclusive = input.[0]
        if apply_prefix then inclusive <- prefix + inclusive

        output.[0] <- prefix
        let exclusive = inclusive

        DefaultInt length inclusive exclusive input output
        
        

    let [<ReflectedDefinition>] inline WithApplyPrefixInt (length:int)
        (input:deviceptr<int>) (output:deviceptr<int>) (prefix:int) (apply_prefix:bool) =
        let mutable inclusive = input.[0]
        if apply_prefix then inclusive <- prefix + inclusive

        output.[0] <- prefix
        let exclusive = inclusive
        
        DefaultInt length inclusive exclusive input output




    let [<ReflectedDefinition>] inline Default (length:int) (scan_op:'T -> 'T -> 'T)
        (inclusive:'T) (exclusive:'T) (input:deviceptr<'T>) (output:deviceptr<'T>) =
        let mutable addend = input.[0]
        let mutable inclusive = inclusive
        output.[0] <- exclusive
        let mutable exclusive = inclusive

        for i = 1 to (length - 1) do
            addend <- input.[i]
            inclusive <- (exclusive, addend) ||> scan_op
            output.[i] <- exclusive
            exclusive <- inclusive

        inclusive
    

    let [<ReflectedDefinition>] inline WithApplyPrefixDefault (length:int) (scan_op:'T -> 'T -> 'T)
        (input:deviceptr<'T>) (output:deviceptr<'T>) (prefix:'T) =
        let apply_prefix = true
        let mutable inclusive = input.[0]
        if apply_prefix then inclusive <- (prefix, inclusive) ||> scan_op

        output.[0] <- prefix
        let exclusive = inclusive

        Default length scan_op inclusive exclusive input output
        
        

    let [<ReflectedDefinition>] inline WithApplyPrefix (length:int) (scan_op:'T -> 'T -> 'T)
        (input:deviceptr<'T>) (output:deviceptr<'T>) (prefix:'T) (apply_prefix:bool) =
        
        let mutable inclusive = input.[0]
        if apply_prefix then inclusive <- (prefix, inclusive) ||> scan_op

        output.[0] <- prefix
        let exclusive = inclusive
        
        Default length scan_op inclusive exclusive input output
        


module ThreadScanInclusive =
    
    let [<ReflectedDefinition>] inline Default (length:int) (scan_op:'T -> 'T -> 'T)
        (inclusive:'T) (input:deviceptr<'T>) (output:deviceptr<'T>) =
        let mutable addend = input.[0]
        let mutable inclusive = (inclusive, addend) ||> scan_op
        output.[0] <- inclusive

        for i = 1 to length - 1 do
            addend <- input.[i]
            inclusive <- (inclusive, addend) ||> scan_op

        inclusive
        

    let [<ReflectedDefinition>] inline ReturnAggregate (length:int) (scan_op:'T -> 'T -> 'T)
        (input:deviceptr<'T>) (output:deviceptr<'T>) =
        let inclusive = input.[0]
        output.[0] <- inclusive
        Default length scan_op inclusive (input + 1) (output + 1)
        

    let [<ReflectedDefinition>] inline WithApplyPrefix (length:int) (scan_op:'T -> 'T -> 'T)
        (input:deviceptr<'T>) (output:deviceptr<'T>) (prefix:'T) (apply_prefix:bool) =
        let mutable inclusive = input.[0]
        if apply_prefix then inclusive <- (prefix,inclusive) ||> scan_op
        output.[0] <- inclusive
        ReturnAggregate length scan_op (input + 1) (output + 1)
        

    let [<ReflectedDefinition>] inline WithApplyPrefixDefault (length:int) (scan_op:'T -> 'T -> 'T)
        (input:deviceptr<'T>) (output:deviceptr<'T>) (prefix:'T) =
        WithApplyPrefix length scan_op input output prefix true
        

//    let [<ReflectedDefinition>] api (length:int) (scan_op:'T -> 'T -> 'T) : API<'T> =
//        lettemplate= _TemplateParams<'T>.Init(length) //, scan_op)
//        {
//            Default                 =   Default length scan_op
//            WithApplyPrefixDefault  =   WithApplyPrefixDefault length scan_op
//            WithApplyPrefix         =   WithApplyPrefix length scan_op
//        }

//    [<Record>]
//    type API3<'T> =
//        {
//            TemplateParams : _TemplateParams<'T>
//        }
//
//        [<ReflectedDefinition>]
//        static member Init(length) =
//            {
//                TemplateParams = _TemplateParams<'T>.Init(length)
//            }
//
//        [<ReflectedDefinition>]
//        member inline this.Default(scan_op:IScanOp<_>, input:deviceptr<_>, output:deviceptr<_>, prefix:'T) =
//            Default (this.TemplateParams.LENGTH) (scan_op.op) 
//                0 0 input output prefix
//    let api2 (length:int) : Template<API2<'T>> = cuda {
//        lettemplate= _TemplateParams<'T>.Init(length) //, scan_op)
//        let! _Default = <@ Defaulttemplate  |> Compiler.DefineFunction
//        let! _WithApplyPrefixDefault = <@ WithApplyPrefixDefaulttemplate  |> Compiler.DefineFunction
//        let! _WithApplyPrefix = <@ WithApplyPrefixtemplate  |> Compiler.DefineFunction
//
//        return {Default = _Default; WithApplyPrefixDefault = _WithApplyPrefixDefault; WithApplyPrefix = _WithApplyPrefix}
//    }


//module ThreadScanInclusive = ()
//type IScanOp =
//    abstract hplus : ('T -> 'T ->'T)
//    abstract dplus : Expr<'T -> 'T -> 'T>

//[<ReflectedDefinition>]
//let [<ReflectedDefinition>] inline scan_op() = (+) //  (ident:int) =
////    { new IScanOp<int> with
////        member this.hplus = (+)
////        member this.dplus = <@ (+) }

//let [<ReflectedDefinition>] inline scan_op (ident:int) = 
//    { new IScanOp<int> with
//        member this.hplus = (+)
//        member this.dplus = <@ (+) }



//
//[<Record>]    
//type ThreadScan =
//    {
//        mutable LENGTH : int
//    }
//
//    [<ReflectedDefinition>]
//    member this.Initialize(length:int) =
//        this.LENGTH <- length
//        this
//
//    [<ReflectedDefinition>]
//    member this.Exclusive(inclusive:int, exclusive:int, input:deviceptr<int>, output:deviceptr<int>, scan_op:(int -> int -> int)) = //, length:bool) =
//        // localize template params & constants
//        let LENGTH = this.LENGTH
//
//        let mutable inclusive = inclusive
//        let mutable exclusive = exclusive
//        let mutable input = input
//        let mutable output = output
//
//        for i = LENGTH to 0 do
//            let addend = input.[0]
//            inclusive <- (exclusive, addend) ||> scan_op
//            output.[0] <- exclusive
//            exclusive <- inclusive
//
//            input <- input + 1
//            output <- output + 1
//
//        inclusive
//
//    [<ReflectedDefinition>]
//    member this.Exclusive(input:deviceptr<int>, output:deviceptr<int>, scan_op:(int -> int -> int), prefix:int, ?apply_prefix:bool) = //, length:bool) =
//        let apply_prefix = if apply_prefix.IsSome then apply_prefix.Value else true
//
//        let mutable inclusive = input.[0]
//        if apply_prefix then inclusive <- (prefix, inclusive) ||> scan_op
//
//        output.[0] <- prefix
//        let exclusive = inclusive
//        
//        this.LENGTH <- this.LENGTH - 1    
//        this.Exclusive(inclusive, exclusive, input + 1, output + 1, scan_op)
//
//
//    [<ReflectedDefinition>]
//    member this.Inclusive(inclusive:int, input:deviceptr<int>, output:deviceptr<int>, scan_op:(int -> int -> int)) = //, length:bool
//        // localize template params & constants
//        let LENGTH = this.LENGTH
//        
//        let mutable inclusive = inclusive
//
//        for i = 0 to LENGTH do 
//            let addend = input.[i]
//            inclusive <- (inclusive, addend) ||> scan_op
//            output.[i] <- inclusive
//        
//
//    [<ReflectedDefinition>]
//    member this.Inclusive(input:deviceptr<int>, output:deviceptr<int>, scan_op:(int -> int -> int)) =
//        let inclusive = input.[0]
//        output.[0] <- inclusive
//
//        this.Inclusive(inclusive, input, output + 1, scan_op)
//
//    [<ReflectedDefinition>]
//    member this.Inclusive(input:deviceptr<int>, output:deviceptr<int>, scan_op:(int -> int -> int), prefix:int, ?apply_prefix:bool) =
//        let apply_prefix = if apply_prefix.IsSome then apply_prefix.Value else true
//
//        let mutable inclusive = input.[0]
//        if apply_prefix then inclusive <- (prefix, inclusive) ||> scan_op
//
//        output.[0] <- inclusive
//
//        this.LENGTH <- this.LENGTH - 1
//        this.Inclusive(inclusive, input + 1, output + 1, scan_op)
//
//
//    [<ReflectedDefinition>]
//    static member Create(length:int) =
//        {
//            LENGTH = length + 1
//        }
//
//
//
////let inclusiveScan length = 
////    fun (inclusive:int option) (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:(int -> int -> int)) (prefix:int option) (apply_prefix:bool option) =
////        match inclusive, prefix, apply_prefix with
////        | None, None, None ->
//            
//let [<ReflectedDefinition>] inline ThreadScanExclusive(length:int) = 
//    fun (input:deviceptr<int>) (output:deviceptr<int>) (scan_op:(int -> int -> int)) (prefix:int) (apply_prefix:bool option) =
//        let apply_prefix = if apply_prefix.IsSome then apply_prefix.Value else true
//
//        let mutable inclusive = input.[0]
//        if apply_prefix then inclusive <- (prefix, inclusive) ||> scan_op
//
//        output.[0] <- prefix
//        let mutable exclusive = inclusive
//
//        for i = 0 to length do
//            let addend = input.[i]
//            inclusive <- (exclusive, addend) ||> scan_op
//            output.[i] <- exclusive
//            exclusive <- inclusive
//
//        inclusive

