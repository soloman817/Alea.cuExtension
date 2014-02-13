[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Scan

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common


//type IScanOp<'T> =
//    abstract hplus : ('T -> 'T ->'T)
//    abstract dplus : Expr<'T -> 'T -> 'T>

//[<ReflectedDefinition>]
//let inline scan_op<'T>() = (+) //  (ident:'T) =
////    { new IScanOp<'T> with
////        member this.hplus = (+)
////        member this.dplus = <@ (+) @>}

//let inline scan_op (ident:'T) = 
//    { new IScanOp<'T> with
//        member this.hplus = (+)
//        member this.dplus = <@ (+) @>}

[<Record>]
type TemplateParameters<'T> =
    {
        mutable LENGTH : int
    }

    [<ReflectedDefinition>]
    static member Init(length:int) =
        {
            LENGTH = length
        }


[<Record>]    
type ThreadScan<'T> =
    {
        TemplateParameters : TemplateParameters<'T>
    }

    [<ReflectedDefinition>]
    member this.Initialize(length:int) =
        this.TemplateParameters.LENGTH <- length
        this

    [<ReflectedDefinition>]
    member this.Exclusive(inclusive:'T, exclusive:'T, input:deviceptr<'T>, output:deviceptr<'T>, scan_op:('T -> 'T -> 'T)) = //, length:bool) =
        
        // localize template params & constants
        let LENGTH = this.TemplateParameters.LENGTH

        let mutable inclusive = inclusive
        let mutable exclusive = exclusive
        let mutable input = input
        let mutable output = output

        for i = this.TemplateParameters.LENGTH to 0 do
            let addend = input.[0]
            inclusive <- (exclusive, addend) ||> scan_op
            output.[0] <- exclusive
            exclusive <- inclusive

            input <- input + 1
            output <- output + 1

        inclusive

    [<ReflectedDefinition>]
    member this.Exclusive(input:deviceptr<'T>, output:deviceptr<'T>, scan_op:('T -> 'T -> 'T), prefix:'T, ?apply_prefix:bool) = //, length:bool) =
        let apply_prefix = if apply_prefix.IsSome then apply_prefix.Value else true

        let mutable inclusive = input.[0]
        if apply_prefix then inclusive <- (prefix, inclusive) ||> scan_op

        output.[0] <- prefix
        let exclusive = inclusive
        
        this.TemplateParameters.LENGTH <- this.TemplateParameters.LENGTH - 1    
        this.Exclusive(inclusive, exclusive, input + 1, output + 1, scan_op)


    [<ReflectedDefinition>]
    member this.Inclusive(inclusive:'T, input:deviceptr<'T>, output:deviceptr<'T>, scan_op:('T -> 'T -> 'T)) = //, length:bool
        // localize template params & constants
        let LENGTH = this.TemplateParameters.LENGTH
        
        let mutable inclusive = inclusive

        for i = 0 to LENGTH do 
            let addend = input.[i]
            inclusive <- (inclusive, addend) ||> scan_op
            output.[i] <- inclusive
        

    [<ReflectedDefinition>]
    member this.Inclusive(input:deviceptr<'T>, output:deviceptr<'T>, scan_op:('T -> 'T -> 'T)) =
        let inclusive = input.[0]
        output.[0] <- inclusive

        this.Inclusive(inclusive, input, output + 1, scan_op)

    [<ReflectedDefinition>]
    member this.Inclusive(input:deviceptr<'T>, output:deviceptr<'T>, scan_op:('T -> 'T -> 'T), prefix:'T, ?apply_prefix:bool) =
        let apply_prefix = if apply_prefix.IsSome then apply_prefix.Value else true

        let mutable inclusive = input.[0]
        if apply_prefix then inclusive <- (prefix, inclusive) ||> scan_op

        output.[0] <- inclusive

        this.TemplateParameters.LENGTH <- this.TemplateParameters.LENGTH - 1
        this.Inclusive(inclusive, input + 1, output + 1, scan_op)



    [<ReflectedDefinition>]
    static member Create<'T>(templateParameters:TemplateParameters<'T>) =
        {
            TemplateParameters = templateParameters
        }

    [<ReflectedDefinition>]
    static member Create(length:int) =
        let tp = TemplateParameters<'T>.Init(length + 1)
        {
            TemplateParameters = tp
        }



//let inclusiveScan length = 
//    fun (inclusive:'T option) (input:deviceptr<'T>) (output:deviceptr<'T>) (scan_op:('T -> 'T -> 'T)) (prefix:'T option) (apply_prefix:bool option) ->
//        match inclusive, prefix, apply_prefix with
//        | None, None, None ->
            
let inline ThreadScanExclusive<'T>(length:int) = 
    fun (input:deviceptr<'T>) (output:deviceptr<'T>) (scan_op:('T -> 'T -> 'T)) (prefix:'T) (apply_prefix:bool option) ->
        let apply_prefix = if apply_prefix.IsSome then apply_prefix.Value else true

        let mutable inclusive = input.[0]
        if apply_prefix then inclusive <- (prefix, inclusive) ||> scan_op

        output.[0] <- prefix
        let mutable exclusive = inclusive

        for i = 0 to length do
            let addend = input.[i]
            inclusive <- (exclusive, addend) ||> scan_op
            output.[i] <- exclusive
            exclusive <- inclusive

        inclusive

