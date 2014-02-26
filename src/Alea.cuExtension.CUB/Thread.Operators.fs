[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Operators

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common




type Op = int -> int -> int
//    
type OpKind = 
    | ADD
    | SUB
    | MUL
    | DIV
    | MIN
    | MAX

let Sum() = <@ fun a b -> a + b @>

//type IScanOp<int> =
//    abstract add : Op<int>
//    abstract sub : Op<int>
//    abstract mul : Op<int>
//    abstract div : Op<int>
//    abstract min : Op<int>
//    abstract max : Op<int>
type IScanOp =
    abstract op : Expr<Op>

type IReductionOp = IScanOp    

//let inline op (id:int) =
//    { new IScanOp<int> with
//        member this.add = <@ (+) @>
//        member this.sub = <@ (-) @>
//        member this.mul = <@ (*) @>
//        member this.div = <@ (/) @>
//        member this.min = <@ min @>
//        member this.max = <@ max @>
//    }

let inline scan_op (opkind:OpKind) (id:int) =
    opkind |> function
    | ADD -> { new IScanOp with member this.op = <@ (+) @> }
    | SUB -> { new IScanOp with member this.op = <@ (-) @> }
    | MUL -> { new IScanOp with member this.op = <@ (*) @> }
    | DIV -> { new IScanOp with member this.op = <@ (/) @> }
    | MIN -> { new IScanOp with member this.op = <@ min @> }
    | MAX -> { new IScanOp with member this.op = <@ max @> }

//[<Record>]
//type Equality<'T when 'T : equality> =
//    {
//        a : int
//        b : int
//    }
//
//    member this.Op = this.a = this.b
//    member this.Init(a,b) = {a = a; b = b} 
//
//[<Record>]
//type Inequality<'T when 'T : equality> =
//    {
//        a : int
//        b : int
//    }
//
//    member this.Op = this.a = this.b
//    member this.Init(a,b) = {a = a; b = b} 
//
//type EqualityOp =
//    | Equality of 'T * 'T
//    | Inequality of 'T * 'T
//
//[<Record>]
//type InequalityWrapper =
//    {
//        op : EqualityOp<int>
//    }



//type ThreadOperators<'T when 'T : equality> =
//    {
//        a : int
//        b : int
//    }
//
//    member this.Equality : bool = this.a = this.b
//    member this.Inequality : bool = this.a <> this.b
//
