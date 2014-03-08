[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Operators

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common




//type Op<'T> = 'T -> 'T -> 'T
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
//type IScanOp<'T> =
//    abstract op : Expr<'T -> 'T -> 'T>

type IScanOp<'T> =
    abstract op : ('T -> 'T -> 'T)

type IReductionOp<'T> = IScanOp<'T>

//let inline op (id:int) =
//    { new IScanOp<int> with
//        member this.add = <@ (+) @>
//        member this.sub = <@ (-) @>
//        member this.mul = <@ (*) @>
//        member this.div = <@ (/) @>
//        member this.min = <@ min @>
//        member this.max = <@ max @>
//    }

let foo x y = x + y

let inline scan_op (opkind:OpKind) (id:'T) =
    opkind |> function
    | ADD -> { new IScanOp<'T> with member this.op = fun (x:'T) (y:'T) -> x + y }
    | SUB -> { new IScanOp<'T> with member this.op = fun (x:'T) (y:'T) -> x - y }
    | MUL -> { new IScanOp<'T> with member this.op = fun (x:'T) (y:'T) -> x * y }
    | DIV -> { new IScanOp<'T> with member this.op = fun (x:'T) (y:'T) -> x / y }
    | MIN -> { new IScanOp<'T> with member this.op = fun (x:'T) (y:'T) -> min x y }
    | MAX -> { new IScanOp<'T> with member this.op = fun (x:'T) (y:'T) -> max x y }


//let inline scan_op (opkind:OpKind) (id:'T) =
//    opkind |> function
//    | ADD -> { new IScanOp<'T> with member this.op = <@ fun (x:'T) (y:'T) -> x + y @> }
//    | SUB -> { new IScanOp<'T> with member this.op = <@ fun (x:'T) (y:'T) -> x + y @> }
//    | MUL -> { new IScanOp<'T> with member this.op = <@ fun (x:'T) (y:'T) -> x + y @> }
//    | DIV -> { new IScanOp<'T> with member this.op = <@ fun (x:'T) (y:'T) -> x + y @> }
//    | MIN -> { new IScanOp<'T> with member this.op = <@ fun (x:'T) (y:'T) -> x + y @> }
//    | MAX -> { new IScanOp<'T> with member this.op = <@ fun (x:'T) (y:'T) -> x + y @> }

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
