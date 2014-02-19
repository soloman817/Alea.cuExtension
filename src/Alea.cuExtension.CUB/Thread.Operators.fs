[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Operators

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common



let [<ReflectedDefinition>] inequality = (<>)
let [<ReflectedDefinition>] equality = (=)
let [<ReflectedDefinition>] inline sum() x y = x + y
let [<ReflectedDefinition>] max = max
let [<ReflectedDefinition>] min = min


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
