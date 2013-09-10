module Alea.CUDA.TestUtilities.TestStruct

open System
open System.Runtime.InteropServices
open Microsoft.FSharp.Quotations
open Alea.CUDA
open NUnit.Framework

#nowarn "9"

// x    y    z
// xxxx xxxx xxxx
[<Struct>]
type Int3 =
    val mutable x : int
    val mutable y : int
    val mutable z : int

    [<ReflectedDefinition>]
    new (x, y, z) = { x = x; y = y; z = z }

    override this.ToString() = sprintf "(%d,%d,%d)" this.x this.y this.z

// x    y    z
// xxxx xxxx xxxx
[<Struct>]
type ConstInt3 =
    val x : int
    val y : int
    val z : int

    [<ReflectedDefinition>]
    new (x, y, z) = { x = x; y = y; z = z }

    override this.ToString() = sprintf "(%d,%d,%d)" this.x this.y this.z

(*
struct __align__(8) Int3A8 {
    int x; int y; int z;
    __device__ Int3A8() {}
    __device__ Int3A8(int _x, int _y, int _z) {
        x = _x; y = _y; z = _z;
    }
};
*)
// x    y    z
// xxxx xxxx xxxx oooo
[<Struct;StructLayout(LayoutKind.Sequential, Size=16);Align(8)>]
type Int3A8 =
    val mutable x : int
    val mutable y : int
    val mutable z : int

    [<ReflectedDefinition>]
    new (x, y, z) = { x = x; y = y; z = z }

    override this.ToString() = sprintf "(%d,%d,%d)" this.x this.y this.z

    [<ReflectedDefinition>]
    static member (+) (lhs:Int3A8, rhs:Int3A8) =
        let x = lhs.x + rhs.x
        let y = lhs.y + rhs.y
        let z = lhs.z + rhs.z
        Int3A8(x, y, z)

    [<ReflectedDefinition>]
    static member (-) (lhs:Int3A8, rhs:Int3A8) =
        let x = lhs.x - rhs.x
        let y = lhs.y - rhs.y
        let z = lhs.z - rhs.z
        Int3A8(x, y, z)

let genRandomInt3A8 minv maxv _ =
    let x = TestUtil.rng.Next(minv, maxv)
    let y = TestUtil.rng.Next(minv, maxv)
    let z = TestUtil.rng.Next(minv, maxv)
    Int3A8(x, y, z)

// x    y
// xxxx xxxx
[<Struct>]
type IntSingle =
    val mutable x : int
    val mutable y : float32

    [<ReflectedDefinition>]
    new (x, y) = { x = x; y = y }

    override this.ToString() = sprintf "(%d,%f)" this.x this.y

    [<ReflectedDefinition>]
    static member (+) (lhs:IntSingle, rhs:IntSingle) =
        IntSingle(lhs.x + rhs.x, lhs.y + rhs.y)

let genRandomIntSingle minv maxv _ =
    let x = TestUtil.genRandomSInt32 (minv |> int) (maxv |> int) ()
    let y = TestUtil.genRandomSingle minv maxv ()
    IntSingle(x, y)

// x         y
// xxxx oooo xxxx xxxx
[<Struct>]
type IntDouble =
    val mutable x : int
    val mutable y : float

    [<ReflectedDefinition>]
    new (x, y) = { x = x; y = y }

    override this.ToString() = sprintf "(%d,%f)" this.x this.y

    [<ReflectedDefinition>]
    static member (+) (lhs:IntDouble, rhs:IntDouble) =
        IntDouble(lhs.x + rhs.x, lhs.y + rhs.y)

let genRandomIntDouble minv maxv _ =
    let x = TestUtil.genRandomSInt32 (minv |> int) (maxv |> int) ()
    let y = TestUtil.genRandomDouble minv maxv ()
    IntDouble(x, y)

// x    y
// xxoo xxxx
[<Struct>]
type ShortSingle =
    val mutable x : int16
    val mutable y : float32

    [<ReflectedDefinition>]
    new (x, y) = { x = x; y = y }

    override this.ToString() = sprintf "(%d,%f)" this.x this.y

let genRandomShortSingle minv maxv _ =
    let x = TestUtil.genRandomSInt16 (minv |> int) (maxv |> int) ()
    let y = TestUtil.genRandomSingle minv maxv ()
    ShortSingle(x, y)

// x    y
// xxoo xxxx
[<Struct>]
type ConstShortSingle =
    val x : int16
    val y : float32

    [<ReflectedDefinition>]
    new (x, y) = { x = x; y = y }

    override this.ToString() = sprintf "(%d,%f)" this.x this.y

let genRandomConstShortSingle minv maxv _ =
    let x = TestUtil.genRandomSInt16 (minv |> int) (maxv |> int) ()
    let y = TestUtil.genRandomSingle minv maxv ()
    ConstShortSingle(x, y)

