module Alea.CUDA.Utilities.TestUtil

open System
open System.Runtime.InteropServices
open Microsoft.FSharp.Quotations
open Alea.CUDA
open NUnit.Framework

#nowarn "9"

let rng = Random(2)

let genRandomBool _ = let n = rng.Next(0, 2) in if n = 0 then false else true
let genRandomSInt8 minv maxv _ = rng.Next(minv, maxv) |> int8
let genRandomUInt8 minv maxv _ = if minv < 0 || maxv < 0 then failwith "minv or maxv < 0" else rng.Next(minv, maxv) |> uint8
let genRandomSInt16 minv maxv _ = rng.Next(minv, maxv) |> int16
let genRandomUInt16 minv maxv _ = if minv < 0 || maxv < 0 then failwith "minv or maxv < 0" else rng.Next(minv, maxv) |> uint16
let genRandomSInt32 minv maxv _ = rng.Next(minv, maxv)
let genRandomUInt32 minv maxv _ = if minv < 0 || maxv < 0 then failwith "minv or maxv < 0" else rng.Next(minv, maxv) |> uint32
let genRandomSInt64 minv maxv _ = rng.Next(minv, maxv) |> int64
let genRandomUInt64 minv maxv _ = if minv < 0 || maxv < 0 then failwith "minv or maxv < 0" else rng.Next(minv, maxv) |> uint64
let genRandomDouble minv maxv _ = rng.NextDouble() * (maxv - minv) + minv
let genRandomSingle minv maxv _ = (rng.NextDouble() * (maxv - minv) + minv) |> float32

let assertArrayEqual (eps:float option) (A:'T[]) (B:'T[]) =
    (A, B) ||> Array.iter2 (fun a b -> eps |> function
        | None -> Assert.AreEqual(a, b)
        | Some eps -> Assert.That(b, Is.EqualTo(a).Within(eps)))

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

let genRandomInt3A8 minv maxv _ = Int3A8(rng.Next(minv, maxv), rng.Next(minv, maxv), rng.Next(minv, maxv))
