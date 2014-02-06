[<AutoOpen>]
module Alea.cuExtension.DataStructures


open Alea.CUDA



type long = nativeint
type ulong = unativeint
type double = float
type longlong = int64
type ulonglong = uint64


[<Struct>]
type short4 =
    val mutable x : sbyte
    val mutable y : sbyte
    val mutable z : sbyte
    val mutable w : sbyte

    [<ReflectedDefinition>]
    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w

[<Struct>]
type ushort4 =
    val mutable x : byte
    val mutable y : byte
    val mutable z : byte
    val mutable w : byte

    [<ReflectedDefinition>]
    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w

[<Struct>]
type int2 =
    val mutable x : int
    val mutable y : int

    [<ReflectedDefinition>]
    new (x, y) = {x = x; y = y}
    override this.ToString() = sprintf "(%d,%d)" this.x this.y

[<Struct>]
type uint2 =
    val mutable x : uint32
    val mutable y : uint32
    
    [<ReflectedDefinition>]
    new (x, y) = {x = x; y = y}
    override this.ToString() = sprintf "(%d,%d)" this.x this.y

[<Struct>]
type long2 =
    val mutable x : long
    val mutable y : long

    [<ReflectedDefinition>]
    new (x, y) = {x = x; y = y}
    override this.ToString() = sprintf "(%d,%d)" this.x this.y

[<Struct>]
type ulong2 =
    val mutable x : ulong
    val mutable y : ulong

    [<ReflectedDefinition>]
    new (x, y) = {x = x; y = y}
    override this.ToString() = sprintf "(%d,%d)" this.x this.y

[<Struct>]
type float2 =
    val mutable x : float32
    val mutable y : float32
    
    [<ReflectedDefinition>]
    new (x, y) = {x = x; y = y}
    override this.ToString() = sprintf "(%f,%f)" this.x this.y

[<Struct>]
type int4 =
    val mutable x : int
    val mutable y : int
    val mutable z : int
    val mutable w : int

    [<ReflectedDefinition>]
    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w

[<Struct>]
type uint4 =
    val mutable x : uint32
    val mutable y : uint32
    val mutable z : uint32
    val mutable w : uint32

    [<ReflectedDefinition>]
    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w

[<Struct>]
type float4 =
    val mutable x : float32
    val mutable y : float32
    val mutable z : float32
    val mutable w : float32
    
    [<ReflectedDefinition>]
    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
    override this.ToString() = sprintf "(%f,%f,%f,%f)" this.x this.y this.z this.w

[<Struct>]
type long4 =
    val mutable x : long
    val mutable y : long
    val mutable z : long
    val mutable w : long

    [<ReflectedDefinition>]
    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w

[<Struct>]
type ulong4 =
    val mutable x : ulong
    val mutable y : ulong
    val mutable z : ulong
    val mutable w : ulong

    [<ReflectedDefinition>]
    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w

[<Struct>]
type longlong2 =
    val mutable x : longlong
    val mutable y : longlong

    [<ReflectedDefinition>]
    new (x, y) = {x = x; y = y}
    override this.ToString() = sprintf "(%d,%d)" this.x this.y

[<Struct>]
type ulonglong2 =
    val mutable x : ulonglong
    val mutable y : ulonglong

    [<ReflectedDefinition>]
    new (x, y) = {x = x; y = y}
    override this.ToString() = sprintf "(%d,%d)" this.x this.y

[<Struct>]
type double2 =
    val mutable x : double
    val mutable y : double

    [<ReflectedDefinition>]
    new (x, y) = {x = x; y = y}
    override this.ToString() = sprintf "(%f,%f)" this.x this.y

[<Struct>]
type longlong4 =
    val mutable x : longlong
    val mutable y : longlong
    val mutable z : longlong
    val mutable w : longlong

    [<ReflectedDefinition>]
    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w

[<Struct>]
type ulonglong4 =
    val mutable x : ulonglong
    val mutable y : ulonglong
    val mutable z : ulonglong
    val mutable w : ulonglong

    [<ReflectedDefinition>]
    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w

[<Struct>]
type double4 =
    val mutable x : double
    val mutable y : double
    val mutable z : double
    val mutable w : double

    [<ReflectedDefinition>]
    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
    override this.ToString() = sprintf "(%f,%f,%f,%f)" this.x this.y this.z this.w