[<AutoOpen>]
module Alea.cuExtension.CUB.Utilities.Type

open System
open Microsoft.FSharp.Core.Operators

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension
open Alea.cuExtension.CUB.Common


type InputIterator = deviceptr<int>
type OutputIterator = deviceptr<int>


//[<Record>]
//type NullType =
//    {
//        NULL : deviceptr<int>
//    }
//
//    static member (=) (_null:NullType<int>, b:int) = __null()
//    static member (==) (_null:NullType<int>, b:NullType<int>)

type Pad =
    {
        v : int
        b : byte
    }

    static member Init(value:int) =
        {
            v = value
            b = 0uy
        }

type AlignBytes =
    {
        ALIGN_BYTES : int
    }

    static member Init(value:int) =
        {
            ALIGN_BYTES = sizeof<Pad> - sizeof<int>
        }

let alignBytes() : AlignBytes =
    typeof<int> |> function 
        | ty when ty = typeof<short4>       -> { ALIGN_BYTES = 8 }
        | ty when ty = typeof<ushort4>      -> { ALIGN_BYTES = 8 }
        | ty when ty = typeof<int2>         -> { ALIGN_BYTES = 8 }
        | ty when ty = typeof<uint2>        -> { ALIGN_BYTES = 8 }

        | ty when ty = typeof<long2>        -> { ALIGN_BYTES = 8 }
        | ty when ty = typeof<ulong2>       -> { ALIGN_BYTES = 8 }
            
        | ty when ty = typeof<longlong>     -> { ALIGN_BYTES = 8 }
        | ty when ty = typeof<ulonglong>    -> { ALIGN_BYTES = 8 }
        | ty when ty = typeof<float2>       -> { ALIGN_BYTES = 8 }
        | ty when ty = typeof<double>       -> { ALIGN_BYTES = 8 }

        | ty when ty = typeof<int4>         -> { ALIGN_BYTES = 16 }
        | ty when ty = typeof<uint4>        -> { ALIGN_BYTES = 16 }
        | ty when ty = typeof<float4>       -> { ALIGN_BYTES = 16 }

        | ty when ty = typeof<long4>        -> { ALIGN_BYTES = 16 }
        | ty when ty = typeof<ulong4>       -> { ALIGN_BYTES = 16 }
        | ty when ty = typeof<longlong2>    -> { ALIGN_BYTES = 16 }
        | ty when ty = typeof<ulonglong2>   -> { ALIGN_BYTES = 16 }
        | ty when ty = typeof<double2>      -> { ALIGN_BYTES = 16 }
        | ty when ty = typeof<longlong4>    -> { ALIGN_BYTES = 16 }
        | ty when ty = typeof<ulonglong4>   -> { ALIGN_BYTES = 16 }
        | ty when ty = typeof<double4>      -> { ALIGN_BYTES = 16 }

        | _ -> { ALIGN_BYTES = 8 }

[<Struct>]
type IsMultiple =
    val UNIT_ALIGN_BYTES : int
    val IS_MULTIPLE : bool
    new (align_bytes) =
        let unit_align_bytes = AlignBytes.Init(align_bytes).ALIGN_BYTES
        {
            UNIT_ALIGN_BYTES = unit_align_bytes
            IS_MULTIPLE = ((sizeof<int> % sizeof<unit>) = 0) && ((align_bytes % unit_align_bytes) = 0)
        }

[<Struct>]
type UnitWordStr =
    val ALIGN_BYTES : int
//        
//        ShuffleWord : 'ShuffleWord
//        VolatileWord : 'VolatileWord
//        DeviceWord : 'DeviceWord
//        TextureWord : intextureWord
    val IS_MULTIPLE : bool

    [<ReflectedDefinition>]
    new (_) = 
        let align_bytes = alignBytes().ALIGN_BYTES
        {
            ALIGN_BYTES = align_bytes
            IS_MULTIPLE = IsMultiple(align_bytes).IS_MULTIPLE
        }

module UnitWord =
    type ShuffleWordAttribute() =
        inherit Attribute()

        interface ICustomCallBuilder with
            member this.Build(ctx, irObject, info, irParams) =
                match irObject, info.Name, irParams with
                | Some(_), "Of", irValue :: [] ->
                    let clrParamType = info.GetParameters().[0].ParameterType
                    let irParamType = IRTypeBuilder.Instance.Build(ctx, clrParamType)
                    let clrRealType = info.ReturnType
                    let irRealType = IRTypeBuilder.Instance.Build(ctx, clrRealType)
                    irRealType |> function
                    | irRealType when irRealType.IsFloatingPoint -> irRealType.FloatingPoint.Kind |> function
                        | FloatingPointKind.Double -> irParamType |> function
                            | irParamType when irParamType.IsInteger -> (irParamType.Integer.Bits, irParamType.Integer.Signed) |> function
                                |  8,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : int8) @@>) |> Some
                                |  8, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : uint8) @@>) |> Some
                                | 16,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : int16) @@>) |> Some
                                | 16, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : uint16) @@>) |> Some
                                | 32,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : int) @@>) |> Some
                                | 32, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : uint32) @@>) |> Some
                                | 64,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : int64) @@>) |> Some
                                | 64, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : uint64) @@>) |> Some
                                | _ -> None
                            | irParamType when irParamType.IsFloatingPoint -> irParamType.FloatingPoint.Kind |> function
                                | FloatingPointKind.Double -> irValue |> Some
                                | FloatingPointKind.Single -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : float32) @@>) |> Some
                            | _ -> None
                        | FloatingPointKind.Single -> irParamType |> function
                            | irParamType when irParamType.IsInteger -> (irParamType.Integer.Bits, irParamType.Integer.Signed) |> function
                                |  8,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : int8) @@>) |> Some
                                |  8, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : uint8) @@>) |> Some
                                | 16,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : int16) @@>) |> Some
                                | 16, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : uint16) @@>) |> Some
                                | 32,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : int) @@>) |> Some
                                | 32, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : uint32) @@>) |> Some
                                | 64,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : int64) @@>) |> Some
                                | 64, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : uint64) @@>) |> Some
                                | _ -> None
                            | irParamType when irParamType.IsFloatingPoint -> irParamType.FloatingPoint.Kind |> function
                                | FloatingPointKind.Double -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : float) @@>) |> Some
                                | FloatingPointKind.Single -> irValue |> Some
                            | _ -> None
                    | _ -> None
                | _ -> None

    type ShuffleWord = ShuffleWord of int
    type VolatileWord = VolatileWord of int
    type DeviceWord = DeviceWord of int
    type TextureWord = TextureWord of int

//module ScanOperators =
//    type ScanOpKind =
//        | Add
//        | Mul
//        | Min
//        | Max
//
//    type IScanOp =
//        abstract Identity : int
//        abstract Op : Expr<'T -> 'T -> 'T>
//
//    let inline scanOp (opKind:ScanOpKind) (identity:int) =
//        { new IScanOp<int> with
//            member this.Identity = identity
//            member this.Op =
//                match opKind with
//                | Add -> ( + )
//                | Mul -> ( * )
//                | Min -> min
//                | Max -> max }

    

type Category =
    | NOT_A_NUMBER
    | SIGNED_INTERGER
    | UNSIGNED_INTEGER
    | FLOATING_POINT

/// [omit]
type KeyTraitsAttribute() =
    inherit Attribute()

    interface ICustomCallBuilder with
        member this.Build(ctx, irObject, info, irParams) =
            match irObject, info.Name, irParams with
            | Some(_), "Of", irValue :: [] ->
                let clrParamType = info.GetParameters().[0].ParameterType
                let irParamType = IRTypeBuilder.Instance.Build(ctx, clrParamType)
                let clrRealType = info.ReturnType
                let irRealType = IRTypeBuilder.Instance.Build(ctx, clrRealType)
                irRealType |> function
                | irRealType when irRealType.IsFloatingPoint -> irRealType.FloatingPoint.Kind |> function
                    | FloatingPointKind.Double -> irParamType |> function
                        | irParamType when irParamType.IsInteger -> (irParamType.Integer.Bits, irParamType.Integer.Signed) |> function
                            |  8,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : int8) @@>) |> Some
                            |  8, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : uint8) @@>) |> Some
                            | 16,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : int16) @@>) |> Some
                            | 16, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : uint16) @@>) |> Some
                            | 32,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : int) @@>) |> Some
                            | 32, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : uint32) @@>) |> Some
                            | 64,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : int64) @@>) |> Some
                            | 64, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : uint64) @@>) |> Some
                            | _ -> None
                        | irParamType when irParamType.IsFloatingPoint -> irParamType.FloatingPoint.Kind |> function
                            | FloatingPointKind.Double -> irValue |> Some
                            | FloatingPointKind.Single -> IRInstructionBuilder.Instance.Build(ctx, <@@ float(%%irValue.Expr : float32) @@>) |> Some
                        | _ -> None
                    | FloatingPointKind.Single -> irParamType |> function
                        | irParamType when irParamType.IsInteger -> (irParamType.Integer.Bits, irParamType.Integer.Signed) |> function
                            |  8,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : int8) @@>) |> Some
                            |  8, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : uint8) @@>) |> Some
                            | 16,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : int16) @@>) |> Some
                            | 16, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : uint16) @@>) |> Some
                            | 32,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : int) @@>) |> Some
                            | 32, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : uint32) @@>) |> Some
                            | 64,  true -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : int64) @@>) |> Some
                            | 64, false -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : uint64) @@>) |> Some
                            | _ -> None
                        | irParamType when irParamType.IsFloatingPoint -> irParamType.FloatingPoint.Kind |> function
                            | FloatingPointKind.Double -> IRInstructionBuilder.Instance.Build(ctx, <@@ float32(%%irValue.Expr : float) @@>) |> Some
                            | FloatingPointKind.Single -> irValue |> Some
                        | _ -> None
                | _ -> None
            | _ -> None



[<KeyTraits>]
type KeyTraits<'K> =
    abstract Of : int8 -> 'K
    abstract Of : uint8 -> 'K
    abstract Of : int16 -> 'K
    abstract Of : uint16 -> 'K
    abstract Of : int -> 'K
    abstract Of : uint32 -> 'K
    abstract Of : int64 -> 'K
    abstract Of : uint64 -> 'K
    abstract Of : float32 -> 'K
    abstract Of : float -> 'K
    abstract Zero : 'K
    abstract HIGH_BIT : 'K
    abstract MIN_KEY : 'K
    abstract MAX_KEY : 'K
 


let inline HIGH_BIT (unsignedBits:'_UnsignedBits) = 1G <<< ((sizeof<'_UnsignedBits> * 8) - 1)

let key<'K>() : KeyTraits<'K> = 
    typeof<'K> |> function 
       | ty when ty = typeof<int8> -> 
        { new KeyTraits<int8> with
            member this.Of(x:int8) = x
            member this.Of(x:uint8) = x |> int8
            member this.Of(x:int16) = x |> int8
            member this.Of(x:uint16) = x |> int8
            member this.Of(x:int) = x |> int8
            member this.Of(x:uint32) = x |> int8
            member this.Of(x:int64) = x |> int8
            member this.Of(x:uint64) = x |> int8
            member this.Of(x:float32) = x |> int8
            member this.Of(x:float) = x |> int8 
            member this.Zero = 0 |> int8
            member this.HIGH_BIT = 1 |> int8 |> HIGH_BIT
            member this.MIN_KEY = this.HIGH_BIT
            member this.MAX_KEY = (1 |> int8) <<< (1 |> int8 |> HIGH_BIT)            
        } |> box |> unbox

       | ty when ty = typeof<uint8> ->
        { new KeyTraits<uint8> with
            member this.Of(x:int8) = x |> uint8
            member this.Of(x:uint8) = x
            member this.Of(x:int16) = x |> uint8
            member this.Of(x:uint16) = x |> uint8
            member this.Of(x:int) = x |> uint8
            member this.Of(x:uint32) = x |> uint8
            member this.Of(x:int64) = x |> uint8
            member this.Of(x:uint64) = x |> uint8
            member this.Of(x:float32) = x |> uint8
            member this.Of(x:float) = x |> uint8 
            member this.Zero = 0 |> uint8
            member this.HIGH_BIT = 1 |> uint8 |> HIGH_BIT
            member this.MIN_KEY = this.HIGH_BIT
            member this.MAX_KEY = (1 |> uint8) <<< (1 |> uint8 |> HIGH_BIT)            
        } |> box |> unbox

       | ty when ty = typeof<int16> ->
        { new KeyTraits<int16> with
            member this.Of(x:int8) = x |> int16
            member this.Of(x:uint8) = x |> int16
            member this.Of(x:int16) = x
            member this.Of(x:uint16) = x |> int16
            member this.Of(x:int) = x |> int16
            member this.Of(x:uint32) = x |> int16
            member this.Of(x:int64) = x |> int16
            member this.Of(x:uint64) = x |> int16
            member this.Of(x:float32) = x |> int16
            member this.Of(x:float) = x |> int16 
            member this.Zero = 0 |> int16
            member this.HIGH_BIT = 1 |> int16 |> HIGH_BIT
            member this.MIN_KEY = this.HIGH_BIT
            member this.MAX_KEY = (1 |> int16) <<< (1 |> int16 |> HIGH_BIT)            
        } |> box |> unbox

       | ty when ty = typeof<uint16> -> 
        { new KeyTraits<uint16> with
            member this.Of(x:int8) = x |> uint16
            member this.Of(x:uint8) = x |> uint16
            member this.Of(x:int16) = x |> uint16
            member this.Of(x:uint16) = x
            member this.Of(x:int) = x |> uint16
            member this.Of(x:uint32) = x |> uint16
            member this.Of(x:int64) = x |> uint16
            member this.Of(x:uint64) = x |> uint16
            member this.Of(x:float32) = x |> uint16
            member this.Of(x:float) = x |> uint16 
            member this.Zero = 0 |> uint16
            member this.HIGH_BIT = 1 |> uint16 |> HIGH_BIT
            member this.MIN_KEY = this.HIGH_BIT
            member this.MAX_KEY = (1 |> uint16) <<< (1 |> uint16 |> HIGH_BIT)            
        } |> box |> unbox

       | ty when ty = typeof<int32> ->
        { new KeyTraits<int32> with
            member this.Of(x:int8) = x |> int32
            member this.Of(x:uint8) = x |> int32
            member this.Of(x:int16) = x |> int32
            member this.Of(x:uint16) = x |> int32
            member this.Of(x:int) = x
            member this.Of(x:uint32) = x |> int32
            member this.Of(x:int64) = x |> int32
            member this.Of(x:uint64) = x |> int32
            member this.Of(x:float32) = x |> int32
            member this.Of(x:float) = x |> int32 
            member this.Zero = 0 |> int32
            member this.HIGH_BIT = 1 |> int32 |> HIGH_BIT
            member this.MIN_KEY = this.HIGH_BIT
            member this.MAX_KEY = (1 |> int32) <<< (1 |> int32 |> HIGH_BIT)            
        } |> box |> unbox

       | ty when ty = typeof<uint32> -> 
        { new KeyTraits<uint32> with
            member this.Of(x:int8) = x |> uint32
            member this.Of(x:uint8) = x |> uint32
            member this.Of(x:int16) = x |> uint32
            member this.Of(x:uint16) = x |> uint32
            member this.Of(x:int) = x |> uint32
            member this.Of(x:uint32) = x
            member this.Of(x:int64) = x |> uint32
            member this.Of(x:uint64) = x |> uint32
            member this.Of(x:float32) = x |> uint32
            member this.Of(x:float) = x |> uint32 
            member this.Zero = 0 |> uint32
            member this.HIGH_BIT = 1 |> uint32 |> HIGH_BIT
            member this.MIN_KEY = this.HIGH_BIT
            member this.MAX_KEY = (1 |> uint32) <<< (1 |> uint32 |> HIGH_BIT)            
        } |> box |> unbox

       | ty when ty = typeof<int64> ->
        { new KeyTraits<int64> with
            member this.Of(x:int8) = x |> int64
            member this.Of(x:uint8) = x |> int64
            member this.Of(x:int16) = x |> int64
            member this.Of(x:uint16) = x |> int64
            member this.Of(x:int) = x |> int64
            member this.Of(x:uint32) = x |> int64
            member this.Of(x:int64) = x |> int64
            member this.Of(x:uint64) = x |> int64
            member this.Of(x:float32) = x |> int64
            member this.Of(x:float) = x |> int64 
            member this.Zero = 0 |> int64
            member this.HIGH_BIT = 1 |> int64 |> HIGH_BIT
            member this.MIN_KEY = this.HIGH_BIT
            member this.MAX_KEY = (1 |> int64) <<< (1 |> int64 |> HIGH_BIT)            
        } |> box |> unbox

       | ty when ty = typeof<uint64> -> 
        { new KeyTraits<uint64> with
            member this.Of(x:int8) = x |> uint64
            member this.Of(x:uint8) = x |> uint64
            member this.Of(x:int16) = x |> uint64
            member this.Of(x:uint16) = x |> uint64
            member this.Of(x:int) = x |> uint64
            member this.Of(x:uint32) = x |> uint64
            member this.Of(x:int64) = x |> uint64
            member this.Of(x:uint64) = x
            member this.Of(x:float32) = x |> uint64
            member this.Of(x:float) = x |> uint64 
            member this.Zero = 0 |> uint64
            member this.HIGH_BIT = 1 |> uint64 |> HIGH_BIT
            member this.MIN_KEY = this.HIGH_BIT
            member this.MAX_KEY = (1 |> uint64) <<< (1 |> uint64 |> HIGH_BIT)            
        } |> box |> unbox

//       | ty when ty = typeof<float32> ->
//        { new KeyTraits<float32> with
//            member this.Of(x:int8) = x |> float32
//            member this.Of(x:uint8) = x |> float32
//            member this.Of(x:int16) = x |> float32
//            member this.Of(x:uint16) = x |> float32
//            member this.Of(x:int) = x |> float32
//            member this.Of(x:uint32) = x |> float32
//            member this.Of(x:int64) = x |> float32
//            member this.Of(x:uint64) = x |> float32
//            member this.Of(x:float32) = x
//            member this.Of(x:float) = x |> float32 
//            member this.Zero = 0.f
//            member this.HIGH_BIT = 1 |> HIGH_BIT
//            member this.MIN_KEY = this.HIGH_BIT
//            member this.MAX_KEY = (1.f |> int) <<< (1.f |> int |> HIGH_BIT)            
//        } |> box |> unbox
//
//       | ty when ty = typeof<float> ->
//        { new KeyTraits<int8> with
//            member this.Of(x:int8) = x
//            member this.Of(x:uint8) = x |> int8
//            member this.Of(x:int16) = x |> int8
//            member this.Of(x:uint16) = x |> int8
//            member this.Of(x:int) = x |> int8
//            member this.Of(x:uint32) = x |> int8
//            member this.Of(x:int64) = x |> int8
//            member this.Of(x:uint64) = x |> int8
//            member this.Of(x:float32) = x |> int8
//            member this.Of(x:float) = x |> int8 
//            member this.Zero = 0 |> int8
//            member this.HIGH_BIT = 1 |> int8 |> HIGH_BIT
//            member this.MIN_KEY = this.HIGH_BIT
//            member this.MAX_KEY = (1 |> int8) <<< (1 |> int8 |> HIGH_BIT)            
//        } |> box |> unbox

       | _ -> failwith "unsupported type for KeyTraits<int>"

type IBaseTraits<'_UnsignedBits> =
    abstract CATEGORY : Category
    abstract PRIMITIVE : bool
    abstract NULL_TYPE : bool
    abstract UnsignedBits : '_UnsignedBits
    abstract HIGH_BIT : '_UnsignedBits option
    abstract MIN_KEY : '_UnsignedBits
    abstract MAX_KEY : '_UnsignedBits
    abstract member TwiddleIn : ('_UnsignedBits -> '_UnsignedBits)
    abstract member TwiddleOut : ('_UnsignedBits -> '_UnsignedBits)


let baseTraits (category:Category) (primitive:bool) (null_type:bool) (unsignedBits:'_UnsignedBits) : IBaseTraits<'_UnsignedBits> option = //(key:KeyTraits<'_UnsignedBits>) =
    let CATEGORY = category
    let PRIMITIVE = primitive
    let NULL_TYPE = null_type
    match CATEGORY, PRIMITIVE, NULL_TYPE with
    | Category.UNSIGNED_INTEGER, true, false ->
        { new IBaseTraits<'_UnsignedBits> with
            member this.CATEGORY = category
            member this.PRIMITIVE = primitive
            member this.NULL_TYPE = null_type
            member this.UnsignedBits = unsignedBits
            member this.HIGH_BIT = None
            member this.MIN_KEY = key<'_UnsignedBits>().MIN_KEY
            member this.MAX_KEY = key<'_UnsignedBits>().MAX_KEY            
            member this.TwiddleIn = fun key -> key
            member this.TwiddleOut = fun key -> key } 
        |> Some
    
    | Category.SIGNED_INTERGER, true, false -> 
        { new IBaseTraits<'_UnsignedBits> with
            member this.CATEGORY = category
            member this.PRIMITIVE = primitive
            member this.NULL_TYPE = null_type
            member this.UnsignedBits = unsignedBits
            member this.HIGH_BIT = unsignedBits |> Some//|> HIGH_BIT |> Some
            member this.MIN_KEY = key<'_UnsignedBits>().MIN_KEY
            member this.MAX_KEY = key<'_UnsignedBits>().MAX_KEY          
            member this.TwiddleIn = fun (key:'_UnsignedBits) -> key ^^^ this.HIGH_BIT.Value
            member this.TwiddleOut = fun key -> key ^^^ this.HIGH_BIT.Value }
        |> Some
                    
    | Category.FLOATING_POINT, true, false ->
        { new IBaseTraits<'_UnsignedBits> with
            member this.CATEGORY = category
            member this.PRIMITIVE = primitive
            member this.NULL_TYPE = null_type
            member this.UnsignedBits = unsignedBits
            member this.HIGH_BIT = unsignedBits |> HIGH_BIT |> Some
            member this.MIN_KEY = -1G
            member this.MAX_KEY = -1G ^^^ this.HIGH_BIT.Value            
            member this.TwiddleIn =
                fun key -> 
                    let mask = if (key &&& this.HIGH_BIT.Value) > 0G then -1G else this.HIGH_BIT.Value
                    key ^^^ mask
            member this.TwiddleOut = 
                fun key -> 
                    let mask = if (key &&& this.HIGH_BIT.Value) > 0G then this.HIGH_BIT.Value else -1G
                    key ^^^ mask }
        |> Some

    | _,_,_ ->
        None

type KeyValuePair<'K, 'V> = System.Collections.Generic.KeyValuePair<'K,'V>
let keyValueOp(op:('V -> 'V -> 'V)) = fun (kvp1:KeyValuePair<'K,'V>) (kvp2:KeyValuePair<'K,'V>) -> (kvp1.Value,kvp2.Value) ||> op
//
//let PRIMITIVE<int>() =
//    typeof<int> |> function
//    | ty when ty = typeof<int8> -> true
//    | ty when ty = typeof<uint8> -> true
//    | ty when ty = typeof<int16> -> true
//    | ty when ty = typeof<uint16> -> true
//    | ty when ty = typeof<int> -> true
//    | ty when ty = typeof<uint32> -> true
//    | ty when ty = typeof<int64> -> true
//    | ty when ty = typeof<uint64> -> true
//    | ty when ty = typeof<float32> -> true
//    | ty when ty = typeof<float> -> true
//    | _ -> false




let inline ZeroInitialize() =
    let MULTIPLE = sizeof<int> / sizeof<UnitWord.ShuffleWord>
    let words = __local__.Array(MULTIPLE)
    words |> __array_to_ptr
