[<AutoOpen>]
module Alea.cuExtension.CUB.Utilities.Vector

open Alea.CUDA
open Alea.CUDA.Utilities    



type [<Record>] Vector1<'T> = 
    {
        x : 'T
    }

    static member internal _1 (x:'T) = { x = x }

and [<Record>] Vector2<'T> =
    {
        x : 'T
        y : 'T
    }

    static member _2 (x:'T, y:'T) = { x = x; y = y}

and [<Record>] Vector3<'T> =
    {
        x : 'T
        y : 'T
        z : 'T
    }

    static member _3 (x:'T, y:'T, z:'T) = { x = x; y = y; z = z}

and [<Record>] Vector4<'T> =
    {
        x : 'T
        y : 'T
        z : 'T
        w : 'T
    }

    static member _4 (x:'T, y:'T, z:'T, w:'T) = { x = x; y = y; z = z; w = w}

and [<Record>] Vector<'T> =
    {
        d : deviceptr<'T>
        h : 'T[]
    }

    static member _1 (x:'T) = Vector1._1(x)
    static member _2 (x:'T, y:'T) = Vector2._2(x,y)
    static member _3 (x:'T, y:'T, z:'T) = Vector3._3(x,y,z)
    static member _4 (x:'T, y:'T, z:'T, w:'T) = Vector4._4(x,y,z,w)

type CubVector<'T> =
    | Vector1 of 'T
    | Vector2 of 'T*'T
    | Vector3 of 'T*'T*'T
    | Vector4 of 'T*'T*'T*'T



//type Vector1<'T>(_x:'T) =
//    inherit Vector<'T>(_x)
//    let mutable _x = _x
//    member this.X with get () = _x and set(v) = _x <- v
//
//and Vector2<'T>(_x:'T, _y:'T) =
//    inherit Vector<'T>(_x, _y)
//    member this.X = _x
//    member this.Y = _y

//type Vector<'T> (?x:'T,?y:'T) =
//    abstract x : 'T option
//    abstract y : 'T option
//
//    default this.x = None
//    default this.y = None

    //new (x,?y) =  if y.IsNone then Vector<'T>(x) else Vector<'T>(x,y.Value)
//
//type Vector<'T> (?_x:'T, ?_y:'T, ?_z:'T) =
//    let mutable x = _x 
//    let mutable y = _y
//    let mutable z = _z
//
//    member this.X   with get() = x 
//                    and set(v) = x <- v
//    member this.Y with get() = if y.IsSome then y.Value else None
//
//let f (v:Vector<'T>) = ()
//
//let v2 (x:'T) (y:'T) =
//    { new Vector<'T>() with
//        member this.x = x |> Some
//        member this.y = y |> Some }
//
//let zz = (1,2) ||> v2
//
//printfn "%d" zz.x

//
//type [<AbstractClass>] IVector<'T> =
//    abstract x : 'T
//    abstract y : 'T
//    abstract z : 'T
//    abstract w : 'T
//    
//    static member (+) (v1:IVector<'T>, v2:IVector<'T>) =
//        {new IVector<'T> with
//            member this.x = v1.x + v2.x
//            member this.y = v1.y + v2.y
//            member this.z = v1.z + v2.z
//            member this.w = v1.w + v2.w} 
    


//let mutable MAX_VEC_ELEMENTS = 4
//
//[<Record>]
//type CubVector<'T> = 
//    {   
//        mutable Ptr : deviceptr<'T>
//        VEC_ELEMENTS : int                        
//    }
//
//    member this.Item
//        with [<ReflectedDefinition>] get (idx:int) = this.Ptr.[idx]
//        and  [<ReflectedDefinition>] set (idx:int) (v:'T) = if idx <= (this.VEC_ELEMENTS - 1) then this.Ptr.[idx] <- v
//
//    member this.W
//        with [<ReflectedDefinition>] get () = this.Ptr.[0]
//        and  [<ReflectedDefinition>] set (w:'T) = this.Ptr.[0] <- w
//
//    member this.X
//        with [<ReflectedDefinition>] get () = this.Ptr.[1]
//        and  [<ReflectedDefinition>] set (x:'T) = this.Ptr.[1] <- x
//
//    member this.Y
//        with [<ReflectedDefinition>] get () = this.Ptr.[2]
//        and  [<ReflectedDefinition>] set (y:'T) = this.Ptr.[2] <- y
//
//    member this.Z
//        with [<ReflectedDefinition>] get () = this.Ptr.[3]
//        and  [<ReflectedDefinition>] set (z:'T) = this.Ptr.[3] <- z
//    
//    static member Null() =
//        {
//            Ptr = __null()
//            VEC_ELEMENTS = 0
//        }
//
//    static member Vector1(dptr:deviceptr<'T>) =
//        {
//            Ptr = dptr
//            VEC_ELEMENTS = 1
//        }
//
//    static member Vector2(dptr:deviceptr<'T>) =
//        {
//            Ptr = dptr
//            VEC_ELEMENTS = 2
//        }
//
//    static member Vector3(dptr:deviceptr<'T>) =
//        {
//            Ptr = dptr
//            VEC_ELEMENTS = 3
//        }
//
//    static member Vector4(dptr:deviceptr<'T>) =
//        {
//            Ptr = dptr
//            VEC_ELEMENTS = 4 
//        }