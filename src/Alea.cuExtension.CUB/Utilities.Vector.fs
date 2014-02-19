[<AutoOpen>]
module Alea.cuExtension.CUB.Utilities.Vector

open Alea.CUDA
open Alea.CUDA.Utilities    



type [<Record>] Vector1 = 
    {
        x : int
    }

    static member internal _1 (x:int) = { x = x }

and [<Record>] Vector2 =
    {
        x : int
        y : int
    }

    static member _2 (x:int, y:int) = { x = x; y = y}

and [<Record>] Vector3 =
    {
        x : int
        y : int
        z : int
    }

    static member _3 (x:int, y:int, z:int) = { x = x; y = y; z = z}

and [<Record>] Vector4 =
    {
        x : int
        y : int
        z : int
        w : int
    }

    static member _4 (x:int, y:int, z:int, w:int) = { x = x; y = y; z = z; w = w}

and [<Record>] Vector =
    {
        d : deviceptr<int>
        h : int[]
    }

    static member _1 (x:int) = Vector1._1(x)
    static member _2 (x:int, y:int) = Vector2._2(x,y)
    static member _3 (x:int, y:int, z:int) = Vector3._3(x,y,z)
    static member _4 (x:int, y:int, z:int, w:int) = Vector4._4(x,y,z,w)

//type CubVector =
//    | Vector1 of 'T
//    | Vector2 of 'T*'T
//    | Vector3 of 'T*'T*'T
//    | Vector4 of 'T*'T*'T*'T



//type Vector1(_x:int) =
//    inherit Vector(_x)
//    let mutable _x = _x
//    member this.X with get () = _x and set(v) = _x <- v
//
//and Vector2(_x:int, _y:int) =
//    inherit Vector(_x, _y)
//    member this.X = _x
//    member this.Y = _y

//type Vector<int> (?x:int,?y:int) =
//    abstract x : int option
//    abstract y : int option
//
//    default this.x = None
//    default this.y = None

    //new (x,?y) =  if y.IsNone then Vector(x) else Vector(x,y.Value)
//
//type Vector<int> (?_x:int, ?_y:int, ?_z:int) =
//    let mutable x = _x 
//    let mutable y = _y
//    let mutable z = _z
//
//    member this.X   with get() = x 
//                    and set(v) = x <- v
//    member this.Y with get() = if y.IsSome then y.Value else None
//
//let f (v:Vector<int>) = ()
//
//let v2 (x:int) (y:int) =
//    { new Vector() with
//        member this.x = x |> Some
//        member this.y = y |> Some }
//
//let zz = (1,2) ||> v2
//
//printfn "%d" zz.x

//
//type [<AbstractClass>] IVector =
//    abstract x : int
//    abstract y : int
//    abstract z : int
//    abstract w : int
//    
//    static member (+) (v1:IVector<int>, v2:IVector<int>) =
//        {new IVector<int> with
//            member this.x = v1.x + v2.x
//            member this.y = v1.y + v2.y
//            member this.z = v1.z + v2.z
//            member this.w = v1.w + v2.w} 
    


//let mutable MAX_VEC_ELEMENTS = 4
//
//[<Record>]
//type CubVector = 
//    {   
//        mutable Ptr : deviceptr<int>
//        VEC_ELEMENTS : int                        
//    }
//
//    member this.Item
//        with [<ReflectedDefinition>] get (idx:int) = this.Ptr.[idx]
//        and  [<ReflectedDefinition>] set (idx:int) (v:int) = if idx <= (this.VEC_ELEMENTS - 1) then this.Ptr.[idx] <- v
//
//    member this.W
//        with [<ReflectedDefinition>] get () = this.Ptr.[0]
//        and  [<ReflectedDefinition>] set (w:int) = this.Ptr.[0] <- w
//
//    member this.X
//        with [<ReflectedDefinition>] get () = this.Ptr.[1]
//        and  [<ReflectedDefinition>] set (x:int) = this.Ptr.[1] <- x
//
//    member this.Y
//        with [<ReflectedDefinition>] get () = this.Ptr.[2]
//        and  [<ReflectedDefinition>] set (y:int) = this.Ptr.[2] <- y
//
//    member this.Z
//        with [<ReflectedDefinition>] get () = this.Ptr.[3]
//        and  [<ReflectedDefinition>] set (z:int) = this.Ptr.[3] <- z
//    
//    static member Null() =
//        {
//            Ptr = __null()
//            VEC_ELEMENTS = 0
//        }
//
//    static member Vector1(dptr:deviceptr<int>) =
//        {
//            Ptr = dptr
//            VEC_ELEMENTS = 1
//        }
//
//    static member Vector2(dptr:deviceptr<int>) =
//        {
//            Ptr = dptr
//            VEC_ELEMENTS = 2
//        }
//
//    static member Vector3(dptr:deviceptr<int>) =
//        {
//            Ptr = dptr
//            VEC_ELEMENTS = 3
//        }
//
//    static member Vector4(dptr:deviceptr<int>) =
//        {
//            Ptr = dptr
//            VEC_ELEMENTS = 4 
//        }