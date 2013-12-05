module Alea.CUDA.TestUtilities.TestClass

open System
open System.Runtime.InteropServices
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

// state                                   index
// xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
[<RefClass>]
type StatesAndIndex() =
    
    [<RefClassArrayField(8)>]
    member this.State : int[] = failwith "device only"

    [<RefClassField>]
    member this.Index 
        with get () : int = failwith "device only"
        and set (value:int) : unit = failwith "device only"

    [<ReflectedDefinition>]
    member this.Init(state0:int) =
        for i = 0 to 7 do this.State.[i] <- state0 + i
        this.Index <- state0 + 8

    [<ReflectedDefinition>]
    static member StaticInit (this:StatesAndIndex ref) (state0:int) =
        for i = 0 to 7 do this.Value.State.[i] <- state0 + i
        this.Value.Index <- state0 + 8
