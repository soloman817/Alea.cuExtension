module Alea.CUDA.Extension.Random.Mrg32k3aGold

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension


type XorShift7Rng<'T when 'T:unmanaged> =
    {
        StreamUnit : int
        JumpAheadMatrices : uint32[]
        GenerateStartState : uint32 -> uint32[]
        // hint -> jumpAheadMatrices -> streams -> steps -> startState -> runs -> rank -> result
        Generate : ActionHint -> DevicePtr<uint32> -> int -> int -> DevicePtr<uint32> -> int -> int -> DevicePtr<'T> -> unit
    }

