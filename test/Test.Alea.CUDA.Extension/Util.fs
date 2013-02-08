[<AutoOpen>]
module Test.Alea.CUDA.Extension.Util

open NUnit.Framework
open Alea.CUDA

let rng = System.Random()

let getDefaultWorker() =
    if Device.Count = 0 then Assert.Inconclusive("We need at least one device of compute capability 2.0 or greater.")
    Engine.workers.DefaultWorker


