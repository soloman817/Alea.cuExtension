module Test.Alea.cuBase.MGPU.Benchmark.Sort

open Alea.CUDA
open Alea.cuBase
open Alea.cuBase.MGPU
open Test.Alea.cuBase.MGPU
open NUnit.Framework

let worker = getDefaultWorker()
let pfuncts = new PMergesort()