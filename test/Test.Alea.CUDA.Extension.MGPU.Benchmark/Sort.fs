module Test.Alea.CUDA.Extension.MGPU.Benchmark.Sort

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Test.Alea.CUDA.Extension.MGPU
open NUnit.Framework

let worker = getDefaultWorker()
let pfuncts = new PMergesort()