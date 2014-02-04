[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Store

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common

type BlockStoreAlgorithm =
    | BLOCK_STORE_DIRECT
    | BLOCK_STORE_VECTORIZE
    | BLOCK_STORE_TRANSPOSE
    | BLOCK_STORE_WARP_TRANSPOSE