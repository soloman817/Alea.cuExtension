[<AutoOpen>]
module Alea.cuExtension.CUB.Thread.Load

open System
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common


type CacheLoadModifier =
    | LOAD_DEFAULT
    | LOAD_CA
    | LOAD_CS
    | LOAD_CV
    | LOAD_LDG
    | LOAD_VOLATILE

let ThreadLoad modifier

let iterateThreadLoad count max =
    fun modifier ->
        fun ptr vals ->
            vals.[count] <-  