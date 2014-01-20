[<AutoOpen>]
module Alea.cuExtension.CUB.Common

open Alea.CUDA

type InputIterator<'T> = deviceptr<'T>