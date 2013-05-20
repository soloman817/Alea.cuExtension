﻿module Alea.CUDA.Extension.MGPU.LoadStore

// this file maps to loadstore.cuh. which acturally did the different matrix
// order changing. Please check http://www.moderngpu.com/scan/globalscan.html#Scan
// and check the concept of transposeValues. 

open Alea.CUDA

let deviceSharedToReg (NT:int) (VT:int) =
    <@ fun (count:int) (data:RPtr<'T>) (tid:int) (reg:RWPtr<'T>) (sync:bool) ->
        if count >= NT * VT then
            for i = 0 to VT - 1 do
                reg.[i] <- data.[NT * i + tid]
        else
            for i = 0 to VT - 1 do
                let index = NT * i + tid
                if index < count then reg.[i] <- data.[index]
        if sync then __syncthreads() @>

let deviceGlobalToReg (NT:int) (VT:int) = deviceSharedToReg NT VT
