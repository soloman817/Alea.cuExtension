﻿module Alea.CUDA.Extension.Transform

open Alea.CUDA
open Alea.CUDA.Extension.Util

let transform name transform = cuda {
    let! kernel =
        <@ fun (n:int) (input:DevicePtr<'T>) (output:DevicePtr<'U>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < n do
                output.[i] <- (%transform) input.[i]
                i <- i + stride @>
        |> defineKernelFuncWithName name

    let calcLaunchParam (m:Module) (n:int) =
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min 64 (divup n blockSize)
        LaunchParam(gridSize, blockSize)

    return PFunc(fun (m:Module) lpmod (n:int) (input:DevicePtr<'T>) (output:DevicePtr<'U>) ->
        let kernel = kernel.Apply m
        let calcLaunchParam = calcLaunchParam m
        let lp = calcLaunchParam n |> lpmod
        kernel.Launch lp n input output) }

