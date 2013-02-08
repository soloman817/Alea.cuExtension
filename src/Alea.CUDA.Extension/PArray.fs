[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]        
module Alea.CUDA.Extension.PArray

open Alea.CUDA
open Alea.CUDA.Extension.Util

let transform transform = cuda {
    let! kernel =
        <@ fun (n:int) (input:DevicePtr<'T>) (output:DevicePtr<'U>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < n do
                output.[i] <- (%transform) input.[i]
                i <- i + stride @>
        |> defineKernelFuncWithName "transfrom"

    let calcLaunchParam (m:Module) (n:int) =
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min 64 (divup n blockSize)
        LaunchParam(gridSize, blockSize)

    return PFunc(fun (m:Module) (input:PArray<'T>) (output:PArray<'U>) ->
        let kernel = kernel.Apply m
        let calcLaunchParam = calcLaunchParam m
        if input.Length > output.Length then failwithf "transform: input.Length(%d) > output.Length(%d)" input.Length output.Length
        let n = input.Length
        let lp = calcLaunchParam n
        kernel.Launch lp n input.Ptr output.Ptr) }

let transformi transform = cuda {
    let! kernel =
        <@ fun (n:int) (input:DevicePtr<'T>) (output:DevicePtr<'U>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < n do
                output.[i] <- (%transform) i input.[i]
                i <- i + stride @>
        |> defineKernelFuncWithName "transformi"

    let calcLaunchParam (m:Module) (n:int) =
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min 64 (divup n blockSize)
        LaunchParam(gridSize, blockSize)

    return PFunc(fun (m:Module) (input:PArray<'T>) (output:PArray<'U>) ->
        let kernel = kernel.Apply m
        let calcLaunchParam = calcLaunchParam m
        if input.Length > output.Length then failwithf "transformi: input.Length(%d) > output.Length(%d)" input.Length output.Length
        let n = input.Length
        let lp = calcLaunchParam n
        kernel.Launch lp n input.Ptr output.Ptr) }

let transform2 transform = cuda {
    let! kernel =
        <@ fun (n:int) (input1:DevicePtr<'T1>) (input2:DevicePtr<'T2>) (output:DevicePtr<'U>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < n do
                output.[i] <- (%transform) input1.[i] input2.[i]
                i <- i + stride @>
        |> defineKernelFuncWithName "transfrom2"

    let calcLaunchParam (m:Module) (n:int) =
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min 64 (divup n blockSize)
        LaunchParam(gridSize, blockSize)

    return PFunc(fun (m:Module) (input1:PArray<'T1>) (input2:PArray<'T2>) (output:PArray<'U>) ->
        let kernel = kernel.Apply m
        let calcLaunchParam = calcLaunchParam m
        if input1.Length > input2.Length then failwithf "transform2: input1.Length(%d) > input2.Length(%d)" input1.Length input2.Length
        if input1.Length > output.Length then failwithf "transform2: input1.Length(%d) > output.Length(%d)" input1.Length output.Length
        let n = input1.Length
        let lp = calcLaunchParam n
        kernel.Launch lp n input1.Ptr input2.Ptr output.Ptr) }

let transform2i transform = cuda {
    let! kernel =
        <@ fun (n:int) (input1:DevicePtr<'T1>) (input2:DevicePtr<'T2>) (output:DevicePtr<'U>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < n do
                output.[i] <- (%transform) i input1.[i] input2.[i]
                i <- i + stride @>
        |> defineKernelFuncWithName "transfrom2i"

    let calcLaunchParam (m:Module) (n:int) =
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min 64 (divup n blockSize)
        LaunchParam(gridSize, blockSize)

    return PFunc(fun (m:Module) (input1:PArray<'T1>) (input2:PArray<'T2>) (output:PArray<'U>) ->
        let kernel = kernel.Apply m
        let calcLaunchParam = calcLaunchParam m
        if input1.Length > input2.Length then failwithf "transform2i: input1.Length(%d) > input2.Length(%d)" input1.Length input2.Length
        if input1.Length > output.Length then failwithf "transform2i: input1.Length(%d) > output.Length(%d)" input1.Length output.Length
        let n = input1.Length
        let lp = calcLaunchParam n
        kernel.Launch lp n input1.Ptr input2.Ptr output.Ptr) }

