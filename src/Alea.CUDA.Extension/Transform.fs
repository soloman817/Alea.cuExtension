module Alea.CUDA.Extension.Transform

open Alea.CUDA

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

    let launchParam (m:Module) (lphint:LPHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> lphint.Modify

    return PFunc(fun (m:Module) ->
        let kernel = kernel.Apply m
        let launchParam = launchParam m
        fun (lphint:LPHint) (n:int) (input:DevicePtr<'T>) (output:DevicePtr<'U>) ->
            let lp = launchParam lphint n
            kernel.Launch lp n input output) }

let transform2 name transform = cuda {
    let! kernel =
        <@ fun (n:int) (input1:DevicePtr<'T1>) (input2:DevicePtr<'T2>) (output:DevicePtr<'U>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < n do
                output.[i] <- (%transform) input1.[i] input2.[i]
                i <- i + stride @>
        |> defineKernelFuncWithName name

    let launchParam (m:Module) (lphint:LPHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> lphint.Modify

    return PFunc(fun (m:Module) ->
        let kernel = kernel.Apply m
        let launchParam = launchParam m
        fun (lphint:LPHint) (n:int) (input1:DevicePtr<'T1>) (input2:DevicePtr<'T2>) (output:DevicePtr<'U>) ->
            let lp = launchParam lphint n
            kernel.Launch lp n input1 input2 output) }

let transformi name transform = cuda {
    let! kernel =
        <@ fun (n:int) (input:DevicePtr<'T>) (output:DevicePtr<'U>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < n do
                output.[i] <- (%transform) i input.[i]
                i <- i + stride @>
        |> defineKernelFuncWithName name

    let launchParam (m:Module) (lphint:LPHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> lphint.Modify

    return PFunc(fun (m:Module) ->
        let kernel = kernel.Apply m
        let launchParam = launchParam m
        fun (lphint:LPHint) (n:int) (input:DevicePtr<'T>) (output:DevicePtr<'U>) ->
            let lp = launchParam lphint n
            kernel.Launch lp n input output) }

let transformi2 name transform = cuda {
    let! kernel =
        <@ fun (n:int) (input1:DevicePtr<'T1>) (input2:DevicePtr<'T2>) (output:DevicePtr<'U>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < n do
                output.[i] <- (%transform) i input1.[i] input2.[i]
                i <- i + stride @>
        |> defineKernelFuncWithName name

    let launchParam (m:Module) (lphint:LPHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> lphint.Modify

    return PFunc(fun (m:Module) ->
        let kernel = kernel.Apply m
        let launchParam = launchParam m
        fun (lphint:LPHint) (n:int) (input1:DevicePtr<'T1>) (input2:DevicePtr<'T2>) (output:DevicePtr<'U>) ->
            let lp = launchParam lphint n
            kernel.Launch lp n input1 input2 output) }
