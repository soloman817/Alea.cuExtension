module Alea.CUDA.Extension.Transform

open Alea.CUDA

/// <summary>Transform.fill</summary>
/// <remarks></remarks>
let fill name transform = cuda {
    let! kernel =
        <@ fun (n:int) (data:DevicePtr<'T>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < n do
                data.[i] <- %transform
                i <- i + stride @>
        |> defineKernelFuncWithName name

    let launchParam (m:Module) (hint:ActionHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let kernel = kernel.Apply m
        let launchParam = launchParam m
        fun (hint:ActionHint) (n:int) (data:DevicePtr<'T>) ->
            let lp = launchParam hint n
            kernel.Launch lp n data) }

/// <summary>Transform.fillp</summary>
/// <remarks></remarks>
let fillp name transform = cuda {
    let! param = defineConstantArray<'P>(1)

    let! kernel =
        <@ fun (n:int) (data:DevicePtr<'T>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let param = param.[0]
            let mutable i = start
            while i < n do
                data.[i] <- (%transform) param
                i <- i + stride @>
        |> defineKernelFuncWithName name

    let launchParam (m:Module) (hint:ActionHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernel = kernel.Apply m
        let param = param.Apply m
        let launchParam = launchParam m
        fun (hint:ActionHint) (n:int) (param':'P) (data:DevicePtr<'T>) ->
            let lp = launchParam hint n
            fun () ->
                param.Scatter([| param' |])
                kernel.Launch lp n data
            |> worker.Eval) }

/// <summary>Transform.filli</summary>
/// <remarks></remarks>
let filli name transform = cuda {
    let! kernel =
        <@ fun (n:int) (data:DevicePtr<'T>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < n do
                data.[i] <- (%transform) i
                i <- i + stride @>
        |> defineKernelFuncWithName name

    let launchParam (m:Module) (hint:ActionHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let kernel = kernel.Apply m
        let launchParam = launchParam m
        fun (hint:ActionHint) (n:int) (data:DevicePtr<'T>) ->
            let lp = launchParam hint n
            kernel.Launch lp n data) }

/// <summary>Transform.fillip</summary>
/// <remarks></remarks>
let fillip name transform = cuda {
    let! param = defineConstantArray<'P>(1)

    let! kernel =
        <@ fun (n:int) (data:DevicePtr<'T>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let param = param.[0]
            let mutable i = start
            while i < n do
                data.[i] <- (%transform) i param
                i <- i + stride @>
        |> defineKernelFuncWithName name

    let launchParam (m:Module) (hint:ActionHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernel = kernel.Apply m
        let param = param.Apply m
        let launchParam = launchParam m
        fun (hint:ActionHint) (n:int) (param':'P) (data:DevicePtr<'T>) ->
            let lp = launchParam hint n
            fun () ->
                param.Scatter([| param' |])
                kernel.Launch lp n data
            |> worker.Eval) }

/// <summary>Transform.transform</summary>
/// <remarks></remarks>
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

    let launchParam (m:Module) (hint:ActionHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let kernel = kernel.Apply m
        let launchParam = launchParam m
        fun (hint:ActionHint) (n:int) (input:DevicePtr<'T>) (output:DevicePtr<'U>) ->
            let lp = launchParam hint n
            kernel.Launch lp n input output) }

/// <summary>Transform.transformp</summary>
/// <remarks></remarks>
let transformp name transform = cuda {
    let! param = defineConstantArray<'P>(1)

    let! kernel =
        <@ fun (n:int) (input:DevicePtr<'T>) (output:DevicePtr<'U>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let param = param.[0]
            let mutable i = start
            while i < n do
                output.[i] <- (%transform) param input.[i]
                i <- i + stride @>
        |> defineKernelFuncWithName name

    let launchParam (m:Module) (hint:ActionHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernel = kernel.Apply m
        let param = param.Apply m
        let launchParam = launchParam m
        fun (hint:ActionHint) (n:int) (param':'P) (input:DevicePtr<'T>) (output:DevicePtr<'U>) ->
            let lp = launchParam hint n
            fun () ->
                param.Scatter([| param' |])
                kernel.Launch lp n input output
            |> worker.Eval ) }

/// <summary>Transform.transformi</summary>
/// <remarks></remarks>
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

    let launchParam (m:Module) (hint:ActionHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let kernel = kernel.Apply m
        let launchParam = launchParam m
        fun (hint:ActionHint) (n:int) (input:DevicePtr<'T>) (output:DevicePtr<'U>) ->
            let lp = launchParam hint n
            kernel.Launch lp n input output) }

/// <summary>Transform.transformip</summary>
/// <remarks></remarks>
let transformip name transform = cuda {
    let! param = defineConstantArray<'P>(1)

    let! kernel =
        <@ fun (n:int) (input:DevicePtr<'T>) (output:DevicePtr<'U>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let param = param.[0]
            let mutable i = start
            while i < n do
                output.[i] <- (%transform) i param input.[i]
                i <- i + stride @>
        |> defineKernelFuncWithName name

    let launchParam (m:Module) (hint:ActionHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernel = kernel.Apply m
        let param = param.Apply m
        let launchParam = launchParam m
        fun (hint:ActionHint) (n:int) (param':'P) (input:DevicePtr<'T>) (output:DevicePtr<'U>) ->
            let lp = launchParam hint n
            fun() ->
                param.Scatter([| param' |])
                kernel.Launch lp n input output
            |> worker.Eval ) }

/// <summary>Transform.transform2</summary>
/// <remarks></remarks>
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

    let launchParam (m:Module) (hint:ActionHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let kernel = kernel.Apply m
        let launchParam = launchParam m
        fun (hint:ActionHint) (n:int) (input1:DevicePtr<'T1>) (input2:DevicePtr<'T2>) (output:DevicePtr<'U>) ->
            let lp = launchParam hint n
            kernel.Launch lp n input1 input2 output) }

/// <summary>Transform.transformp2</summary>
/// <remarks></remarks>
let transformp2 name transform = cuda {
    let! param = defineConstantArray<'P>(1)

    let! kernel =
        <@ fun (n:int) (input1:DevicePtr<'T1>) (input2:DevicePtr<'T2>) (output:DevicePtr<'U>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let param = param.[0]
            let mutable i = start
            while i < n do
                output.[i] <- (%transform) param input1.[i] input2.[i]
                i <- i + stride @>
        |> defineKernelFuncWithName name

    let launchParam (m:Module) (hint:ActionHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernel = kernel.Apply m
        let param = param.Apply m
        let launchParam = launchParam m
        fun (hint:ActionHint) (n:int) (param':'P) (input1:DevicePtr<'T1>) (input2:DevicePtr<'T2>) (output:DevicePtr<'U>) ->
            let lp = launchParam hint n
            fun() ->
                param.Scatter([| param' |])
                kernel.Launch lp n input1 input2 output
            |> worker.Eval )  }

/// <summary>Transform.transformi2</summary>
/// <remarks></remarks>
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

    let launchParam (m:Module) (hint:ActionHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let kernel = kernel.Apply m
        let launchParam = launchParam m
        fun (hint:ActionHint) (n:int) (input1:DevicePtr<'T1>) (input2:DevicePtr<'T2>) (output:DevicePtr<'U>) ->
            let lp = launchParam hint n
            kernel.Launch lp n input1 input2 output) }

/// <summary>Transform.transformip2</summary>
/// <remarks></remarks>
let transformip2 name transform = cuda {
    let! param = defineConstantArray<'P>(1)
    
    let! kernel =
        <@ fun (n:int) (input1:DevicePtr<'T1>) (input2:DevicePtr<'T2>) (output:DevicePtr<'U>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let param = param.[0]
            let mutable i = start
            while i < n do
                output.[i] <- (%transform) i param input1.[i] input2.[i]
                i <- i + stride @>
        |> defineKernelFuncWithName name

    let launchParam (m:Module) (hint:ActionHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernel = kernel.Apply m
        let param = param.Apply m
        let launchParam = launchParam m
        fun (hint:ActionHint) (n:int) (param':'P) (input1:DevicePtr<'T1>) (input2:DevicePtr<'T2>) (output:DevicePtr<'U>) ->
            let lp = launchParam hint n
            fun() ->
                param.Scatter([| param' |])
                kernel.Launch lp n input1 input2 output
            |> worker.Eval ) }