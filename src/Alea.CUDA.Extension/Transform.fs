module Alea.CUDA.Extension.Transform

open Alea.CUDA

let private divup num den = (num + den - 1) / den

type IMap<'T> =
    abstract Map : int * DevicePtr<'T> * DevicePtr<'T> -> unit
    abstract Map : int * DevicePtr<'T> -> unit
    abstract Map : 'T[] -> 'T[]

let map transform = cuda {
    let! kernel =
        <@ fun (n:int) (input:DevicePtr<'T>) (output:DevicePtr<'T>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < n do
                input.[i] <- (%transform) output.[i]
                i <- i + stride @>
        |> defineKernelFuncWithName "map"

    let calcLaunchParam (m:Module) (n:int) =
        let blockSize = 256 // TODO: more advanced calcuation due to fine tune
        let gridSize = min 64 (divup n blockSize)
        LaunchParam(gridSize, blockSize)

    let launch (m:Module) (n:int) (input:DevicePtr<'T>) (output:DevicePtr<'T>) =
        let kernel = kernel.Apply m
        let calcLaunchParam = calcLaunchParam m
        let lp = calcLaunchParam n
        kernel.Launch lp n input output

    return PFunc(fun (m:Module) ->
        let launch = launch m
        { new IMap<'T> with
            member this.Map(n:int, input:DevicePtr<'T>, output:DevicePtr<'T>) =
                launch n input output
            member this.Map(n:int, data:DevicePtr<'T>) =
                launch n data data
            member this.Map(input:'T[]) =
                let n = input.Length
                if n = 0 then Array.zeroCreate<'T> 0
                else
                    m.Worker.Eval(fun () ->
                        use data = m.Worker.Malloc(input)
                        launch n data.Ptr data.Ptr
                        data.ToHost()) }) }
                    
         

