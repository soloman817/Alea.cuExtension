module Alea.CUDA.Extension.MatrixTransform

open Microsoft.FSharp.Quotations
open Alea.CUDA

open Util

let filli name transform = cuda {
    // this is internal transform in : major -> minor -> 'T
    let kernel (transform:Expr<int -> int -> 'T>) = 
        <@ fun (leadingDimension:int) (majors:int) (minors:int) (data:DevicePtr<'T>) ->
            let minorStart = blockIdx.x * blockDim.x + threadIdx.x
            let majorStart = blockIdx.y * blockDim.y + threadIdx.y
            
            let minorStride = gridDim.x * blockDim.x
            let majorStride = gridDim.y * blockDim.y

            let mutable major = majorStart
            while major < majors do
                let mutable minor = minorStart
                while minor < minors do
                    let i = major * leadingDimension + minor
                    data.[i] <- (%transform) major minor
                    minor <- minor + minorStride
                major <- major + majorStride @>

    let! kernelRowMajorOrder = RowMajorOrder.ToStorageOrder(transform) |> kernel |> defineKernelFunc
    let! kernelColMajorOrder = ColMajorOrder.ToStorageOrder(transform) |> kernel |> defineKernelFunc

    let launchParam (m:Module) =
        let worker = m.Worker
        fun (hint:ActionHint) ->
            let blockSize = Dim3(32, 8)
            let gridSize = Dim3(32, 32) // TODO fine tune
            LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let kernelRowMajorOrder = kernelRowMajorOrder.Apply m
        let kernelColMajorOrder = kernelColMajorOrder.Apply m
        let launchParam = launchParam m
        fun (hint:ActionHint) (order:MatrixStorageOrder) (rows:int) (cols:int) (data:DevicePtr<'T>) ->
            let lp = launchParam hint
            let leadingDimension, majors, minors = order.ToStorageOrder(rows, cols)
            match order with
            | RowMajorOrder -> kernelRowMajorOrder.Launch lp leadingDimension majors minors data
            | ColMajorOrder -> kernelColMajorOrder.Launch lp leadingDimension majors minors data) }

let fillip name transform = cuda {
    let! param = defineConstantArray<'P>(1)

    // this is internal transform in : major -> minor -> 'T
    let kernel (transform:Expr<int -> int -> 'P -> 'T>) = 
        <@ fun (leadingDimension:int) (majors:int) (minors:int) (data:DevicePtr<'T>) ->
            let minorStart = blockIdx.x * blockDim.x + threadIdx.x
            let majorStart = blockIdx.y * blockDim.y + threadIdx.y
            
            let minorStride = gridDim.x * blockDim.x
            let majorStride = gridDim.y * blockDim.y

            let param = param.[0]

            let mutable major = majorStart
            while major < majors do
                let mutable minor = minorStart
                while minor < minors do
                    let i = major * leadingDimension + minor
                    data.[i] <- (%transform) major minor param
                    minor <- minor + minorStride
                major <- major + majorStride @>

    // Here Alea.cuBase has a bug, cannot use the following function transformation, need to be fix : @BUG@
    //let! kernelRowMajorOrder = RowMajorOrder.ToStorageOrder(transform) |> kernel |> defineKernelFunc
    //let! kernelColMajorOrder = ColMajorOrder.ToStorageOrder(transform) |> kernel |> defineKernelFunc

    let toStorageOrder (order:MatrixStorageOrder) (transform:Expr<int -> int -> 'P -> 'T>) =
        match order with
        | RowMajorOrder -> transform
        | ColMajorOrder -> <@ fun c r p -> (%transform) r c p @>
    let! kernelRowMajorOrder = toStorageOrder RowMajorOrder transform |> kernel |> defineKernelFunc
    let! kernelColMajorOrder = toStorageOrder ColMajorOrder transform |> kernel |> defineKernelFunc

    let launchParam (m:Module) =
        let worker = m.Worker
        fun (hint:ActionHint) ->
            let blockSize = Dim3(32, 8)
            let gridSize = Dim3(32, 32) // TODO fine tune
            LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let param = param.Apply m
        let kernelRowMajorOrder = kernelRowMajorOrder.Apply m
        let kernelColMajorOrder = kernelColMajorOrder.Apply m
        let launchParam = launchParam m
        fun (hint:ActionHint) (order:MatrixStorageOrder) (rows:int) (cols:int) (param':'P) (data:DevicePtr<'T>) ->
            fun () ->
                let lp = launchParam hint
                let leadingDimension, majors, minors = order.ToStorageOrder(rows, cols)
                param.Scatter([| param' |])
                match order with
                | RowMajorOrder -> kernelRowMajorOrder.Launch lp leadingDimension majors minors data
                | ColMajorOrder -> kernelColMajorOrder.Launch lp leadingDimension majors minors data
            |> worker.Eval) }

let transformip name transform = cuda {
    let! param = defineConstantArray<'P>(1)

    let kernel (transform:Expr<int -> int -> 'P -> 'T -> 'U>) =
        <@ fun (leadingDimension:int) (majors:int) (minors:int) (input:DevicePtr<'T>) (output:DevicePtr<'U>) ->
            let minorStart = blockIdx.x * blockDim.x + threadIdx.x
            let majorStart = blockIdx.y * blockDim.y + threadIdx.y
            
            let minorStride = gridDim.x * blockDim.x
            let majorStride = gridDim.y * blockDim.y

            let param = param.[0]

            let mutable major = majorStart
            while major < majors do
                let mutable minor = minorStart
                while minor < minors do
                    let i = major * leadingDimension + minor
                    output.[i] <- (%transform) major minor param input.[i]
                    minor <- minor + minorStride
                major <- major + majorStride @>

    // Here Alea.cuBase has a bug, cannot use the following function transformation, need to be fix : @BUG@
    //let! kernelRowMajorOrder = RowMajorOrder.ToStorageOrder(transform) |> kernel |> defineKernelFunc
    //let! kernelColMajorOrder = ColMajorOrder.ToStorageOrder(transform) |> kernel |> defineKernelFunc

    let toStorageOrder (order:MatrixStorageOrder) (transform:Expr<int -> int -> 'P -> 'T -> 'U>) =
        match order with
        | RowMajorOrder -> transform
        | ColMajorOrder -> <@ fun c r p v -> (%transform) r c p v @>
    let! kernelRowMajorOrder = toStorageOrder RowMajorOrder transform |> kernel |> defineKernelFunc
    let! kernelColMajorOrder = toStorageOrder ColMajorOrder transform |> kernel |> defineKernelFunc

    let launchParam (m:Module) =
        let worker = m.Worker
        fun (hint:ActionHint) ->
            let blockSize = Dim3(32, 8)
            let gridSize = Dim3(32, 32) // TODO fine tune
            LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let param = param.Apply m
        let kernelRowMajorOrder = kernelRowMajorOrder.Apply m
        let kernelColMajorOrder = kernelColMajorOrder.Apply m
        let launchParam = launchParam m
        fun (hint:ActionHint) (order:MatrixStorageOrder) (rows:int) (cols:int) (param':'P) (input:DevicePtr<'T>) (output:DevicePtr<'U>) ->
            fun () ->
                let lp = launchParam hint
                let leadingDimension, majors, minors = order.ToStorageOrder(rows, cols)
                param.Scatter([| param' |])
                match order with
                | RowMajorOrder -> kernelRowMajorOrder.Launch lp leadingDimension majors minors input output
                | ColMajorOrder -> kernelColMajorOrder.Launch lp leadingDimension majors minors input output
            |> worker.Eval) }

