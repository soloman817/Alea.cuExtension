module Sample.Sobol.GPU

open Microsoft.FSharp.Quotations
open Alea.CUDA

let kernel (convertExpr:Expr<uint32 -> 'T>) =
    <@ fun (numDimensions:int) (numVectors:int) (offset:int) (directions:deviceptr<uint32>) (numbers:deviceptr<'T>) ->
        let v = __shared__.Array<int>(Common.numDirections)

        let directions = directions + Common.numDirections * blockIdx.y
        let numbers = numbers + numVectors * blockIdx.y

        if threadIdx.x < Common.numDirections then
            v.[threadIdx.x] <- directions.[threadIdx.x] |> int
        __syncthreads()

        let i0 = threadIdx.x + blockIdx.x * blockDim.x
        let stride = gridDim.x * blockDim.x

        let mutable g = (i0 + offset) ^^^ ((i0 + offset) >>> 1)
        let mutable X = 0
        let mutable mask = 0
        let mutable k = 0
        while k < Common.numDirections do
            mask <- -(g &&& 1)
            X <- X ^^^ (mask &&& v.[k])
            g <- g >>> 1
            k <- k + 1

        if i0 < numVectors then numbers.[i0] <- X |> uint32 |> (%convertExpr)

        let v_log2stridem1 = v.[__nv_ffs(stride) - 2]
        let v_stridemask = stride - 1

        let mutable i = i0 + stride
        while i < numVectors do
            X <- X ^^^ v_log2stridem1 ^^^ v.[__nv_ffs(~~~((i + offset - stride) ||| v_stridemask)) - 1]
            numbers.[i] <- X |> uint32 |> (%convertExpr)
            i <- i + stride @>

let numThreadsPerBlock = 64

let launchParam numDimensions numVectors =
    let mutable dimGrid = dim3(1)
    let mutable dimBlock = dim3(1)

    dimGrid.y <- numDimensions

    dimGrid.x <- 1 + 31 / numDimensions
    if dimGrid.x > (numVectors / numThreadsPerBlock) then
        dimGrid.x <- (numVectors + numThreadsPerBlock - 1) / numThreadsPerBlock

    dimBlock.x <- numThreadsPerBlock

    LaunchParam(dimGrid, dimBlock)


