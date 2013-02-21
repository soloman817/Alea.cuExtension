module Alea.CUDA.Extension.Sobol

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension

type ISobol<'T when 'T:unmanaged> =
    abstract Directions : uint32[]
    // lpmod -> vectors -> offset -> directions -> output -> unit
    abstract Generate : LPModifier -> int -> int -> DevicePtr<uint32> -> DevicePtr<'T> -> unit

let [<ReflectedDefinition>] internal nDirections = 32

let [<ReflectedDefinition>] toUInt32 (x:uint32) = x

/// Transforms an uint32 random number to a float value 
/// on the interval [0,1] by dividing by 2^32-1
let [<ReflectedDefinition>] toFloat32 (x:uint32) = float32(x) * 2.3283064E-10f

/// Transforms an uint32 random number to a float value 
/// on the interval [0,1] by dividing by 2^32-1
let [<ReflectedDefinition>] toFloat64 (x:uint32) = float(x) * 2.328306437080797e-10   

let directions nDimensions =
    let directions = Array.zeroCreate<uint32> (nDimensions * nDirections)
        
    let directionsOffset = ref 0

    let setV (idx:int) (value:uint32) = directions.[!directionsOffset + idx] <- value
    let getV (idx:int) = directions.[!directionsOffset + idx]

    for dim = 0 to nDimensions - 1 do
        if dim = 0 then
            for i = 0 to nDirections - 1 do
                setV i (1u <<< (31 - i))
        else
            let d = SobolJoeKuo_6_21201.d.[dim]
            let a = SobolJoeKuo_6_21201.a.[dim]
            let o = SobolJoeKuo_6_21201.o.[dim]
            // the following is the old method, which is slow on x64 platform
            //let _, d, a, m = SobolJoeKuo_6_21201.primitivePolynomials.[dim]
            //let d = int(d)

            // the first direction numbers (up to the degree of the polynomial)
            // are simply v[i] = m[i] / 2^i (stored in Q0.32 format)
            for i = 0 to (d - 1) do
                setV i (SobolJoeKuo_6_21201.m.[o + i] <<< (31 - i))
                // the following is the old method, which is slow on x64 platform
                //setV i (m.[i] <<< (31 - i))

            // the remaining direction numbers are computed as described in the Bratley and Fox paper according to
            // v[i] = a[1]v[i-1] ^ a[2]v[i-2] ^ ... ^ a[v-1]v[i-d+1] ^ v[i-d] ^ v[i-d]/2^d
            for i = d to nDirections - 1 do
                setV i (getV(i-d) ^^^ (getV(i-d) >>> d))

                for j = 1 to d - 1 do
                    setV i (getV(i) ^^^ (((a >>> (d - 1 - j)) &&& 1u) * getV(i-j)))

        directionsOffset := !directionsOffset + nDirections

    directions

/// Create a Sobol kernel interface for a given random number type, 
/// either uint32, float32, or float64, as specified by the converter expression.
/// When generating n points of dimension d, the output will consist of n results 
/// from dimension 1, followed by n results from dimension 2, and so on up to dimension d.
let generator (convertExpr:Expr<uint32 -> 'T>) = cuda {
    let nThreadsPerBlock = 64

    let! kernel =
        <@ fun nDimensions nVectors offset (dDirections:DevicePtr<uint32>) (dOutput:DevicePtr<'T>) ->
            let convert = %convertExpr
            let v = __shared__(nDirections)

            let directionIdx idx = nDirections * blockIdx.y + idx

            // the ordering of the output is done as describe above, in order to have coalescing memory writes 
            let outputIdx idx = nVectors * blockIdx.y + idx

            if (threadIdx.x < nDirections) then
                v.[threadIdx.x] <- dDirections.[directionIdx threadIdx.x]
            __syncthreads()

            let i0 = threadIdx.x + blockIdx.x * blockDim.x
            let stride = gridDim.x * blockDim.x

            let mutable g:uint32 = (uint32(i0) + uint32(offset)) ^^^ ((uint32(i0) + uint32(offset)) >>> 1)

            let mutable X = 0u
            let mutable mask = 0u
            let mutable k = 0
            while k < nDirections do
                mask <- uint32(-(int(g) &&& 1))
                X <- X ^^^ (mask &&& v.[k])
                g <- g >>> 1
                k <- k + 1

            if (i0 < nVectors) then dOutput.[outputIdx i0] <- convert X

            let v_log2stridem1 = v.[DeviceFunction.__ffs(stride) - 2]
            let v_stridemask = stride - 1

            let mutable i = i0 + stride
            while i < nVectors do
                X <- X ^^^ v_log2stridem1 ^^^ v.[DeviceFunction.__ffs(~~~((i + offset - stride) ||| v_stridemask)) - 1]
                dOutput.[outputIdx i] <- convert X
                i <- i + stride @>
        |> defineKernelFuncWithName "sobol"

    let launchParam nDimensions nVectors =
        let mutable dimGrid = dim3(1)
        let mutable dimBlock = dim3(1)

        dimGrid.y <- nDimensions

        dimGrid.x <- 1 + 31 / nDimensions
        if dimGrid.x > (nVectors / nThreadsPerBlock) then
            dimGrid.x <- (nVectors + nThreadsPerBlock - 1) / nThreadsPerBlock

        dimBlock.x <- nThreadsPerBlock

        LaunchParam(dimGrid, dimBlock)

    return PFunc(fun (m:Module) ->
        let kernel = kernel.Apply m

        fun dimensions ->
            let directions = directions dimensions

            let launch lpmod vectors offset (directions:DevicePtr<uint32>) (output:DevicePtr<'T>) =
                let lp = launchParam dimensions vectors |> lpmod
                let offset = offset + 1
                kernel.Launch lp dimensions vectors offset directions output

            { new ISobol<'T> with
                member this.Directions = directions
                member this.Generate lpmod vectors offset directions output = launch lpmod vectors offset directions output
            } ) }
