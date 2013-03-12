module Alea.CUDA.Extension.MovingAverage

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Util

let inline normalizedDifferenceKernel () =
    <@ fun n windowSize (normalizer:'T) (x:DevicePtr<'T>) (y:DevicePtr<'T>) ->
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start
        while i < n do
            y.[i] <- (x.[i + windowSize] - x.[i]) / normalizer
            i <- i + stride @>

let inline movingAverageTemplate () = cuda {

    let! kernel = normalizedDifferenceKernel () |> defineKernelFunc

    return PFunc(fun (m:Module) windowSize -> 
        let normalizer:float = NumericLiteralG.FromInt32 windowSize

        ()
        ) }



