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

    let! scanner = Scan.sum Scan.Planner.Default 
    let! windowDifference = normalizedDifferenceKernel () |> defineKernelFunc

    return PFunc(fun (m:Module) windowSize -> 
        let worker = m.Worker
        let scanner = scanner.Apply m
        let normalizer:float = NumericLiteralG.FromInt32 windowSize
        fun (inclusive:bool) (values:DArray<'T>) ->
            let n = values.Length
            let scanner = scanner n 
            pcalc {
                let! ranges = DArray.scatterInBlob worker scanner.Ranges
                let! rangeTotals = DArray.createInBlob worker scanner.NumRangeTotals
                let! results = DArray.createInBlob worker n
                do! PCalc.action (fun hint -> scanner.Scan hint ranges.Ptr rangeTotals.Ptr values.Ptr results.Ptr inclusive)
                return results } ) }




