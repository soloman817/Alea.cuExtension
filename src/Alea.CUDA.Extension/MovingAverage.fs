module Alea.CUDA.Extension.MovingAverage

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Util

// TODO 
//    - extend to multiple windows at once, as we have the running sums 
//      just use the normalized difference kernel multiple times
//      use shared memory here, similar in spirit as in the direct implementation below
//    - write a matrix version, which operates on multiple rows of data
let movingAverager () = cuda {

    let! scanner = Scan.sum Scan.Planner.Default 
    let! normalizedDifferenceKernel = 
        <@ fun n windowSize (normalizer:float) (x:DevicePtr<float>) (y:DevicePtr<float>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start + windowSize
            while i < n do
                y.[i - windowSize] <- (x.[i] - x.[i - windowSize]) / normalizer
                i <- i + stride
        @> |> defineKernelFunc

    let launchParam (m:Module) (hint:ActionHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) windowSize -> 
        let worker = m.Worker
        let scanner = scanner.Apply m
        let normalizer:float = NumericLiteralG.FromInt32 windowSize
        fun (values:DArray<float>) ->
            let n = values.Length
            let scanner = scanner n            
            pcalc {
                let! ranges = DArray.scatterInBlob worker scanner.Ranges
                let! rangeTotals = DArray.createInBlob worker scanner.NumRangeTotals
                let! sums = DArray.createInBlob worker n
                // note that we assume that values have an additonal zero a the end so n - windowSize 
                // is the right length of the moving average vector, see comment below as well
                let! results = DArray.createInBlob worker (n - windowSize)
                do! PCalc.action (fun hint -> 
                    // the values must have a zero appended in order to get full exclusive scan
                    // attention this means in particular that n is actually original n + 1
                    scanner.Scan hint ranges.Ptr rangeTotals.Ptr values.Ptr sums.Ptr false
                    let lp = launchParam m hint n
                    normalizedDifferenceKernel.Launch m lp n windowSize normalizer sums.Ptr results.Ptr                  
                )
                return results } ) }

module MovingAvDirect =

    /// Moving average 
    /// Slightly different version than above scan based implementation as it produces a vector of same length as the initial data, e.g.
    /// values = [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0|]
    /// scan based moving averag:
    ///              [|2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0|]
    /// this version: 
    ///    [|1.0; 1.5; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0|]
    let movingAverage () =
        <@ fun windowSize blockSize (dValues:DevicePtr<float>) (dResults:DevicePtr<float>) ->
            let idx = threadIdx.x
            let iGlobal = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x
                      
            let shared = __extern_shared__<float>()

            // load one data element from global to shared memory
            shared.[idx + (windowSize - 1)] <- dValues.[iGlobal]

            // load additional data elements into shared memory if block size < windowSize
            let mutable j = 0
            while j < (windowSize-2)/blockSize + 1 do
                if idx + j*blockSize < windowSize - 1 then
                    shared.[idx + j*blockSize] <- dValues.[iGlobal + j * blockSize - (windowSize - 1)]
                j <- j + 1

            __syncthreads()

            let mutable temp = 0.0
            let mutable k = 0
            while k <= min (windowSize - 1) iGlobal do
                temp <- temp + shared.[idx - k + (windowSize - 1)]
                k <- k + 1

            dResults.[iGlobal] <- temp / float(min windowSize (iGlobal + 1)) 
        @>

    let movingAverager () = cuda {
        let! movingAverageKernel = movingAverage () |> defineKernelFunc
            
        return PFunc(fun (m:Module)  -> 
            let worker = m.Worker
            let maxBlockSize = 256
            fun (windowSize:int) (values:DArray<float>) ->
                let n = values.Length
                let sharedMem = (maxBlockSize + windowSize - 1) * sizeof<float>
                let blockSize = min n maxBlockSize
                let gridSizeX = (n - 1) / blockSize + 1
                let lp = 
                    if gridSizeX <= 65535 then 
                        LaunchParam(gridSizeX, blockSize, sharedMem)
                    else
                        let gridSizeY = 1 + (n - 1) / (blockSize * 65535)
                        let gridSizeX = 1 + (n - 1) / (blockSize * gridSizeY)
                        LaunchParam(dim3(gridSizeX, gridSizeY), dim3(blockSize), sharedMem)
                let movingAverage = movingAverageKernel.Launch m lp
                pcalc {
                    let! results = DArray.createInBlob worker n
                    do! PCalc.action (fun hint -> movingAverage windowSize blockSize values.Ptr results.Ptr)
                    return results } ) }
