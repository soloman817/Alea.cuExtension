module Alea.CUDA.Extension.Random.XorShift7

#nowarn "9"
#nowarn "51"

open System.Runtime.InteropServices
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.NativeInterop
open Alea.CUDA
open Alea.CUDA.Extension

[<ReflectedDefinition>]
let jumpAhead (numThreads:int) (threadRank:int)
              (stateStart:RPtr<uint32>) (jumpAheadMatrices:RPtr<uint32>)
              (jumpAheadMatrixCache:RWPtr<uint32>) (state:RWPtr<uint32>) =

    let numThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z
    let threadRankInBlock = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x

    let mutable p = 0
    while (1 <<< p) < numThreads do p <- p + 1
    let matrixSize = 256 * 8
    let mutable matrix = jumpAheadMatrices + (32 - p) * matrixSize

    // init rng state to start state
    for i = 0 to 7 do state.[i] <- stateStart.[i]

    for i = 0 to p - 1 do
        let statePrev = __local__<uint32>(8)
        for j = 0 to 7 do statePrev.[j] <- state.[j]

        for j = 0 to 7 do
            let mutable stateWord = 0u

            for k = 0 to 31 do
                __syncthreads()
                let mutable l = threadRankInBlock
                while l < 8 do
                    jumpAheadMatrixCache.[l] <- matrix.[l]
                    l <- l + numThreadsPerBlock
                matrix <- matrix + 8

                __syncthreads()
                if (threadRank &&& (1 <<< i) <> 0) then
                    let mutable partialSums = 0u
                    for l = 0 to 7 do
                        partialSums <- partialSums ^^^ (jumpAheadMatrixCache.[l] &&& statePrev.[l])

                    let sum = partialSums
                    let sum = (sum >>> 16) ^^^ (sum &&& 0xffffu)
                    let sum = (sum >>> 8) ^^^ (sum &&& 0xffu)
                    let sum = (sum >>> 4) ^^^ (sum &&& 0xfu)
                    let sum = (sum >>> 2) ^^^ (sum &&& 0x3u)
                    let sum = (sum >>> 1) ^^^ (sum &&& 0x1u)

                    stateWord <- stateWord <<< 1
                    stateWord <- stateWord ||| sum

            if (threadRank &&& (1 <<< i) <> 0) then state.[j] <- stateWord

[<Struct;StructLayout(LayoutKind.Explicit)>]
type XorShift7 =
    [<FieldOffset(0);FixedArrayField(8)>] val mutable _state : uint32
    [<FieldOffset(32)>] val mutable index : int 

    [<FixedArrayProperty("_state")>]
    member this.state
        with get i =
            if i >= 0 && i < 8 then NativePtr.get &&this._state i
            else failwith "index out of range"
        and set i v =
            if i >= 0 && i < 8 then NativePtr.set &&this._state i v
            else failwith "index out of range"

    [<ReflectedDefinition>]
    static member Size = sizeof<XorShift7>

    [<ReflectedDefinition>]
    static member Bless(buffer:RWPtr<byte>) = buffer.Reinterpret<XorShift7>()

    [<ReflectedDefinition>]
    static member Init(xorshift7:XorShift7 ref, buffer:RPtr<uint32>) =
        for i = 0 to 7 do xorshift7.contents.state(i) <- buffer.[i]
        xorshift7.contents.index <- 0

    [<ReflectedDefinition>]
    static member Next(xorshift7:XorShift7 ref) =
        let mutable r = 0u
        let mutable t = 0u
        let index = xorshift7.contents.index

        t <- xorshift7.contents.state((index + 7) &&& 0x7)
        t <- t ^^^ (t <<< 13)
        r <- t ^^^ (t <<< 9)
        t <- xorshift7.contents.state((index + 4) &&& 0x7)
        r <- r ^^^ (t ^^^ (t <<< 7))
        t <- xorshift7.contents.state((index + 3) &&& 0x7)
        r <- r ^^^ (t ^^^ (t >>> 3))
        t <- xorshift7.contents.state((index + 1) &&& 0x7)
        r <- r ^^^ (t ^^^ (t >>> 10))
        t <- xorshift7.contents.state(index)
        t <- t ^^^ (t >>> 7)
        r <- r ^^^ (t ^^^ (t <<< 24))
        xorshift7.contents.state(index) <- r
        xorshift7.contents.index <- (index + 1) &&& 0x7

        r

let LCG_A = 1664525u
let LCG_C = 1013904223u

let generateStartState (seed:uint32) =
    let state = Array.zeroCreate 8
    state.[0] <- seed
    for i = 1 to 7 do state.[i] <- LCG_A * state.[i - 1] + LCG_C
    state

type XorShift7Rng<'T when 'T:unmanaged> =
    {
        StreamUnit : int
        JumpAheadMatrices : uint32[]
        GenerateStartState : uint32 -> uint32[]
        // hint -> jumpAheadMatrices -> streams -> steps -> startState -> runs -> rank -> result
        Generate : ActionHint -> DevicePtr<uint32> -> int -> int -> DevicePtr<uint32> -> int -> int -> DevicePtr<'T> -> unit
    }

let generator (convertExpr:Expr<uint32 -> 'T>) = cuda {
    let! kernel =
        <@ fun (numRuns:int) (runRank:int) (stateStart:DevicePtr<uint32>) (jumpAheadMatrices:DevicePtr<uint32>) (numSteps:int) (results:DevicePtr<'T>) ->
            let sharedData = __extern_align_shared__<byte>(4)

            let numBlocks = gridDim.x * gridDim.y * gridDim.z
            let blockRank = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x
            let numThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z
            let numThreads = numBlocks * numThreadsPerBlock
            let threadRankInBlock = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x
            let threadRank = blockRank * numThreadsPerBlock + threadRankInBlock

            let state = __local__<uint32>(8).Ptr(0)
            jumpAhead (numRuns * numThreads) (runRank * numThreads + threadRank) stateStart jumpAheadMatrices (sharedData.Reinterpret<uint32>()) state

            let rng = XorShift7.Bless(sharedData + threadRankInBlock * XorShift7.Size).Ref(0)
            XorShift7.Init(rng, state)

            let mutable index = threadRank
            for i = 0 to numSteps - 1 do
                results.[index] <- (%convertExpr) (XorShift7.Next(rng))
                index <- index + numThreads @>
        |> defineKernelFunc

    let blockSize = dim3(32, 8)

    let launchParam numStreams =
        let numThreadsPerBlock = blockSize.Size
        if numStreams % numThreadsPerBlock <> 0 then failwithf "streams should be multiple of %d" numThreadsPerBlock
        let gridSize = dim3(numStreams / numThreadsPerBlock)
        let sharedMemorySize = sizeof<XorShift7> * numThreadsPerBlock
        LaunchParam(gridSize, blockSize, sharedMemorySize)

    return PFunc(fun (m:Module) ->
        let kernel = kernel.Apply m

        let launch (hint:ActionHint) jumpAheadMatrices numStreams numSteps startState numRuns runRank (results:DevicePtr<'T>) =
            let lp = launchParam numStreams |> hint.ModifyLaunchParam
            kernel.Launch lp numRuns runRank startState jumpAheadMatrices numSteps results

        { StreamUnit = blockSize.Size
          JumpAheadMatrices = XorShift7Data.jumpAheadMatrices
          GenerateStartState = generateStartState
          Generate = launch } ) }

/// Transforms an uint32 random number to a float value 
/// on the interval [0,1] by dividing by 2^32-1
let [<ReflectedDefinition>] toFloat32 (x:uint32) = float32(x) * 2.3283064E-10f

/// Transforms an uint32 random number to a float value 
/// on the interval [0,1] by dividing by 2^32-1
let [<ReflectedDefinition>] toFloat64 (x:uint32) = float(x) * 2.328306437080797e-10   
