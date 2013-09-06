module Sample.XorShift7.GPU

open System
open System.Runtime.InteropServices
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.NativeInterop
open Alea.CUDA
open Alea.CUDA.Utilities

#nowarn "9"
#nowarn "51"

[<Struct;StructLayout(LayoutKind.Explicit)>]
type XorShift7 =
    [<FieldOffset(0);EmbeddedArrayField(8)>] val mutable _state : uint32
    [<FieldOffset(32)>] val mutable Index : int

    [<EmbeddedArrayProperty("_state")>]
    member this.State
        with get (i:int) : uint32 = failwith "device only"
        and set (i:int) (value:uint32) : unit = failwith "device only"

    [<ReflectedDefinition>]
    static member Size = __sizeof<XorShift7>()

    [<ReflectedDefinition>]
    static member Bless (buffer:deviceptr<byte>) =
        buffer.Reinterpret<XorShift7>().Ref(0)

    [<ReflectedDefinition>]
    static member Init (xorshift7:XorShift7 ref) (buffer:deviceptr<uint32>) =
        for i = 0 to 7 do xorshift7.contents.State(i) <- buffer.[i]
        xorshift7.contents.Index <- 0

    [<ReflectedDefinition>]
    static member Next (xorshift7:XorShift7 ref) =
        let mutable r = 0u
        let mutable t = 0u
        let index = xorshift7.contents.Index

        t <- xorshift7.contents.State((index + 7) &&& 0x7)
        t <- t ^^^ (t <<< 13)
        r <- t ^^^ (t <<< 9)
        t <- xorshift7.contents.State((index + 4) &&& 0x7)
        r <- r ^^^ (t ^^^ (t <<< 7))
        t <- xorshift7.contents.State((index + 3) &&& 0x7)
        r <- r ^^^ (t ^^^ (t >>> 3))
        t <- xorshift7.contents.State((index + 1) &&& 0x7)
        r <- r ^^^ (t ^^^ (t >>> 10))
        t <- xorshift7.contents.State(index)
        t <- t ^^^ (t >>> 7)
        r <- r ^^^ (t ^^^ (t <<< 24))
        xorshift7.contents.State(index) <- r
        xorshift7.contents.Index <- (index + 1) &&& 0x7

        r

[<ReflectedDefinition>]
let jumpAhead (numThreads:int) (threadRank:int)
              (stateStart:deviceptr<uint32>) (jumpAheadMatrices:deviceptr<uint32>)
              (jumpAheadMatrixCache:deviceptr<uint32>) (state:deviceptr<uint32>) =

    let numThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z
    let threadRankInBlock = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x

    let mutable p = 0
    while (1 <<< p) < numThreads do p <- p + 1
    let matrixSize = 256 * 8
    let mutable matrix = jumpAheadMatrices + (32 - p) * matrixSize

    // init rng state to start state
    for i = 0 to 7 do state.[i] <- stateStart.[i]

    for i = 0 to p - 1 do
        let statePrev = __local__.Array<uint32>(8)
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

let kernel (convertExpr:Expr<uint32 -> 'T>) =
    <@ fun (numRuns:int) (runRank:int) (stateStart:deviceptr<uint32>) (jumpAheadMatrices:deviceptr<uint32>) (numSteps:int) (results:deviceptr<'T>) ->
        // Shared memory declaration; aligned to 4 because of the
        // intended usage as the cache for current row of the jump-ahead
        // matrix. 
        let sharedData = __shared__.Extern<byte>(4)

        // Calculate ranks of the block within the grid, and the thread
        // within the block, as well as rank of the thread within the
        // whole grid.
        let numBlocks = gridDim.x * gridDim.y * gridDim.z
        let blockRank = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x
        let numThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z
        let numThreads = numBlocks * numThreadsPerBlock
        let threadRankInBlock = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x
        let threadRank = blockRank * numThreadsPerBlock + threadRankInBlock

        // Perform jumping ahead, taking into account the total number
        // of thread on all devices, as well as current thread rank
        // within all thread on all devices.
        let state = __local__.Array<uint32>(8) |> __ptrofarray
        jumpAhead (numRuns * numThreads) (runRank * numThreads + threadRank) stateStart jumpAheadMatrices (sharedData.Reinterpret<uint32>()) state

        // Use corresponding piece of shared memory for xorshift7 RNG
        // data structure, and intialized RNG with the state calculated
        // through above jump-ahead procedure.
        let rng = XorShift7.Bless(sharedData + threadRankInBlock * XorShift7.Size)
        XorShift7.Init rng state

        let mutable index = threadRank
        for i = 0 to numSteps - 1 do
            results.[index] <- (%convertExpr) (XorShift7.Next rng)
            index <- index + numThreads @>

let blockSize = dim3(32, 8)

let launchParam numStreams =
    let numThreadsPerBlock = blockSize.Size
    if numStreams % numThreadsPerBlock <> 0 then failwithf "streams should be multiple of %d" numThreadsPerBlock
    let gridSize = dim3(numStreams / numThreadsPerBlock)
    let sharedMemorySize = sizeof<XorShift7> * numThreadsPerBlock
    LaunchParam(gridSize, blockSize, sharedMemorySize)
