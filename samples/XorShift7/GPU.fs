﻿module Sample.XorShift7.GPU

open System
open System.Runtime.InteropServices
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.NativeInterop
open Alea.CUDA
open Alea.CUDA.Utilities

#nowarn "9"
#nowarn "51"

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

// In this algorithm, acturally the struct is not used to carry data
// from host to device, and we just reinterpret part of our shared
// memory into a struct to operate easier. So we could use ref class
// here. The RefClass attribute is an attribute builder in Alea.CUDA.Utilites,
// it can automatically lookup all properties that is marked as 
// RefClassField (or its child class RefClassArrayField), and build the
// underlying data storage struct. But inside kernel, all ref class
// are used as a pointer to the underlying struct, so you should be
// careful, like when you need know the size of it, see below for more
// detail.
[<RefClass>]
type XorShift7() =

    // with this mark, the builder will build an array ref class
    [<RefClassArrayField(8)>]
    member this.State : uint32[] = failwith "device only"

    // with this mark, the builder will build property get/set handles
    [<RefClassField>]
    member this.Index 
        with get () : int = failwith "device only"
        and set (value:int) : unit = failwith "device only"

    // Important here, we should use __sizeofclass instead of __sizeof
    // because now XorShift7 is ref class, so in ir, it is represented
    // by pointer, so if you use __sizeof, you get 4 or 8 (on 64bit),
    // that is just size of pointer! Use __sizeofclass to get the
    // content size!
    [<ReflectedDefinition>]
    static member Size = __sizeofclass<XorShift7>()

    // Bless is reinterpret a buffer into our ref class type, then use
    // __ptrtoobj to construct the ref class object, which will use
    // the memory pointed by the pointer as the underlying storage.
    [<ReflectedDefinition>]
    static member Bless(buffer:deviceptr<byte>) =
        buffer.Reinterpret<XorShift7>() |> __ptrtoobj

    // Now, you can code method, and this pointer because an ir pointer
    // points to the underlying storage. You can use __debug(_) to show
    // what it is like in ir.
    [<ReflectedDefinition>]
    member this.Init(buffer:deviceptr<uint32>) =
        //__debug(this) // will print out the underlying ir value
        for i = 0 to 7 do this.State.[i] <- buffer.[i]
        this.Index <- 0

    // With this pointer, you now got a real ref struct, so you don't 
    // bother using xxxx.contents.xx stuff. Those are used in value
    // type struct, because struct is value type, so you cannot add
    // reflected definition on its member method cause that doesn't 
    // have a ref this pointer!
    // Also, ref class allows to be generic! This is quite powerful
    // But the bad thing is ref class is not suit for transfer data
    // from host to device. Although Alea.cuBase provides 
    // To/From Unmanaged Marshaller build framework, But that is
    // very slow. So the data that will be exchanged between host and
    // device should better be declared as value class, but once it is
    // in device, we can eaisly reinterpret and recreate ref class object
    // by functions like __ptrtoobj. value class struct is unmanaged
    // so it don't need to be reinterpret before it is send into device,
    // but struct cannot be generic. To make generic usage on struct,
    // I think we should look for Type Provider feature of F#.
    [<ReflectedDefinition>]
    member this.Next() =
        let mutable r = 0u
        let mutable t = 0u
        let index = this.Index

        t <- this.State.[(index + 7) &&& 0x7]
        t <- t ^^^ (t <<< 13)
        r <- t ^^^ (t <<< 9)
        t <- this.State.[(index + 4) &&& 0x7]
        r <- r ^^^ (t ^^^ (t <<< 7))
        t <- this.State.[(index + 3) &&& 0x7]
        r <- r ^^^ (t ^^^ (t >>> 3))
        t <- this.State.[(index + 1) &&& 0x7]
        r <- r ^^^ (t ^^^ (t >>> 10))
        t <- this.State.[index]
        t <- t ^^^ (t >>> 7)
        r <- r ^^^ (t ^^^ (t <<< 24))
        this.State.[index] <- r
        this.Index <- (index + 1) &&& 0x7

        r

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
        rng.Init(state)

        let mutable index = threadRank
        for i = 0 to numSteps - 1 do
            results.[index] <- (%convertExpr) (rng.Next())
            index <- index + numThreads @>

let blockSize = dim3(32, 8)

let launchParam numStreams =
    let numThreadsPerBlock = blockSize.Size
    if numStreams % numThreadsPerBlock <> 0 then failwithf "streams should be multiple of %d" numThreadsPerBlock
    let gridSize = dim3(numStreams / numThreadsPerBlock)
    let sharedMemorySize = XorShift7.Size * numThreadsPerBlock
    LaunchParam(gridSize, blockSize, sharedMemorySize)
