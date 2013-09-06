module Test.Sample.XorShift7

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework
open Sample.XorShift7

// this template shows the full usage of this algorithm, it includes
// all memory management so you can get a feeling of how to use it
let template (convertExpr:Expr<uint32 -> 'T>) = cuda {
    let! kernel = GPU.kernel convertExpr |> Compiler.DefineKernel

    return Entry(fun (program:Program) ->
        let worker = program.Worker
        let kernel = program.Apply(kernel)

        // Copy pre-calculated bit-matrices, needed for jump-ahead
        // calculations, to the device memory.
        let jumpAheadMatrices = worker.Malloc(Data.jumpAheadMatrices)

        let generate (streams:int) (steps:int) (seed:uint32) (runs:int) (rank:int) =
            // first create state0 and scatter into device
            let state0 = Common.generateStartState seed
            use state0 = worker.Malloc(state0)
            // then create random number memory
            use numbers = worker.Malloc<'T>(streams * steps)
            // calculate the launch param
            let lp = GPU.launchParam streams
            // just launch
            kernel.Launch lp runs rank state0.Ptr jumpAheadMatrices.Ptr steps numbers.Ptr
            numbers.Gather()

        generate ) }

let correctness (convertD:Expr<uint32 -> 'T>) (convertH:uint32 -> 'T) (eps:float option) =
    use program = template convertD |> Util.load Worker.Default

    let streams = 114688
    let steps = 5
    let seed = 42u
    let runs = 1
    let rank = 0

    let hOutputs =
        let result = Array.zeroCreate (streams * steps)
        let mutable p = 0
        while (1 <<< p) < streams do p <- p + 1
        let state = Common.generateStartState seed
        let m = CPU.XorShift7Rng.Matrix.PowPow2(256 - p)
        let mutable v = CPU.Vector256(state)
        for i = 0 to streams - 1 do
            let rng = CPU.XorShift7Rng(v.Bits)
            for j = 0 to steps - 1 do
                let number = rng.NextUniformUInt32() |> convertH
                result.[j * streams + i] <- number
            v <- m * v
        result

    let dOutputs = program.Run streams steps seed runs rank

    if Util.debug then
        printfn "%A" hOutputs
        printfn "%A" dOutputs

    TestUtil.assertArrayEqual eps hOutputs dOutputs

let [<Test>] ``correctness on uint32``() = correctness <@ uint32 @> uint32 None
let [<Test>] ``correctness on float32``() = correctness <@ Common.toFloat32 @> Common.toFloat32 None
let [<Test>] ``correctness on float64``() = correctness <@ Common.toFloat64 @> Common.toFloat64 None

let test (convertD:Expr<uint32 -> 'T>) (convertH:uint32 -> 'T) streams steps seed runs rank =
    use program = template convertD |> Util.load Worker.Default

    let resultGPU = program.Run streams steps seed runs rank

    let m = 
        let mutable p = 0
        while (1 <<< p) < (streams * runs) do p <- p + 1
        CPU.XorShift7Rng.Matrix.PowPow2(256 - p)

    let mutable v =
        let state = Common.generateStartState seed
        CPU.Vector256(state)

    for i = 0 to rank * streams - 1 do
        v <- m * v

    for i = 0 to streams - 1 do
        let rng = CPU.XorShift7Rng(v.Bits)
        for j = 0 to steps - 1 do
            let numberCPU = rng.NextUniformUInt32() |> convertH
            let numberGPU = resultGPU.[j * streams + i]
            Assert.AreEqual(numberCPU, numberGPU)
        v <- m * v

let [<Test>] ``uint32: 4096x1000 seed=42u runs=1 rank=0``() = test <@ uint32 @> uint32 4096 1000 42u 1 0
let [<Test>] ``uint32: 4096x1000 seed=42u runs=5 rank=3``() = test <@ uint32 @> uint32 4096 1000 42u 5 3
let [<Test>] ``float32: 4096x1000 seed=42u runs=1 rank=0``() = test <@ Common.toFloat32 @> Common.toFloat32 4096 1000 42u 1 0
let [<Test>] ``float32: 4096x1000 seed=42u runs=5 rank=3``() = test <@ Common.toFloat32 @> Common.toFloat32 4096 1000 42u 5 3
let [<Test>] ``float64: 4096x1000 seed=42u runs=1 rank=0``() = test <@ Common.toFloat64 @> Common.toFloat64 4096 1000 42u 1 0
let [<Test>] ``float64: 4096x1000 seed=42u runs=5 rank=3``() = test <@ Common.toFloat64 @> Common.toFloat64 4096 1000 42u 5 3

