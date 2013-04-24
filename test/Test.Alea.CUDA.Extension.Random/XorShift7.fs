module Test.Alea.CUDA.Extension.Random.XorShift7

open System.IO
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Random
open Test.Alea.CUDA.Extension

let worker = getDefaultWorker()
let xorshift7UniformUInt32 = worker.LoadPModule(PRandom.xorshift7 <@ Util.identity @>).Invoke
let xorshift7UniformFloat32 = worker.LoadPModule(PRandom.xorshift7 <@ XorShift7.toFloat32 @>).Invoke
let xorshift7UniformFloat64 = worker.LoadPModule(PRandom.xorshift7 <@ XorShift7.toFloat64 @>).Invoke

[<Test>]
let simple() =
    let streams = 114688
    let steps = 5
    let runs = 1
    let rank = 0
    let seed = 42u

    let xorshift7 = worker.LoadPModule(PRandom.xorshift7 <@ Util.identity @>).Invoke

    let calc = pcalc {
        let! rn = xorshift7 streams steps seed runs rank
        return! rn.Numbers.Gather() }

    let rn = calc |> PCalc.run
    for i = 0 to 10 do
        printf "%d " rn.[i]
    printfn ""

    let calcCPU () =
        let result = Array.zeroCreate (streams * steps)
        let mutable p = 0
        while (1 <<< p) < streams do p <- p + 1
        let state = XorShift7.generateStartState seed
        let m = XorShift7Gold.XorShift7Rng.Matrix.PowPow2(256 - p)
        let mutable v = XorShift7Gold.Vector256(state)
        for i = 0 to streams - 1 do
            let rng = XorShift7Gold.XorShift7Rng(v.Bits)
            for j = 0 to steps - 1 do
                let number = rng.NextUniformUInt32()
                result.[j * streams + i] <- number
            v <- m * v
        result

    let rncpu = calcCPU()
    for i = 0 to 10 do
        printf "%d " rncpu.[i]
    printfn ""

let testUniformUInt32 streams steps seed runs rank =
    let calc = pcalc {
        let! rn = xorshift7UniformUInt32 streams steps seed runs rank
        return! rn.Numbers.Gather() }

    let resultGPU = calc |> PCalc.run

    let verify () =
        let m =
            let mutable p = 0
            while (1 <<< p) < (streams * runs) do p <- p + 1
            XorShift7Gold.XorShift7Rng.Matrix.PowPow2(256 - p)

        let mutable v =
            let state = XorShift7.generateStartState seed
            XorShift7Gold.Vector256(state)

        for i = 0 to rank * streams - 1 do
            v <- m * v

        for i = 0 to streams - 1 do
            let rng = XorShift7Gold.XorShift7Rng(v.Bits)
            for j = 0 to steps - 1 do
                let numberCPU = rng.NextUniformUInt32()
                let numberGPU = resultGPU.[j * streams + i]
                Assert.AreEqual(numberCPU, numberGPU)
            v <- m * v

    verify()

let [<Test>] ``TestUniformUInt32: 4096 1000 42u 1 0`` () = testUniformUInt32 4096 1000 42u 1 0
let [<Test>] ``TestUniformUInt32: 4096 1000 42u 5 3`` () = testUniformUInt32 4096 1000 42u 5 3

let testUniformFloat32 streams steps seed runs rank =
    let calc = pcalc {
        let! rn = xorshift7UniformFloat32 streams steps seed runs rank
        return! rn.Numbers.Gather() }

    let resultGPU = calc |> PCalc.run

    let verify () =
        let m =
            let mutable p = 0
            while (1 <<< p) < (streams * runs) do p <- p + 1
            XorShift7Gold.XorShift7Rng.Matrix.PowPow2(256 - p)

        let mutable v =
            let state = XorShift7.generateStartState seed
            XorShift7Gold.Vector256(state)

        for i = 0 to rank * streams - 1 do
            v <- m * v

        for i = 0 to streams - 1 do
            let rng = XorShift7Gold.XorShift7Rng(v.Bits)
            for j = 0 to steps - 1 do
                let numberCPU = rng.NextUniformFloat32()
                let numberGPU = resultGPU.[j * streams + i]
                Assert.AreEqual(numberCPU, numberGPU)
            v <- m * v

    verify()

let [<Test>] ``TestUniformFloat32: 4096 1000 42u 1 0`` () = testUniformFloat32 4096 1000 42u 1 0
let [<Test>] ``TestUniformFloat32: 4096 1000 42u 5 3`` () = testUniformFloat32 4096 1000 42u 5 3

let testUniformFloat64 streams steps seed runs rank =
    let calc = pcalc {
        let! rn = xorshift7UniformFloat64 streams steps seed runs rank
        return! rn.Numbers.Gather() }

    let resultGPU = calc |> PCalc.run

    let verify () =
        let m =
            let mutable p = 0
            while (1 <<< p) < (streams * runs) do p <- p + 1
            XorShift7Gold.XorShift7Rng.Matrix.PowPow2(256 - p)

        let mutable v =
            let state = XorShift7.generateStartState seed
            XorShift7Gold.Vector256(state)

        for i = 0 to rank * streams - 1 do
            v <- m * v

        for i = 0 to streams - 1 do
            let rng = XorShift7Gold.XorShift7Rng(v.Bits)
            for j = 0 to steps - 1 do
                let numberCPU = rng.NextUniformFloat64()
                let numberGPU = resultGPU.[j * streams + i]
                Assert.AreEqual(numberCPU, numberGPU)
            v <- m * v

    verify()

let [<Test>] ``TestUniformFloat64: 4096 1000 42u 1 0`` () = testUniformFloat64 4096 1000 42u 1 0
let [<Test>] ``TestUniformFloat64: 4096 1000 42u 5 3`` () = testUniformFloat64 4096 1000 42u 5 3

let performanceFloat32 streams steps seed runs rank =
    let calc = pcalc {
        let! rn = xorshift7UniformUInt32 streams steps seed runs rank
        do! PCalc.force() }

    let _, ktc = calc |> PCalc.runWithKernelTiming 3
    ktc.Dump()

let [<Test>] ``PerformanceUniformFloat32: 2048  1000 42u 1 0`` () = performanceFloat32 2048  1000 42u 1 0
let [<Test>] ``PerformanceUniformFloat32: 2048  5000 42u 1 0`` () = performanceFloat32 2048  5000 42u 1 0
let [<Test>] ``PerformanceUniformFloat32: 2048 10000 42u 1 0`` () = performanceFloat32 2048 10000 42u 1 0
let [<Test>] ``PerformanceUniformFloat32: 2048 20000 42u 1 0`` () = performanceFloat32 2048 30000 42u 1 0
let [<Test>] ``PerformanceUniformFloat32: 2048 30000 42u 1 0`` () = performanceFloat32 2048 30000 42u 1 0
let [<Test>] ``PerformanceUniformFloat32: 2048 40000 42u 1 0`` () = performanceFloat32 2048 40000 42u 1 0
let [<Test>] ``PerformanceUniformFloat32: 2048 50000 42u 1 0`` () = performanceFloat32 2048 50000 42u 1 0


