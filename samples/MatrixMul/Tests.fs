module Test.Sample.MatrixMul

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework
open Sample.MatrixMul

[<Test>]
let correctness() =
    let template (blockSize:int) = cuda {
        let! kernel = blockSize |> GPU.kernel |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply(kernel)
            let cpuCalc = CPU.calc ()
            let gpuCalc = GPU.calc blockSize worker kernel
            
            let run (dimA:int*int) (dimB:int*int) =
                let wA, hA = dimA
                let wB, hB = dimB

                let sizeA = wA * hA
                let sizeB = wB * hB

                let A = Array.init sizeA (TestUtil.genRandomSingle -5.0 5.0)
                let B = Array.init sizeB (TestUtil.genRandomSingle -4.0 4.0)

                let hOutput = cpuCalc A B wA wB
                let dOutput = gpuCalc A B wA wB
                TestUtil.assertArrayEqual (Some 1e-3) hOutput dOutput

            run ) }

    let blockSize = 32
    let program = blockSize |> template |> Util.load Worker.Default
    ((320, 320), (640, 320)) ||> program.Run
    ((320, 320), (640, 320)) ||> program.Run
    ((320, 320), (640, 320)) ||> program.Run
    ((640, 640), (640, 640)) ||> program.Run

[<Test>]
let ``just compile``() =
    let template (blockSize:int) = cuda {
        let! kernel = blockSize |> GPU.kernel |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply(kernel)
            let cpuCalc = CPU.calc ()
            let gpuCalc = GPU.calc blockSize worker kernel
            
            let run (dimA:int*int) (dimB:int*int) =
                let wA, hA = dimA
                let wB, hB = dimB

                let sizeA = wA * hA
                let sizeB = wB * hB

                let A = Array.init sizeA (TestUtil.genRandomSingle -5.0 5.0)
                let B = Array.init sizeB (TestUtil.genRandomSingle -4.0 4.0)

                let hOutput = cpuCalc A B wA wB
                let dOutput = gpuCalc A B wA wB
                TestUtil.assertArrayEqual (Some 1e-3) hOutput dOutput

            run ) }

    let blockSize = 32
    let program = blockSize |> template |> Util.load Worker.Default
    ()
