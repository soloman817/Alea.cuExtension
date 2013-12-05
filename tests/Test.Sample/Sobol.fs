module Test.Sample.Sobol

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.TestUtilities
open NUnit.Framework
open Sample.Sobol

let correctness (convertD:Expr<uint32 -> 'T>) (convertH:uint32 -> 'T) =
    let template = cuda {
        let! kernel = GPU.kernel convertD |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply(kernel)

            let generate (numDimensions:int) (numVectors:int) (offset:int) =
                let directions = Common.directions numDimensions
                use directions = worker.Malloc(directions)
                use numbers = worker.Malloc<'T>(numDimensions * numVectors)
                let offset = offset + 1
                let lp = GPU.launchParam numDimensions numVectors
                kernel.Launch lp numDimensions numVectors offset directions.Ptr numbers.Ptr
                numbers.Gather()

            generate ) }

    use program = template |> Util.load Worker.Default

    //let numDimensions = 4096
    let numDimensions = 1024
    let numVectors = 4096
    let offset = 234

    let hOutputs = 
        let rng = CPU.Sobol(numDimensions, offset)
        Array.init numVectors (fun _ -> rng.NextPoint)
        |> Array.concat
        |> CPU.reorderPoints numDimensions numVectors        
        |> Array.map convertH

    let dOutputs = program.Run numDimensions numVectors offset

    TestUtil.assertArrayEqual None hOutputs dOutputs

let [<Test>] ``correctness on uint32``() = correctness <@ uint32 @> uint32
let [<Test>] ``correctness on float32``() = correctness <@ Common.toFloat32 @> Common.toFloat32
let [<Test>] ``correctness on float64``() = correctness <@ Common.toFloat64 @> Common.toFloat64

let test (convertD:Expr<uint32 -> 'T>) (convertH:uint32 -> 'T) (numDimensions:int) (numVectors:int) (numIters:int) =
    let template = cuda {
        let! kernel = GPU.kernel convertD |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply(kernel)

            let test () =
                let directions = Common.directions numDimensions
                use directions = worker.Malloc(directions)
                use numbers = worker.Malloc<'T>(numDimensions * numVectors)
                let lp = GPU.launchParam numDimensions numVectors

                for i = 0 to numIters - 1 do
                    let offset = i * numVectors

                    let hOutputs =
                        let rng = CPU.Sobol(numDimensions, offset)
                        Array.init numVectors (fun _ -> rng.NextPoint)
                        |> Array.concat
                        |> CPU.reorderPoints numDimensions numVectors
                        |> Array.map convertH

                    let dOutputs =
                        let offset = offset + 1
                        kernel.Launch lp numDimensions numVectors offset directions.Ptr numbers.Ptr
                        numbers.Gather()

                    //for i = 0 to numVectors - 1 do
                    //    for d = 0 to numDimensions - 1 do
                    //        let rCuda = dOutputs.[d * numVectors + i]
                    //        let rGold = hOutputs.[d * numVectors + i]
                    //        printfn "%d.%d: rGold(%A) rCuda(%A)" i d rGold rCuda

                    TestUtil.assertArrayEqual None hOutputs dOutputs

            test ) }

    use program = template |> Util.load Worker.Default
    program.Run()

let [<Test>]``uint32: 32x4 iter=1``() = test <@ uint32 @> uint32 32 4 1
let [<Test>]``uint32: 32x256 iter=5``() = test <@ uint32 @> uint32 32 256 5
let [<Test>]``uint32: 32x4096 iter=5``() = test <@ uint32 @> uint32 32 4096 5
let [<Test>]``uint32: 32x65535 iter=5``() = test <@ uint32 @> uint32 32 65535 5
let [<Test>]``uint32: 1024x256 iter=5``() = test <@ uint32 @> uint32 1024 256 5
let [<Test>]``uint32: 1024x4096 iter=5``() = test <@ uint32 @> uint32 1024 4096 5
let [<Test>]``uint32: 4096x256 iter=5``() = test <@ uint32 @> uint32 4096 256 5
let [<Test>]``uint32: 4096x4096 iter=5``() = test <@ uint32 @> uint32 4096 4096 5
let [<Test>]``float32: 1024x256 iter=5``() = test <@ Common.toFloat32 @> Common.toFloat32 1024 256 5
let [<Test>]``float32: 1024x4096 iter=5``() = test <@ Common.toFloat32 @> Common.toFloat32 1024 4096 5
let [<Test>]``float64: 1024x256 iter=5``() = test <@ Common.toFloat64 @> Common.toFloat64 1024 256 5
let [<Test>]``float64: 1024x4096 iter=5``() = test <@ Common.toFloat64 @> Common.toFloat64 1024 4096 5


