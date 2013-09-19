module Test.Sample.Mrg32k3a

open System.Reflection
open System.IO
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.TestUtilities
open Sample.Mrg32k3a
open NUnit.Framework

let ``test raw`` (nb:int) (nt:int) (np:int) (v1:uint32[]) (v2:uint32[]) (offset:int) =
    let template = cuda {
        let! core = GPU.core
        let! raw = GPU.raw core

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply(raw.Kernel)
            let init = raw.Init program

            let test () =
                let hOutputs =
                    let generator = CPU.Generator(v1, v2, offset)
                    let points = Array.zeroCreate<uint32> (nb * nt * np)
                    CPU.raw generator points
                    GPU.reorder nb nt np points

                let dOutputs =
                    use points = worker.Malloc<uint32>(nb * nt * np)
                    let lp = LaunchParam(nb, nt)
                    init v1 v2 offset
                    kernel.Launch lp np points.Ptr
                    points.Gather()

                TestUtil.assertArrayEqual None hOutputs dOutputs

            test ) }

    use program = template |> Util.load Worker.Default
    program.Run()

let [<Test>] ``raw 1500x128x100 [1;2;3] [1;2;3] offset=0``() = ``test raw`` 1500 128 100 [| 1u; 2u; 3u |] [| 1u; 2u; 3u |] 0
let [<Test>] ``raw 1500x128x100 [1;2;3] [7;8;9] offset=100``() = ``test raw`` 1500 128 100 [| 1u; 2u; 3u |] [| 7u; 8u; 9u |] 100

let inline ``test uniform`` (real:RealConverter<'T>) (nb:int) (nt:int) (np:int) (v1:uint32[]) (v2:uint32[]) (offset:int) =
    let template = cuda {
        let! core = GPU.core
        let! uniform = GPU.uniform real core

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply(uniform.Kernel)
            let init = uniform.Init program

            let test () =
                let hOutputs =
                    let generator = CPU.Generator(v1, v2, offset)
                    let numbers = Array.zeroCreate<'T> (nb * nt * np)
                    CPU.uniform real generator numbers
                    GPU.reorder nb nt np numbers

                let dOutputs =
                    use numbers = worker.Malloc<'T>(nb * nt * np)
                    let lp = LaunchParam(nb, nt)
                    init v1 v2 offset
                    kernel.Launch lp np numbers.Ptr
                    numbers.Gather()

                //printfn "%A" hOutputs
                //printfn "%A" dOutputs

                TestUtil.assertArrayEqual None hOutputs dOutputs

            test ) }

    use program = template |> Util.load Worker.Default
    program.Run()

let [<Test>] ``uniform<float32> 1500x128x100 [1;2;3] [1;2;3] offset=0``() = ``test uniform`` RealConverter.Real32 1500 128 100 [| 1u; 2u; 3u |] [| 1u; 2u; 3u |] 0
let [<Test>] ``uniform<float32> 1500x128x100 [1;2;3] [7;8;9] offset=100``() = ``test uniform`` RealConverter.Real32 1500 128 100 [| 1u; 2u; 3u |] [| 7u; 8u; 9u |] 100
let [<Test>] ``uniform<float64> 1500x128x100 [1;2;3] [1;2;3] offset=0``() = ``test uniform`` RealConverter.Real64 1500 128 100 [| 1u; 2u; 3u |] [| 1u; 2u; 3u |] 0
let [<Test>] ``uniform<float64> 1500x128x100 [1;2;3] [7;8;9] offset=100``() = ``test uniform`` RealConverter.Real64 1500 128 100 [| 1u; 2u; 3u |] [| 7u; 8u; 9u |] 100

let inline ``test exponential`` (real:RealConverter<'T>) (nb:int) (nt:int) (np:int) (v1:uint32[]) (v2:uint32[]) (offset:int) (eps:float option) =
    let template = cuda {
        let! core = GPU.core
        let! exponential = GPU.exponential real core

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply(exponential.Kernel)
            let init = exponential.Init program

            let test () =
                let hOutputs =
                    let generator = CPU.Generator(v1, v2, offset)
                    let numbers = Array.zeroCreate<'T> (nb * nt * np)
                    CPU.exponential real generator numbers
                    GPU.reorder nb nt np numbers

                let dOutputs =
                    use numbers = worker.Malloc<'T>(nb * nt * np)
                    let lp = LaunchParam(nb, nt)
                    init v1 v2 offset
                    kernel.Launch lp np numbers.Ptr
                    numbers.Gather()

                //printfn "%A" hOutputs
                //printfn "%A" dOutputs

                TestUtil.assertArrayEqual eps hOutputs dOutputs

            test ) }

    use program = template |> Util.load Worker.Default
    program.Run()

let [<Test>] ``exponential<float32> 1500x128x100 [1;2;3] [1;2;3] offset=0``() = ``test exponential`` RealConverter.Real32 1500 128 100 [| 1u; 2u; 3u |] [| 1u; 2u; 3u |] 0 (Some 1e-5)
let [<Test>] ``exponential<float32> 1500x128x100 [1;2;3] [7;8;9] offset=100``() = ``test exponential`` RealConverter.Real32 1500 128 100 [| 1u; 2u; 3u |] [| 7u; 8u; 9u |] 100 (Some 1e-5)
let [<Test>] ``exponential<float64> 1500x128x100 [1;2;3] [1;2;3] offset=0``() = ``test exponential`` RealConverter.Real64 1500 128 100 [| 1u; 2u; 3u |] [| 1u; 2u; 3u |] 0 (Some 1e-12)
let [<Test>] ``exponential<float64> 1500x128x100 [1;2;3] [7;8;9] offset=100``() = ``test exponential`` RealConverter.Real64 1500 128 100 [| 1u; 2u; 3u |] [| 7u; 8u; 9u |] 100 (Some 1e-12)

let inline ``test normal`` (real:RealConverter<'T>) (nb:int) (nt:int) (np:int) (v1:uint32[]) (v2:uint32[]) (offset:int) (eps:float option) =
    let template = cuda {
        let! core = GPU.core
        let! normal = GPU.normal real core

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply(normal.Kernel)
            let init = normal.Init program

            let test () =
                let hOutputs =
                    let generator = CPU.Generator(v1, v2, offset)
                    let numbers = Array.zeroCreate<'T> (nb * nt * np)
                    CPU.normal real generator numbers
                    GPU.reorder nb nt np numbers

                let dOutputs =
                    use numbers = worker.Malloc<'T>(nb * nt * np)
                    let lp = LaunchParam(nb, nt)
                    init v1 v2 offset
                    kernel.Launch lp np numbers.Ptr
                    numbers.Gather()

                //printfn "%A" hOutputs
                //printfn "%A" dOutputs

                TestUtil.assertArrayEqual eps hOutputs dOutputs

            test ) }

    use program = template |> Util.load Worker.Default
    program.Run()

let [<Test>] ``normal<float32> 1500x128x100 [1;2;3] [1;2;3] offset=0``() = ``test normal`` RealConverter.Real32 1500 128 100 [| 1u; 2u; 3u |] [| 1u; 2u; 3u |] 0 (Some 1e-5)
let [<Test>] ``normal<float32> 1500x128x100 [1;2;3] [7;8;9] offset=100``() = ``test normal`` RealConverter.Real32 1500 128 100 [| 1u; 2u; 3u |] [| 7u; 8u; 9u |] 100 (Some 1e-5)
let [<Test>] ``normal<float64> 1500x128x100 [1;2;3] [1;2;3] offset=0``() = ``test normal`` RealConverter.Real64 1500 128 100 [| 1u; 2u; 3u |] [| 1u; 2u; 3u |] 0 (Some 1e-12)
let [<Test>] ``normal<float64> 1500x128x100 [1;2;3] [7;8;9] offset=100``() = ``test normal`` RealConverter.Real64 1500 128 100 [| 1u; 2u; 3u |] [| 7u; 8u; 9u |] 100 (Some 1e-12)

let inline ``test gamma`` (real:RealConverter<'T>) (nb:int) (nt:int) (np:int) (v1:uint32[]) (v2:uint32[]) (offset:int) (alpha:'T) (eps:float option) =
    let template = cuda {
        let! core = GPU.core
        let! gamma = GPU.gamma real core

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply(gamma.Kernel)
            let init = gamma.Init program

            let test () =
                let hOutputs =
                    let generator = CPU.Generator(v1, v2, offset)
                    let numbers = Array.zeroCreate<'T> (nb * nt * np)
                    CPU.gamma real generator alpha numbers
                    GPU.reorder nb nt np numbers

                let dOutputs =
                    use numbers = worker.Malloc<'T>(nb * nt * np)
                    let lp = LaunchParam(nb, nt)
                    init v1 v2 offset
                    kernel.Launch lp np alpha numbers.Ptr
                    numbers.Gather()

                printfn "%A" hOutputs
                printfn "%A" dOutputs

                TestUtil.assertArrayEqual eps hOutputs dOutputs

            test ) }

    use program = template |> Util.load Worker.Default
    program.Run()

//let [<Test>] ``gamma<float32> 2x64x2 [1;2;3] [1;2;3] offset=0 alpha=0.5``() = ``test gamma`` RealConverter.Real32 2 64 2 [| 1u; 2u; 3u |] [| 1u; 2u; 3u |] 0 0.5f (Some 1e-5)
//let [<Test>] ``gamma<float32> 1000x64x50 [1;2;3] [1;2;3] offset=0 alpha=0.5``() = ``test gamma`` RealConverter.Real32 1000 64 50 [| 1u; 2u; 3u |] [| 1u; 2u; 3u |] 0 0.5f (Some 1e-5)
