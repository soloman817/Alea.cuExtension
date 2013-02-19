module Test.Alea.CUDA.Extension.Sobol

open System.IO
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Test.Alea.CUDA.Extension.Util

//let worker = getDefaultWorker()
//
//let generator = worker.LoadPModule(PRandom.sobol <@ Sobol.toUInt32 @>)
//let generate = generator.Invoke
//
//let testSobol verify (dimensions:int) vectors offset =
//    let dOutput =
//        use output = PArray.Create(worker, dimensions * vectors)
//        generate dimensions vectors offset output
//        output.ToHost()
//
//    if verify then
//        let generator = SobolGold.Sobol(dimensions, offset)
//        let hOutput = Array.init vectors (fun _ -> generator.NextPoint) |> Array.concat
//        let dOutput = dOutput |> SobolGold.reorderPoints dimensions vectors
//        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))
//
//let generatorFloat32 = worker.LoadPModule(PRandom.sobol <@ Sobol.toFloat32 @>)
//let generateFloat32 = generatorFloat32.Invoke
//
//let testSobolFloat32 verify (dimensions:int) vectors offset =
//    let dOutput =
//        use output = PArray.Create(worker, dimensions * vectors)
//        generateFloat32 dimensions vectors offset output
//        output.ToHost()
//
//    if verify then
//        let generator = SobolGold.Sobol(dimensions, offset)
//        let hOutput = Array.init vectors (fun _ -> generator.NextPoint |> Array.map Sobol.toFloat32) |> Array.concat
//        let dOutput = dOutput |> SobolGold.reorderPoints dimensions vectors
//        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))
//
//let generatorFloat64 = worker.LoadPModule(PRandom.sobol <@ Sobol.toFloat64 @>)
//let generateFloat64 = generatorFloat64.Invoke
//
//let testSobolFloat64 verify (dimensions:int) vectors offset =
//    let dOutput =
//        use output = PArray.Create(worker, dimensions * vectors)
//        generateFloat64 dimensions vectors offset output
//        output.ToHost()
//
//    if verify then
//        let generator = SobolGold.Sobol(dimensions, offset)
//        let hOutput = Array.init vectors (fun _ -> generator.NextPoint |> Array.map Sobol.toFloat64) |> Array.concat
//        let dOutput = dOutput |> SobolGold.reorderPoints dimensions vectors
//        (hOutput, dOutput) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))
//
//// Tests for uint32
//let [<Test>] ``[V] 32 x 256 5`` () = testSobol true 32 256 5
//let [<Test>] ``[V] 32 x 256 9`` () = testSobol true 32 256 9
//let [<Test>] ``[V] 32 x 256 1`` () = testSobol true 32 256 1
//let [<Test>] ``[V] 32 x 4096 1`` () = testSobol true 32 4096 1
//let [<Test>] ``[V] 32 x 65536 1`` () = testSobol true 32 65536 1
//let [<Test>] ``[_] 32 x 1048576 1`` () = testSobol false 32 1048576 1
//let [<Test>] ``[V] 1024 x 256 1`` () = testSobol true 1024 256 1
//let [<Test>] ``[V] 1024 x 4096 1`` () = testSobol true 1024 4096 1
//let [<Test>] ``[_] 1024 x 65536 1`` () = testSobol false 1024 65536 1
//let [<Test>] ``[V] 4096 x 256 1`` () = testSobol true 4096 256 1
//let [<Test>] ``[V] 4096 x 4096 1`` () = testSobol true 4096 4096 1
//let [<Test>] ``[_] 4096 x 8192 1`` () = testSobol false 4096 8192 1
//let [<Test>] ``[_] 4096 x 16384 1`` () = testSobol false 4096 16384 1
//
//// Tests for float32
//let [<Test>] ``Float32: [V] 1024 x 256 1`` () = testSobolFloat32 true 1024 256 1
//let [<Test>] ``Float32: [V] 1024 x 4096 1`` () = testSobolFloat32 true 1024 4096 1
//
//// Tests for float64
//let [<Test>] ``Float64: [V] 1024 x 256 1`` () = testSobolFloat64 true 1024 256 1
//let [<Test>] ``Float64: [V] 1024 x 4096 1`` () = testSobolFloat64 true 1024 4096 1