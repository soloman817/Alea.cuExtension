module Sample.MatrixMul.Program

open Alea.CUDA
open Alea.CUDA.Utilities

[<EntryPoint>]
let main argv = 

    let template (blockSize:int) = cuda {
        let! kernel = blockSize |> GPU.kernel |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply(kernel)
            let cpuCalc = CPU.calc ()
            let gpuCalc = GPU.calc blockSize worker kernel

            let run (dimA:int*int) (dimB:int*int) =
                printfn "[Matrix Multiply Using CUDA] - Starting..."
                printfn "GPU Device %d: %A with compute capability %d.%d"
                        worker.Device.ID worker.Device.Name
                        worker.Device.Arch.Major worker.Device.Arch.Minor
                printfn ""

                let wA, hA = dimA
                let wB, hB = dimB

                let sizeA = wA * hA
                let sizeB = wB * hB

                printfn "MatrixA(%d,%d), MatrixB(%d,%d)" wA hA wB hB

                let A = Array.init sizeA (fun _ -> 1.0f)
                let B = Array.init sizeB (fun _ -> 0.01f)

                let hOutput = cpuCalc A B wA wB
                printfn "Computing result using CUDA Kernel..."
                let dOutput = gpuCalc A B wA wB
                printfn "done"

                // do performance test
                let wC = wB
                let hC = A.Length / wA

                use A = worker.Malloc(A)
                use B = worker.Malloc(B)
                use C = worker.Malloc<float32>(wC * hC)

                let threads = dim3(blockSize, blockSize)
                let grid = dim3(wB / threads.x, hC / threads.y)
                let lp = LaunchParam(grid, threads)

                worker.Synchronize()
                let nIter = 300
                use start = worker.CreateEvent()
                use stop = worker.CreateEvent()
                start.Record()
                for i = 1 to nIter do
                    kernel.Launch lp C.Ptr A.Ptr B.Ptr wA wB
                stop.Record()
                stop.Synchronize()
                let msecTotal = Event.ElapsedMilliseconds(start, stop)

                // Compute and print the performance
                let msecPerMatrixMul = msecTotal / float(nIter)
                let flopsPerMatrixMul = 2.0 * float(wA) * float(hA) * float(wB)
                let gigaFlops = (flopsPerMatrixMul * 1.0e-9) / (msecPerMatrixMul / 1000.0)
                printfn "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block"
                        gigaFlops msecPerMatrixMul flopsPerMatrixMul (threads.x * threads.y)

                printf "Checking computed result for correctness: "
                TestUtil.assertArrayEqual (Some 1e-5) hOutput dOutput
                printfn "Result = PASS"

            run ) }

    let blockSize = 32
    let program = blockSize |> template |> Util.load Worker.Default
    ((320, 320), (640, 320)) ||> program.Run

    0 // return an integer exit code
