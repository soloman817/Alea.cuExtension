﻿open Alea.CUDA
open Alea.CUDA.TestUtilities

let test (input:'T) =
    let template = cuda {
        let! kernel =
            <@ fun (outputs:deviceptr<'T>) (input:'T) ->
                outputs.[0] <- input @>
            |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply(kernel)

            let test (iter:int) =
                fun () ->
                    use start = worker.CreateEvent()
                    use stop = worker.CreateEvent()
                    use outputs = worker.Malloc<'T>(1)
                    let lp = LaunchParam(1, 1)

                    // warmup
                    for i = 1 to 10 do
                        kernel.Launch lp outputs.Ptr input

                    // now test
                    // idea is, the kernel is very small, so
                    // all time should be launching time
                    worker.Synchronize()
                    start.Record()
                    for i = 1 to iter do
                        kernel.Launch lp outputs.Ptr input
                    stop.Record()
                    stop.Synchronize()
                    let msecTotal = Event.ElapsedMilliseconds(start, stop)

                    let msec = msecTotal / float(iter)
                    printfn "%s %d: %f ms" (typeof<'T>.Name) iter msec
                |> worker.Thread.Eval

            test ) }

    use program = Worker.Default.LoadProgram(template)
    1000 |> program.Run
    5000 |> program.Run

[<EntryPoint>]
let main argv = 
    TestUtil.genRandomSInt32 -100 100 () |> test
    TestUtil.genRandomDouble -100.0 100.0 () |> test
    TestStruct.genRandomInt3A8 -100 100 () |> test
    0 // return an integer exit code
