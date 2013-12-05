open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Sample.Sobol

let performance (convertD:Expr<uint32 -> 'T>) (convertH:uint32 -> 'T) (numDimensions:int) (numVectors:int) (offset:int) (numIters:int) =
    let template = cuda {
        let! kernel = GPU.kernel convertD |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply(kernel)

            let test () =
                use start = worker.CreateEvent()
                use stop = worker.CreateEvent()
                let directions = Common.directions numDimensions
                use directions = worker.Malloc(directions)
                use numbers = worker.Malloc<'T>(numDimensions * numVectors)
                let lp = GPU.launchParam numDimensions numVectors

                // warmup
                for i = 1 to 10 do
                    kernel.Launch lp numDimensions numVectors offset directions.Ptr numbers.Ptr

                // performance
                worker.Synchronize()
                start.Record()                
                for i = 1 to numIters do
                    kernel.Launch lp numDimensions numVectors offset directions.Ptr numbers.Ptr
                stop.Record()
                stop.Synchronize()
                let msecTotal = Event.ElapsedMilliseconds(start, stop)
                let msec = msecTotal / (numIters |> float)

                printfn "%s %d: %4dx%7d offset=%d %9.6f ms (%10.6f ms total)" (typeof<'T>.Name) numIters numDimensions numVectors offset msec msecTotal

            let run () = test |> worker.Thread.Eval

            test ) }

    use program = template |> Util.load Worker.Default
    program.Run()


[<EntryPoint>]
let main argv = 
    let numIters = 100
    
    performance <@ uint32 @> uint32 (1<<< 5) (1<<< 8) 1 numIters
    performance <@ uint32 @> uint32 (1<<< 5) (1<<<12) 1 numIters
    performance <@ uint32 @> uint32 (1<<< 5) (1<<<16) 1 numIters
    performance <@ uint32 @> uint32 (1<<< 5) (1<<<20) 1 numIters

    performance <@ uint32 @> uint32 (1<<<10) (1<<< 8) 1 numIters
    performance <@ uint32 @> uint32 (1<<<10) (1<<<12) 1 numIters
    performance <@ uint32 @> uint32 (1<<<10) (1<<<16) 1 numIters

    performance <@ uint32 @> uint32 (1<<<12) (1<<< 8) 1 numIters
    performance <@ uint32 @> uint32 (1<<<12) (1<<<12) 1 numIters
    performance <@ uint32 @> uint32 (1<<<12) (1<<<13) 1 numIters
    performance <@ uint32 @> uint32 (1<<<12) (1<<<14) 1 numIters

    0 // return an integer exit code
