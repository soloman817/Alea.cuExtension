module Sample.XorShift7.Program

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities

// this template didn't malloc memories, it is in-place updater,
// used for performance test
let template (convertExpr:Expr<uint32 -> 'T>) = cuda {
    let! kernel = GPU.kernel convertExpr |> Compiler.DefineKernel

    return Entry(fun (program:Program) ->
        let worker = program.Worker
        let kernel = program.Apply(kernel)

        // Copy pre-calculated bit-matrices, needed for jump-ahead
        // calculations, to the device memory.
        let jumpAheadMatrices = worker.Malloc(Data.jumpAheadMatrices)

        let test (streams:int) (steps:int) (seed:uint32) (runs:int) (rank:int) (iter:int) =
            let state0 = Common.generateStartState seed
            use state0 = worker.Malloc(state0)
            use numbers = worker.Malloc<'T>(streams * steps)
            let lp = GPU.launchParam streams
            use start = worker.CreateEvent()
            use stop = worker.CreateEvent()

            // warmup
            for i = 1 to 10 do
                kernel.Launch lp runs rank state0.Ptr jumpAheadMatrices.Ptr steps numbers.Ptr

            worker.Synchronize()
            start.Record()
            for i = 1 to iter do
                kernel.Launch lp runs rank state0.Ptr jumpAheadMatrices.Ptr steps numbers.Ptr
            stop.Record()
            stop.Synchronize()
            let msecTotal = Event.ElapsedMilliseconds(start, stop)

            let msec = msecTotal / float(iter)
            printfn "%s iter=%d %dx%5d seed=%d runs=%d rank=%d: %9.6f ms" (typeof<'T>.Name) iter streams steps seed runs rank msec

        test ) }

let test convertExpr streams steps seed runs rank iter =
    use program = template convertExpr |> Util.load Worker.Default
    program.Run streams steps seed runs rank iter

[<EntryPoint>]
let main argv = 

    test <@ uint32 @>           2048   1000  42u 1 0 100
    test <@ uint32 @>           2048   5000  42u 1 0 100
    test <@ uint32 @>           2048  10000  42u 1 0 100
    test <@ uint32 @>           2048  20000  42u 1 0 100
    test <@ uint32 @>           2048  30000  42u 1 0 100
    test <@ uint32 @>           2048  40000  42u 1 0 100
    test <@ uint32 @>           2048  50000  42u 1 0 100

    test <@ Common.toFloat32 @> 2048   1000  42u 1 0 100
    test <@ Common.toFloat32 @> 2048   5000  42u 1 0 100
    test <@ Common.toFloat32 @> 2048  10000  42u 1 0 100
    test <@ Common.toFloat32 @> 2048  20000  42u 1 0 100
    test <@ Common.toFloat32 @> 2048  30000  42u 1 0 100
    test <@ Common.toFloat32 @> 2048  40000  42u 1 0 100
    test <@ Common.toFloat32 @> 2048  50000  42u 1 0 100

    test <@ Common.toFloat64 @> 2048   1000  42u 1 0 100
    test <@ Common.toFloat64 @> 2048   5000  42u 1 0 100
    test <@ Common.toFloat64 @> 2048  10000  42u 1 0 100
    test <@ Common.toFloat64 @> 2048  20000  42u 1 0 100
    test <@ Common.toFloat64 @> 2048  30000  42u 1 0 100
    test <@ Common.toFloat64 @> 2048  40000  42u 1 0 100
    test <@ Common.toFloat64 @> 2048  50000  42u 1 0 100

    0 // return an integer exit code
