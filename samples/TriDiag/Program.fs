open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Sample.TriDiag

let kernelByLambda = cuda {
    let! kernel =
        // signature is (n, l, d, u, b, x) 
        // we must use dx to store, because we will run performance test, that will launch
        // this kernel for many times, so that requires all input should be non-mutable.
        <@ fun (n:int) (dl:deviceptr<float>) (dd:deviceptr<float>) (du:deviceptr<float>) (db:deviceptr<float>) (dx:deviceptr<float>) ->
            // we will do multiple solving on different blocks
            let tid = threadIdx.x
            let gid = blockIdx.x * blockDim.x + tid

            let shared = __shared__.Extern<float>()
            let l = shared
            let d = l + n
            let u = d + n
            let b = u + n
        
            l.[tid] <- dl.[gid]
            d.[tid] <- dd.[gid]
            u.[tid] <- du.[gid]
            b.[tid] <- db.[gid]
        
            __syncthreads()

            GPU.solve n l d u b

            dx.[gid] <- b.[tid] @>
        |> Compiler.DefineKernel

    return kernel }

let kernelByFunction = cuda {
    let! solve = <@ GPU.solve @> |> Compiler.DefineFunction

    let! kernel =
        <@ fun (n:int) (dl:deviceptr<float>) (dd:deviceptr<float>) (du:deviceptr<float>) (db:deviceptr<float>) (dx:deviceptr<float>) ->
            let tid = threadIdx.x
            let gid = blockIdx.x * blockDim.x + tid

            let shared = __shared__.Extern<float>()
            let l = shared
            let d = l + n
            let u = d + n
            let b = u + n
        
            l.[tid] <- dl.[gid]
            d.[tid] <- dd.[gid]
            u.[tid] <- du.[gid]
            b.[tid] <- db.[gid]
        
            __syncthreads()

            solve.Invoke n l d u b

            dx.[gid] <- b.[tid] @>
        |> Compiler.DefineKernel

    return kernel }

let test (kernelName:string) (kernel:Template<Resources.Kernel<int -> deviceptr<float> -> deviceptr<float> -> deviceptr<float> -> deviceptr<float> -> deviceptr<float> -> unit>>) =
    let template = cuda {
        let! kernel = kernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply(kernel)
            let maxBlockDimX = worker.Device.Attributes.MAX_BLOCK_DIM_X
            let maxGridDimX = worker.Device.Attributes.MAX_GRID_DIM_X
            let maxThreads = worker.Device.Attributes.MAX_THREADS_PER_BLOCK
            let maxSharedSize = worker.Device.Attributes.MAX_SHARED_MEMORY_PER_BLOCK
            let typeName = typeof<float>.Name

            let test (generatorName:string) (generate:Common.Generator) (grids:int) (iters:int) (n:int) =
                let sharedSize = 9 * n * __sizeof<float>()
                let lp = LaunchParam(grids, n, sharedSize)

                let report (str:string) =
                    printfn "%s %s %s iters=%d grids=%d n=%4d : %s" typeName kernelName generatorName iters grids n str

                if grids > maxGridDimX then sprintf "grids(%d) > maxGridDimX(%d)" grids maxGridDimX |> report
                elif n > maxBlockDimX then sprintf "n(%d) > maxBlockDimX(%d)" n maxBlockDimX |> report
                elif n > maxThreads then sprintf "n(%d) > maxThreads(%d)" n maxThreads |> report
                elif sharedSize > maxSharedSize then sprintf "sharedSize(%d) > maxSharedSize(%d)" sharedSize maxSharedSize |> report
                else
                    let inputs = Array.init grids (fun _ -> generate n)

                    let xs, msec =
                        fun () ->
                            use start = worker.CreateEvent()
                            use stop = worker.CreateEvent()

                            let l = inputs |> Array.map (fun (A, _, _) -> let l, _, _ = A in l) |> Array.concat
                            let d = inputs |> Array.map (fun (A, _, _) -> let _, d, _ = A in d) |> Array.concat
                            let u = inputs |> Array.map (fun (A, _, _) -> let _, _, u = A in u) |> Array.concat
                            let b = inputs |> Array.map (fun (_, b, _) -> b) |> Array.concat

                            use l = worker.Malloc(l)
                            use d = worker.Malloc(d)
                            use u = worker.Malloc(u)
                            use b = worker.Malloc(b)
                            use x = worker.Malloc<float>(n * grids)

                            // warmup
                            for i = 1 to 10 do
                                kernel.Launch lp n l.Ptr d.Ptr u.Ptr b.Ptr x.Ptr
                            worker.Synchronize()

                            // performance
                            start.Record()
                            for i = 1 to iters do
                                kernel.Launch lp n l.Ptr d.Ptr u.Ptr b.Ptr x.Ptr
                            stop.Record()
                            stop.Synchronize()
                            let msecTotal = Event.ElapsedMilliseconds(start, stop)
                            let msec = msecTotal / (iters |> float)

                            // retrieve results
                            let xs =
                                let x = x.Gather()
                                Array.init grids (fun i -> Array.sub x (i * n) n)

                            xs, msec
                        |> worker.Eval

                    let err =
                        (inputs, xs)
                        ||> Array.map2 (fun (A, b, tol) x ->
                            let b' = Common.apply A x
                            Common.maxError b b')
                        |> Array.max

                    sprintf "err=%.15f %10.6f ms" err msec |> report

            test ) }

    use program = template |> Util.load Worker.Default
    let grids = 10240
    let iters = 200
    let generate1 = Common.generate1 0.1 0.01
    let generate2 = Common.generate2
    let nsight = false
    if nsight then
        program.Run "generate1" generate1 grids iters 640
    else
        let ns = [ 64..64..1024 ]
        ns |> List.iter (program.Run "generate1" generate1 grids iters)
        ns |> List.iter (program.Run "generate2" generate2 grids iters)

[<EntryPoint>]
let main argv = 

    test "ByLambda  " kernelByLambda
    test "ByFunction" kernelByFunction

    0 // return an integer exit code
