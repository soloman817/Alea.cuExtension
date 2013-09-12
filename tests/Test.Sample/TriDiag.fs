module Test.Sample.TriDiag

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.TestUtilities
open NUnit.Framework
open Sample.TriDiag

let correctness (solve:Common.Solver) (generate:Common.Generator) (ns:int list) =
    let test n =
        let A, b, tol = generate n
        let x = solve A b
        let b' = Common.apply A x
        let err = Common.maxError b b'
        if Util.debug then printfn "n=%9d err=%.9f" n err
        Assert.IsTrue(err < tol)
    ns |> List.iter test

[<Test>]
let ``correctness of CPU solver``() =
    let solve = CPU.solve
    let generate1 = Common.generate1 0.1 0.01
    let generate2 = Common.generate2
    let ns = [4; 8; 16; 18; 20; 22] |> List.map (fun e -> (2<<<e))
    correctness solve generate1 ns
    correctness solve generate2 ns

[<Test>]
let ``correctness of GPU solver (by lambda)``() =
    let template = cuda {
        let! kernel =
            <@ fun (n:int) (dl:deviceptr<float>) (dd:deviceptr<float>) (du:deviceptr<float>) (db:deviceptr<float>) ->
                let tid = threadIdx.x

                let shared = __shared__.Extern<float>()
                let l = shared
                let d = l + n
                let u = d + n
                let b = u + n
        
                l.[tid] <- dl.[tid]
                d.[tid] <- dd.[tid]
                u.[tid] <- du.[tid]
                b.[tid] <- db.[tid]
        
                __syncthreads()

                GPU.solve n l d u b

                db.[tid] <- b.[tid] @>
            |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply(kernel)

            let solve (l:float[], d:float[], u:float[]) (b:float[]) =
                let n = d.Length
                use l = worker.Malloc(l)
                use d = worker.Malloc(d)
                use u = worker.Malloc(u)
                use b = worker.Malloc(b)
                let sharedSize = 9 * n * __sizeof<float>()
                let lp = LaunchParam(1, n, sharedSize)
                kernel.Launch lp n l.Ptr d.Ptr u.Ptr b.Ptr
                b.Gather()

            solve ) }

    use program = template |> Util.load Worker.Default
    let solve = program.Run
    let generate1 = Common.generate1 0.1 0.01
    let generate2 = Common.generate2
    let ns = [ 32; 64; 128; 256 ]
    correctness solve generate1 ns
    correctness solve generate2 ns

[<Test>]
let ``correctness of GPU solver (by function)``() =
    let template = cuda {
        let! solve = <@ GPU.solve @> |> Compiler.DefineFunction

        let! kernel =
            <@ fun (n:int) (dl:deviceptr<float>) (dd:deviceptr<float>) (du:deviceptr<float>) (db:deviceptr<float>) ->
                let tid = threadIdx.x

                let shared = __shared__.Extern<float>()
                let l = shared
                let d = l + n
                let u = d + n
                let b = u + n
        
                l.[tid] <- dl.[tid]
                d.[tid] <- dd.[tid]
                u.[tid] <- du.[tid]
                b.[tid] <- db.[tid]
        
                __syncthreads()

                solve.Invoke n l d u b

                db.[tid] <- b.[tid] @>
            |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply(kernel)

            let solve (l:float[], d:float[], u:float[]) (b:float[]) =
                let n = d.Length
                use l = worker.Malloc(l)
                use d = worker.Malloc(d)
                use u = worker.Malloc(u)
                use b = worker.Malloc(b)
                let sharedSize = 9 * n * __sizeof<float>()
                let lp = LaunchParam(1, n, sharedSize)
                kernel.Launch lp n l.Ptr d.Ptr u.Ptr b.Ptr
                b.Gather()

            solve ) }

    use program = template |> Util.load Worker.Default
    let solve = program.Run
    let generate1 = Common.generate1 0.1 0.01
    let generate2 = Common.generate2
    let ns = [ 32; 64; 128; 256 ]
    correctness solve generate1 ns
    correctness solve generate2 ns
