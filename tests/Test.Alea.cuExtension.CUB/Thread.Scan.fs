module Test.Alea.cuExtension.CUB.Thread.Scan

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework
open Microsoft.FSharp.Quotations
open Alea.cuExtension.CUB
open Alea.cuExtension.CUB.Thread

open Test.Alea.cuExtension.CUB.Utilities

let BLOCKS = 1
let BLOCK_THREADS = 16
let ITEMS_PER_THREAD = 4
let N = BLOCKS * BLOCK_THREADS

let [<ReflectedDefinition>] scan input output prefix = ThreadScanExclusive.WithApplyPrefixDefault BLOCK_THREADS (+) input output prefix

let hinput = Array.init N (fun _ -> 1)


[<Test>]
let ``thread scan exclusive - int`` () =
    let template = cuda {

        let! kernel =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
                let mutable tid = threadIdx.x

                
                let x = scan input output 0
                ()

            @> |> Compiler.DefineKernel
    
        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            fun (input:int[]) ->
                use d_in = worker.Malloc(input)
                use d_out = worker.Malloc(input.Length)

                let lp = lp11 //LaunchParam(BLOCKS, BLOCK_THREADS)
                
                kernel.Launch lp d_in.Ptr d_out.Ptr
                
                d_out.Gather()
        )}


    let program = template |> Compiler.load Worker.Default
    let output = program.Run hinput
    printfn "Device:%A" output

    let houtput = hinput |> Array.scan (+) 0
    printfn "Host:%A" houtput
    printfn "%A" houtput.[houtput.Length - 1]