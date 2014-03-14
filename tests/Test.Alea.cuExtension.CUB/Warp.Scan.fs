module Test.Alea.cuExtension.CUB.Warp.Scan

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Common

open NUnit.Framework

open Alea.cuExtension.CUB
open Alea.cuExtension.CUB.Warp


let BLOCKS = 1
let THREADS = 128

let BLOCK_THREADS = THREADS
let ITEMS_PER_THREAD = 4

let N = BLOCK_THREADS * ITEMS_PER_THREAD

[<Test>]
let ``exclusive sum int`` () =
    
    let template = cuda {
        let ws_h = WarpScan.HostApi.Init()

        let! kernel =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
                let r = output.[threadIdx.x] |> __obj_to_ref
                let ws_d = WarpScan.API<int>.Create(ws_h) //.ExclusiveSumInt(ws_h, input.[threadIdx.x], r)
                let aggregate = __local__.Variable<int>()
                WarpScan.ExclusiveSum.WithAggregateInt ws_h ws_d.DeviceApi input.[threadIdx.x] (output.Ref(threadIdx.x)) aggregate
            
            @> |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply kernel
            
            fun (input:int[]) ->
                use input = worker.Malloc(input)
                use output = worker.Malloc<int>(N)

                let lp = LaunchParam(BLOCK_THREADS, BLOCKS)

                kernel.Launch lp input.Ptr output.Ptr

                output.Gather()
            )}

    let program = template |> Compiler.load Worker.Default
    let input = Array.init N (fun i -> i)
    let output = program.Run input
    printfn "%A" output