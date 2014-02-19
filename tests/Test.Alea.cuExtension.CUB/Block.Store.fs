module Test.Alea.cuExtension.CUB.Block.Store

open Alea.CUDA
open Alea.CUDA.Utilities

open NUnit.Framework

open Alea.cuExtension.CUB.Block


type BlockStore = Alea.cuExtension.CUB.Block.Store.BlockStore

[<Test>]
let ``block store basic`` () =
    
    let template block_threads items_per_thread = cuda {
        let BlockStore = BlockStore.Create(block_threads, items_per_thread, BLOCK_STORE_WARP_TRANSPOSE)
        
 

        let! kernel = 
            <@ fun (blockStore:BlockStore) (d_in:deviceptr<int>) (d_out:deviceptr<int>) (d_elapsed:deviceptr<float>) ->
                let temp_storage_store = __shared__.Extern()

                let data = __local__.Array<int>(items_per_thread)
                
                __syncthreads()
                
                blockStore.Initialize(temp_storage_store).Store(d_out, data |> __array_to_ptr)

            @> |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            let run (input:int[]) =
                use d_in = worker.Malloc(input)
                use d_out = worker.Malloc<int>(input.Length + 1)
                use d_elapsed = worker.Malloc<float>(input.Length)

                kernel.Launch (LaunchParam(1, 128, items_per_thread)) BlockStore d_in.Ptr d_out.Ptr d_elapsed.Ptr

                d_out.Gather()

            run
        )}

    let BLOCK_THREADS = 32
    let ITEMS_PER_THREAD = 32
    let TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD

    let h_in = Array.init TILE_SIZE (fun i -> i)
    let h_out = Array.scan (fun r e -> r + e) 0

    printfn "Input: %A" h_in
    printfn "Host Output: %A" h_out 

    let program = template BLOCK_THREADS ITEMS_PER_THREAD |> Compiler.load Worker.Default
    let output = program.Run h_in

    printfn "Device Output: %A" output