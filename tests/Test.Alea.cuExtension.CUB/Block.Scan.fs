module Test.Alea.cuExtension.CUB.Block.Scan

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

open Alea.cuExtension.CUB

//
//type TempStorage<'T> =
//    {
//        load    : deviceptr<'T>
//        store   : deviceptr<'T>
//        scan    : deviceptr<'T>
//    }
//
//
//[<Test>]
//let ``block scan basic`` () =
//    
//    let template block_threads items_per_thread algorithm = cuda {
//        let! blockLoad = blockLoad block_threads items_per_thread BLOCK_LOAD_WARP_TRANSPOSE
//        let! blockStore = blockStore block_threads items_per_thread BLOCK_STORE_WARP_TRANSPOSE
//        let! blockScan = blockScan block_threads algorithm
//        
//        let temp_storage =
//            {
//                load = blockLoad.tempStorage<'T>()
//                store = blockStore.tempStorage<'T>()
//                scan = blockScan.tempStorage<'T>()
//            } 
//
//        let! kernel = 
//            <@ fun (d_in:deviceptr<int>) (d_out:deviceptr<int>) (d_elapsed:deviceptr<float>) ->
//                let data = __local__.Array<int>(items_per_thread)
//                blockLoad(temp_storage.load).Load(d_in, data)
//
//                __syncthreads()
//
//                let start = 0.0 //clock()
//                let aggregate = 0
//                blockScan(temp_storage.scan).ExclusiveSum(data, data, aggregate)
//
//                let stop = 0.0 //clock()
//
//                __syncthreads()
//
//                blockStore(temp_storage.store).Store(d_out, data)
//
//                if threadIdx.x = 0 then
//                    d_elapsed.[0] <- if start > stop then start - stop else stop - start
//                    d_out.[block_threads * items_per_thread] <- aggregate
//            @> |> Compiler.DefineKernel
//
//        return Entry(fun program ->
//            let worker = program.Worker
//            let kernel = program.Apply kernel
//
//            let run (input:int[]) =
//                use d_in = worker.Malloc(input)
//                use d_out = worker.Malloc<int>(input.Length + 1)
//                use d_elapsed = worker.Malloc<float>(input.Length)
//
//                kernel.Launch (LaunchParam(1, 128)) d_in.Ptr d_out.Ptr d_elapsed.Ptr
//
//                d_out.Gather()
//
//            run
//        )}
//
//    let BLOCK_THREADS = 32
//    let ITEMS_PER_THREAD = 32
//    let TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD
//
//    let h_in = Array.init TILE_SIZE (fun i -> i)
//    let h_out = Array.scan (fun r e -> r + e) 0
//
//    printfn "Input: %A" h_in
//    printfn "Host Output: %A" h_out 
//
//    let program = template BLOCK_THREADS ITEMS_PER_THREAD BLOCK_SCAN_RAKING |> Compiler.load Worker.Default
//    let output = program.Run h_in
//
//    printfn "Device Output: %A" output