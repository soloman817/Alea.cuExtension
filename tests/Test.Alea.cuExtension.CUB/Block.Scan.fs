module Test.Alea.cuExtension.CUB.Block.Scan

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

open Alea.cuExtension.CUB
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Block


type BlockLoadAlgorithm = Alea.cuExtension.CUB.Block.Load.Template.BlockLoadAlgorithm
type BlockStoreAlgorithm = Alea.cuExtension.CUB.Block.Store.Template.BlockStoreAlgorithm
type BlockScanAlgorithm = Alea.cuExtension.CUB.Block.Scan.Template.BlockScanAlgorithm


[<Record>]
type TempStorage<'T> =
    {
        load    : Alea.cuExtension.CUB.Block.Load.Template._TempStorage<'T>
        store   : Alea.cuExtension.CUB.Block.Store.Template._TempStorage<'T>
        scan    : Alea.cuExtension.CUB.Block.Scan.Template._TempStorage<'T>
    }

    [<ReflectedDefinition>]
    static member Init(block_threads, items_per_thread, warp_time_slicing, algorithm, scan_op) =
        let scantp = Alea.cuExtension.CUB.Block.Scan.Template._TemplateParams.Init(block_threads, algorithm, scan_op)
        {
            load = Alea.cuExtension.CUB.Block.Load.Template._TempStorage<'T>.Init(block_threads, items_per_thread, warp_time_slicing)
            store = Alea.cuExtension.CUB.Block.Store.Template._TempStorage<'T>.Init(block_threads, items_per_thread, warp_time_slicing)
            scan = Alea.cuExtension.CUB.Block.Scan.Template._TempStorage<'T>.Init(scantp)
        }

[<Test>]
let ``block scan basic`` () =
    
    let template block_threads items_per_thread (algorithm:BlockScanAlgorithm) = cuda {
        let scan_op = (scan_op ADD 0)
        let! blockLoad   =   <@ BlockLoad.API<int>.Create(block_threads, items_per_thread, BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE, false).Default @> |> Compiler.DefineFunction
        let! blockStore  =   <@ BlockStore.API<int>.Create(block_threads, items_per_thread, BlockStoreAlgorithm.BLOCK_STORE_WARP_TRANSPOSE, false).Default @> |> Compiler.DefineFunction
        let! blockScan   =   <@ BlockScan.API<int>.Create(block_threads, algorithm, items_per_thread, scan_op).ExclusiveSum.MultipleDataPerThread.WithAggregate @> |> Compiler.DefineFunction
        let TempStorage =   TempStorage<int>.Init(block_threads, items_per_thread, false, algorithm, scan_op)

        let! kernel = 
            <@ fun (d_in:deviceptr<int>) (d_out:deviceptr<int>) (d_elapsed:deviceptr<float>) ->
                let data = __local__.Array<int>(items_per_thread) |> __array_to_ptr
                blockLoad.Invoke items_per_thread d_in data

                __syncthreads()

                let aggregate = __local__.Variable<int>()

                blockScan.Invoke data data aggregate

                __syncthreads()

                blockStore.Invoke threadIdx.x d_out data

                if threadIdx.x = 0 then d_out.[block_threads * items_per_thread] <- !aggregate

            @> |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            let run (input:int[]) =
                use d_in = worker.Malloc(input)
                use d_out = worker.Malloc<int>(input.Length + 1)
                use d_elapsed = worker.Malloc<float>(input.Length)

                kernel.Launch (LaunchParam(1, 128, items_per_thread)) d_in.Ptr d_out.Ptr d_elapsed.Ptr

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

    let program = template BLOCK_THREADS ITEMS_PER_THREAD BlockScanAlgorithm.BLOCK_SCAN_RAKING |> Compiler.load Worker.Default
    let output = program.Run h_in

    printfn "Device Output: %A" output
//
//type BlockLoad = Alea.cuExtension.CUB.Block.Load.BlockLoad
//type BlockStore = Alea.cuExtension.CUB.Block.Store.BlockStore
//type BlockScan = Alea.cuExtension.CUB.Block.Scan.BlockScan
//
//
//[<Test>]
//let ``block scan basic`` () =
//    
//    
//    let template block_threads items_per_thread (algorithm:BlockScanAlgorithm) = cuda {
//        let BlockLoad = BlockLoad.Create(block_threads, items_per_thread, BLOCK_LOAD_WARP_TRANSPOSE)
//        let BlockStore = BlockStore.Create(block_threads, items_per_thread, BLOCK_STORE_WARP_TRANSPOSE)
//        let BlockScan = BlockScan.Create(block_threads, algorithm)
//        
// 
//
//        let! kernel = 
//            <@ fun (blockLoad:BlockLoad) (blockStore:BlockStore) (blockScan:BlockScan) (d_in:deviceptr<int>) (d_out:deviceptr<int>) (d_elapsed:deviceptr<float>) ->
//                let temp_storage_load = __shared__.Extern()
//                let temp_storage_store = __shared__.Extern()
//                let temp_storage_scan = __shared__.Extern()
//
//                let data = __local__.Array<int>(items_per_thread)
//                blockLoad.Initialize(temp_storage_load).Load(d_in, data |> __array_to_ptr)
//
//                __syncthreads()
//
//                let start = 0.0 //clock()
//                let aggregate = __local__.Variable()
//                let x = blockScan.Initialize(temp_storage_scan).ExclusiveSum(items_per_thread, data |> __array_to_ptr, data |> __array_to_ptr, aggregate)
//
//                let stop = 0.0 //clock()
//
//                __syncthreads()
//
//                blockStore.Initialize(temp_storage_store).Store(d_out, data |> __array_to_ptr)
//
//                if threadIdx.x = 0 then
//                    d_elapsed.[0] <- if start > stop then start - stop else stop - start
//                    d_out.[block_threads * items_per_thread] <- !aggregate
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
//                kernel.Launch (LaunchParam(1, 128, items_per_thread)) BlockLoad BlockStore BlockScan d_in.Ptr d_out.Ptr d_elapsed.Ptr
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