module Test.Alea.cuExtension.CUB.Block.Scan

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

open Alea.cuExtension.CUB
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Block
open Alea.cuExtension.CUB.Block.BlockSpecializations

let BLOCKS = 1
let THREADS = 32
let N = BLOCKS * THREADS

let BLOCK_THREADS = THREADS
let ITEMS_PER_THREAD = 4

let cppResults = @"
Using device 0: GeForce GTX 560 Ti (PTX version 100, SM210, 8 SMs, 519 free / 1024 total MB physmem, ECC off)
BlockScan algorithm BLOCK_SCAN_RAKING on 1024 items (100 timing iterations, 1 blocks, 1024 threads, 1 items per thread, 1 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 14948.500
        Average BlockScan::Sum clocks per item: 14.598
        Average kernel millis: 0.0114
        Average million items / sec: 89.9079
BlockScan algorithm BLOCK_SCAN_RAKING on 1024 items (100 timing iterations, 1 blocks, 512 threads, 2 items per thread, 3 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 2402.480
        Average BlockScan::Sum clocks per item: 2.346
        Average kernel millis: 0.0039
        Average million items / sec: 263.5047
BlockScan algorithm BLOCK_SCAN_RAKING on 1024 items (100 timing iterations, 1 blocks, 256 threads, 4 items per thread, 4 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 1646.740
        Average BlockScan::Sum clocks per item: 1.608
        Average kernel millis: 0.0037
        Average million items / sec: 278.1641
BlockScan algorithm BLOCK_SCAN_RAKING on 1024 items (100 timing iterations, 1 blocks, 128 threads, 8 items per thread, 7 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 1388.140
        Average BlockScan::Sum clocks per item: 1.356
        Average kernel millis: 0.0041
        Average million items / sec: 250.0001
BlockScan algorithm BLOCK_SCAN_RAKING on 1024 items (100 timing iterations, 1 blocks, 64 threads, 16 items per thread, 8 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 1457.100
        Average BlockScan::Sum clocks per item: 1.423
        Average kernel millis: 0.0050
        Average million items / sec: 203.0845
BlockScan algorithm BLOCK_SCAN_RAKING on 1024 items (100 timing iterations, 1 blocks, 32 threads, 32 items per thread, 8 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 1398.000
        Average BlockScan::Sum clocks per item: 1.365
        Average kernel millis: 0.0097
        Average million items / sec: 105.9568
-------------
BlockScan algorithm BLOCK_SCAN_RAKING_MEMOIZE on 1024 items (100 timing iterations, 1 blocks, 1024 threads, 1 items per thread, 0 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 1398.000
        Average BlockScan::Sum clocks per item: 1.365
        Average kernel millis: 0.0018
        Average million items / sec: 574.5065
BlockScan algorithm BLOCK_SCAN_RAKING_MEMOIZE on 1024 items (100 timing iterations, 1 blocks, 512 threads, 2 items per thread, 2 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 1943.400
        Average BlockScan::Sum clocks per item: 1.898
        Average kernel millis: 0.0037
        Average million items / sec: 279.3295
BlockScan algorithm BLOCK_SCAN_RAKING_MEMOIZE on 1024 items (100 timing iterations, 1 blocks, 256 threads, 4 items per thread, 4 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 1532.180
        Average BlockScan::Sum clocks per item: 1.496
        Average kernel millis: 0.0036
        Average million items / sec: 286.7127
BlockScan algorithm BLOCK_SCAN_RAKING_MEMOIZE on 1024 items (100 timing iterations, 1 blocks, 128 threads, 8 items per thread, 7 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 1337.020
        Average BlockScan::Sum clocks per item: 1.306
        Average kernel millis: 0.0041
        Average million items / sec: 252.3065
BlockScan algorithm BLOCK_SCAN_RAKING_MEMOIZE on 1024 items (100 timing iterations, 1 blocks, 64 threads, 16 items per thread, 8 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 1424.960
        Average BlockScan::Sum clocks per item: 1.392
        Average kernel millis: 0.0050
        Average million items / sec: 203.6144
BlockScan algorithm BLOCK_SCAN_RAKING_MEMOIZE on 1024 items (100 timing iterations, 1 blocks, 32 threads, 32 items per thread, 8 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 1398.000
        Average BlockScan::Sum clocks per item: 1.365
        Average kernel millis: 0.0097
        Average million items / sec: 105.8096
-------------
BlockScan algorithm BLOCK_SCAN_WARP_SCANS on 1024 items (100 timing iterations, 1 blocks, 1024 threads, 1 items per thread, 1 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 4002.280
        Average BlockScan::Sum clocks per item: 3.908
        Average kernel millis: 0.0049
        Average million items / sec: 206.9456
BlockScan algorithm BLOCK_SCAN_WARP_SCANS on 1024 items (100 timing iterations, 1 blocks, 512 threads, 2 items per thread, 3 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 1657.140
        Average BlockScan::Sum clocks per item: 1.618
        Average kernel millis: 0.0035
        Average million items / sec: 288.7305
BlockScan algorithm BLOCK_SCAN_WARP_SCANS on 1024 items (100 timing iterations, 1 blocks, 256 threads, 4 items per thread, 5 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 1173.960
        Average BlockScan::Sum clocks per item: 1.146
        Average kernel millis: 0.0034
        Average million items / sec: 299.2611
BlockScan algorithm BLOCK_SCAN_WARP_SCANS on 1024 items (100 timing iterations, 1 blocks, 128 threads, 8 items per thread, 7 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 996.180
        Average BlockScan::Sum clocks per item: 0.973
        Average kernel millis: 0.0038
        Average million items / sec: 267.8049
BlockScan algorithm BLOCK_SCAN_WARP_SCANS on 1024 items (100 timing iterations, 1 blocks, 64 threads, 16 items per thread, 8 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 1146.420
        Average BlockScan::Sum clocks per item: 1.120
        Average kernel millis: 0.0049
        Average million items / sec: 207.0395
BlockScan algorithm BLOCK_SCAN_WARP_SCANS on 1024 items (100 timing iterations, 1 blocks, 32 threads, 32 items per thread, 8 SM occupancy):
        Output items: PASS
        Aggregate: PASS
        Average BlockScan::Sum clocks: 1536.000
        Average BlockScan::Sum clocks per item: 1.500
        Average kernel millis: 0.0097
        Average million items / sec: 105.6838
"
type ScanTimingParams =
    {
        items               : int
        iterations          : int
        blocks              : int
        threads             : int
        items_per_thread    : int
        sm_occupancy        : int
    }
    static member Init(items, iterations, blocks, threads, items_per_thread, sm_occupancy) =
        { items = items; iterations = iterations; blocks = blocks; threads = threads; items_per_thread = items_per_thread; sm_occupancy = sm_occupancy }

type ScanTimingResults =
    {
        DeviceInfo              : string
        ALGORITHM               : BlockScanAlgorithm
        TimingParams            : ScanTimingParams
        OutputPass              : bool
        AggregatePass           : bool
        AvgClocks               : float
        AvgClocksPerItem        : float
        AvgKernelMillis         : float
        AvgMillionItemsPerSec   : float
    }

    static member Init(resultString:string) = ()


[<Record>]
type TempStorage<'T> =
    {
        load    : BlockLoad.TempStorage<'T>
        store   : BlockStore.TempStorage<'T>
        scan    : BlockScan.TempStorage<'T>
//        scan    : BlockScanWarpScans.TempStorage<'T>
    }

    [<ReflectedDefinition>]
    static member Init(bload_h:BlockLoad.StaticParam, bstore_h:BlockStore.StaticParam, bscan_h:BlockScan.StaticParam) =
        {
            load = __shared__.Array<'T>(bload_h.BlockExchangeParam.SharedMemoryLength) |> __array_to_ptr
            store = __shared__.Array<'T>(bload_h.BlockExchangeParam.SharedMemoryLength) |> __array_to_ptr
            scan = BlockScan.TempStorage<'T>.Init(bscan_h.BlockScanRakingParam)
        }


//let [<ReflectedDefinition>] ExclusiveSum (h:BlockScan.StaticParam) temp_storage linear_tid input output = 
//    BlockScan.ExclusiveScan.MultipleDataPerThread.Identityless.Default h (+) ITEMS_PER_THREAD 
//        temp_storage linear_tid 
//        input output


let inline intTest (block_threads:int) (items_per_thread:int) = 
    cuda {
        let bload_h = BlockLoad.StaticParam.Init(block_threads, items_per_thread, BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE, false)
        let bstore_h = BlockStore.StaticParam.Init(block_threads, items_per_thread, BlockStoreAlgorithm.BLOCK_STORE_WARP_TRANSPOSE, false)
        let bscan_h = BlockScan.StaticParam.Init(block_threads, BlockScanAlgorithm.BLOCK_SCAN_WARP_SCANS)
            
        let bsws_h = BlockScanWarpScans.StaticParam.Init(block_threads)    



        let! kernel = 
            <@ fun (d_in:deviceptr<int>) (d_out:deviceptr<int>) ->
                let tid = threadIdx.x
                        
//                let temp_storage = TempStorage<int>.Init(bload_h, bstore_h, bsws_h)
                let temp_storage = TempStorage<int>.Init(bload_h, bstore_h, bscan_h)

                let data = __local__.Array<int>(items_per_thread)
                let dptr = data |> __array_to_ptr
                        
//                BlockLoad.API<int>.Init(bload_h, temp_storage.load).Load(bload_h, d_in, dptr)
                __syncthreads()

                let aggregate = __local__.Variable<int>()
                
                //BlockScan.ExclusiveSum.MultipleDataPerThread.WithAggregate bscan_h items_per_thread temp_storage.scan tid dptr dptr aggregate
                
//                BlockScan.IntApi.Init(bscan_h, temp_storage.scan).ExclusiveSum(bscan_h, items_per_thread, dptr, dptr, aggregate)
                
                //BlockScanWarpScans.API<int>.Init(bsws_h, temp_storage.scan, tid).ExclusiveSum(bsws_h, temp_storage.scan, dptr.[tid], dptr.Ref(tid), aggregate)
                __syncthreads()


//                BlockStore.API<int>.Init(bstore_h, temp_storage.store).Store(bstore_h, d_out, dptr)

                if threadIdx.x = 0 then d_out.[block_threads * items_per_thread] <- !aggregate

            @> |> Compiler.DefineKernel


        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            fun (d_in:deviceptr<int>) (d_out:deviceptr<int>) ->
                let lp = LaunchParam(1,block_threads)
                kernel.Launch lp d_in d_out

        )}

let scanAlgorithms = [
    BlockScanAlgorithm.BLOCK_SCAN_RAKING
    BlockScanAlgorithm.BLOCK_SCAN_RAKING_MEMOIZE
    BlockScanAlgorithm.BLOCK_SCAN_WARP_SCANS]


[<Test>]
let ``BlockScan exclusive sum`` () =
    let worker = Worker.Default
    let run block_threads items_per_thread =
        let TILE_SIZE = block_threads * items_per_thread
        let h_in = Array.zeroCreate<int> TILE_SIZE
        let h_reference = Array.zeroCreate<int> TILE_SIZE
        let h_gpu = Array.zeroCreate<int> (TILE_SIZE + 1)

        let mutable inclusive = 0

        let h_aggregate =
            for i = 0 to TILE_SIZE - 1 do
                h_in.[i] <- i % 17
                h_reference.[i] <- inclusive
                inclusive <- inclusive + h_in.[i]
            inclusive

        //let input = Array.init N (fun _ -> 1)

        

        let runTest() = 
            use program = intTest block_threads items_per_thread |> Compiler.load worker
            use d_in  = worker.Malloc(h_in)
            use d_out = worker.Malloc<int>(TILE_SIZE + 1)
            worker.Eval <| fun _ -> program.Run d_in.Ptr d_out.Ptr
            d_out.Gather()
        
        let d_out = runTest()
        let d_aggregate = d_out.[TILE_SIZE]
        printfn "H_in:\n%A\nH_Reference:\n%A\nH_Aggregate:\n%A\nD_Out:\n%A\nD_Aggegate:\n%A\n" h_in h_reference h_aggregate d_out d_aggregate

//    run 128 4
//    run 1024 1
//    run 512 2
//    run 128 8
//    run 64 16
//    run 32 32
    run 32 4