module Test.Alea.cuExtension.CUB.Block.Scan

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

open Alea.cuExtension.CUB
open Alea.cuExtension.CUB.Thread
open Alea.cuExtension.CUB.Block

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
        mutable load    : BlockLoad.TempStorage<'T>
        mutable store   : BlockStore.TempStorage<'T>
        mutable scan    : BlockScan.TempStorage<'T>
    }
    [<ReflectedDefinition>] static member Init(load, store, scan) = { load = load; store = store; scan = scan }

[<Record>]
type DeviceApi<'T> =
    {
        mutable Load    : BlockLoad.API<'T> 
        mutable Store   : BlockStore.API<'T>
        mutable Scan    : BlockScan.API<'T>
    }
    [<ReflectedDefinition>] static member Init(load,store,scan) = { Load = load; Store = store; Scan = scan }

//[<Record>]
//type FunctionApi<'T> =
//    {
//        Load    : BlockLoad.FunctionApi<'T>
//        Store   : BlockStore.FunctionApi<'T>
//        Scan    : BlockScan.FunctionApi<'T>
//    }
//    [<ReflectedDefinition>] static member Init(load,store,scan) = { Load = load; Store = store; Scan = scan }


let inline test<'T> (block_threads:int) (algorithm:BlockScanAlgorithm) (scan_op:Expr<'T -> 'T -> 'T>) (items_per_thread:int) = 
        cuda {
            let bload_h  = BlockLoad.HostApi.Init(block_threads, items_per_thread, BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE, false)
            let bstore_h = BlockStore.HostApi.Init(block_threads, items_per_thread, BlockStoreAlgorithm.BLOCK_STORE_WARP_TRANSPOSE, false)
            let bscan_h  = BlockScan.HostApi.Init(block_threads, algorithm)


//            let! scan = 
//                <@ fun (d:DeviceApi<'T>) (input:deviceptr<'T>) (output:deviceptr<'T>) (agg:Ref<'T>) -> 
//                    BlockScan.ExclusiveSum.MDPT.WithAggregate bscan_h %scan_op items_per_thread d.Scan.DeviceApi input output agg
//                @> |> Compiler.DefineFunction

            let! kernel = 
                <@ fun (d_in:deviceptr<'T>) (d_out:deviceptr<'T>) ->
                    let tid = threadIdx.x
                    
                    let dApi = 
                        DeviceApi<'T>.Init(
                            BlockLoad.API<'T>.Create(bload_h),
                            BlockStore.API<'T>.Create(bstore_h),
                            BlockScan.API<'T>.Create(bscan_h))
                
                    let temp_storage = 
                        TempStorage<'T>.Init(
                            dApi.Load.DeviceApi.temp_storage,
                            dApi.Store.DeviceApi.temp_storage,
                            dApi.Scan.DeviceApi.temp_storage)

                    //let fApi = FunctionApi<'T>.Init(bload_fApi, bstore_fApi, bscan_fApi)

                    let data = __local__.Array<'T>(items_per_thread)
                    let dptr = data |> __array_to_ptr
                    dApi.Load.Load(bload_h, d_in, dptr)//dApi.Load.device d_in (dptr)
                    
                    __syncthreads()

                    let aggregate = __local__.Variable<'T>()
                    //BlockScan.ExclusiveSum.MDPT.WithAggregate bscan_h %scan_op items_per_thread dApi.Scan.DeviceApi dptr dptr aggregate
//                    scan.Invoke dApi dptr dptr aggregate

                    __syncthreads()

                    dApi.Store.Store(bstore_h, d_out, dptr)

                    if threadIdx.x = 0 then d_out.[block_threads * items_per_thread] <- !aggregate

                @> |> Compiler.DefineKernel


            return Entry(fun (program:Program) ->
                let worker = program.Worker
                let kernel = program.Apply kernel

                fun (input:'T[]) ->
                    use d_in = worker.Malloc(input)
                    use d_out = worker.Malloc<'T>(input.Length)

                    let lp = LaunchParam(BLOCKS,THREADS)
                    kernel.Launch lp d_in.Ptr d_out.Ptr

                    d_out.Gather()
            )}

let scanAlgorithms = [
    BlockScanAlgorithm.BLOCK_SCAN_RAKING
    BlockScanAlgorithm.BLOCK_SCAN_RAKING_MEMOIZE
    BlockScanAlgorithm.BLOCK_SCAN_WARP_SCANS]

let [<ReflectedDefinition>] inline Sum() = fun (x:'T) (y:'T) -> x + y

[<Test>]
let ``block scan example`` () =
    let input = Array.init N (fun i -> i)
    
    //let sum() = fun x y -> x + y
    
    let hresult = Array.scan (+) 0 input

    
    for a in scanAlgorithms do
        let program = test<int> BLOCK_THREADS a <@ Sum() @> ITEMS_PER_THREAD |> Compiler.load Worker.Default
        printfn "HostResult:\n%A\n" hresult
        printfn "DeviceResult:\n%A\n" (program.Run input)



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