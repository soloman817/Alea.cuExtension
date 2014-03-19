module Test.Alea.cuExtension.CUB.Block.Specializations
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework
open Alea.CUDA.Utilities.NumericLiteralG
open Alea.cuExtension.CUB
open Alea.cuExtension.CUB.Block
open Alea.cuExtension.CUB.Block.BlockSpecializations

let BLOCKS = 1
let THREADS = 32
let N = BLOCKS * THREADS

let BLOCK_THREADS = THREADS
let ITEMS_PER_THREAD = 4

[<Record>]
type ScanTempStorage<'T> =
    {
        load    : BlockLoad.TempStorage<'T>
        store   : BlockStore.TempStorage<'T>
        scan    : BlockScanWarpScans.TempStorage<'T>
    }

    [<ReflectedDefinition>]
    static member Init(bload_h:BlockLoad.HostApi, bstore_h:BlockStore.HostApi, bsws_h:BlockScanWarpScans.HostApi) =
        {
            load = __shared__.Array<'T>(bload_h.SharedMemoryLength) |> __array_to_ptr
            store = __shared__.Array<'T>(bload_h.SharedMemoryLength) |> __array_to_ptr
            scan = BlockScanWarpScans.TempStorage<'T>.Uninitialized(bsws_h)
        }
//
//[<Record>]
//type DeviceApi<'T> =
//    {
//        load : BlockLoad.API<'T>
//        store : BlockStore.API<'T>
//        scan : BlockScanWarpScans.API<'T>    
//    }
//
//    [<ReflectedDefinition>]
//    static member Init(bload_h:BlockLoad.HostApi, bstore_h:BlockStore.HostApi, bsws_h:BlockScanWarpScans.HostApi ) =
//        let temp_storage = ScanTempStorage<'T>.Init(bsws_h)
//        {
//            load = BlockLoad.API<'T>.Init(bload_h, temp_storage.load, threadIdx.x)
//            store = BlockStore.API<'T>.Init(bstore_h, temp_storage.store, threadIdx.x)
//            scan = BlockScanWarpScans.API<'T>.Init(bsws_h, temp_storage.scan, threadIdx.x)
        //}


module HistogramAtomic =
    let f() = "histogram atomic"

module HistogramSort =
    let f() = "histogram sort"

module ReduceRanking =
    let f() = "reduce ranking"

module ReduceWarpReduction =
    let f() = "reduce warp reduction"

module BlockScanRaking =
    let block_threads = 16
    let items_per_thread = 1
    let n = block_threads * items_per_thread
    
    [<Test>]
    let ``BlockScanRaking exclusive sum - int`` () =
        let template = cuda {
            let bload_h = BlockLoad.HostApi.Init(block_threads, items_per_thread)
            let bstore_h = BlockStore.HostApi.Init(block_threads, items_per_thread)
            let bsr_h = BlockScanRaking.HostApi.Init(block_threads, true)

            let! kernel =
                <@ fun (d_in:deviceptr<int>) (d_out:deviceptr<int>) ->
                    let tid = threadIdx.x
                    let temp_storage = BlockScanRaking.TempStorage<int>.Uninitialized(bsr_h)
                    let cached_segment = __local__.Array<int>(bsr_h.Constants.SEGMENT_LENGTH)
                    
                        
                    //BlockLoad.API<int>.Init(bload_h, __null()).Load(bload_h, d_in, dptr)
                    let thread_data = __local__.Variable<int>(d_in.[tid])
                    __syncthreads()

                    let aggregate = __local__.Variable<int>()


                    BlockScanRaking.ExclusiveSum.WithAggregateInt bsr_h temp_storage tid cached_segment !thread_data thread_data aggregate
                    
                    __syncthreads()

                    d_out.[tid] <- !aggregate
                    //if threadIdx.x = 0 then d_out.[block_threads * items_per_thread] <- !aggregate                    
                    
                @> |> Compiler.DefineKernel
            
            return Entry(fun (program:Program) ->
                let worker = program.Worker
                let kernel = program.Apply kernel                
                fun (input:int[]) ->
                    use d_in = worker.Malloc(input)
                    use d_out = worker.Malloc(d_in.Length)
                    let lp = LaunchParam(1,block_threads)
                    kernel.Launch lp d_in.Ptr d_out.Ptr
                    d_out.Gather()
            )}

        let program = template |> Compiler.load Worker.Default
        let hinput = Array.init n (fun _ -> 1)
        let doutput = program.Run hinput
        printfn "%A" doutput


module BlockScanWarpScans =
    open Alea.cuExtension.CUB.Utilities
    open Alea.cuExtension.CUB.Warp

    [<Test>]
    let ``BlockScanWarpScans - initialization`` () =
        let template = cuda {
            let h = BlockScanWarpScans.HostApi.Init(BLOCK_THREADS)

            let! kernel =
                <@ fun (d_in:deviceptr<int>) (d_out:deviceptr<int>) ->
                    let tid = threadIdx.x
                    let temp_storage = BlockScanWarpScans.TempStorage<int>.Uninitialized(h)
                    let bsws = BlockScanWarpScans.API<int>.Init(h, temp_storage, threadIdx.x)
                    temp_storage.block_prefix := 99
                    d_out.[tid] <- !bsws.temp_storage.block_prefix
                @> |> Compiler.DefineKernel
            
            return Entry(fun (program:Program) ->
                let worker = program.Worker
                let kernel = program.Apply kernel                
                fun (input:int[]) ->
                    use d_in = worker.Malloc(input)
                    use d_out = worker.Malloc(d_in.Length)
                    let lp = LaunchParam(BLOCKS,THREADS)
                    kernel.Launch lp d_in.Ptr d_out.Ptr
                    d_out.Gather()
            )}

        let program = template |> Compiler.load Worker.Default
        let hinput = Array.init N (fun _ -> 1)
        let doutput = program.Run hinput
        printfn "%A" doutput



    let inline test (block_threads:int) (items_per_thread:int) = 
            cuda {
                let bload_h = BlockLoad.HostApi.Init(BLOCK_THREADS, ITEMS_PER_THREAD, BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE, false)
                let bstore_h = BlockStore.HostApi.Init(BLOCK_THREADS, ITEMS_PER_THREAD, BlockStoreAlgorithm.BLOCK_STORE_WARP_TRANSPOSE, false)
                let bsws_h = BlockScanWarpScans.HostApi.Init(BLOCK_THREADS)
                

                let! kernel = 
                    <@ fun (d_in:deviceptr<int>) (d_out:deviceptr<int>) ->
                        let tid = threadIdx.x
                        
                        let temp_storage = ScanTempStorage<int>.Init(bload_h, bstore_h, bsws_h)
                        //let temp_load = SharedRecord<int>.Init(N)
                        //let temp_store = SharedRecord<int>.Init(N)
                        let temp_scan = BlockScanWarpScans.TempStorage<int>.Uninitialized(bsws_h)

                        let warp_id = BlockScanWarpScans.warp_id block_threads tid
                        let lane_id = BlockScanWarpScans.lane_id block_threads tid

                        let data = __local__.Array<int>(items_per_thread)
                        let dptr = data |> __array_to_ptr
                        
                        BlockLoad.API<int>.Init(bload_h, temp_storage.load, tid).Load(bload_h, d_in, dptr)
                        __syncthreads()

                        let aggregate = __local__.Variable<int>()
                        //BlockScanWarpScans.IntApi.Init(bsws_h, temp_storage.scan, tid).ExclusiveSum(bsws_h, dptr.[tid], dptr.Ref(tid), aggregate)
                        BlockScanWarpScans.ExclusiveSum.WithAggregateInt bsws_h temp_scan tid warp_id lane_id d_in.[tid] (d_out.Ref(tid)) aggregate
                        __syncthreads()

                        BlockStore.API<int>.Init(bstore_h, temp_storage.store, tid).Store(bstore_h, d_out, dptr)

                        if threadIdx.x = 0 then d_out.[block_threads * items_per_thread] <- !aggregate

                    @> |> Compiler.DefineKernel


                return Entry(fun (program:Program) ->
                    let worker = program.Worker
                    let kernel = program.Apply kernel

                    fun (d_in:deviceptr<int>) (d_out:deviceptr<int>) ->
                        let lp = LaunchParam(BLOCKS,THREADS)
                        kernel.Launch lp d_in d_out

                )}

    let scanAlgorithms = [
        BlockScanAlgorithm.BLOCK_SCAN_RAKING
        BlockScanAlgorithm.BLOCK_SCAN_RAKING_MEMOIZE
        BlockScanAlgorithm.BLOCK_SCAN_WARP_SCANS]

    let [<ReflectedDefinition>] inline Sum() = fun (x:int) (y:int) -> x + y

    [<Test>]
    let ``BlockScanWarpScans exclusive sum`` () =
        let TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD
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

        let input = Array.init N (fun _ -> 1)
    
        //let sum() = fun x y -> x + y
    
        
        let run() = 
            let worker = Worker.Default
            let program = test BLOCK_THREADS ITEMS_PER_THREAD |> Compiler.load Worker.Default
            use d_in  = worker.Malloc(h_in)
            use d_out = worker.Malloc<int>(TILE_SIZE + 1)
            program.Run d_in.Ptr d_out.Ptr
            d_out.Gather()
        
        let d_out = run()
        let d_aggregate = d_out.[TILE_SIZE]
        printfn "H_in:\n%A\nH_Refence:\n%A\nH_Aggregate:\n%A\nD_Out:\n%A\nD_Aggegate:\n%A\n" h_in h_reference h_aggregate d_out d_aggregate
    
//        for a in scanAlgorithms do
//            let program = test BLOCK_THREADS ITEMS_PER_THREAD |> Compiler.load Worker.Default
//            printfn "HostResult:\n%A\n" hresult
//            printfn "DeviceResult:\n%A\n" (program.Run input)