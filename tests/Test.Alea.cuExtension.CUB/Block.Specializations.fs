module Test.Alea.cuExtension.CUB.Block.Specializations
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

open Alea.cuExtension.CUB
open Alea.cuExtension.CUB.Block
open Alea.cuExtension.CUB.Block.BlockSpecializations

let BLOCKS = 1
let THREADS = 32
let N = BLOCKS * THREADS

let BLOCK_THREADS = THREADS
let ITEMS_PER_THREAD = 4

[<Record>]
type TempStorage =
    {
        mutable load    : BlockLoad.TempStorage<int>
        mutable store   : BlockStore.TempStorage<int>
        mutable scan    : BlockScanWarpScans.TempStorage<int>
    }
    [<ReflectedDefinition>] static member Init(load, store, scan) = { load = load; store = store; scan = scan }

[<Record>]
type DeviceApi =
    {
        mutable Load    : BlockLoad.API<int> 
        mutable Store   : BlockStore.API<int>
        mutable Scan    : BlockScanWarpScans.API<int>
    }
    [<ReflectedDefinition>] static member Init(load,store,scan) = { Load = load; Store = store; Scan = scan }


module HistogramAtomic =
    let f() = "histogram atomic"

module HistogramSort =
    let f() = "histogram sort"

module ReduceRanking =
    let f() = "reduce ranking"

module ReduceWarpReduction =
    let f() = "reduce warp reduction"

module BlockScanRaking =
    let f() = "scan raking"

module BlockScanWarpScans =
    let inline test (block_threads:int) (items_per_thread:int) = 
            cuda {
                let bsws_h  = BlockScanWarpScans.HostApi.Init(block_threads)
                let bload_h = BlockLoad.HostApi.Init(block_threads, items_per_thread)
                let bstore_h = BlockStore.HostApi.Init(block_threads, items_per_thread)
                
                //let! sum = <@ fun (x:int) (y:int) -> x + y @> |> Compiler.DefineFunction
                
//                let! scan = 
//                    <@ fun (d:BlockScanWarpScans.DeviceApi<int>) (input:int) 
//                            (output:Ref<int>) (block_aggregate:Ref<int>) -> 
//                        BlockScanWarpScans.ExclusiveSum.WithAggregateInt bsws_h d input output block_aggregate 
//                    @> |> Compiler.DefineFunction

                let! kernel = 
                    <@ fun (d_in:deviceptr<int>) (d_out:deviceptr<int>) ->
                        let tid = threadIdx.x
                    
                        let dApi = 
                            DeviceApi.Init(
                                BlockLoad.API<int>.Create(bload_h),
                                BlockStore.API<int>.Create(bstore_h),
                                BlockScanWarpScans.API<int>.Create(bsws_h, BlockScanWarpScans.TempStorage<int>.Uninitialized(bsws_h), threadIdx.x))
                
                        let temp_storage = 
                            TempStorage.Init(
                                dApi.Load.DeviceApi.temp_storage,
                                dApi.Store.DeviceApi.temp_storage,
                                dApi.Scan.DeviceApi.temp_storage)

                        //let fApi = FunctionApi<int>.Init(bload_fApi, bstore_fApi, bscan_fApi)

                        let data = __local__.Array<int>(items_per_thread)
                        let dptr = data |> __array_to_ptr
                        dApi.Load.Load(bload_h, d_in, dptr)//dApi.Load.device d_in (dptr)
                    
                        __syncthreads()

                        let aggregate = __local__.Variable<int>()
                        //BlockScan.ExclusiveSum.MDPT.WithAggregate bscan_h %scan_op items_per_thread dApi.Scan.DeviceApi dptr dptr aggregate
                        BlockScanWarpScans.ExclusiveSum.WithAggregateInt bsws_h dApi.Scan.DeviceApi dptr.[threadIdx.x] (dptr.Ref(0)) aggregate

                        __syncthreads()

                        dApi.Store.Store(bstore_h, d_out, dptr)

                        if threadIdx.x = 0 then d_out.[block_threads * items_per_thread] <- !aggregate

                    @> |> Compiler.DefineKernel


                return Entry(fun (program:Program) ->
                    let worker = program.Worker
                    let kernel = program.Apply kernel

                    fun (input:int[]) ->
                        use d_in = worker.Malloc(input)
                        use d_out = worker.Malloc<int>(input.Length)

                        let lp = LaunchParam(BLOCKS,THREADS)
                        kernel.Launch lp d_in.Ptr d_out.Ptr

                        d_out.Gather()
                )}

    let scanAlgorithms = [
        BlockScanAlgorithm.BLOCK_SCAN_RAKING
        BlockScanAlgorithm.BLOCK_SCAN_RAKING_MEMOIZE
        BlockScanAlgorithm.BLOCK_SCAN_WARP_SCANS]

    let [<ReflectedDefinition>] inline Sum() = fun (x:int) (y:int) -> x + y

    [<Test>]
    let ``block scan example`` () =
        let input = Array.init N (fun i -> i)
    
        //let sum() = fun x y -> x + y
    
        let hresult = Array.scan (+) 0 input

    
        for a in scanAlgorithms do
            let program = test BLOCK_THREADS ITEMS_PER_THREAD |> Compiler.load Worker.Default
            printfn "HostResult:\n%A\n" hresult
            printfn "DeviceResult:\n%A\n" (program.Run input)