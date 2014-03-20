module Test.Alea.cuExtension.CUB.Block.LoadStore
    
open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities

open NUnit.Framework

open Alea.cuExtension.CUB.Block

open Test.Alea.cuExtension.CUB.Utilities

let BLOCKS = 1
let THREADS = 128
let N = BLOCKS * THREADS

let BLOCK_THREADS = THREADS
let ITEMS_PER_THREAD = 4
//
let stripedData() = stripedData BLOCK_THREADS ITEMS_PER_THREAD

let loadAlgorithms = [
    BlockLoadAlgorithm.BLOCK_LOAD_DIRECT, "BLOCK_LOAD_DIRECT";
    BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE, "BLOCK_LOAD_VECTORIZE";
    BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE, "BLOCK_LOAD_TRANSPOSE";
    BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE, "BLOCK_LOAD_WARP_TRANSPOSE" ]

let StoreAlgorithms = [
    BlockStoreAlgorithm.BLOCK_STORE_DIRECT, "BLOCK_STORE_DIRECT";
    BlockStoreAlgorithm.BLOCK_STORE_VECTORIZE, "BLOCK_STORE_VECTORIZE";
    BlockStoreAlgorithm.BLOCK_STORE_TRANSPOSE, "BLOCK_STORE_TRANSPOSE";
    BlockStoreAlgorithm.BLOCK_STORE_WARP_TRANSPOSE, "BLOCK_STORE_WARP_TRANSPOSE" ]

[<Record>]
type TempStorage<'T> =
    {
        load : BlockLoad.TempStorage<'T>
        store : BlockStore.TempStorage<'T>
    }

    [<ReflectedDefinition>] static member Init(load, store) = { load = load; store = store }


let inline template
    (block_threads:int)
    (items_per_thread:int)
    (load_algorithm:BlockLoadAlgorithm) 
    (store_algorithm:BlockStoreAlgorithm)
    (warp_time_slicing:bool) = cuda {
        
    let TILE_SIZE = block_threads * items_per_thread

    let bload_h = BlockLoad.StaticParam.Init(block_threads, items_per_thread, load_algorithm, warp_time_slicing)
    let bstore_h = BlockStore.StaticParam.Init(block_threads, items_per_thread, store_algorithm, warp_time_slicing)

    let! kernel =
        <@ fun (d_in:deviceptr<int>) (d_out_unguarded:deviceptr<int>) (d_out_guarded:deviceptr<int>) (num_items:int) ->
            
            let temp_storage = TempStorage<int>.Init(BlockLoad.PrivateStorage<int>(bload_h.BlockExchangeParam), BlockStore.PrivateStorage<int>(bstore_h))

            let block_offset = blockIdx.x * TILE_SIZE
            let guarded_elements = num_items - block_offset

            let data = __local__.Array<int>(items_per_thread) |> __array_to_ptr

//            BlockLoad.API<int>.Init(bload_h, temp_storage.load).Load(bload_h, d_in + block_offset, data)
                
            __syncthreads()

//            BlockStore.API<int>.Init(bstore_h, temp_storage.store).Store(bstore_h, d_out_unguarded + block_offset, data)

            __syncthreads()

            for ITEM = 0 to items_per_thread - 1 do data.[ITEM] <- 0

            __syncthreads()

//            BlockLoad.API<int>.Init(bload_h, temp_storage.load).Load(bload_h, d_in + block_offset, data, guarded_elements)

            __syncthreads()

//            BlockStore.API<int>.Init(bstore_h, temp_storage.store).Store(bstore_h, d_out_guarded + block_offset, data, guarded_elements)
                
        @> |> Compiler.DefineKernel
        
    return Entry(fun (program:Program) ->
        let worker = program.Worker
        let kernel = program.Apply kernel

            
        fun (d_in:deviceptr<int>) (d_out_unguarded:deviceptr<int>) (d_out_guarded:deviceptr<int>) (grid_size:int) (guarded_elements:int) ->
            let lp = LaunchParam(grid_size,block_threads)
            kernel.Launch lp d_in d_out_unguarded d_out_guarded guarded_elements
                
    )}

let program = template BLOCK_THREADS ITEMS_PER_THREAD BlockLoadAlgorithm.BLOCK_LOAD_DIRECT BlockStoreAlgorithm.BLOCK_STORE_DIRECT false |> Compiler.load Worker.Default
let h_in = Array.init N (fun i -> i)
let run (h_in:int[]) block_threads items_per_thread grid_size fraction_valid =
    let unguarded_elements = grid_size * block_threads * items_per_thread
    let guarded_elements = fraction_valid * (float unguarded_elements) |> int
    let worker = Worker.Default
    use d_in = worker.Malloc(h_in)
    use d_out_unguarded = worker.Malloc<int>(unguarded_elements)
    use d_out_guarded = worker.Malloc<int>(guarded_elements)
    program.Run d_in.Ptr d_out_unguarded.Ptr d_out_guarded.Ptr grid_size guarded_elements
    (d_in.Gather(), d_out_unguarded.Gather(), d_out_guarded.Gather())



[<Test>]
let ``load store test`` () = 
    let d_in, d_out_unguarded, d_out_guarded = run h_in BLOCK_THREADS ITEMS_PER_THREAD 1 1.0
    printfn "d_in:\n%A\n\nd_out_unguarded:\n%A\n\nd_out_guarded:\n%A" d_in d_out_unguarded d_out_guarded


[<Test>]
let ``load test - int`` () =
    let block_threads = 16
    let items_per_thread = 4
    let n = block_threads * items_per_thread

    let template = cuda{
        let h = BlockLoad.StaticParam.Init(block_threads, items_per_thread, BlockLoadAlgorithm.BLOCK_LOAD_DIRECT, false)

        let! kernel =
            <@ fun (d_in:deviceptr<int>) (d_out:deviceptr<int>) ->
                let tid = threadIdx.x
                let thread_data = __local__.Array<int>(items_per_thread)
//                BlockLoad.API<int>.Init(h).Load(h, d_in, (thread_data |> __array_to_ptr))
                d_out.[tid] <- thread_data.[items_per_thread - 1]
            @> |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            fun (h_in:int[]) ->
                let lp = LaunchParam(1,block_threads)
                use d_in = worker.Malloc(h_in)
                use d_out = worker.Malloc<int>(h_in.Length)
                kernel.Launch lp d_in.Ptr d_out.Ptr

                d_out.Gather()
        )}
    
    let program = template |> Compiler.load Worker.Default
    let input = Array.init n (fun i -> i % 4)
    let output = program.Run input
    printfn "Host:\n%A\nDevice:\n%A\n" input output