module Test.Alea.cuExtension.CUB.Block.Exchange

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

open Alea.cuExtension.CUB.Block.Exchange

let BLOCKS = 2
let THREADS = 32
let N = BLOCKS * THREADS

let BLOCK_THREADS = THREADS
let ITEMS_PER_THREAD = 4


[<Test>]
let ``block exchange verification`` () =
    let template block_threads items_per_thread = cuda {
        let! hApi, fApi = BlockExchange.template<int> block_threads items_per_thread false
        // blocked to striped
        let! kbts = 
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
                let tid = threadIdx.x
                let dApi = BlockExchange.API<int>.Create(hApi)
                fApi.BlockedToStriped.Invoke dApi.device input
                let f = dApi.device.ThreadFields
                output.[tid] <- f.temp_storage.[tid]                
            @> |> Compiler.DefineKernel
        // blocked to warp striped
        let! kbtws = 
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
                let tid = threadIdx.x
                let dApi = BlockExchange.API<int>.Create(hApi)
                fApi.BlockedToWarpStriped.Invoke dApi.device input
                let f = dApi.device.ThreadFields
                output.[tid] <- f.temp_storage.[tid]                
            @> |> Compiler.DefineKernel
        // striped to blocked
        let! kstb = 
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
                let tid = threadIdx.x
                let dApi = BlockExchange.API<int>.Create(hApi)
                fApi.StripedToBlocked.Invoke dApi.device input
                let f = dApi.device.ThreadFields
                output.[tid] <- f.temp_storage.[tid]                
            @> |> Compiler.DefineKernel
        // warp striped to blocked
        let! kwstb = 
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
                let tid = threadIdx.x
                let dApi = BlockExchange.API<int>.Create(hApi)
                fApi.WarpStripedToBlocked.Invoke dApi.device input
                let f = dApi.device.ThreadFields
                output.[tid] <- f.temp_storage.[tid]                
            @> |> Compiler.DefineKernel
//        // scatter to blocked
//        let! ksctb = 
//            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
//                let tid = threadIdx.x
//                let dApi = BlockExchange.API<int>.Create(hApi)
//                fApi.ScatterToBlocked.Invoke dApi.device input
//                let f = dApi.device.ThreadFields
//                output.[tid] <- f.temp_storage.[tid]                
//            @> |> Compiler.DefineKernel
//        // scatter to striped
//        let! kscts = 
//            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
//                let tid = threadIdx.x
//                let dApi = BlockExchange.API<int>.Create(hApi)
//                fApi.BlockedToWarpStriped.Invoke dApi.device input
//                let f = dApi.device.ThreadFields
//                output.[tid] <- f.temp_storage.[tid]                
//            @> |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernels = [
                program.Apply kbts;
                program.Apply kbtws;
                program.Apply kstb;
                program.Apply kwstb]
//            let ksctb = program.Apply ksctb
//            let kscts = program.Apply kscts

            fun (input:int[]) ->
                use input = worker.Malloc(input)
                use output = worker.Malloc(input.Length)

                let lp = LaunchParam(BLOCKS,THREADS)
                [for k in kernels do
                    k.Launch lp input.Ptr output.Ptr
                    yield output.Gather()]
        )}

    let program = template BLOCK_THREADS ITEMS_PER_THREAD |> Compiler.load Worker.Default
    let input = Array.init N (fun i -> i)
    let output = program.Run input
    
    //printfn "%A" output
    for o in output do printfn "%A\n" o