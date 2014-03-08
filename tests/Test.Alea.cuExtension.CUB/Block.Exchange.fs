module Test.Alea.cuExtension.CUB.Block.Exchange

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

open Alea.cuExtension.CUB.Block.Exchange

//type [<Record>] _Template<'T> = Alea.cuExtension.CUB.Block.Exchange._Template<'T>

[<Test>]
let ``block exchange template verification`` () =
    //let BlockExchange = BlockExchange.template<int>
    
    let template block_threads items_per_thread = cuda {
        let hostapi = Template.Host.API.Init(block_threads, items_per_thread, false)
        let! kernel = 
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
                let betp = _Template<int>.Init(hostapi)
                ()
            @> |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            fun (input:int[]) ->
                use input = worker.Malloc(input)
                use output = worker.Malloc(input.Length)

                let lp = LaunchParam(1,1)
                
                kernel.Launch lp input.Ptr output.Ptr

                output.Gather()
        )}

    let program = template 1 8 |> Compiler.load Worker.Default
    let input = Array.init 32 (fun i -> i)
    let output = program.Run input
    
    printfn "%A" output


[<Test>]
let ``block exchange verification`` () =
    //let BlockExchange = BlockExchange.template<int>
    let template block_threads items_per_thread = cuda {
        
        //let! blockLoad = BlockLoad block_threads items_per_thread BlockLoadAlgorithm.BLOCK_LOAD_DIRECT false
        let BlockExchange = BlockExchange.API<int>.Init(block_threads, items_per_thread).BlockToStriped

        let! kernel = 
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
                let tid = threadIdx.x
                BlockExchange.Default input
                output.[tid] <- BlockExchange.template.ThreadFields.temp_storage.[tid]                
            @> |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            fun (input:int[]) ->
                use input = worker.Malloc(input)
                use output = worker.Malloc(input.Length)

                let lp = LaunchParam(1,1)
                
                kernel.Launch lp input.Ptr output.Ptr

                output.Gather()
        )}

    let program = template 1 8 |> Compiler.load Worker.Default
    let input = Array.init 32 (fun i -> i)
    let output = program.Run input
    
    printfn "%A" output