module Test.Alea.cuExtension.CUB.Thread.Load

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Thread.Load


let debug = Test.Alea.cuExtension.CUB.Utilities.Util.debug

[<Test>]
let ``info of stuff`` () =
    if debug then
        BuilderRegistry.Instance.Ping()
        let irType = IRTypeBuilder.Instance.Build(typeof<uint4>)
        printfn "%A" irType
        printfn ""
        irType.Struct.Dump()


[<Test>]
let ``test cub_load`` () =
    let foo() = cuda {
        let! kernel = 
            <@ fun (x:deviceptr<int>) (output:deviceptr<int>) ->
                let tid = threadIdx.x
                let thing = cub_load x.[0]
                output.[tid] <- thing
            @> |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            let run (output:deviceptr<int>) =
                //let x = uint4(1u,2u,3u,4u)
                //use d_x = worker.Malloc<uint4>([|x|])
                let x = [| 99 |]
                use d_x = worker.Malloc(x)
                kernel.Launch (LaunchParam(1,128)) d_x.Ptr output
            run
        )}

    let program = foo() |> Compiler.load Worker.Default
    use output = Worker.Default.Malloc<int>(128)
    program.Run output.Ptr
    let output = output.Gather()
    printfn "%A" output


[<Test>]
let ``test cub_div`` () =
    let foo() = cuda {
        let! kernel = 
            <@ fun (x:int) (y:int) (output:deviceptr<int>) ->
                let tid = threadIdx.x
                let r = cub_div x y
                output.[tid] <- r //|> __ptr_to_obj
            @> |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            let run (output:deviceptr<int>) =
                let x = 10
                let y = 5
                kernel.Launch (LaunchParam(1,128)) x y output
            run
        )}

    let program = foo() |> Compiler.load Worker.Default
    use output = Worker.Default.Malloc<int>(128)
    program.Run output.Ptr
    let output = output.Gather()
    printfn "%A" output