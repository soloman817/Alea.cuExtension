module Test.Alea.cuExtension.CUB.Thread.Scan

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

open Alea.cuExtension.CUB
open Alea.cuExtension.CUB.Thread



[<Test>]
let ``thread scan : verification`` () =
    let inline template() = cuda {
        //let scan_op = scan_op 0
        let! kernel = 
            <@ fun (scan:ThreadScan<int>) ->
                let s = scan
                ()
            @> |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker    
            let kernel = program.Apply kernel

            fun _ ->
                let scan = ThreadScan<int>.Create(1)
                kernel.Launch (LaunchParam(1,1)) scan )}

    let program = template() |> Compiler.load Worker.Default
    program.Run()

    printfn "done"
    


[<Test>]
let ``thread scan : inclusive : int`` () =
    
    let inline template() = cuda {
        //let op = <@ op @>
        let! kernel = 
            <@ fun (scan:ThreadScan<int>) (input:deviceptr<int>) (output:deviceptr<int>) ->
                //let op = scan_op.plus
               (scan.Inclusive(input, output,(+)))
            @> |> Compiler.DefineKernel


        return Entry(fun program ->
            let worker = program.Worker    
            let kernel = program.Apply kernel

            let run (input:int[]) =
                let scan = ThreadScan<int>.Create(input.Length)
                use d_in = worker.Malloc(input)
                use d_out = worker.Malloc<int>(input.Length + 1)
                use d_out2 = worker.Malloc<int>(input.Length + 1)

                kernel.Launch (LaunchParam(1,1)) scan d_in.Ptr d_out.Ptr

                d_out.Gather(), d_out2.Gather()

            run )}


    let n = 10
    let h_in = Array.init n (fun i -> i)
    let h_out = h_in |> Array.scan (+) 0
    
    let program = template() |> Compiler.load Worker.Default
    let d_out, d_out2 = program.Run h_in
    
    printfn "Host Input:\n%A\n" h_in
    printfn "Host Output:\n%A\n" h_out
    printfn "Device Output:\n%A\n" d_out
    printfn "Out2:\n%A\n" d_out2.[0]

    (h_out, d_out) ||> Array.iter2 (fun h d -> Assert.AreEqual(h,d)) 


[<Test>]
let ``thread scan : inclusive : int64`` () =
    ()


[<Test>]
let ``thread scan : exclusive : int`` () =
    ()


[<Test>]
let ``thread scan : exclusive : int64`` () =
    ()


[<Test>]
let ``thread scan : inclusive : float32`` () =
    ()


[<Test>]
let ``thread scan : inclusive : float`` () =
    ()


[<Test>]
let ``thread scan : exclusive : float32`` () =
    ()


[<Test>]
let ``thread scan : exclusive : float`` () =
    ()