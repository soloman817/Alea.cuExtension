module Test.Alea.cuExtension.CUB.Thread.Scan

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

open Alea.cuExtension.CUB
open Alea.cuExtension.CUB.Thread

[<Test>]
let ``thread scan : inclusive : int`` () =
    
    let inline template() = cuda {
        let! foo = <@ fun (x:'T) (y:'T) (op:('T -> 'T -> 'T)) ->
                        (x,y) ||> op
                    @> |> Compiler.DefineFunction
        
//        let! kernel = 
//            <@ fun (scan:ThreadScan<int>) (input:deviceptr<int>) (output:deviceptr<int>) (output2:deviceptr<int>) ->
//                output2.[0] <- scan.Exclusive(input, output, (+), 0)
//            @> |> Compiler.DefineKernel

        let! kernel = 
            <@ fun (scan:ThreadScan<'T>) (input:deviceptr<'T>) (output:deviceptr<'T>) (output2:deviceptr<'T>) ->
                output.[0] <- foo.Invoke input.[0] input.[0] (+)
                //output2.[0] <- scan.Exclusive(input, output, (+), 0)
            @> |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker    
            let kernel = program.Apply kernel

            let run (input:'T[]) =
                let scan = ThreadScan<'T>.Create(input.Length)
                use d_in = worker.Malloc(input)
                use d_out = worker.Malloc<'T>(input.Length)
                use d_out2 = worker.Malloc<'T>(1)

                kernel.Launch (LaunchParam(1,1)) scan d_in.Ptr d_out.Ptr d_out2.Ptr

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
    printfn "Out2:\n%A\n" d_out2

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