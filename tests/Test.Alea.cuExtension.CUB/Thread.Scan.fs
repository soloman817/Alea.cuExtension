module Test.Alea.cuExtension.CUB.Thread.Scan

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework
open Microsoft.FSharp.Quotations
open Alea.cuExtension.CUB
open Alea.cuExtension.CUB.Thread

let N = 32

//let [<ReflectedDefinition>] inline Default (length:int) (scan_op:'T -> 'T -> 'T)
//    (inclusive:'T) (exclusive:'T) (input:deviceptr<'T>) (output:deviceptr<'T>) =
//        //let scan_op = tp.scan_op
//        let mutable addend = input.[0]
//        let mutable inclusive = inclusive
//        output.[0] <- exclusive
//        let mutable exclusive = inclusive
//
//        for i = 1 to (length - 1) do
//            addend <- input.[i]
//            inclusive <- (exclusive, addend) ||> scan_op
//            output.[i] <- exclusive
//            exclusive <- inclusive
//
//        inclusive

[<Test>]
let ``thread scan : verification`` () =
    let inline template (length:int) = cuda {
        //let scan_op = (scan_op ADD 0)
        //let tp = Thread.Scan.Template._TemplateParams<int>.Init length
        //let! threadScan = <@ fun () -> Default length (+) @> |> Compiler.DefineFunction
        let! kernel = 
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
                let tid = threadIdx.x
                let ThreadScan = ThreadScanExclusive.API3<int>.Init(length)
                let inclusive = ThreadScan.Default (+) 0 0 input output
                //let inclusive = threadScan.WithApplyPrefixDefault.Invoke (+) input output input.[0]
                //let inclusive = threadScan.Invoke() 0 0 input output
                //let inclusive = (%foo) 0 0 input output
                //let inclusive = Default length (+) 0 0 input output
                ()
            @> |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker    
            let kernel = program.Apply kernel

            fun (input:int[]) ->
                use input = worker.Malloc(input)
                use output = worker.Malloc(input.Length)
                kernel.Launch (LaunchParam(1,length)) input.Ptr output.Ptr
                output.Gather()
        )}

    let program = template N |> Compiler.load Worker.Default
    let input = Array.init N (fun i -> i)
    let output = program.Run input
    let h_output = Array.scan (+) 0 input

    printfn "host: %A" h_output
    printfn "device: %A" output
//    
//
//
//[<Test>]
//let ``thread scan : inclusive : int`` () =
//    
//    let inline template() = cuda {
//        //let op = <@ op @>
//        let! kernel = 
//            <@ fun (scan:ThreadScan) (input:deviceptr<int>) (output:deviceptr<int>) ->
//                //let op = scan_op.plus
//               (scan.Inclusive(input, output,(+)))
//            @> |> Compiler.DefineKernel
//
//
//        return Entry(fun program ->
//            let worker = program.Worker    
//            let kernel = program.Apply kernel
//
//            let run (input:int[]) =
//                let scan = ThreadScan.Create(input.Length)
//                use d_in = worker.Malloc(input)
//                use d_out = worker.Malloc<int>(input.Length + 1)
//                use d_out2 = worker.Malloc<int>(input.Length + 1)
//
//                kernel.Launch (LaunchParam(1,1)) scan d_in.Ptr d_out.Ptr
//
//                d_out.Gather(), d_out2.Gather()
//
//            run )}
//
//
//    let n = 10
//    let h_in = Array.init n (fun i -> i)
//    let h_out = h_in |> Array.scan (+) 0
//    
//    let program = template() |> Compiler.load Worker.Default
//    let d_out, d_out2 = program.Run h_in
//    
//    printfn "Host Input:\n%A\n" h_in
//    printfn "Host Output:\n%A\n" h_out
//    printfn "Device Output:\n%A\n" d_out
//    printfn "Out2:\n%A\n" d_out2.[0]
//
//    (h_out, d_out) ||> Array.iter2 (fun h d -> Assert.AreEqual(h,d)) 
//
//
//[<Test>]
//let ``thread scan : inclusive : int64`` () =
//    ()
//
//
//[<Test>]
//let ``thread scan : exclusive : int`` () =
//    ()
//
//
//[<Test>]
//let ``thread scan : exclusive : int64`` () =
//    ()
//
//
//[<Test>]
//let ``thread scan : inclusive : float32`` () =
//    ()
//
//
//[<Test>]
//let ``thread scan : inclusive : float`` () =
//    ()
//
//
//[<Test>]
//let ``thread scan : exclusive : float32`` () =
//    ()
//
//
//[<Test>]
//let ``thread scan : exclusive : float`` () =
//    ()