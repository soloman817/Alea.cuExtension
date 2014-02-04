module Test.Alea.cuExtension.CUB.Utilities.Test


open Alea.CUDA
open Alea.CUDA.Utilities

open NUnit.Framework

open Alea.cuExtension.CUB.Utilities.Ptx

[<Test>]
let ``shr_add test`` () =
    let x = 0
    ()


[<Test>]
let ``bfi test`` () =
    let x = 0
    ()


[<Test>]
let ``dynamic attribute test`` () =
    let x,y,bit,numBits = 1,2,2,3

//    let r = __ptx__.BFI(x,y,bit,numBits)
//    printfn "%A" r

    let template = cuda {
        let! kernel = 
            <@ fun  (inputX:deviceptr<int>) 
                    (inputY:deviceptr<int>) 
                    (inputBit:deviceptr<int>) 
                    (inputNumBits:deviceptr<int>) 
                    (output:deviceptr<int>) ->
                
                let tid = threadIdx.x
                output.[tid] <- __ptx__.BFI(    inputX.[tid], 
                                                inputY.[tid],
                                                inputBit.[tid],
                                                inputNumBits.[tid])
            @> |> Compiler.DefineKernel
            
        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply(kernel)

            let run (xs:int[]) (ys:int[]) (bits:int[]) (numBitss:int[]) =
                let n = xs.Length
                //let xs,ys,bits,numBitss = input |> Array.map (fun e -> e)
                use d_xs = worker.Malloc(xs)
                use d_ys = worker.Malloc(ys)
                use d_bits = worker.Malloc(bits)
                use d_numBitss = worker.Malloc(numBitss)
                
                use output = worker.Malloc(n)

                let lp = LaunchParam(1, 128)
                
                kernel.Launch lp d_xs.Ptr d_ys.Ptr d_bits.Ptr d_numBitss.Ptr output.Ptr

                output.Gather()
            run          
    )}

    let xs = Array.init 128 (fun i -> i)
    let ys = xs |> Array.rev
    let bits = Array.init 128 (fun _ -> System.Random().Next(31))
    let numBits = Array.init 128 (fun _ -> System.Random().Next(31))

    //let h_output = [for i = 0 to 127 do yield __ptx__.BFI(xs.[i], ys.[i], bits.[i], numBits.[i])] |> Array.ofSeq

    let program = template |> Compiler.load Worker.Default
    let d_output = program.Run xs ys bits numBits

    //(h_output, d_output) ||> Array.iter2 (fun h d -> Assert.AreEqual(h,d))
    //(h_output, d_output) ||> Array.iter2 (fun h d -> printfn "[h][d] == [%d] [%d]" h d)
    printfn "%A" d_output