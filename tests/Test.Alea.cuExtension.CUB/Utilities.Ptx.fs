module Test.Alea.cuExtension.CUB.Utilities.Ptx

open System

open Alea.CUDA
open Alea.CUDA.Utilities

open NUnit.Framework

open Alea.cuExtension.CUB.Utilities.Ptx


let BLOCKS = 1
let BLOCK_THREADS = 128
let N = BLOCKS * BLOCK_THREADS

[<Test>]
let ``shr_add test`` () =
    
    let shradd x shift addend = (x >>> shift) + addend
    
    
    let template = cuda {
    
        let! kernel =
            <@ fun (x:deviceptr<int>) (shift:deviceptr<int>) (addend:deviceptr<int>) (output:deviceptr<int>) ->
                let tid = threadIdx.x
                output.[tid] <- __ptx__.SHR_ADD(x.[tid], shift.[tid], addend.[tid])
            @> |> Compiler.DefineKernel
            
        
        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            fun (x:int[]) (shift:int[]) (addend:int[]) ->
                use dx = worker.Malloc(x)
                use dshift = worker.Malloc(shift)
                use daddend = worker.Malloc(addend)

                use dout = worker.Malloc(x.Length)

                let lp = LaunchParam(BLOCKS,BLOCK_THREADS)

                kernel.Launch lp dx.Ptr dshift.Ptr daddend.Ptr dout.Ptr

                dout.Gather()
            )}


    let program = template |> Compiler.load Worker.Default
    let rng = new System.Random()
    let hx = Array.init N (fun _ -> rng.Next(100))
    let hshift = Array.init N (fun _ -> rng.Next(5))
    let haddend = Array.init N (fun _ -> rng.Next(1000))
    let hout = (hx,hshift, haddend) |||> Array.zip3 |> Array.map (fun e -> e |||> shradd)
    let dout = program.Run hx hshift haddend
    printfn "hout:\n%A\ndout:\n%A\n" hout dout
    (hout,dout) ||> Array.iter2 (fun h d -> Assert.AreEqual(h,d))


[<Test>]
let ``bfi test`` () =
    let template = cuda {
    
        let! kernel =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
                let x = __ptx__.BFI(1,1,1,1)
                output.[threadIdx.x] <- x
            @> |> Compiler.DefineKernel
            
        
        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            fun (input:int[]) ->
                use d_in = worker.Malloc(input)
                use d_out = worker.Malloc(input.Length)

                let lp = LaunchParam(1,1)

                kernel.Launch lp d_in.Ptr d_out.Ptr

                d_out.Gather()
            )}


    let program = template |> Compiler.load Worker.Default
    let hinput = Array.init N (fun i -> i)
    let output = program.Run hinput
    printfn "%A"     output


[<Test>]
let ``LaneId test`` () =
    let template = cuda {
    
        let! kernel =
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
                let x = __ptx__.LaneId() |> int
                output.[threadIdx.x] <- x
            @> |> Compiler.DefineKernel
            
        
        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            fun (input:int[]) ->
                use d_in = worker.Malloc(input)
                use d_out = worker.Malloc(input.Length)

                let lp = LaunchParam(BLOCKS,BLOCK_THREADS)

                kernel.Launch lp d_in.Ptr d_out.Ptr

                d_out.Gather()
            )}


    let program = template |> Compiler.load Worker.Default
    let hinput = Array.init N (fun i -> i)
    let output = program.Run hinput
    printfn "%A"     output

//[<Test>]
//let ``dynamic attribute test`` () =
//    let x,y,bit,numBits = 1,2,2,3
//
////    let r = __ptx__.BFI(x,y,bit,numBits)
////    printfn "%A" r
//
//    let template = cuda {
//        let! kernel = 
//            <@ fun  (inputX:deviceptr<int>) 
//                    (inputY:deviceptr<int>) 
//                    (inputBit:deviceptr<int>) 
//                    (inputNumBits:deviceptr<int>) 
//                    (output:deviceptr<int>) ->
//                
//                let tid = threadIdx.x
//                output.[tid] <- __ptx__.BFI(    inputX.[tid], 
//                                                inputY.[tid],
//                                                inputBit.[tid],
//                                                inputNumBits.[tid])
//            @> |> Compiler.DefineKernel
//            
//        return Entry(fun program ->
//            let worker = program.Worker
//            let kernel = program.Apply(kernel)
//
//            let run (xs:int[]) (ys:int[]) (bits:int[]) (numBitss:int[]) =
//                let n = xs.Length
//                //let xs,ys,bits,numBitss = input |> Array.map (fun e -> e)
//                use d_xs = worker.Malloc(xs)
//                use d_ys = worker.Malloc(ys)
//                use d_bits = worker.Malloc(bits)
//                use d_numBitss = worker.Malloc(numBitss)
//                
//                use output = worker.Malloc(n)
//
//                let lp = LaunchParam(1, 128)
//                
//                kernel.Launch lp d_xs.Ptr d_ys.Ptr d_bits.Ptr d_numBitss.Ptr output.Ptr
//
//                output.Gather()
//            run          
//    )}
//
//    let xs = Array.init 128 (fun i -> i)
//    let ys = xs |> Array.rev
//    let bits = Array.init 128 (fun _ -> System.Random().Next(31))
//    let numBits = Array.init 128 (fun _ -> System.Random().Next(31))
//
//    //let h_output = [for i = 0 to 127 do yield __ptx__.BFI(xs.[i], ys.[i], bits.[i], numBits.[i])] |> Array.ofSeq
//
//    let program = template |> Compiler.load Worker.Default
//    let d_output = program.Run xs ys bits numBits
//
//    //(h_output, d_output) ||> Array.iter2 (fun h d -> Assert.AreEqual(h,d))
//    //(h_output, d_output) ||> Array.iter2 (fun h d -> printfn "[h][d] == [%d] [%d]" h d)
//    printfn "%A" d_output