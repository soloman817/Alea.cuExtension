module Test.Alea.CUDA.Extension.SegmentedScan

open System
open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.Interop.LLVM
open Alea.CUDA
open Alea.CUDA.IRBuilderUtil
open Alea.CUDA.Extension

let worker = getDefaultWorker()

let deviceOnly () = failwith "Device only!"

let [<LLVMFunctionBuilder>] bfi(x:uint32, y:uint32, bit:uint32, numBits:uint32):uint32 = deviceOnly()
let ``bfi [BUILDER]``(ctx:LLVMFunctionBuilderContext) =
    let args = ctx.LLVMValueArgs
    let funct = LLVMFunctionTypeEx(ctx.LLVMHelper.i32_t, [| ctx.LLVMHelper.i32_t; ctx.LLVMHelper.i32_t; ctx.LLVMHelper.i32_t; ctx.LLVMHelper.i32_t |], 0)
    let funcp = LLVMConstInlineAsm(funct, "bfi.b32 \t$0, $2, $1, $3, $4;", "=r,r,r,r,r", 0, 0)
    Value(LLVMBuildCallEx(ctx.Builder, funcp, args, ""))

let [<LLVMFunctionBuilder>] __ballot(a:int):uint32 = deviceOnly()
let ``__ballot [BUILDER]``(ctx:LLVMFunctionBuilderContext) =
    let args = ctx.LLVMValueArgs
    let funct = LLVMFunctionTypeEx(ctx.LLVMHelper.i32_t, [| ctx.LLVMHelper.i32_t |], 0)
    let cmdstr = "{ \n\t\
                  .reg .pred \t%p1; \n\t\
                  setp.ne.u32 \t%p1, $1, 0; \n\t\
                  vote.ballot.b32 \t$0, %p1; \n\t\
                  }"
    let funcp = LLVMConstInlineAsm(funct, cmdstr, "=r,r", 1, 0)
    Value(LLVMBuildCallEx(ctx.Builder, funcp, [| args.[0] |], ""))

let [<LLVMFunctionBuilder>] __clz(x:uint32):int = deviceOnly()
let ``__clz [BUILDER]``(ctx:LLVMFunctionBuilderContext) =
    let args = ctx.LLVMValueArgs
    let funct = LLVMFunctionTypeEx(ctx.LLVMHelper.i32_t, [| ctx.LLVMHelper.i32_t |], 0)
    let funcp = LLVMConstInlineAsm(funct, "clz.b32 \t$0, $1;", "=r,r", 0, 0)
    Value(LLVMBuildCallEx(ctx.Builder, funcp, args, ""))
    
let [<ReflectedDefinition>] segscanWarp (input:DevicePtr<int>) (output:DevicePtr<int>) (debug:DevicePtr<int>) =
    let tid = threadIdx.x
    let packed = input.[tid]

    // the start flag is in the high bit
    let flag = 0x80000000 &&& packed

    // get the start flags for each thread in the warp
    let mutable flags = __ballot(flag)

    // mask out the bits above the current thread
    flags <- flags &&& bfi(0u, 0xffffffffu, 0u, uint32(tid + 1))

    // find the distance from the current thread to the thread at the start of
    // the segment
    let distance = DeviceFunction.__clz(int(flags)) + tid - 31

    let shared = __shared__<int>(Util.WARP_SIZE).Ptr(0).Volatile()

    let x0 = 0x7fffffff &&& packed
    let mutable x = x0
    shared.[tid] <- x

    // perform the parallel scan. Note the conditional if(offset < distance)
    // replaces the ordinary scan conditional if(offset <= tid)
    for i = 0 to Util.LOG_WARP_SIZE - 1 do
        let offset = 1 <<< i
        if offset <= distance then x <- x + shared.[tid - offset]
        shared.[tid] <- x

    // turn inclusive scan into exclusive scan
    x <- x - x0

    output.[tid] <- x
    debug.[tid] <- distance

[<Test>]
let test() =
    let blockSize = 256
    let numWarps = blockSize / Util.WARP_SIZE

    let pfunct = cuda {
        let s x = x ||| (1 <<< 31)
        let! segscanWarp = <@ segscanWarp @> |> defineKernelFunc

        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let segscanWarp = segscanWarp.Apply m
            pcalc {
                let hInput =
                    [|
                        s 3;   0;   3;   3;   0; s 1;   2;   0;   3;   3;   3;   2;   3;   0;   3;   1;
                          0;   0;   2;   3;   2; s 3;   1;   0;   2;   1;   2;   1;   1;   0;   1; s 3;
                    |]

                let! dInput = DArray.scatterInBlob worker hInput
                let! dOutput = DArray.createInBlob worker hInput.Length
                let! dDebug = DArray.createInBlob worker hInput.Length

                do! PCalc.action (fun hint ->
                    let lp = LaunchParam(1, Util.WARP_SIZE) |> hint.ModifyLaunchParam
                    segscanWarp.Launch lp dInput.Ptr dOutput.Ptr dDebug.Ptr)

                let! hOutput = dOutput.Gather()
                let! hDebug = dDebug.Gather()
                for i =  0 to 15 do printf "%2d; " hOutput.[i]
                printfn ""
                for i = 16 to 31 do printf "%2d; " hOutput.[i]
                printfn ""
                for i =  0 to 15 do printf "%2d; " hDebug.[i]
                printfn ""
                for i = 16 to 31 do printf "%2d; " hDebug.[i]
                printfn "" } ) }
                
    let pfunc, irm = genirm pfunct
    let pfunc, ptxm = genptxm (2, 0) (pfunc, irm)
    //ptxm.Dump()
    let calc = worker.LoadPModule(pfunc, ptxm).Invoke

    calc |> PCalc.run

        


//let rng = System.Random()
//
//let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608; 33554432]
//
//[<Test>]  
//let ``segmented scan reduce test max<int>`` () =
//    let worker = getDefaultWorker()   
//    let test = worker.LoadPModule(Sum.reduceTest <@(fun () -> -10)@> <@(max)@>).Invoke
//
//    let n = plan32.NumThreads
//    let v = Array.init n (fun _ -> rng.Next(-5, 5))
//    let d = test v
//    let expected = Array.max v
//
//    printfn "v = %A" v
//    printfn "d = %A" d
//    printfn "expected = %A" expected
//
//[<Test>]
//let ``segmented scan reduce test sum<int>`` () =
//    let worker = getDefaultWorker()
//    let test = worker.LoadPModule(Sum.reduceTest <@(fun () -> 0)@> <@(+)@>).Invoke
//
//    let n = plan32.NumThreads
//    let v = Array.init n (fun _ -> rng.Next(-5, 5))
//    let d = test v
//    let expected = Array.sum v
//
//    printfn "v = %A" v
//    printfn "d = %A" d
//    printfn "expected = %A" expected
//
//[<Test>]
//let ``segmented scan sum<int>`` () =
//    let worker = getDefaultWorker()
//    let scan = worker.LoadPModule(segScan ()).Invoke
//
//    let n = 200
//    let values = Array.init n (fun _ -> 1)
//    let flags = Array.zeroCreate n
//    flags.[0] <- 1
//    flags.[50] <- 1
//    flags.[100] <- 1
//    flags.[150] <- 1
//
//    let segScan = scan values flags false
//
//    printfn "segScan = %A" segScan
//
//[<Test>]
//let ``segmented scan upsweep`` () =
//    let worker = getDefaultWorker()
//    let scan = worker.LoadPModule(segScan ()).Invoke
//
//    let n = 20*1024
//    let values = Array.init n (fun _ -> 1)
//    let flags = Array.zeroCreate n
//    flags.[0] <- 1
//    flags.[512] <- 1
//    flags.[1024] <- 1
//    flags.[2000] <- 1
//    flags.[3000] <- 1
//    flags.[5000] <- 1
//    flags.[7000] <- 1
//
//    let segScan = scan values flags false
//
//    printfn "segScan = %A" segScan
//
//
//

