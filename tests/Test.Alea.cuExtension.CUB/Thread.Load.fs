module Test.Alea.cuExtension.CUB.Thread.Load

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

open Alea.cuExtension
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Thread

open Test.Alea.cuExtension.CUB.Utilities

let BLOCKS = 1
let BLOCK_THREADS = 128
let ITEMS_PER_THREAD = 4
let N = BLOCKS * BLOCK_THREADS

let minv, maxv = -10, 10


let CacheLoadModifiers = [
    "LOAD_DEFAULT"
    "LOAD_VOLATILE"
    "LOAD_CA"
    "LOAD_CG"
    "LOAD_CS"
    "LOAD_CV"
    "LOAD_LDG"
    ]


let inline testLoad<'T>() = cuda {

    let! kLOAD_DEFAULT =
        <@ fun (d_in:deviceptr<'T>) (d_out:deviceptr<'T>) ->
            let tid = threadIdx.x
            d_out.[tid]  <- ThreadLoad.LOAD_DEFAULT (d_in + tid)
        @> |> Compiler.DefineKernel    

    let! kLOAD_VOLATILE =
        <@ fun (d_in:deviceptr<'T>) (d_out:deviceptr<'T>) ->
            let tid = threadIdx.x
            d_out.[tid]  <- ThreadLoad.LOAD_VOLATILE (d_in + tid)
        @> |> Compiler.DefineKernel

    let! kLOAD_CA =
        <@ fun (d_in:deviceptr<'T>) (d_out:deviceptr<'T>) ->
            let tid = threadIdx.x
            d_out.[tid]  <- ThreadLoad.LOAD_CA (d_in + tid)
        @> |> Compiler.DefineKernel

    let! kLOAD_CG =
        <@ fun (d_in:deviceptr<'T>) (d_out:deviceptr<'T>) ->
            let tid = threadIdx.x
            d_out.[tid]  <- ThreadLoad.LOAD_CG (d_in + tid)
        @> |> Compiler.DefineKernel

    let! kLOAD_CS =
        <@ fun (d_in:deviceptr<'T>) (d_out:deviceptr<'T>) ->
            let tid = threadIdx.x
            d_out.[tid]  <- ThreadLoad.LOAD_CS (d_in + tid)
        @> |> Compiler.DefineKernel

    let! kLOAD_CV =
        <@ fun (d_in:deviceptr<'T>) (d_out:deviceptr<'T>) ->
            let tid = threadIdx.x
            d_out.[tid]  <- ThreadLoad.LOAD_CV (d_in + tid)
        @> |> Compiler.DefineKernel

    let! kLOAD_LDG =
        <@ fun (d_in:deviceptr<'T>) (d_out:deviceptr<'T>) ->
            let tid = threadIdx.x
            d_out.[tid]  <- ThreadLoad.LOAD_LDG (d_in + tid)
        @> |> Compiler.DefineKernel


    return Entry(fun (program:Program) ->
        let worker = program.Worker
        let kernels = [
            kLOAD_DEFAULT
            kLOAD_VOLATILE
            kLOAD_CA
            kLOAD_CG
            kLOAD_CS
            kLOAD_CV
            kLOAD_LDG 
                            ] |> List.map (fun k -> program.Apply k)
        
        fun (input:'T[]) ->
            use din = worker.Malloc(input)
            use dout = worker.Malloc<'T>(input.Length)
            let lp = LaunchParam(BLOCKS, BLOCK_THREADS)
            [ for k in kernels do
                k.Launch lp din.Ptr dout.Ptr
                yield dout.Gather() ] |> Array.ofList
    )}
     

[<Test>]
let ``thread Load - all modifiers, int`` () =
    let hinput = Array.init N (fun i -> i)
    let program = testLoad<int>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput
    
    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

[<Test>]
let ``thread Load - all modifiers, int64`` () =
    let hinput = Array.init N (fun i -> i |> int64)
    let program = testLoad<int64>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput
    
    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

[<Test>]
let ``thread Load - all modifiers, float32`` () =
    let hinput = Array.init N (fun i -> i |> float32)
    let program = testLoad<float32>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput
    
    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

[<Test>]
let ``thread Load - all modifiers, float`` () =
    let hinput = Array.init N (fun i -> i |> float)
    let program = testLoad<float>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput
    
    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))



////////////// V2 ////////////////////////////////////////////////////////////////////
let [<Test>] ``thread Load - all modifiers, int2`` () =
    let hinput = Array.init<int2> N (fun _ -> GenRandomTypeUtil.v2.rint2(minv,maxv))
    let program = testLoad<int2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]        
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, uint2`` () =
    let hinput = Array.init<uint2> N (fun _ -> GenRandomTypeUtil.v2.ruint2(minv,maxv))
    let program = testLoad<uint2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput
    
    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, long2`` () =
    let hinput = Array.init<long2> N (fun _ -> GenRandomTypeUtil.v2.rlong2(minv,maxv))
    let program = testLoad<long2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, ulong2`` () =
    let hinput = Array.init<ulong2> N (fun _ -> GenRandomTypeUtil.v2.rulong2(minv,maxv))
    let program = testLoad<ulong2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, float2`` () =
    let hinput = Array.init<float2> N (fun _ -> GenRandomTypeUtil.v2.rfloat2(minv,maxv))
    let program = testLoad<float2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput
    
    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, longlong2`` () =
    let hinput = Array.init<longlong2> N (fun _ -> GenRandomTypeUtil.v2.rlonglong2(minv,maxv))
    let program = testLoad<longlong2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, ulonglong2`` () =
    let hinput = Array.init<ulonglong2> N (fun _ -> GenRandomTypeUtil.v2.rulonglong2(minv,maxv))
    let program = testLoad<ulonglong2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, double2`` () =
    let hinput = Array.init<double2> N (fun _ -> GenRandomTypeUtil.v2.rdouble2(minv,maxv))
    let program = testLoad<double2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

////////////// V4 ////////////////////////////////////////////////////////////////////
let [<Test>] ``thread Load - all modifiers, short4`` () =
    let hinput = Array.init<short4> N (fun _ -> GenRandomTypeUtil.v4.rshort4(minv,maxv))
    let program = testLoad<short4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, ushort4`` () =
    let hinput = Array.init<ushort4> N (fun _ -> GenRandomTypeUtil.v4.rushort4(minv,maxv))
    let program = testLoad<ushort4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, int4`` () =
    let hinput = Array.init<int4> N (fun _ -> GenRandomTypeUtil.v4.rint4(minv,maxv))
    let program = testLoad<int4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, uint4`` () =
    let hinput = Array.init<uint4> N (fun _ -> GenRandomTypeUtil.v4.ruint4(minv,maxv))
    let program = testLoad<uint4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, float4`` () =
    let hinput = Array.init<float4> N (fun _ -> GenRandomTypeUtil.v4.rfloat4(minv,maxv))
    let program = testLoad<float4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, long4`` () =
    let hinput = Array.init<long4> N (fun _ -> GenRandomTypeUtil.v4.rlong4(minv,maxv))
    let program = testLoad<long4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, ulong4`` () =
    let hinput = Array.init<ulong4> N (fun _ -> GenRandomTypeUtil.v4.rulong4(minv,maxv))
    let program = testLoad<ulong4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, longlong4`` () =
    let hinput = Array.init<longlong4> N (fun _ -> GenRandomTypeUtil.v4.rlonglong4(minv,maxv))
    let program = testLoad<longlong4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, ulonglong4`` () =
    let hinput = Array.init<ulonglong4> N (fun _ -> GenRandomTypeUtil.v4.rulonglong4(minv,maxv))
    let program = testLoad<ulonglong4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread Load - all modifiers, double4`` () =
    let hinput = Array.init<double4> N (fun _ -> GenRandomTypeUtil.v4.rdouble4(minv,maxv))
    let program = testLoad<double4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheLoadModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))