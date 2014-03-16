module Test.Alea.cuExtension.CUB.Thread.Store

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


let CacheStoreModifiers = [
    "STORE_DEFAULT"
    "STORE_VOLATILE"
    "STORE_CG"
    "STORE_CS"
    "STORE_WB"
    "STORE_WT"]


let inline testStore<'T>() = cuda {
    
    let! kSTORE_DEFAULT =
        <@ fun (d_in:deviceptr<'T>) (d_out:deviceptr<'T>) ->
            let tid = threadIdx.x
            ThreadStore.STORE_DEFAULT (d_out + tid) d_in.[tid]
        @> |> Compiler.DefineKernel

    let! kSTORE_VOLATILE =
        <@ fun (d_in:deviceptr<'T>) (d_out:deviceptr<'T>) ->
            let tid = threadIdx.x
            ThreadStore.STORE_VOLATILE (d_out + tid) d_in.[tid]
        @> |> Compiler.DefineKernel            

    let! kSTORE_CG =
        <@ fun (d_in:deviceptr<'T>) (d_out:deviceptr<'T>) ->
            let tid = threadIdx.x
            ThreadStore.STORE_CG (d_out + tid) d_in.[tid]
        @> |> Compiler.DefineKernel

    let! kSTORE_CS =
        <@ fun (d_in:deviceptr<'T>) (d_out:deviceptr<'T>) ->
            let tid = threadIdx.x
            ThreadStore.STORE_CS (d_out + tid) d_in.[tid]
        @> |> Compiler.DefineKernel

    let! kSTORE_WB =
        <@ fun (d_in:deviceptr<'T>) (d_out:deviceptr<'T>) ->
            let tid = threadIdx.x
            ThreadStore.STORE_WB (d_out + tid) d_in.[tid]
        @> |> Compiler.DefineKernel

    let! kSTORE_WT =
        <@ fun (d_in:deviceptr<'T>) (d_out:deviceptr<'T>) ->
            let tid = threadIdx.x
            ThreadStore.STORE_WT (d_out + tid) d_in.[tid]
        @> |> Compiler.DefineKernel


    return Entry(fun (program:Program) ->
        let worker = program.Worker
        let kernels = [
            kSTORE_DEFAULT
            kSTORE_VOLATILE
            kSTORE_CG
            kSTORE_CS
            kSTORE_WB
            kSTORE_WT 
                        ] |> List.map (fun k -> program.Apply k)
        
        fun (input:'T[]) ->
            use din = worker.Malloc(input)
            use dout = worker.Malloc<'T>(input.Length)
            let lp = LaunchParam(BLOCKS, BLOCK_THREADS)
            [ for k = 0 to kernels.Length - 1 do
                kernels.[k].Launch lp din.Ptr dout.Ptr
                yield dout.Gather() ]
    )}



[<Test>]
let ``thread store - all modifiers, int`` () =
    let hinput = Array.init N (fun i -> i)
    let program = testStore<int>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput
    
    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

[<Test>]
let ``thread store - all modifiers, int64`` () =
    let hinput = Array.init N (fun i -> i |> int64)
    let program = testStore<int64>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput
    
    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

[<Test>]
let ``thread store - all modifiers, float32`` () =
    let hinput = Array.init N (fun i -> i |> float32)
    let program = testStore<float32>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput
    
    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

[<Test>]
let ``thread store - all modifiers, float`` () =
    let hinput = Array.init N (fun i -> i |> float)
    let program = testStore<float>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput
    
    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))



////////////// V2 ////////////////////////////////////////////////////////////////////
let [<Test>] ``thread store - all modifiers, int2`` () =
    let hinput = Array.init<int2> N (fun _ -> GenRandomTypeUtil.v2.rint2(minv,maxv))
    let program = testStore<int2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]        
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, uint2`` () =
    let hinput = Array.init<uint2> N (fun _ -> GenRandomTypeUtil.v2.ruint2(minv,maxv))
    let program = testStore<uint2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput
    
    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, long2`` () =
    let hinput = Array.init<long2> N (fun _ -> GenRandomTypeUtil.v2.rlong2(minv,maxv))
    let program = testStore<long2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, ulong2`` () =
    let hinput = Array.init<ulong2> N (fun _ -> GenRandomTypeUtil.v2.rulong2(minv,maxv))
    let program = testStore<ulong2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, float2`` () =
    let hinput = Array.init<float2> N (fun _ -> GenRandomTypeUtil.v2.rfloat2(minv,maxv))
    let program = testStore<float2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput
    
    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, longlong2`` () =
    let hinput = Array.init<longlong2> N (fun _ -> GenRandomTypeUtil.v2.rlonglong2(minv,maxv))
    let program = testStore<longlong2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, ulonglong2`` () =
    let hinput = Array.init<ulonglong2> N (fun _ -> GenRandomTypeUtil.v2.rulonglong2(minv,maxv))
    let program = testStore<ulonglong2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, double2`` () =
    let hinput = Array.init<double2> N (fun _ -> GenRandomTypeUtil.v2.rdouble2(minv,maxv))
    let program = testStore<double2>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

////////////// V4 ////////////////////////////////////////////////////////////////////
let [<Test>] ``thread store - all modifiers, short4`` () =
    let hinput = Array.init<short4> N (fun _ -> GenRandomTypeUtil.v4.rshort4(minv,maxv))
    let program = testStore<short4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, ushort4`` () =
    let hinput = Array.init<ushort4> N (fun _ -> GenRandomTypeUtil.v4.rushort4(minv,maxv))
    let program = testStore<ushort4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, int4`` () =
    let hinput = Array.init<int4> N (fun _ -> GenRandomTypeUtil.v4.rint4(minv,maxv))
    let program = testStore<int4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, uint4`` () =
    let hinput = Array.init<uint4> N (fun _ -> GenRandomTypeUtil.v4.ruint4(minv,maxv))
    let program = testStore<uint4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, float4`` () =
    let hinput = Array.init<float4> N (fun _ -> GenRandomTypeUtil.v4.rfloat4(minv,maxv))
    let program = testStore<float4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, long4`` () =
    let hinput = Array.init<long4> N (fun _ -> GenRandomTypeUtil.v4.rlong4(minv,maxv))
    let program = testStore<long4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, ulong4`` () =
    let hinput = Array.init<ulong4> N (fun _ -> GenRandomTypeUtil.v4.rulong4(minv,maxv))
    let program = testStore<ulong4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, longlong4`` () =
    let hinput = Array.init<longlong4> N (fun _ -> GenRandomTypeUtil.v4.rlonglong4(minv,maxv))
    let program = testStore<longlong4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, ulonglong4`` () =
    let hinput = Array.init<ulonglong4> N (fun _ -> GenRandomTypeUtil.v4.rulonglong4(minv,maxv))
    let program = testStore<ulonglong4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))

let [<Test>] ``thread store - all modifiers, double4`` () =
    let hinput = Array.init<double4> N (fun _ -> GenRandomTypeUtil.v4.rdouble4(minv,maxv))
    let program = testStore<double4>() |> Compiler.load Worker.Default
    let doutput = program.Run hinput

    for i = 0 to doutput.Length - 1 do 
        printfn "************ %s **************" (CacheStoreModifiers.[i])
        printfn "Host:\n%A\nDevice:\n:%A\n" hinput doutput.[i]
        (hinput, doutput.[i]) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))