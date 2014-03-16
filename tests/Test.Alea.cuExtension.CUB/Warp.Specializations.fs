module Test.Alea.cuExtension.CUB.Warp.Specializations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Common

open NUnit.Framework

open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Warp.WarpSpecializations

let BLOCKS = 1    
let BLOCK_THREADS = 128
let ITEMS_PER_THREAD = 4
let N = BLOCK_THREADS * BLOCKS

module ReduceShfl =
       let f()       = "reduce shfl"

module ReduceSmem =
       let f()       = "reduce smem"

module ScanShfl   =
       let f()       = "scan shfl"




//////////////////////////////////////////////////////////////// WARP SCAN SMEM
[<Test>]
let ``warp scan smem - initialization`` () =
    let template = cuda{
        let h = WarpScanSmem.HostApi.Init(1)

        let! kernel = 
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
                let tid = threadIdx.x
                let warp_id = threadIdx.x / h.Params.LOGICAL_WARP_THREADS
                let lane_id = __ptx__.LaneId()
                let temp_storage = WarpScanSmem.TempStorage<int>.Uninitialized(h)
                let warpscan = WarpScanSmem.IntApi.Init(temp_storage, warp_id |> uint32, lane_id)
                ()
            @> |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            fun (input:int[]) ->
                use d_in = worker.Malloc(input)
                use d_out = worker.Malloc<int>(input.Length)
                let lp = LaunchParam(BLOCKS, BLOCK_THREADS)
                kernel.Launch lp d_in.Ptr d_out.Ptr
                d_out.Gather()
        )}

    let program = template |> Compiler.load Worker.Default
    let hinput = Array.init N (fun _ -> 1)
    let doutput = program.Run hinput
    printfn "%A" doutput
    

[<Test>]
let ``warp scan smem - int`` () =
    let template = cuda{
        let h = WarpScanSmem.HostApi.Init(4)

        let! kernel = 
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
                let tid = threadIdx.x
                let warp_id = threadIdx.x / h.Params.LOGICAL_WARP_THREADS
                let lane_id = __ptx__.LaneId()
                let temp_storage = WarpScanSmem.TempStorage<int>.Uninitialized(h)
                let thread_data = __local__.Variable<int>(input.[tid])
                //if threadIdx.x < 32 then
                //WarpScanSmem.InclusiveSum.DefaultInt h temp_storage warp_id lane_id !thread_data thread_data
                WarpScanSmem.IntApi.Init(temp_storage, warp_id |> uint32, lane_id).ExclusiveScan(h, input.[tid], (output.Ref(tid)))


                //output.[tid] <- temp_storage.[tid]
//                let wsSmem_d = WarpScanSmem.DeviceApi.Init()
//                let scan = WarpScanSmem.API<int>.Create(wsSmem_h)
            @> |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            fun (input:int[]) ->
                use d_in = worker.Malloc(input)
                use d_out = worker.Malloc<int>(input.Length)

                let lp = LaunchParam(BLOCKS, BLOCK_THREADS)

                kernel.Launch lp d_in.Ptr d_out.Ptr

                d_out.Gather()
        )}

    let program = template |> Compiler.load Worker.Default
    let hinput = Array.init N (fun _ -> 1)
    let doutput = program.Run hinput
    let houtput() = 
        [for i = 0 to ITEMS_PER_THREAD - 1 do 
            let b = 1 + i * 32
            
            let h = hinput |> Array.sub <|| (b,31) |> Array.scan (+) 0
            printfn "%A" h
            for item in h do yield item]
    
    let hout = houtput() |> Array.ofList

    printfn "Host:\n%A\nDevice:\n%A\n" hout doutput
    (hout, doutput) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))