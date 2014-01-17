module Alea.cuExtension.MGPU.Reduce

// this file maps to reduce.cuh

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension
//open Alea.cuExtension.Util
open Alea.cuExtension.MGPU
//open Alea.cuExtension.MGPU.QuotationUtil
open Alea.cuExtension.MGPU.DeviceUtil
open Alea.cuExtension.MGPU.LoadStore
open Alea.cuExtension.MGPU.CTAScan

// in mgpu, it uses a template type Tuning for different tuning parameters,
// which we call it plan. In most of mgpu kernel, a plan need a NT (means the
// numbers of thread in one block), and a VT (means how many values will be
// handles by one thread, this is used in the local values you can see below in
// the kernel)


let kernelReduce (plan:Plan) (op:IScanOp<'TI, 'TV, 'TR>) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    // so here we dynamically created a shared array, which matches the requiement
    // of a union's alignment and size
    //	union Shared {
    //	    typename R::Storage reduce;
    //      input_type inputs[NV];
    //  };
    // R::Storage is TV[Capacity]
    // Inputs is TI[NV]
    let capacity, reduce = ctaReduce NT op
    let alignOfTI, sizeOfTI = TypeUtil.cudaAlignOf typeof<'TI>, sizeof<'TI>
    let alignOfTV, sizeOfTV = TypeUtil.cudaAlignOf typeof<'TV>, sizeof<'TV>
    let sharedAlign = max alignOfTI alignOfTV
    let sharedSize = max (sizeOfTI * NV) (sizeOfTV * capacity)
    //let createSharedExpr = createSharedExpr sharedAlign sharedSize

    // get neccesary quotations for later doing slicing in quotation
    let commutative = op.Commutative
    let identity = op.Identity
    let extract = op.DExtract
    let plus = op.DPlus
    let deviceGlobalToReg = deviceGlobalToReg NT VT

    <@ fun (data_global:deviceptr<'TI>) (count:int) (task:int2) (reduction_global:deviceptr<'TV>) ->
        // quotation slicing (you can google what it means)
        let extract = %extract
        let plus = %plus
        let deviceGlobalToReg = %deviceGlobalToReg
        let reduce = %reduce

        //let shared = %(createSharedExpr)
        let sharedReduce = __shared__.Array<'TV>(sharedSize)
        let sharedInputs = __shared__.Array<'TI>(sharedSize)

        let tid = threadIdx.x
        let block = blockIdx.x
        let first = VT * tid

        let mutable range = computeTaskRangeEx block task NV count

        let mutable total = extract identity -1
        let mutable totalDefined = false

        while range.x < range.y do
            let count2 = min NV (count - range.x)

            // here is what VT means, one thread can handle multiple values
            // which could make the device busy enough
            // but you need be very careful about the storage order,
            // so that is why deviceGlobalToReg is used. 
            // Please check http://www.moderngpu.com/scan/globalscan.html#Scan
            // and check the concept of transposeValues. 
            let inputs = __local__.Array<'TI>(VT) |> __array_to_ptr
            deviceGlobalToReg count2 (data_global + range.x) tid inputs false

            if commutative <> 0 then
                for i = 0 to VT - 1 do
                    let index = NT * i + tid
                    if index < count2 then
                        let x = extract inputs.[i] (range.x + index)
                        total <- if i > 0 || totalDefined then plus total x else x
            else 
                // TODO
                ()

            range.x <- range.x + NV
            totalDefined <- true

        if commutative <> 0 then
            total <- reduce tid total (sharedReduce |> __array_to_ptr)

        if tid = 0 then reduction_global.[block] <- total @>

// each raw implmenetation could return an interface (or api), which doesn't contain
// any memory management, this will be used later for wrapping with pcalc
//type IReduce<'TI, 'TV> =
//    {
//        NumBlocks : int // how may 'TV
//        Action : ActionHint -> deviceptr<'TI> ->deviceptr<'TV> -> unit
//        Result : 'TV[] -> 'TV
//    }

let reduce (op:IScanOp<'TI, 'TV, 'TR>) = cuda {
    let cutOff = 20000
    let plan1 = { NT = 512; VT = 5 }
    let plan2 = { NT = 128; VT = 9 }
    let! kernelReduce1 = kernelReduce plan1 op |> Compiler.DefineKernel
    let! kernelReduce2 = kernelReduce plan2 op |> Compiler.DefineKernel
    let hplus = op.HPlus

    return Entry(fun program ->
        let worker = program.Worker
        let kernelReduce1 = program.Apply kernelReduce1
        let kernelReduce2 = program.Apply kernelReduce2

        fun (count:int) ->
            let numBlocks, task, lp, kernelReduce =
                if count < cutOff then
                    let plan = plan1
                    let kernelReduce = kernelReduce1
                    let NV = plan.NT * plan.VT
                    let numTiles = divup count NV
                    let numBlocks = 1
                    let task = int2(numTiles, 1)
                    let lp = LaunchParam(1, plan.NT)
                    numBlocks, task, lp, kernelReduce
                else
                    let plan = plan2
                    let kernelReduce = kernelReduce2
                    let NV = plan.NT * plan.VT
                    let numTiles = divup count NV
                    let numBlocks = min (worker.Device.Attributes.MULTIPROCESSOR_COUNT * 25) numTiles
                    let task = divideTaskRange numTiles numBlocks
                    let lp = LaunchParam(numBlocks, plan.NT)
                    numBlocks, task, lp, kernelReduce

            let run (data:deviceptr<'TI>) (reduction:deviceptr<'TV>) =
                fun () ->
                    
                    kernelReduce.Launch lp data count task reduction
                |> worker.Eval

            let result (reduction:'TV[]) =
                reduction |> Array.reduce hplus

            { NumBlocks = numBlocks; Result = result } ) }


        

