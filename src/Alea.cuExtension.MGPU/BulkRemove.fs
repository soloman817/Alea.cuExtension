module Alea.cuExtension.MGPU.BulkRemove

// this maps to bulkremove.cuh
open System.Diagnostics
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
open Alea.cuExtension.MGPU.CTASearch
open Alea.cuExtension.MGPU.CTAScan
open Alea.cuExtension.MGPU.Search




let kernelBulkRemove (plan:Plan) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    // @COMMENTS@: well, 'TI means TYPE OF INPUT, 'TR means TYPE OF RESULT, 'TV means TYPE OF VALUE
    // but acturally in bulkRemove, I think 'TI should = 'TR, right? So I change it to be 'T. Tell me
    // if I was wrong that there is a conversion from 'TI to 'TR

    // in reduce algorithm, the input data is of type 'TI, and we use extract() to get a 'TV from input
    // data 'TI[], then we do calcualtion on 'TV, and then we use Combine to convert 'TV to 'TR. This
    // is a very good approach, for example, we want to do sum, and use the reduce algorithm, the input
    // is a float[], so 'TI is float, but we would like to use more complex 'TV for internal computing
    // which can also count how many numbers, so 'TV would be a struct, then we combine 'TV to 'TR for
    // the final result. But in bulkremove, this is not needed, it is just remove, so no 'TV, and 'TI = 'TR
    // so we use 'T.

    // for more detail of these types, you can referenece test case Reduce.'mean float (iteratively)'.

    // and I use createSharedExpr to simulate when 'TI <> 'TR, but if you check the code, the :
    //typedef CTAScan<NT, ScanOpAdd> S;
    //union Shared {
    //	int indices[NV];
    //	typename S::Storage scan;
    //};
    //__shared__ Shared shared;
    // and ScanOpAdd is typedef ScanOp<ScanOpTypeAdd, int> ScanOpAdd
    // so it means this scan is just used to calculate index, and to make things easier, we can assume
    // that index type are always int, so here we don't need to use this way to create shared memory
    // once more, why I need use createSharedExpr to create shared memory, is because I want to simulate
    // the union, it has two cases, int for indices, or S::Storage for storage. The reason I must use
    // createSharedExpr (if you check its implementation, you can see that it is most about the alignment)
    // For example, if Storage type is float64, then its size is 8 bytes, but int is 4 bytes, So to 
    // make it work, we should define the shared memory (which will be shared as union) with alignment of 8
    // otherwise it will generate error.

    // but here it is very simple, we only use scan for index calculation, so storage type is int, so
    // no need to use createSharedExpr to do that

    let capacity, scan2 = ctaScan2 NT (scanOp ScanOpTypeAdd 0)
    let sharedSize = max NV capacity
    
    let deviceGlobalToReg = deviceGlobalToReg NT VT
    let deviceSharedToReg = deviceSharedToReg NT VT
    let deviceRegToGlobal = deviceRegToGlobal NT VT
    let deviceGather = deviceGather NT VT
        
    <@ fun (source_global:deviceptr<'T>) (sourceCount:int) (indices_global:deviceptr<int>) (indicesCount:int) (p_global:deviceptr<int>) (dest_global:deviceptr<'T>) ->
        
        let deviceGather = %deviceGather
        
        let deviceGlobalToReg = %deviceGlobalToReg
        let deviceSharedToReg = %deviceSharedToReg
        let deviceRegToGlobal = %deviceRegToGlobal

        let scan = %scan2

        // @COMMENTS@ so now I only need use int directly (but remember, bulkremove is the most simple one,
        // for other algorithms such as reduce, scan, we need use createSharedExpr, please check my reduce
        // example and think again why I need it (hint, for alignment)

        let shared = __shared__.Array<int>(sharedSize) |> __array_to_ptr
        let sharedScan = shared
        let sharedIndices = shared

        let tid = threadIdx.x
        let block = blockIdx.x
        let gid = block * NV
        
        let mutable sourceCount = sourceCount
        sourceCount <- min NV (sourceCount - gid)

        let mutable source_global = source_global

        // search for begin and end iterators of interval to load
        let p0 = p_global.[block]
        let p1 = p_global.[block + 1]

        // Set the flags to 1. The default is to copy a value
        for i = 0 to VT - 1 do
            let index = NT * i + tid
            if index < sourceCount then sharedIndices.[index] <- 1 else sharedIndices.[index] <- 0
        __syncthreads()

        // Load the indices into register
        let begin' = p0
        let indexCount = p1 - begin'
        let indices = __local__.Array<int>(VT) |> __array_to_ptr
        deviceGlobalToReg indexCount (indices_global + begin') tid indices false

        // Set the counter to 0 for each index we've loaded
        for i = 0 to VT - 1 do
            if (NT * i + tid) < indexCount then
                sharedIndices.[indices.[i] - gid] <- 0
        __syncthreads()

        // Run a raking scan over the flags.  We count the set flags - this is the
        // number of elements to load in per thread
        let mutable x = 0
        for i = 0 to VT - 1 do
            indices.[i] <- sharedIndices.[VT * tid + i]
            x <- x + indices.[i]
        __syncthreads()

        // Run a CTA scan and scatter the gather indices to shared memory
        let mutable s = scan tid x sharedScan
        for i = 0 to VT - 1 do
            if indices.[i] = 1 then                
                sharedIndices.[s] <- VT * tid + i
                s <- s + 1                
        __syncthreads()

        // Load the gather indices into register
        deviceSharedToReg NV sharedIndices tid indices true
        
        // Gather the data into register.  The number of values to copy
        // is sourceCount - indexCount
        source_global <- source_global + gid
        let count = sourceCount - indexCount
        let values = __local__.Array<'T>(VT) |> __array_to_ptr
        deviceGather count source_global indices tid values false

        // Store all the valid registers to dest_global
        deviceRegToGlobal count values tid (dest_global + gid - begin') false  @>

//type IBulkRemove<'T> =
//    {
//        Action : ActionHint -> int -> deviceptr<int> -> deviceptr<'T> -> deviceptr<int> -> deviceptr<'T> -> unit        
//        NumPartitions : int
//    }


// @COMMENTS@ : actrually, bulkRemove just need use a comp of int (for index), so you don't
// need to input a indent, right? cause for index, we assumed it is always int. for user
// of bulkremove, input type is 'TI, output type is 'TR, but we don't give it a function
// to convert 'TI to 'TR, so, here we simply use 'T for both
// and when create bsp, we just create comp CompTypeLess 0, and that gives the comp of int
// just used for int cacluation. so again, you don't even need use inline, cause no need.
let bulkRemove(plan:Plan) = cuda {
    //let plan = { NT = 128; VT = 11 }
    let NV = plan.NT * plan.VT
    
    let! kernelBulkRemove = kernelBulkRemove plan |> Compiler.DefineKernel //"br"
    let! bsp = Search.binarySearchPartitions MgpuBoundsLower (comp CompTypeLess 0)

    return Entry(fun program ->
        let worker = program.Worker
        let kernelBulkRemove = program.Apply kernelBulkRemove
        //let bsp = program.Apply bsp

        // @COMMENTS@: you need understand why I need only sourceCount, but not indicesCount here.
        // because we need numBlocks, and to calculate numBlocks we only need sourceCount.
        // and with numBlocks, we can let the wrapper prepare the memory, and then we create
        // bsp inside action (also because bsp don't need some internal memory)
        // so the rule is, once the interface is created, it should contains informations which 
        // could be used for wrapper to prepare the memory (here is the NumPartitions), and some
        // actions which don't do any memory allocation internally. So the wrapper will response
        // for how to create memory and launch the action.
        fun (sourceCount:int) ->    
            let numBlocks = divup sourceCount NV
            let lp = LaunchParam(numBlocks, plan.NT)

            // @COMMENTS@ :IMPORTANT, this malloc breaks a rule that DONOT do memory operation
            // during here, all memory operation should be handled in PCALC, otherwise, you will
            // get many small malloc (think if you call this many times).
            
            //let parts = worker.Malloc(numBlocks + 1)

            // so the correct way is to expose the size for partitions from the interface, and let
            // the wrapper do the memory.
            
            let run (indicesCount:int) (parts:deviceptr<int>) (source_global:deviceptr<'T>) (indices_global:deviceptr<int>) (dest_global:deviceptr<'T>) =
                // @COMMENTS@ : here I use worker.Eval, because otherwise, it will do thread switching, make them
                // all do in one eval, ignore the switching.
                fun () ->
                    
                    let bsp = bsp sourceCount indicesCount NV
                    let partitions = bsp indices_global parts
                    kernelBulkRemove.Launch lp source_global sourceCount indices_global indicesCount parts dest_global
                |> worker.Eval

            { NumPartitions = numBlocks + 1 } ) }
            
