module Test.Alea.cuBase.MGPU.Debug

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.cuBase.MGPU.QuotationUtil
open Alea.cuBase
open Alea.cuBase.Util
open Alea.cuBase.MGPU
open Alea.cuBase.MGPU.Intrinsics
open Alea.cuBase.MGPU.QuotationUtil
open Alea.cuBase.MGPU.DeviceUtil
open Alea.cuBase.MGPU.LoadStore
open Alea.cuBase.MGPU.CTASearch

open NUnit.Framework


let worker = getDefaultWorker()

type Plan =
    {
        NT : int
        VT : int
    }

let serialMerge (VT:int) (rangeCheck:bool) (comp:IComp<'TV>) =
    let comp = comp.Device
    <@ fun  (keys_shared    :RWPtr<'TV>) 
            (aBegin         :int) 
            (aEnd           :int) 
            (bBegin         :int) 
            (bEnd           :int) 
            (results        :RWPtr<'TV>) 
            (indices        :RWPtr<int>) 
            ->
        
        let comp = %comp
        let mutable aKey = keys_shared.[aBegin]
        let mutable bKey = keys_shared.[bBegin]
        let mutable aBegin = aBegin
        let mutable bBegin = bBegin

        for i = 0 to VT - 1 do
            let mutable p = false
            if rangeCheck then
                p <- (bBegin >= bEnd) || ((aBegin < aEnd) && not(comp bKey aKey))
            else
                p <- not (comp bKey aKey)
            results.[i] <- if p then aKey else bKey
            indices.[i] <- if p then aBegin else bBegin
            if p then
                aBegin <- aBegin + 1
                aKey <- keys_shared.[aBegin]                
            else
                bBegin <- bBegin + 1
                bKey <- keys_shared.[bBegin]                
        __syncthreads() @>

let ctaBlocksortPass (NT:int) (VT:int) (compOp:IComp<'TV>) =
    let mergePath = (mergePath MgpuBoundsLower compOp).DMergePath
    let serialMerge = serialMerge VT true compOp
    let comp = compOp.Device
    <@ fun  (keys_shared    :RWPtr<'TV>) 
            (tid            :int) 
            (count          :int) 
            (coop           :int) 
            (keys           :RWPtr<'TV>) 
            (indices        :RWPtr<int>) 
            ->

        let comp = %comp
        let mergePath = %mergePath
        let serialMerge = %serialMerge

        let list = ~~~(coop - 1) &&& tid
        let diag = min count (VT * ((coop - 1) &&& tid))
        let start = VT * list
        let a0 = min count start
        let b0 = min count (start + VT * (coop / 2))
        let b1 = min count (start + VT * coop)
                
        let p = mergePath (keys_shared + a0) (b0 - a0) (keys_shared + b0) (b1 - b0) diag
        serialMerge keys_shared (a0 + p) b0 (b0 + diag - p) b1 keys indices 
    @>


let ctaBlocksortLoop (NT:int) (VT:int) (hasValues:bool) (compOp:IComp<'TV>) =
    let ctaBlocksortPass = ctaBlocksortPass NT VT compOp
    let deviceThreadToShared = deviceThreadToShared VT
    let deviceGather = deviceGather NT VT

    <@ fun  (threadValues   :RWPtr<'TV>) 
            (keys_shared    :RWPtr<'TV>) 
            (values_shared  :RWPtr<'TV>) 
            (tid            :int) 
            (count          :int) 
            ->
        
        let ctaBlocksortPass = %ctaBlocksortPass
        let deviceThreadToShared = %deviceThreadToShared
        let deviceGather = %deviceGather

        let mutable coop = 2
        while coop <= NT do
            let indices = __local__<int>(VT).Ptr(0)
            let keys = __local__<'TV>(VT).Ptr(0)
            ctaBlocksortPass keys_shared tid count coop keys indices            
            deviceThreadToShared keys tid keys_shared true
            coop <- coop * 2
    @>


let ctaMergesort (NT:int) (VT:int) (hasValues:bool) (compOp:IComp<'TV>) =    
    let deviceThreadToShared = deviceThreadToShared VT
    let ctaBlocksortLoop = ctaBlocksortLoop NT VT hasValues compOp
    //let oddEvenTransposeSort = oddEvenTransposeSort VT compOp
    let comp = compOp.Device
    <@ fun (threadKeys      :RWPtr<'TV>) 
           (threadValues    :RWPtr<'TV>) 
           (keys_shared     :RWPtr<'TV>) 
           (values_shared   :RWPtr<'TV>) 
           (count           :int) 
           (tid             :int) 
           ->

        let comp = %comp
        let deviceThreadToShared = %deviceThreadToShared
        let ctaBlocksortLoop = %ctaBlocksortLoop
        //let oddEvenTransposeSort = %oddEvenTransposeSort
        if VT * tid < count then
            //oddEvenTransposeSort threadKeys threadValues
            let mutable level = 0
            while level < VT do
                let mutable i = 1 &&& level
                while i < VT - 1 do                    
                    if ( comp threadKeys.[i + 1] threadKeys.[i] ) then
                        swap threadKeys.[i] threadKeys.[i + 1]
                        swap threadValues.[i] threadValues.[i + 1]
                    i <- i + 2
                level <- level + 1            
        deviceThreadToShared threadKeys tid keys_shared true
        ctaBlocksortLoop threadValues keys_shared values_shared tid count 
        @>

let kernelBlocksort (plan:Plan) (hasValues:int) (compOp:IComp<'TV>) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    let hasValues = if hasValues = 1 then true else false

    let sharedSize = max NV (NT * (VT + 1))
    let comp = compOp.Device
    
    let deviceGlobalToShared = deviceGlobalToShared NT VT
    let deviceSharedToThread = deviceSharedToThread VT
    let deviceSharedToGlobal = deviceSharedToGlobal NT VT
    let deviceThreadToShared = deviceThreadToShared VT
    let ctaMergesort = ctaMergesort NT VT hasValues compOp
                                                    
    <@ fun  (keysSource_global:DevicePtr<'TV>) 
            (valsSource_global:DevicePtr<'TV>) 
            (count:int) 
            (keysDest_global:DevicePtr<'TV>) 
            (valsDest_global:DevicePtr<'TV>) 
            ->

        let comp = %comp
        let deviceGlobalToShared = %deviceGlobalToShared
        let deviceSharedToThread = %deviceSharedToThread
        let deviceSharedToGlobal = %deviceSharedToGlobal
        let deviceThreadToShared = %deviceThreadToShared
        let ctaMergesort = %ctaMergesort

        let shared = __shared__<'TV>(sharedSize).Ptr(0)
        let sharedKeys = shared
        let sharedValues = shared

        let tid = threadIdx.x
        let block = blockIdx.x
        let gid = NV * block
        let count2 = min NV (count - gid)            

        let threadValues = __local__<'TV>(VT).Ptr(0) 
                
        let threadKeys = __local__<'TV>(VT).Ptr(0)
        deviceGlobalToShared count2 (keysSource_global + gid) tid sharedKeys true
        deviceSharedToThread sharedKeys tid threadKeys true

        let first = VT * tid
        let mutable maxKey = threadKeys.[0]
        if ((first + VT) > count2) && (first < count2) then            
            for i = 1 to VT - 1 do
                if (first + i) < count2 then
                    maxKey <- if comp maxKey threadKeys.[i] then threadKeys.[i] else maxKey
            for i = 0 to VT - 1 do
                if (first + i) >= count2 then threadKeys.[i] <- maxKey

        ctaMergesort threadKeys threadValues sharedKeys sharedValues count2 tid
        deviceSharedToGlobal count2 sharedKeys tid (keysDest_global + gid) true

        @>

type IBlocksort<'TV> =
    {
        Action : ActionHint -> DevicePtr<'TV> -> unit
    }


let mergesortKeys (compOp:IComp<'TV>) = cuda {
    let plan = { NT = 257; VT = 7 }
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT
    
    let! kernelBlocksort = kernelBlocksort plan 0 compOp |> defineKernelFunc

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelBlocksort = kernelBlocksort.Apply m
        
        fun (count:int) ->
            let numBlocks = divup count NV
            let numPasses = findLog2 numBlocks true
            let lp = LaunchParam(numBlocks, NT)

            let action (hint:ActionHint) (source:DevicePtr<'TV>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
                    kernelBlocksort.Launch lp source (DevicePtr<'TV>(0n)) count source (DevicePtr<'TV>(0n))
                |> worker.Eval
            { Action = action } ) }

let pMergesortKeys (compOp:IComp<'TV>) = cuda {
    let! api = mergesortKeys compOp

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m
        fun (count:int) ->
            let api = api count
            pcalc {                
                let merger (source:DArray<'TV>) = 
                    pcalc { do! PCalc.action (fun hint -> api.Action hint source.Ptr) }
                return merger } ) }


[<Test>]
let `` simple MergeSort Keys test`` () =
    let compOp = (comp CompTypeLess 0)
    let pfunct = worker.LoadPModule(pMergesortKeys compOp).Invoke
// Input:
    let hSource = [|   81;   13;   90;   83;   12;   96;   91;   22;   63;   30;
                        9;   54;   27;   18;   54;   99;   95;   99;   96;   96;
                       15;   72;   97;   98;   95;   10;   48;   79;   80;   29;
                       14;    0;   42;   11;   91;   63;   79;   87;   95;   50;
                       65;   79;    3;   36;   84;   21;   93;   68;   67;   39;
                       75;   74;   74;   47;   39;   42;   65;   17;   17;   30;
                       70;   79;    3;   31;   27;   87;    4;   14;    9;   99;
                       82;   82;   69;   12;   31;   76;   95;   49;    3;   66;
                       43;   12;   38;   21;   76;    5;   79;    3;   18;   40;
                       48;   45;   44;   48;   64;   79;   70;   92;   75;   80 |]

//Sorted output:
    let answer = [| 0;    3;    3;    3;    3;    4;    5;    9;    9;   10;
                   11;   12;   12;   12;   13;   14;   14;   15;   17;   17;
                   18;   18;   21;   21;   22;   27;   27;   29;   30;   30;
                   31;   31;   36;   38;   39;   39;   40;   42;   42;   43;
                   44;   45;   47;   48;   48;   48;   49;   50;   54;   54;
                   63;   63;   64;   65;   65;   66;   67;   68;   69;   70;
                   70;   72;   74;   74;   75;   75;   76;   76;   79;   79;
                   79;   79;   79;   79;   80;   80;   81;   82;   82;   83;
                   84;   87;   87;   90;   91;   91;   92;   93;   95;   95;
                   95;   95;   96;   96;   96;   97;   98;   99;   99;   99 |]
    
    let count = hSource.Length

    let dResult = pcalc {
        let! dSource = DArray.scatterInBlob worker hSource
        let! merge = pfunct count
        do! merge dSource 
        let! results = dSource.Gather()
        return results } |> PCalc.run

    let source = dResult
    printfn "source %A" source

