module Test.Alea.CUDA.Extension.MGPU.CTAMerge

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.CTAMerge
open Alea.CUDA.Extension.MGPU.LoadStore

open NUnit.Framework

let worker = getDefaultWorker()

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


//type IOETS =
//    abstract I : int
//    abstract VT : int
//    abstract Sort : Expr<RWPtr<int> -> RWPtr<int> -> unit>
//
//type IOETS2 =
//    abstract I : int
//    abstract VT : int
//    abstract Sort : Expr<RWPtr<int> -> RWPtr<int> -> IOETS>
//
////[<Struct;Align(8)>]
////type OddEvenTransposeSort =
////    val mutable I : int
////    val mutable VT : int
////    val mutable Sort : Expr<RWPtr<int> -> RWPtr<int> -> unit>
////    [<ReflectedDefinition>]
////    new (i,vt,s) = { I = i; VT = vt; Sort = s }
//
//let rOETS (i:int) (vt:int) =
//    let comp = (comp CompTypeLess 0).Device
//    {new IOETS2 with
//        member oo.I = i
//        member oo.VT = vt
//        member oo.Sort =
//            <@ fun (keys:RWPtr<int>) (vals:RWPtr<int>) ->
//                let comp = %comp
//                let mutable i = 1 &&& oo.I
//                while i < oo.VT - 1 do                    
//                    if ( comp keys.[i + 1] keys.[i] ) then
//                        swap keys.[i] keys.[i + 1]
//                        swap vals.[i] vals.[i + 1]
//                    i <- i + 2 
//                {new IOETS with
//                    member o.I = oo.I + 1
//                    member o.VT = oo.VT
//                    member o.Sort = oo.Sort } 
//             @> }
//
//let sort (I:int) (VT:int) =
//    let comp = (comp CompTypeLess 0).Device
//    <@ fun (keys:RWPtr<int>) (vals:RWPtr<int>) ->
//        let comp = %comp
//        let mutable i = 1 &&& I
//        while i < VT - 1 do                    
//            if ( comp keys.[i + 1] keys.[i] ) then
//                swap keys.[i] keys.[i + 1]
//                swap vals.[i] vals.[i + 1]
//            i <- i + 2        
//    @>
//
//let oets (i:int) (vt:int) =
//    let srt =
//        let sort = sort i vt
//        <@ fun (keys:RWPtr<int>) (vals:RWPtr<int>) ->
//            OddEvenTransposeSort((i + 1), vt, sort)            
//        @>
//    OddEvenTransposeSort(i, vt, (%(srt) (DevicePtr(0n)) (DevicePtr(0n))))
//let sortA (I:int) (VT:int) (sortFun:int -> int -> Expr<int -> int -> RWPtr<int> -> RWPtr<int> -> unit>) =
//    let comp = (comp CompTypeLess 0).Device
//    let sortRef = ref (sortFun I VT)
//    <@ fun (I:int) (VT:int) (keys:RWPtr<int>) (vals:RWPtr<int>) ->
//        let comp = %comp
//        let sortRef2 = ref(!sortRef)
//        let sort = %(!sortRef)
//        let mutable i = 1 &&& I
//        while i < VT - 1 do                    
//            if ( comp keys.[i + 1] keys.[i] ) then
//                swap keys.[i] keys.[i + 1]
//                swap vals.[i] vals.[i + 1]
//            i <- i + 2 
//        if I < VT then sort (I + 1) VT keys vals
//        //sortRef := sort (I + 1) VT keys vals 
//        @>
//
//let sortB (VT:int) =
//    let comp = (comp CompTypeLess 0).Device
//    let sortRef = ref (sortA 0 VT (<@ fun (i:int) (vt:int) (keys:RWPtr<int>) (vals:RWPtr<int>) -> () @>))
//    let sort = 
//        sortA 0 VT
//            <@ fun (I:int) (VT:int) (keys:RWPtr<int>) (vals:RWPtr<int>) ->
//                let sort = %(!sortRef)
//                if I < VT then sort (I + 1) VT keys vals
//                @>
//    sortRef := sort
//    <@ fun (keys:RWPtr<int>) (vals:RWPtr<int>) ->
//        let sort = %sort
//        sort I VT keys vals @>

//let sortBox (I:int) (VT:int) = 
//    //let mutable I = 0
//    //let i = ref 0
//    //let sort = sort I VT
//    <@ fun (keys:RWPtr<int>) (vals:RWPtr<int>) ->
//        let sort = %sort
//        let sort = sort I VT keys vals (ref (sort I VT keys vals))
//        sort
//        @>
        
    
let ctaMergesort (NT:int) (VT:int) (hasValues:bool) (compOp:IComp<int>) =
    let deviceThreadToShared = deviceThreadToShared VT
    let ctaBlocksortLoop = ctaBlocksortLoop NT VT hasValues compOp
    //let sort = sortB NT VT
    let comp = compOp.Device
    <@ fun  (threadKeys      :RWPtr<int>) 
            (threadValues    :RWPtr<int>) 
            (keys_shared     :RWPtr<int>) 
            (values_shared   :RWPtr<int>) 
            (count           :int) 
            (tid             :int) 
            (wasHere         :DevicePtr<int>)
            ->

        let comp = %comp
        let deviceThreadToShared = %deviceThreadToShared
        let ctaBlocksortLoop = %ctaBlocksortLoop
        //let sort = %sort
        
//        if VT * tid < count then            
//            sort threadKeys threadValues          
            
        deviceThreadToShared threadKeys tid keys_shared true
        ctaBlocksortLoop threadValues keys_shared values_shared tid count 
        @>


[<Test>]
let ``CTAMergesort test`` () =
    let NT = 256
    let VT = 7
    let NV = NT * VT

    let pfunct = cuda {        
        let ctaMergesort = ctaMergesort NT VT false (comp CompTypeLess 0)
        let comp = (comp CompTypeLess 0).Device
        let deviceGlobalToShared = deviceGlobalToShared NT VT
        let deviceSharedToThread = deviceSharedToThread VT
        let deviceSharedToGlobal = deviceSharedToGlobal NT VT
        let! kernel = 
            <@ fun  (keysSource_global  :DevicePtr<int>) 
                    (valsSource_global  :DevicePtr<int>) 
                    (count              :int) 
                    (keysDest_global    :DevicePtr<int>) 
                    (valsDest_global    :DevicePtr<int>) 
                    (output             :DevicePtr<int>)
                    ->
                let comp = %comp
                let ctaMergesort = %ctaMergesort
                let deviceGlobalToShared = %deviceGlobalToShared
                let deviceSharedToThread = %deviceSharedToThread
                let deviceSharedToGlobal = %deviceSharedToGlobal

                let shared = __shared__<int>(NT * (VT + 1)).Ptr(0)
                let sharedKeys = shared
                let sharedValues = shared

                let tid = threadIdx.x
                let block = blockIdx.x
                let gid = NV * block
                let count2 = min NV (count - gid)

                let threadValues = __local__<int>(VT).Ptr(0) 
                
        
                let threadKeys = __local__<int>(VT).Ptr(0)
                deviceGlobalToShared count2 (keysSource_global + gid) tid sharedKeys true
                deviceSharedToThread sharedKeys tid threadKeys true

                let first = VT * tid
                if ((first + VT) > count2) && (first < count2) then
                    let mutable maxKey = threadKeys.[0]    
                    for i = 1 to VT - 1 do
                        if (first + i) < count2 then
                            maxKey <- if comp maxKey threadKeys.[i] then threadKeys.[i] else maxKey
                    for i = 0 to VT - 1 do
                        if (first + i) >= count2 then threadKeys.[i] <- maxKey

                ctaMergesort threadKeys threadValues sharedKeys sharedValues count2 tid output
                deviceSharedToGlobal count2 sharedKeys tid (keysDest_global + gid) true
                   
            @> |> defineKernelFunc

        return PFunc(fun (m:Module) ->
            use source = worker.Malloc hSource
            let count = hSource.Length
            let numBlocks = divup count NV
            let lp = LaunchParam(numBlocks, NT)
            use output = worker.Malloc NT
            kernel.Launch m lp source.Ptr (DevicePtr(0n)) count source.Ptr (DevicePtr(0n)) output.Ptr
            source.ToHost() ) }

    let pfuncm = worker.LoadPModule(pfunct)
    let output = pfuncm.Invoke
    printfn "output:\n%A" output
            



//[<Test>]
//let ``compute merge range`` () =
//    let pfunct (plan:Plan) = cuda {
//        let NT = plan.NT
//        let VT = plan.VT
//        let NV = NT * VT
//
//        let capacity, scan2 = ctaScan2 (scanOp ScanOpTypeAdd 0)
//        let sharedSize = max NV capacity
//
//        let deviceGlobalToReg = deviceGlobalToReg NT VT
//        let computeMergeRange = computeMergeRange.Device
//        
//        let! kernel = 
//            <@ fun (aCount:int) (bCount:int) (block:int) (coop:int) (nv:int) (mp_global:DevicePtr<int>) (indices_global:DevicePtr<int>) ->
//                let deviceGlobalToReg = %deviceGlobalToReg
//                let computeMergeRange = %computeMergeRange
//                let deviceTransferMergeValues = %deviceTransferMergeValues
//                let S = %scan2
//        
//
//                let shared = __shared__<int>(sharedSize).Ptr(0)
//                let sharedScan = shared
//                let sharedIndices = shared
//
//                let tid = threadIdx.x
//                let block = blockIdx.x
//
//                let range = cmr aCount bCount block 0 nv mp_global
//                let a0 = range.x
//                let a1 = range.y
//                let b0 = range.z
//                let b1 = range.w
//
//                let mutable aCount = aCount
//                let mutable bCount = bCount
//
//                aCount <- a1 - a0
//                bCount <- b1 - b0
//
//                for i = 0 to VT - 1 do
//                    sharedIndices.[NT * i + tid] <- 0
//                __synchthreads()
//
//                deviceGlobalToReg aCount (indices_global + a0) tid indices true 
//                @> |> defineKernelFunc
//        
//
//        return PFunc(fun (m:Module) (indicesCount:int) ->
//            use indices_global = m.Worker.Malloc(indicesCount)
//            use data = m.Worker.Malloc([|2..5..100|])
//            let numBlocks = divup (aCount + bCount) NV
//            let lp = LaunchParam(numBlocks, plan.NT)
//            kernel.Launch m lp data.Length 400 0 NV 

