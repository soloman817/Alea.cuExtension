module Test.Alea.CUDA.Extension.MGPU.Debug

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Intrinsics
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTAScan
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.CTAMerge
open Alea.CUDA.Extension.MGPU.CTASortedSearch

open NUnit.Framework


//let worker = getDefaultWorker()
//
//
//
//module SortedSearch =
//    type Plan =
//        {
//            NT : int
//            VT : int
//        }
//
//    let deviceLoadSortedSearch (NT:int) (VT:int) (bounds:int) (indexA:bool) (matchA:bool) (indexB:bool) (matchB:bool) (compOp:IComp<int>) =
//    
//        let deviceLoad2ToShared = deviceLoad2ToSharedB NT VT (VT + 1) 
//        let ctaSortedSearch = ctaSortedSearch NT VT bounds indexA matchA indexB matchB compOp
//
//        <@ fun (range:int4) (a_global:DevicePtr<int>) (aCount:int) (b_global:DevicePtr<int>) (bCount:int) (tid:int) (block:int) (keys_shared:RWPtr<int>) (indices_shared:RWPtr<int>) ->
//            let deviceLoad2ToShared = %deviceLoad2ToShared
//            let ctaSortedSearch = %ctaSortedSearch
//
//            let a0 = range.x
//            let a1 = range.y
//            let b0 = range.z
//            let b1 = range.w
//            let aCount2 = a1 - a0
//            let bCount2 = b1 - b0
//
//            let leftA = if (matchB && (bounds = MgpuBoundsLower)) && (a0 > 0) then 1 else 0
//            let leftB = if (matchA && (bounds = MgpuBoundsUpper)) && (b0 > 0) then 1 else 0
//            let rightA = if a1 < aCount then 1 else 0
//            let rightB = if b1 < bCount then 1 else 0
//
//            let aStart = leftA
//            let aEnd = aStart + aCount2 + rightA
//            let bStart = aEnd + leftB
//            let bEnd = bStart + bCount2 + rightB
//
//            deviceLoad2ToShared (a_global + a0 - leftA) aEnd (b_global + b0 - leftB) (bEnd - aEnd) tid keys_shared true
//
//            let extended = 
//                let mutable x = (rightA = 1) && (rightB = 1) 
//                x <- x && ((not matchA) || (leftB = 1))
//                x <- x && ((not matchB) || (leftA = 1))
//                x
//            
//            let matchCount = ctaSortedSearch keys_shared aStart aCount2 aEnd a0 bStart bCount2 bEnd b0 extended tid indices_shared
//
//            matchCount
//        @>
//
//
//
//    let kernelSortedSearch (plan:Plan) (bounds:int) (indexA:int) (matchA:int) (indexB:int) (matchB:int) (compOp:IComp<int>) =
//        let NT = plan.NT
//        let VT = plan.VT
//        let NV = plan.NT * plan.VT
//
//        let indexA = if indexA = 0 then false else true
//        let matchA = if matchA = 0 then false else true
//        let indexB = if indexB = 0 then false else true
//        let matchB = if matchB = 0 then false else true
//    
//        let capacity, reduce = ctaReduce NT (scanOp ScanOpTypeAdd 0)
//        let sharedSize = max capacity (NT * (VT + 1))
//
//        let computeMergeRange = computeMergeRange.Device
//        let deviceLoadSortedSearch = deviceLoadSortedSearch NT VT bounds indexA matchA indexB matchB compOp
//        let deviceMemToMemLoop = deviceMemToMemLoop NT
//
//        <@ fun (a_global:DevicePtr<int>) (aCount:int) (b_global:DevicePtr<int>) (bCount:int) (mp_global:DevicePtr<int>) (aIndices_global:DevicePtr<int>) (bIndices_global:DevicePtr<int>) -> //(aMatchCount:DevicePtr<int>) (bMatchCount:DevicePtr<int>) ->
//            let reduce = %reduce
//            let computeMergeRange = %computeMergeRange
//            let deviceLoadSortedSearch = %deviceLoadSortedSearch
//            let deviceMemToMemLoop = %deviceMemToMemLoop
//
//            let shared = __shared__<int>(sharedSize).Ptr(0)
//            let sharedKeys = shared
//            let sharedIndices = shared
//            let sharedReduce = shared
//
//            
//            let tid = threadIdx.x
//            let block = blockIdx.x
//            let range = computeMergeRange aCount bCount block 0 NV mp_global
//
//            
//            let matchCount = deviceLoadSortedSearch range a_global aCount b_global bCount tid block sharedKeys sharedIndices
//            let aCount = range.y - range.x
//            let bCount = range.w - range.z
//
//            if (indexA || matchA) then
//                deviceMemToMemLoop aCount sharedIndices tid (aIndices_global + range.x) true
//
//            if (indexB || matchB) then
//                deviceMemToMemLoop bCount (sharedIndices + aCount) tid (bIndices_global + range.z) true
//
////            if ((matchA || matchB) && (aMatchCount.[0] <> 0 || bMatchCount.[0] <> 0)) then
////                let x = bfi (uint32 matchCount.y) (uint32 matchCount.x) 16u 16u
////                let total = reduce tid x (sharedReduce.Reinterpret<uint32>())
////                if (tid = 0) && (aMatchCount.[0] <> 0) then atomicAdd aMatchCount (0xffff &&& total)
////                if (tid = 0) && (bMatchCount.[0] <> 0) then atomicAdd bMatchCount (total >>> 16)
//            @>
//
//
//    type ISortedSearch =
//        {
//            Action : ActionHint -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> unit //DevicePtr<int> -> DevicePtr<int> -> unit
//            NumPartitions : int
//        }
//
//
//    let sortedSearch (bounds:int) (typeA:MgpuSearchType) (typeB:MgpuSearchType) (compOp:IComp<int>) = cuda {
//        let plan = {NT = 128; VT = 7}
//        let NT = plan.NT
//        let VT = plan.VT
//        let NV = NT * VT
//
//        let indexA = if (typeA = MgpuSearchTypeIndex) || (typeA = MgpuSearchTypeIndexMatch) then 1 else 0
//        let matchA = if (typeA = MgpuSearchTypeMatch) || (typeA = MgpuSearchTypeIndexMatch) then 1 else 0
//        let indexB = if (typeB = MgpuSearchTypeIndex) || (typeB = MgpuSearchTypeIndexMatch) then 1 else 0
//        let matchB = if (typeB = MgpuSearchTypeMatch) || (typeB = MgpuSearchTypeIndexMatch) then 1 else 0
//                
//
//        let! kernelSortedSearch = kernelSortedSearch plan bounds indexA matchA indexB matchB compOp |> defineKernelFuncWithName "ss"
//        let! mpp = Search.mergePathPartitions bounds compOp
//
//        return PFunc(fun (m:Module) ->
//            let worker = m.Worker
//            let kernelSortedSearch = kernelSortedSearch.Apply m
//            let mpp = mpp.Apply m
//
//            fun (aCount:int) (bCount:int) ->
//                let numBlocks = divup (aCount + bCount) NV
//                let lp = LaunchParam(numBlocks, NT)
//
//
//    //                 Action : ActionHint -> a_global input       ->      b_global input    ->   patitions         ->       aindices output          ->        bindices output        ->        amatchCount ptr    ->      bmatchCount ptr     -> unit
//                let action (hint:ActionHint) (a_global:DevicePtr<int>) (b_global:DevicePtr<int>) (parts:DevicePtr<int>) (aIndices_global:DevicePtr<int>) (bIndices_global:DevicePtr<int>) = //(aMatchCount:DevicePtr<int>) (bMatchCount:DevicePtr<int>) =
//                    fun () ->
//                        let lp = lp |> hint.ModifyLaunchParam
//                        let mpp = mpp aCount bCount NV 0
//                        let partitions = mpp.Action hint a_global b_global parts
//                        kernelSortedSearch.Launch lp a_global aCount b_global bCount parts aIndices_global bIndices_global //aMatchCount bMatchCount
//                    |> worker.Eval
//                { Action = action; NumPartitions = numBlocks + 1 } ) }
//
//    let pSortedSearch (bounds:int) = cuda {
//        let! api = sortedSearch bounds MgpuSearchTypeIndex MgpuSearchTypeNone (comp CompTypeLess 0)
//        return PFunc(fun (m:Module) ->
//            let worker = m.Worker
//            let api = api.Apply m
//            fun (aCount:int) (bCount:int) ->                
//                //let sequence = Array.init aCount (fun i -> i)
//                pcalc {
//                    let api = api aCount bCount
//                    let! partition = DArray.createInBlob<int> worker api.NumPartitions
//                    //let! counter = DArray.scatterInBlob worker sequence
//                    let sortedSearch (aData:DArray<int>) (bData:DArray<int>) (aIndices:DArray<int>) = 
//                        pcalc { do! PCalc.action (fun hint -> api.Action hint aData.Ptr bData.Ptr partition.Ptr aIndices.Ptr (DevicePtr(0n)) ) } //amc.Ptr bmc.Ptr ) }
//                    return sortedSearch } ) } 
//
//
//[<Test>]
//let ``SortedSearch mgpu website example 1`` () =
////Haystack array:
//    let haystack =  [|   0;    5;    5;    7;    7;    7;    7;    8;    9;    9;
//                        10;   11;   12;   14;   15;   15;   16;   17;   19;   19;
//                        20;   24;   25;   28;   28;   29;   31;   33;   36;   36;
//                        37;   38;   40;   42;   42;   43;   45;   46;   49;   50;
//                        51;   51;   51;   52;   53;   55;   56;   57;   60;   60;
//                        61;   61;   62;   62;   64;   66;   68;   69;   73;   74;
//                        79;   81;   82;   84;   85;   88;   90;   90;   95;   97;
//                        99;  101;  105;  108;  108;  111;  115;  118;  118;  119;
//                       119;  119;  119;  122;  122;  123;  125;  126;  126;  130;
//                       133;  133;  135;  135;  139;  140;  143;  145;  145;  146;
//                       147;  149;  149;  149;  154;  158;  160;  161;  165;  166;
//                       168;  169;  170;  172;  172;  174;  174;  174;  175;  175;
//                       175;  177;  179;  182;  183;  184;  186;  187;  188;  190;
//                       192;  193;  194;  196;  198;  199;  199;  205;  205;  208;
//                       209;  215;  217;  218;  218;  218;  220;  220;  221;  221;
//                       223;  224;  225;  230;  234;  234;  235;  240;  240;  243;
//                       244;  249;  250;  251;  252;  253;  253;  254;  255;  255;
//                       255;  257;  258;  258;  259;  262;  263;  265;  267;  270;
//                       270;  274;  278;  278;  278;  279;  280;  281;  284;  284;
//                       284;  285;  285;  292;  294;  295;  296;  296;  296;  298 |]
//
////Needles array:
//    let needles = [|    3;    3;   12;   16;   16;   17;   17;   19;   20;   21;
//                       24;   27;   27;   28;   30;   31;   35;   39;   40;   42;
//                       52;   52;   53;   53;   54;   55;   57;   58;   62;   63;
//                       72;   75;   83;   86;   86;   89;   92;   95;   98;   98;
//                       99;   99;   99;  100;  104;  105;  107;  109;  110;  111;
//                      112;  117;  118;  121;  124;  126;  129;  132;  133;  139;
//                      140;  148;  156;  160;  161;  167;  168;  173;  179;  186;
//                      191;  198;  202;  202;  212;  212;  214;  220;  223;  229;
//                      233;  239;  245;  254;  256;  256;  260;  268;  269;  269;
//                      271;  271;  272;  273;  277;  285;  296;  296;  299;  299 |]
//
////Lower bound array:
//    let lowerBoundArray = [|    1;    1;   12;   16;   16;   17;   17;   18;   20;   21;
//                               21;   23;   23;   23;   26;   26;   28;   32;   32;   33;
//                               43;   43;   44;   44;   45;   45;   47;   48;   52;   54;
//                               58;   60;   63;   65;   65;   66;   68;   68;   70;   70;
//                               70;   70;   70;   71;   72;   72;   73;   75;   75;   75;
//                               76;   77;   77;   83;   86;   87;   89;   90;   90;   94;
//                               95;  101;  105;  106;  107;  110;  110;  115;  122;  126;
//                              130;  134;  137;  137;  141;  141;  141;  146;  150;  153;
//                              154;  157;  161;  167;  171;  171;  175;  179;  179;  179;
//                              181;  181;  181;  181;  182;  191;  196;  196;  200;  200 |]
//
//    let pfunct = SortedSearch.pSortedSearch MgpuBoundsLower
//    let sortedSearch = worker.LoadPModule(pfunct).Invoke
//
//    let dResult = pcalc {
//        let! dNeedles = DArray.scatterInBlob worker needles
//        let! dHaystack = DArray.scatterInBlob worker haystack
//        
//        let needlesSize, haystackSize = needles.Length, haystack.Length
//        
//        //let! aIndices = DArray.createInBlob worker needlesSize
//        //let! bIndices = DArray.scatterInBlob worker [|0|]
//        
//        let! sortedSearch = sortedSearch needlesSize haystackSize
//        
//        do! sortedSearch dNeedles dHaystack dNeedles 
//        
//        let! results = dNeedles.Gather()
//        return results } |> PCalc.run
//
//    printfn "%A" dResult


//[<Test>]
//let ``SortedSearch demo 2`` () =
//    let aCount = 100
//    let bCount = 100
//    let aData = Array.init aCount (fun _ -> rng.Next(299)) |> Array.sort
//    let bData = Array.init bCount (fun _ -> rng.Next(299)) |> Array.sort
//    let aIndices = Array.init aCount (fun _ -> 0)
//    let bIndices = Array.init bCount (fun _ -> 0)
//
//    let pfunct = SortedSearch.pSortedSearch MgpuBoundsLower
//    let sortedSearch = worker.LoadPModule(pfunct).Invoke
//
//    let dResult = pcalc {
//        let! dAData = DArray.scatterInBlob worker aData
//        let! dAIndices = DArray.scatterInBlob worker aIndices
//
//        let! dBData = DArray.scatterInBlob worker bData
//        let! dBIndices = DArray.scatterInBlob worker bIndices
//                
//        let! sortedSearch = sortedSearch aCount bCount
//        
//        do! sortedSearch dAData dBData dAIndices dBIndices
//        
//        let! results = dAIndices.Gather()
//        return results } |> PCalc.runInWorker worker
//
//    printfn "%A" dResult
//
//[<Test>]
//let ``SortedSearch demo 2 copy`` () = 
//    let aCount = 100
//    let bCount = 100
//
//    let aData = [| 1;    9;   10;   10;   10;   13;   15;   29;   29;   32;
//                   33;   37;   37;   38;   40;   42;   44;   47;   51;   52;
//                   56;   56;   63;   63;   66;   83;   83;   89;   90;   92;
//                   94;   95;  108;  114;  117;  119;  122;  126;  126;  131;
//                   133;  137;  142;  145;  146;  146;  147;  151;  164;  164;
//                   189;  191;  193;  196;  196;  199;  203;  204;  208;  211;
//                   212;  217;  222;  222;  226;  227;  229;  229;  237;  238;
//                   238;  239;  239;  239;  240;  242;  244;  246;  247;  250;
//                   254;  261;  263;  271;  274;  274;  276;  280;  285;  287;
//                   287;  287;  289;  290;  290;  291;  294;  297;  298;  298 |]
//
//    let bData = [|  0;    3;    4;    5;   16;   20;   22;   35;   38;   41;
//                   44;   46;   47;   48;   58;   62;   65;   67;   69;   73;
//                   75;   76;   76;   77;   82;   85;   89;   99;  101;  102;
//                   104;  105;  105;  106;  114;  116;  116;  117;  118;  121;
//                   123;  126;  136;  140;  141;  149;  151;  158;  159;  164;
//                   164;  168;  170;  170;  172;  174;  175;  175;  177;  178;
//                   184;  193;  196;  203;  203;  208;  209;  211;  213;  218;
//                   219;  225;  226;  227;  228;  232;  233;  233;  238;  242;
//                   242;  244;  246;  246;  249;  252;  254;  263;  267;  272;
//                   275;  278;  280;  282;  286;  287;  287;  289;  296;  296 |]
//
////Lower bound of A into B (* for match):
////    0:     1;    4;    4;    4;    4;    4;    4;    7;    7;    7;
////   10:     7;    8;    8; *  8;    9;   10; * 10; * 12;   14;   14;
////   20:    14;   14;   16;   16;   17;   25;   25; * 26;   27;   27;
////   30:    27;   27;   34; * 34; * 37;   39;   40; * 41; * 41;   42;
////   40:    42;   43;   45;   45;   45;   45;   45; * 46; * 49; * 49;
////   50:    61;   61; * 61; * 62; * 62;   63; * 63;   65; * 65; * 67;
////   60:    68;   69;   71;   71; * 72; * 73;   75;   75;   78; * 78;
////   70:  * 78;   79;   79;   79;   79; * 79; * 81; * 82;   84;   85;
////   80:  * 86;   87; * 87;   89;   90;   90;   91; * 92;   94; * 95;
////   90:  * 95; * 95; * 97;   98;   98;   98;   98;  100;  100;  100;
////
////Upper bound of B into A (* for match):
////    0:     0;    1;    1;    1;    7;    7;    7;   11; * 14;   15;
////   10:  * 17;   17; * 18;   18;   22;   22;   24;   25;   25;   25;
////   20:    25;   25;   25;   25;   25;   27; * 28;   32;   32;   32;
////   30:    32;   32;   32;   32; * 34;   34;   34; * 35;   35;   36;
////   40:    37; * 39;   41;   42;   42;   47; * 48;   48;   48; * 50;
////   50:  * 50;   50;   50;   50;   50;   50;   50;   50;   50;   50;
////   60:    50; * 53; * 55; * 57; * 57; * 59;   59; * 60;   61;   62;
////   70:    62;   64; * 65; * 66;   66;   68;   68;   68; * 71; * 76;
////   80:  * 76; * 77; * 78; * 78;   79;   80; * 81; * 83;   83;   84;
////   90:    86;   87; * 88;   88;   89; * 92; * 92; * 93;   97;   97;
//
//    
//    let pfunct = SortedSearch.pSortedSearch MgpuBoundsLower
//    let sortedSearch = worker.LoadPModule(pfunct).Invoke
//
//    let dResult = pcalc {
//        let! dAData = DArray.scatterInBlob worker aData
//        let! dAIndices = DArray.createInBlob worker aCount
//
//        let! dBData = DArray.scatterInBlob worker bData
//        let! dBIndices = DArray.createInBlob worker bCount
//                
//        let! sortedSearch = sortedSearch aCount bCount
//        
//        do! sortedSearch dAData dBData dAIndices dBIndices
//        
//        let! results = dAIndices.Gather()
//        return results } |> PCalc.runInWorker worker
//
//    printfn "rA: %A" dResult
//    //printfn "rB: %A" rB


