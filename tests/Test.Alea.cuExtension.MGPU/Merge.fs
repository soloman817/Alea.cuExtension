module Test.Alea.cuBase.MGPU.Merge

open Alea.CUDA
open Alea.cuBase
open Alea.cuBase.MGPU
open NUnit.Framework

let worker = Engine.workers.DefaultWorker
let pfuncts = new PMerge()

let sourceCounts = [512; 1024; 2048; 3000; 6000; 12000; 24000; 100000; 1000000]
let aib count =
    let aCount = count / 2
    let bCount = count - aCount
    aCount,bCount

let aibCounts = sourceCounts |> List.map (fun x -> aib x)

let split (arr : _ array) =
    let n = arr.Length
    arr.[0..n/2-1], arr.[n/2..n-1]
 
let rec hMerge (l : 'a array) (r : 'a array) =
    let n = l.Length + r.Length
    let res = Array.zeroCreate<'a> n
    let mutable i, j = 0, 0
    for k = 0 to n-1 do
        if i >= l.Length   then res.[k] <- r.[j]; j <- j + 1
        elif j >= r.Length then res.[k] <- l.[i]; i <- i + 1
        elif l.[i] < r.[j] then res.[k] <- l.[i]; i <- i + 1
        else res.[k] <- r.[j]; j <- j + 1
 
    res
 
let rec mergeSort = function
    | [||]  -> [||]
    | [|a|] -> [|a|]
    | arr   -> let (x, y) = split arr
               hMerge (mergeSort x) (mergeSort y)



let testMergeKeys() =
    let test verify eps (aData:'T[]) (bData:'T[]) (compOp:IComp<'T>)= pcalc {
        let merge = worker.LoadPModule(pfuncts.MergeKeys(compOp)).Invoke
        
        let aData = aData |> Array.sort
        let bData = bData |> Array.sort

        let aCount = aData.Length
        let bCount = bData.Length

        printfn "Testing %d items merged with %d..." aCount bCount

        let! dA = DArray.scatterInBlob worker aData
        let! dB = DArray.scatterInBlob worker bData
        let! dC = DArray.createInBlob<'T> worker (aCount + bCount)
        let! merge = merge aCount bCount
        do! merge dA dB dC

        if verify then
            let hResults = hMerge aData bData
            let! dResults = dC.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
                    
        else 
            do! PCalc.force() }

    let eps = 1e-10

    
    let values1 n na nb = 
        let (hA:int[]) = rngGenericArrayBounded na n 
        let (hB:int[]) = rngGenericArrayBounded nb n
        hA,hB

    let values2 n na nb = 
        let (hA:int[]) = rngGenericArrayBounded na n
        let (hB:int[]) = rngGenericArrayBounded nb n
        hA,hB

    let values3 n na nb = 
        let (hA:float[]) = rngGenericArrayBounded na n
        let (hB:float[]) = rngGenericArrayBounded nb n
        hA,hB
        
    let compOp = comp CompTypeLess 0
    (sourceCounts, aibCounts) ||> List.iter2 (fun ns (na,nb) -> let test = test true eps
                                                                values1 ns na nb ||> test <| compOp |> PCalc.run)

    (sourceCounts, aibCounts) ||> List.iter2 (fun ns (na,nb) -> let test = test true eps
                                                                values2 ns na nb ||> test <| compOp |> PCalc.run)

    let compOp = comp CompTypeLess 0.0
    (sourceCounts, aibCounts) ||> List.iter2 (fun ns (na,nb) -> let test = test true eps
                                                                values3 ns na nb ||> test <| compOp |> PCalc.run)
        
    let n = 2097152
    let na,nb = aib n
    let test = (values1 n na nb) ||> test false eps <| (comp CompTypeLess 0)

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))


[<Test>]
let ``merge keys moderngpu website example`` () =
//A:
    let aData = [|  0;    0;    3;    4;    4;    7;    7;    7;    8;    8;
                    9;   10;   11;   12;   13;   13;   13;   14;   14;   15;
                   16;   16;   18;   18;   19;   22;   23;   23;   25;   25;
                   26;   26;   28;   31;   34;   34;   35;   36;   38;   39;
                   40;   43;   43;   43;   44;   44;   45;   46;   47;   49;
                   50;   50;   50;   51;   52;   52;   53;   53;   54;   54;
                   55;   57;   60;   60;   62;   62;   62;   65;   66;   67;
                   68;   68;   71;   72;   74;   74;   76;   77;   79;   80;
                   80;   81;   82;   82;   85;   85;   85;   86;   86;   86;
                   91;   91;   91;   92;   96;   97;   97;   98;   98;   99 |]
 
//B:
    let bData = [|  1;    3;    4;    4;    4;    5;    5;    8;    9;   10;
                   11;   12;   13;   16;   16;   18;   18;   21;   22;   23;
                   24;   24;   25;   27;   28;   29;   30;   30;   30;   31;
                   32;   33;   34;   34;   35;   36;   36;   36;   37;   37;
                   38;   38;   39;   40;   40;   41;   43;   43;   44;   45;
                   45;   48;   48;   48;   49;   49;   49;   49;   50;   51;
                   54;   54;   55;   57;   62;   62;   64;   64;   65;   66;
                   68;   71;   73;   74;   75;   75;   77;   78;   78;   79;
                   80;   81;   81;   81;   82;   82;   87;   87;   88;   90;
                   90;   90;   91;   91;   92;   94;   94;   95;   95;   98 |]

//Merged array:
    let answer = [| 0;    0;    1;    3;    3;    4;    4;    4;    4;    4;
                    5;    5;    7;    7;    7;    8;    8;    8;    9;    9;
                   10;   10;   11;   11;   12;   12;   13;   13;   13;   13;
                   14;   14;   15;   16;   16;   16;   16;   18;   18;   18;
                   18;   19;   21;   22;   22;   23;   23;   23;   24;   24;
                   25;   25;   25;   26;   26;   27;   28;   28;   29;   30;
                   30;   30;   31;   31;   32;   33;   34;   34;   34;   34;
                   35;   35;   36;   36;   36;   36;   37;   37;   38;   38;
                   38;   39;   39;   40;   40;   40;   41;   43;   43;   43;
                   43;   43;   44;   44;   44;   45;   45;   45;   46;   47;
                   48;   48;   48;   49;   49;   49;   49;   49;   50;   50;
                   50;   50;   51;   51;   52;   52;   53;   53;   54;   54;
                   54;   54;   55;   55;   57;   57;   60;   60;   62;   62;
                   62;   62;   62;   64;   64;   65;   65;   66;   66;   67;
                   68;   68;   68;   71;   71;   72;   73;   74;   74;   74;
                   75;   75;   76;   77;   77;   78;   78;   79;   79;   80;
                   80;   80;   81;   81;   81;   81;   82;   82;   82;   82;
                   85;   85;   85;   86;   86;   86;   87;   87;   88;   90;
                   90;   90;   91;   91;   91;   91;   91;   92;   92;   94;
                   94;   95;   95;   96;   97;   97;   98;   98;   98;   99 |]
    
    let pfunct = pfuncts.MergeKeys((comp CompTypeLess 0))
    let mergeKeys = worker.LoadPModule(pfunct).Invoke
    let aCount = aData.Length
    let bCount = bData.Length

    let dResult = pcalc {
        let! dA = DArray.scatterInBlob worker aData
        let! dB = DArray.scatterInBlob worker bData
        let! merge = mergeKeys aCount bCount
        let! dC = DArray.createInBlob worker (aCount + bCount)
        do! merge dA dB dC
        let! results = dC.Gather()
        return results } |> PCalc.runInWorker worker

    printfn "%A" dResult
    
    (answer, dResult) ||> Array.iter2 (fun a d -> Assert.AreEqual(a, d))


[<Test>]
let ``MergeKeys 3 value test`` () =
    testMergeKeys()


[<Test>]
let ``MergePairs vs mgpu demo output`` () =
//MERGE PAIRS DEMONSTRATION:
//A:
    let aKeys = [|  0;    3;    3;    3;    3;    4;    5;    9;    9;   10;
                   11;   12;   12;   12;   13;   14;   14;   15;   17;   17;
                   18;   18;   21;   21;   22;   27;   27;   29;   30;   30;
                   31;   31;   36;   38;   39;   39;   40;   42;   42;   43;
                   44;   45;   47;   48;   48;   48;   49;   50;   54;   54;
                   63;   63;   64;   65;   65;   66;   67;   68;   69;   70;
                   70;   72;   74;   74;   75;   75;   76;   76;   79;   79;
                   79;   79;   79;   79;   80;   80;   81;   82;   82;   83;
                   84;   87;   87;   90;   91;   91;   92;   93;   95;   95;
                   95;   95;   96;   96;   96;   97;   98;   99;   99;   99 |]

//B:
    let bKeys = [|  0;    1;    1;    1;    5;    6;    7;   11;   12;   13;
                   14;   15;   15;   16;   19;   20;   21;   22;   23;   24;
                   25;   25;   25;   25;   27;   28;   29;   33;   33;   34;
                   34;   35;   35;   35;   38;   38;   38;   39;   39;   40;
                   41;   42;   45;   46;   47;   49;   50;   52;   53;   54;
                   54;   56;   56;   56;   57;   58;   58;   58;   59;   59;
                   61;   64;   65;   67;   67;   69;   69;   70;   71;   72;
                   73;   75;   75;   75;   76;   77;   77;   77;   79;   80;
                   80;   81;   82;   82;   83;   84;   84;   87;   89;   90;
                   91;   92;   93;   94;   95;   95;   95;   96;   98;   98 |]

//Merged keys:
    let mergedKeys = [|     0;    0;    1;    1;    1;    3;    3;    3;    3;    4;
                            5;    5;    6;    7;    9;    9;   10;   11;   11;   12;
                           12;   12;   12;   13;   13;   14;   14;   14;   15;   15;
                           15;   16;   17;   17;   18;   18;   19;   20;   21;   21;
                           21;   22;   22;   23;   24;   25;   25;   25;   25;   27;
                           27;   27;   28;   29;   29;   30;   30;   31;   31;   33;
                           33;   34;   34;   35;   35;   35;   36;   38;   38;   38;
                           38;   39;   39;   39;   39;   40;   40;   41;   42;   42;
                           42;   43;   44;   45;   45;   46;   47;   47;   48;   48;
                           48;   49;   49;   50;   50;   52;   53;   54;   54;   54;
                           54;   56;   56;   56;   57;   58;   58;   58;   59;   59;
                           61;   63;   63;   64;   64;   65;   65;   65;   66;   67;
                           67;   67;   68;   69;   69;   69;   70;   70;   70;   71;
                           72;   72;   73;   74;   74;   75;   75;   75;   75;   75;
                           76;   76;   76;   77;   77;   77;   79;   79;   79;   79;
                           79;   79;   79;   80;   80;   80;   80;   81;   81;   82;
                           82;   82;   82;   83;   83;   84;   84;   84;   87;   87;
                           87;   89;   90;   90;   91;   91;   91;   92;   92;   93;
                           93;   94;   95;   95;   95;   95;   95;   95;   95;   96;
                           96;   96;   96;   97;   98;   98;   98;   99;   99;   99 |]

//Merged values (0-99 are A indices, 100-199 are B indices).
    let mergedValues = [|   0;  100;  101;  102;  103;    1;    2;    3;    4;    5;
                            6;  104;  105;  106;    7;    8;    9;   10;  107;   11;
                           12;   13;  108;   14;  109;   15;   16;  110;   17;  111;
                          112;  113;   18;   19;   20;   21;  114;  115;   22;   23;
                          116;   24;  117;  118;  119;  120;  121;  122;  123;   25;
                           26;  124;  125;   27;  126;   28;   29;   30;   31;  127;
                          128;  129;  130;  131;  132;  133;   32;   33;  134;  135;
                          136;   34;   35;  137;  138;   36;  139;  140;   37;   38;
                          141;   39;   40;   41;  142;  143;   42;  144;   43;   44;
                           45;   46;  145;   47;  146;  147;  148;   48;   49;  149;
                          150;  151;  152;  153;  154;  155;  156;  157;  158;  159;
                          160;   50;   51;   52;  161;   53;   54;  162;   55;   56;
                          163;  164;   57;   58;  165;  166;   59;   60;  167;  168;
                           61;  169;  170;   62;   63;   64;   65;  171;  172;  173;
                           66;   67;  174;  175;  176;  177;   68;   69;   70;   71;
                           72;   73;  178;   74;   75;  179;  180;   76;  181;   77;
                           78;  182;  183;   79;  184;   80;  185;  186;   81;   82;
                          187;  188;   83;  189;   84;   85;  190;   86;  191;   87;
                          192;  193;   88;   89;   90;   91;  194;  195;  196;   92;
                           93;   94;  197;   95;   96;  198;  199;   97;   98;   99 |]

    let pfunct = pfuncts.MergePairs((comp CompTypeLess 0))
    let mergePairs = worker.LoadPModule(pfunct).Invoke

    let N = 100

    let dResult = pcalc {
        let! dAk = DArray.scatterInBlob worker aKeys
        let! dBk = DArray.scatterInBlob worker bKeys
        let! dAv = DArray.scatterInBlob worker (Array.init N (fun i -> i))
        let! dBv = DArray.scatterInBlob worker (Array.init N (fun i -> i + N))
        let! mergePairs = mergePairs N N
        let! dCk = DArray.createInBlob worker (2 * N)
        let! dCv = DArray.createInBlob worker (2 * N)

        do! mergePairs dAk dAv dBk dBv dCk dCv
        let! result = pcalc { let! dCk = dCk.Gather()
                              let! dCv = dCv.Gather() 
                              return dCk, dCv}
        return result } |> PCalc.runInWorker worker

    let dMk, dMv = dResult
    let hMk, hMv = mergedKeys, mergedValues
    printfn "hMk: %A" hMk
    printfn "dMk: %A" dMk
    printfn "hMv: %A" hMv
    printfn "dMv: %A" dMv
    (hMk, dMk) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
    (hMv, dMv) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
