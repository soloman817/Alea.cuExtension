module Test.Alea.cuBase.MGPU.Mergesort
//
//open Alea.CUDA
////open Alea.cuBase
////open Alea.cuBase.MGPU
////open Alea.cuBase.MGPU.DeviceUtil
//
//open NUnit.Framework
//
//let worker = getDefaultWorker()
//let pfuncts = new PMergesort()
//
//[<Test>]
//let `` Simple MergeSort Keys Test`` () =
//    let pfunct = worker.LoadPModule(pfuncts.MergesortKeys()).Invoke
//// Input:
//    let hSource = [|   81;   13;   90;   83;   12;   96;   91;   22;   63;   30;
//                        9;   54;   27;   18;   54;   99;   95;   99;   96;   96;
//                       15;   72;   97;   98;   95;   10;   48;   79;   80;   29;
//                       14;    0;   42;   11;   91;   63;   79;   87;   95;   50;
//                       65;   79;    3;   36;   84;   21;   93;   68;   67;   39;
//                       75;   74;   74;   47;   39;   42;   65;   17;   17;   30;
//                       70;   79;    3;   31;   27;   87;    4;   14;    9;   99;
//                       82;   82;   69;   12;   31;   76;   95;   49;    3;   66;
//                       43;   12;   38;   21;   76;    5;   79;    3;   18;   40;
//                       48;   45;   44;   48;   64;   79;   70;   92;   75;   80 |]
//
////Sorted output:
//    let hAnswer = [| 0;    3;    3;    3;    3;    4;    5;    9;    9;   10;
//                   11;   12;   12;   12;   13;   14;   14;   15;   17;   17;
//                   18;   18;   21;   21;   22;   27;   27;   29;   30;   30;
//                   31;   31;   36;   38;   39;   39;   40;   42;   42;   43;
//                   44;   45;   47;   48;   48;   48;   49;   50;   54;   54;
//                   63;   63;   64;   65;   65;   66;   67;   68;   69;   70;
//                   70;   72;   74;   74;   75;   75;   76;   76;   79;   79;
//                   79;   79;   79;   79;   80;   80;   81;   82;   82;   83;
//                   84;   87;   87;   90;   91;   91;   92;   93;   95;   95;
//                   95;   95;   96;   96;   96;   97;   98;   99;   99;   99 |]
//    
//    let count = hSource.Length
//
//    let dResult = pcalc {
//        let! dSource = DArray.scatterInBlob worker hSource
//        let! mergesort = pfunct dSource        
//        let! results = mergesort.Gather()
//        return results } |> PCalc.runInWorker worker
//    (hAnswer, dResult) ||> Array.iter2 (fun h d -> Assert.AreEqual(h,d))
//
//
//
//[<Test>]
//let ``Simple Mergesort Pairs Test`` () =
////SORT PAIRS DEMONSTRATION:
//    let mergesort = worker.LoadPModule(pfuncts.MergesortPairs(0)).Invoke
//    let N = 100
//
//    let hInputKeys = [|    81;   13;   90;   83;   12;   96;   91;   22;   63;   30;
//                            9;   54;   27;   18;   54;   99;   95;   99;   96;   96;
//                           15;   72;   97;   98;   95;   10;   48;   79;   80;   29;
//                           14;    0;   42;   11;   91;   63;   79;   87;   95;   50;
//                           65;   79;    3;   36;   84;   21;   93;   68;   67;   39;
//                           75;   74;   74;   47;   39;   42;   65;   17;   17;   30;
//                           70;   79;    3;   31;   27;   87;    4;   14;    9;   99;
//                           82;   82;   69;   12;   31;   76;   95;   49;    3;   66;
//                           43;   12;   38;   21;   76;    5;   79;    3;   18;   40;
//                           48;   45;   44;   48;   64;   79;   70;   92;   75;   80 |]
//
//    let hInputVals = Array.init N (fun i -> i)
//
//
//    // mgpu result
//    let hSortedKeys = [|    0;    3;    3;    3;    3;    4;    5;    9;    9;   10;
//                           11;   12;   12;   12;   13;   14;   14;   15;   17;   17;
//                           18;   18;   21;   21;   22;   27;   27;   29;   30;   30;
//                           31;   31;   36;   38;   39;   39;   40;   42;   42;   43;
//                           44;   45;   47;   48;   48;   48;   49;   50;   54;   54;
//                           63;   63;   64;   65;   65;   66;   67;   68;   69;   70;
//                           70;   72;   74;   74;   75;   75;   76;   76;   79;   79;
//                           79;   79;   79;   79;   80;   80;   81;   82;   82;   83;
//                           84;   87;   87;   90;   91;   91;   92;   93;   95;   95;
//                           95;   95;   96;   96;   96;   97;   98;   99;   99;   99 |]
//    // mgpu result
//    let hSortedVals = [|   31;   42;   62;   78;   87;   66;   85;   10;   68;   25;
//                           33;    4;   73;   81;    1;   30;   67;   20;   57;   58;
//                           13;   88;   45;   83;    7;   12;   64;   29;    9;   59;
//                           63;   74;   43;   82;   49;   54;   89;   32;   55;   80;
//                           92;   91;   53;   26;   90;   93;   77;   39;   11;   14;
//                            8;   35;   94;   40;   56;   79;   48;   47;   72;   60;
//                           96;   21;   51;   52;   50;   98;   75;   84;   27;   36;
//                           41;   61;   86;   95;   28;   99;    0;   70;   71;    3;
//                           44;   37;   65;    2;    6;   34;   97;   46;   16;   24;
//                           38;   76;    5;   18;   19;   22;   23;   15;   17;   69 |]
//    let N = 100
//
//    let dSortedKeys, dSortedVals = 
//        pcalc { let! dInputKeys = DArray.scatterInBlob worker hInputKeys
//                let! dInputVals = DArray.scatterInBlob worker hInputVals
//                return! mergesort dInputKeys dInputVals } |> PCalc.run
//
//    printfn "Verifying sorted keys result..."
//    (hSortedKeys, dSortedKeys) ||> Array.iter2 (fun h d -> Assert.AreEqual(h,d))
//    printfn "Pass\n"
//    printfn "Verifying sorted vals result..."
//    (hSortedVals, dSortedVals) ||> Array.iter2 (fun h d -> Assert.AreEqual(h,d))
//    printfn "Pass\n"