module Test.Alea.CUDA.Extension.MGPU.Mergesort

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.DeviceUtil

open NUnit.Framework

let worker = Engine.workers.DefaultWorker
let pfuncts = new PMergesort()

[<Test>]
let `` simple MergeSort Keys test`` () =
    let compOp = (comp CompTypeLess 0)
    let pfunct = worker.LoadPModule(pfuncts.MergesortKeys()).Invoke
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
        let! mergesort = pfunct dSource        
        let! results = mergesort.Gather()
        return results } |> PCalc.runInWorker worker

//    let dest, source = dResult
//    printfn "dest %A" dest
    let source = dResult
    printfn "source %A" source


//[<Test>]
//let `` simple MergeSort Keys test #2`` () =
//    let compOp = (comp CompTypeLess 0)
//    let pfunct = worker.LoadPModule(MGPU.PArray.mergesortKeys2()).Invoke
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
//    let answer = [| 0;    3;    3;    3;    3;    4;    5;    9;    9;   10;
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
//        let! dDest = DArray.createInBlob worker count
//        let! mergesort = pfunct count
//        let! mergesort = mergesort dSource dDest
//        do mergesort
//        let! results = pcalc { let! d = dDest.Gather()
//                               let! s = dSource.Gather() 
//                               return d, s}
//        return results } |> PCalc.runInWorker worker
//
//    let dest, source = dResult
//    printfn "dest %A" dest
//    printfn "source %A" source