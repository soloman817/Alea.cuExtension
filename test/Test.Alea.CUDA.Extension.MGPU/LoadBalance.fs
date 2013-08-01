module Test.Alea.CUDA.Extension.MGPU.LoadBalance

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.LoadBalance
open Alea.CUDA.Extension.MGPU.CTAScan
open Alea.CUDA.Extension.MGPU.PArray
open Test.Alea.CUDA.Extension.MGPU.Util


open NUnit.Framework


let worker = getDefaultWorker()
let rng = System.Random()
let pScanner = new PScan()
let pLBS = new PLoadBalanceSearch()

[<Test>]
let ``Load Balance Search simple`` () =
//LOAD-BALANCING SEARCH DEMONSTRATION:
    let op = scanOp ScanOpTypeAdd 0
    let scanner = pScanner.Scan(op) |> worker.LoadPModule
    let scan = scanner.Invoke

    let search = worker.LoadPModule(pLBS.Search()).Invoke

    //Object hCounts
    let hCounts = [|    4;    0;    5;    5;    0;    5;    5;    1;    3;    1;
                       0;    3;    1;    1;    3;    5;    5;    5;    5;    5;
                       0;    4;    5;    5;    5;    0;    2;    4;    4;    1;
                       0;    0;    2;    0;    5;    3;    4;    5;    5;    3;
                       3;    4;    0;    2;    5;    1;    5;    4;    4;    2 |]
    
    printfn "Object hCounts:\n%A" hCounts
        
    let dResult = pcalc {
        let! dCounts = DArray.scatterInBlob worker hCounts        
        let! total, scanned = scan dCounts
        let! scanned = scanned.Gather()
        let! total = total.Gather()        
        return total, scanned } |> PCalc.run

    let dTotal, dScannedCounts = dResult
    let dTotal = dTotal.[0]

    //Scan of object counts:
    let hScannedCounts = [|   0;    4;    4;    9;   14;   14;   19;   24;   25;   28;
                            29;   29;   32;   33;   34;   37;   42;   47;   52;   57;
                            62;   62;   66;   71;   76;   81;   81;   83;   87;   91;
                            92;   92;   92;   94;   94;   99;  102;  106;  111;  116;
                           119;  122;  126;  126;  128;  133;  134;  139;  143;  147 |]
    //Total:  149
    let hTotal = 149

    printfn "\nScan of object counts:\n%A" dScannedCounts
    printfn "\nTotal: %A" dTotal
    
    (hScannedCounts, dScannedCounts) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
    printfn "Scan Passed"
    if hTotal = dTotal then printfn "Total Passed"

    
    //Object references:
    let hObjRefs = [|    0;    0;    0;    0;    2;    2;    2;    2;    2;    3;
                        3;    3;    3;    3;    5;    5;    5;    5;    5;    6;
                        6;    6;    6;    6;    7;    8;    8;    8;    9;   11;
                       11;   11;   12;   13;   14;   14;   14;   15;   15;   15;
                       15;   15;   16;   16;   16;   16;   16;   17;   17;   17;
                       17;   17;   18;   18;   18;   18;   18;   19;   19;   19;
                       19;   19;   21;   21;   21;   21;   22;   22;   22;   22;
                       22;   23;   23;   23;   23;   23;   24;   24;   24;   24;
                       24;   26;   26;   27;   27;   27;   27;   28;   28;   28;
                       28;   29;   32;   32;   34;   34;   34;   34;   34;   35;
                       35;   35;   36;   36;   36;   36;   37;   37;   37;   37;
                       37;   38;   38;   38;   38;   38;   39;   39;   39;   40;
                       40;   40;   41;   41;   41;   41;   43;   43;   44;   44;
                       44;   44;   44;   45;   46;   46;   46;   46;   46;   47;
                       47;   47;   47;   48;   48;   48;   48;   49;   49      |]
    
    let dObjRefs = 
        pcalc {
            let! dCounts = DArray.scatterInBlob worker hScannedCounts            
            let! results = search dTotal dCounts
            let! results = results.Gather()            
        return results } |> PCalc.run
            
    printfn "\nObject References:\n%A" dObjRefs

    (hObjRefs, dObjRefs) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
    printfn "\nLoad Balance Search Passed\n"
         

