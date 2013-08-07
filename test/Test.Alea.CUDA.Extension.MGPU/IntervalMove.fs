module Test.Alea.CUDA.Extension.MGPU.IntervalMove

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.Output.Util
open Test.Alea.CUDA.Extension.MGPU.BenchmarkStats

open NUnit.Framework

let worker = getDefaultWorker()
let pScanner = new PScan()
let pIntervalMover = new PIntervalMove()
let op = (CTAScan.scanOp ScanOpTypeAdd 0)


let genRandomIntervals (count:int) (numTerms:int) (terms:int[]) =
    let dquot, drem = (count / numTerms), (count % numTerms)
    for i = 0 to numTerms - 1 do
        Array.set terms i (dquot + (if i < drem then 1 else 0))
    for i = 0 to numTerms - 2 do
        let r = rng.Next(numTerms - i - 1)
        let x = min terms.[r] terms.[i + r]
        let r2 = rng.Next(-x,x)
        Array.set terms r (terms.[r] - r2)
        Array.set terms (i + r) (terms.[i + r] + r2)
        swap terms.[r] terms.[i + r]


[<Test>]
let ``Demo Interval Expand`` () =
    let expand = worker.LoadPModule(pIntervalMover.IntervalExpand()).Invoke
    let scan = worker.LoadPModule(pScanner.Scan(op)).Invoke

    printfn "INTERVAL-EXPAND DEMONSTRATION:\n"

    let numInputs = 20
    printfn "Expand counts (n = %d):" numInputs
    let hCounts = [| 2;    5;    7;   16;    0;    1;    0;    0;   14;   10;
                     3;   14;    2;    1;   11;    2;    1;    0;    5;    6 |]
    printfn "%A\n" hCounts

    printfn "Expand values (n = %d):" numInputs
    let hInputs = [|  1;    1;    2;    3;    5;    8;   13;   21;   34;   55;
                     89;  144;  233;  377;  610;  987; 1597; 2584; 4181; 6765 |]
    printfn "%A\n" hInputs

    printfn "Expanded data (MGPU result):"
    let hAnswer = [|     1;    1;    1;    1;    1;    1;    1;    2;    2;    2;
                        2;    2;    2;    2;    3;    3;    3;    3;    3;    3;
                        3;    3;    3;    3;    3;    3;    3;    3;    3;    3;
                        8;   34;   34;   34;   34;   34;   34;   34;   34;   34;
                       34;   34;   34;   34;   34;   55;   55;   55;   55;   55;
                       55;   55;   55;   55;   55;   89;   89;   89;  144;  144;
                      144;  144;  144;  144;  144;  144;  144;  144;  144;  144;
                      144;  144;  233;  233;  377;  610;  610;  610;  610;  610;
                      610;  610;  610;  610;  610;  610;  987;  987; 1597; 4181;
                     4181; 4181; 4181; 4181; 6765; 6765; 6765; 6765; 6765; 6765 |]
    printfn "%A\n" hAnswer

    let dScanResults = pcalc {
        let! dCounts = DArray.scatterInBlob worker hCounts
        let! total, scanned = scan dCounts
        let! scanned = scanned.Gather()
        let! total = total.Gather()
        return total, scanned } |> PCalc.run

    let hTotal, hCounts_Scanned = dScanResults
    
    let total = hTotal.[0]
    let hCounts = hCounts_Scanned
    printfn "total: %A" total
    let dExpandResult = 
        pcalc {
            let! dCounts = DArray.scatterInBlob worker hCounts
            let! dInputs = DArray.scatterInBlob worker hInputs
            let! result = expand total dCounts dInputs
            let! result = result.Gather()
        return result } |> PCalc.run

    printfn "Expanded data (Alea.cuBase result):"
    printfn "%A\n" dExpandResult

    (hAnswer, dExpandResult) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
    printfn "\nInterval Expand Passed\n"

[<Test>]
let ``Demo Interval Expand Func`` () =
    // using the "in place" function from pIntervalMover (the one used for benchmarking)
    let expand = worker.LoadPModule(pIntervalMover.IntervalExpandFunc()).Invoke
    let scan = worker.LoadPModule(pScanner.ScanFuncReturnTotal(op)).Invoke

    printfn "INTERVAL-EXPAND DEMONSTRATION:\n"

    let numInputs = 20
    printfn "Expand counts (n = %d):" numInputs
    let hCounts = [| 2;    5;    7;   16;    0;    1;    0;    0;   14;   10;
                     3;   14;    2;    1;   11;    2;    1;    0;    5;    6 |]
    printfn "%A\n" hCounts

    printfn "Expand values (n = %d):" numInputs
    let hInputs = [|  1;    1;    2;    3;    5;    8;   13;   21;   34;   55;
                     89;  144;  233;  377;  610;  987; 1597; 2584; 4181; 6765 |]
    printfn "%A\n" hInputs

    printfn "Expanded data (MGPU result):"
    let hAnswer = [|     1;    1;    1;    1;    1;    1;    1;    2;    2;    2;
                        2;    2;    2;    2;    3;    3;    3;    3;    3;    3;
                        3;    3;    3;    3;    3;    3;    3;    3;    3;    3;
                        8;   34;   34;   34;   34;   34;   34;   34;   34;   34;
                       34;   34;   34;   34;   34;   55;   55;   55;   55;   55;
                       55;   55;   55;   55;   55;   89;   89;   89;  144;  144;
                      144;  144;  144;  144;  144;  144;  144;  144;  144;  144;
                      144;  144;  233;  233;  377;  610;  610;  610;  610;  610;
                      610;  610;  610;  610;  610;  610;  987;  987; 1597; 4181;
                     4181; 4181; 4181; 4181; 6765; 6765; 6765; 6765; 6765; 6765 |]
    printfn "%A\n" hAnswer
    
    let dExpandResult = 
        pcalc {
            // scan
            let! dCounts = DArray.scatterInBlob worker hCounts
            let! dScannedCounts = DArray.createInBlob worker numInputs
            let! scan = scan dCounts.Length
            let! total = scan dCounts dScannedCounts

            // expand
            let! expand = expand total numInputs
            let! dInputs = DArray.scatterInBlob worker hInputs
            let! dDest = DArray.createInBlob worker total
            
            do! expand dScannedCounts dInputs dDest            
            let! result = dDest.Gather()
        return result } |> PCalc.run

    printfn "Expanded data (Alea.cuBase result):"
    printfn "%A\n" dExpandResult
    (hAnswer, dExpandResult) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))


[<Test>]
let ``Demo Interval Expand Func 2`` () =
    let expand = worker.LoadPModule(pIntervalMover.IntervalExpandFunc()).Invoke
    let scan = worker.LoadPModule(pScanner.ScanFuncReturnTotal(op)).Invoke
    let numInputs = 50
    let count = numInputs * 25
    let terms = Array.zeroCreate numInputs
    genRandomIntervals count numInputs terms
    printfn "terms:\n%A" terms
    let hInputs = Array.init numInputs (fun i -> i)     

    let dExpandResult = 
        pcalc {
            let! dCounts = DArray.scatterInBlob worker terms
            let! dScannedCounts = DArray.createInBlob worker numInputs
            
            let! scan = scan dCounts.Length
            let! total = scan dCounts dScannedCounts

            let! expand = expand total numInputs

            let! dInputs = DArray.scatterInBlob worker hInputs
            let! dDest = DArray.createInBlob worker total
                        
            do! expand dScannedCounts dInputs dDest            
            let! result = dDest.Gather()
        return result } |> PCalc.run

    printfn "Expanded data (Alea.cuBase result):"
    printfn "%A\n" dExpandResult


[<Test>]
let ``Demo Interval Move`` () =
    let scan = worker.LoadPModule(pScanner.Scan(op)).Invoke
    let move = worker.LoadPModule(pIntervalMover.IntervalMove()).Invoke

    printfn "INTERVAL-MOVE DEMONSTRATION:\n"

    let numInputs = 20
    printfn "Interval counts:"
    let hCounts = [|    3;    9;    1;    9;    8;    5;   10;    2;    5;    2;
                        8;    6;    5;    2;    4;    0;    8;    2;    5;    6 |]
    printfn "%A\n" hCounts

    printfn "Interval gather:"
    let hGather = [|    75;   86;   17;    2;   67;   24;   37;   11;   95;   35;
                        52;   18;   47;    0;   13;   75;   78;   60;   62;   29 |]
    printfn "%A\n" hGather

    printfn "Interval scatter:"
    let hScatter = [|   10;   80;   99;   27;   41;   71;   15;    0;   36;   13;
                        89;   49;   66;   97;   76;   76;    2;   25;   61;   55 |]
    printfn "%A\n" hScatter

    printfn "Moved data (MGPU result):"
    let hAnswer = [| 11;   12;   78;   79;   80;   81;   82;   83;   84;   85;
                    75;   76;   77;   35;   36;   37;   38;   39;   40;   41;
                    42;   43;   44;   45;   46;   60;   61;    2;    3;    4;
                     5;    6;    7;    8;    9;   10;   95;   96;   97;   98;
                    99;   67;   68;   69;   70;   71;   72;   73;   74;   18;
                    19;   20;   21;   22;   23;   29;   30;   31;   32;   33;
                    34;   62;   63;   64;   65;   66;   47;   48;   49;   50;
                    51;   24;   25;   26;   27;   28;   13;   14;   15;   16;
                    86;   87;   88;   89;   90;   91;   92;   93;   94;   52;
                    53;   54;   55;   56;   57;   58;   59;    0;    1;   17 |]
    printfn "%A\n" hAnswer


    let dScanResults = 
        pcalc {
            let! dCounts = DArray.scatterInBlob worker hCounts
            let! total, scanned = scan dCounts
            let! scanned = scanned.Gather()
            let! total = total.Gather()
        return total, scanned } |> PCalc.run

    let hTotal, hCounts_Scanned = dScanResults
    
    let total = hTotal.[0]
    let hCounts = hCounts_Scanned

    let dMoveResult = 
        pcalc {
            let! dGather = DArray.scatterInBlob worker hGather
            let! dScatter = DArray.scatterInBlob worker hScatter
            let! dCounts = DArray.scatterInBlob worker hCounts
            let! dInput = DArray.scatterInBlob worker ([| 0..total |])          
            let! result = move total dGather dScatter dCounts dInput
            let! result = result.Gather()
        return result } |> PCalc.run

    printfn "Moved data (Alea.cuBase result):"
    printfn "%A\n" dMoveResult

    (hAnswer, dMoveResult) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
    printfn "\nInterval Move Passed\n"

