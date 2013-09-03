module Test.Alea.CUDA.Extension.MGPU.BulkInsert

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.IO.Util
open NUnit.Framework


let worker = getDefaultWorker()
let pfuncts = new PBulkInsert()
let rng = System.Random()

let aib count =
    let aCount = count / 2
    let bCount = count - aCount
    aCount,bCount

let sourceCounts = [512; 1024; 2048; 3000; 6000; 12000; 24000; 100000; 1000000]
let aibCounts = sourceCounts |> List.map (fun x -> aib x)
let aCounts, bCounts = aibCounts |> List.unzip


let hostBulkInsert (dataA:'T[]) (indices:int[]) (dataB:'T[]) =
    let result : 'T[] = Array.zeroCreate (dataA.Length + dataB.Length)
    Array.blit dataB 0 result 0 indices.[0]
    Array.set result indices.[0] dataA.[0]
    for i = 1 to indices.Length - 1 do
        Array.blit dataB indices.[i - 1] result (indices.[i - 1] + i) (indices.[i] - indices.[i - 1])
        Array.set result (indices.[i] + i) dataA.[i]
    let i = indices.Length - 1
    Array.blit dataB indices.[i] result (indices.[i] + i + 1) (result.Length - (indices.[i] + i + 1))
    result


let testBulkInsert() =
    let test verify eps (dataA:'T[]) (indices:int[]) (dataB:'T[]) = pcalc {
        let bulkin = worker.LoadPModule(pfuncts.BulkInsert()).Invoke
    
        let aCount = dataA.Length
        let bCount = dataB.Length
        printfn "Testing %d items inserted into %d..." aCount bCount

        let! dA = DArray.scatterInBlob worker dataA     // elements to insert
        let! dI = DArray.scatterInBlob worker indices   // where to insert them
        let! dB = DArray.scatterInBlob worker dataB     // collection they are inserted into
        let! dR = bulkin dA dI dB

        if verify then
            let hResults = hostBulkInsert dataA indices dataB
            let! dResults = dR.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
                    
        else 
            do! PCalc.force() }

    let eps = 1e-10
    let values1 na nb = 
        let (r:int[]*int[]*int[]) = rngGenericArrayAIB na nb
        let hA,hI,hB = r
        hA,hI,hB

    let values2 na nb = 
        let (r:int[]*int[]*int[]) = rngGenericArrayAIB na nb
        let hA,hI,hB = r
        hA,hI,hB

    let values3 na nb = 
        let (r:float[]*int[]*float[]) = rngGenericArrayAIB na nb
        let hA,hI,hB = r
        hA,hI,hB  
        

    aibCounts |> List.iter (fun (na,nb) -> let test = test true eps
                                           values1 na nb |||> test |> PCalc.run)

    aibCounts |> List.iter (fun (na,nb) -> let test = test true eps
                                           values2 na nb |||> test |> PCalc.run)

    aibCounts |> List.iter (fun (na,nb) -> let test = test true eps
                                           values3 na nb |||> test |> PCalc.run)
        
    let n = 2097152
    let na,nb = aib n
    let test = (values1 na nb) |||> test false eps

    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
        



[<Test>]
let ``bulkInsert simple example`` () =
    let hB = Array.init 20 (fun i -> i)     // source data
    let hI = [| 3; 7; 11; 14; 19 |]         // indices of insertions
    let hA = [| 93; 97; 911; 914; 919 |]    // data to insert into hB

    let pfunct = pfuncts.BulkInsert()
    let bulkin = worker.LoadPModule(pfunct).Invoke

    let dResult = pcalc {
        let! dA = DArray.scatterInBlob worker hA
        let! dB = DArray.scatterInBlob worker hB
        let! dI = DArray.scatterInBlob worker hI
        let! dR = bulkin dA dI dB
        let! results = dR.Gather()
        return results } |> PCalc.run

    printfn "%A" dResult


[<Test>]
let ``bulkInsert moderngpu website example`` () =
    let hI = [|2..5..100|]
    let aCount, bCount = hI.Length, 100
    
    let hA = [|1000..10..((aCount*10+1000)-10)|]
    let hB = Array.init bCount int
    
    let answer = [|     0;  1; 1000;  2;     3;  4;    5;  6; 1010;  7;
                        8;  9;   10; 11;  1020; 12;   13; 14;   15; 16;
                     1030; 17;   18; 19;    20; 21; 1040; 22;   23; 24;
                       25; 26; 1050; 27;    28; 29;   30; 31; 1060; 32;
                       33; 34;   35; 36;  1070; 37;   38; 39;   40; 41;
                     1080; 42;   43; 44;    45; 46; 1090; 47;   48; 49;
                       50; 51; 1100; 52;    53; 54;   55; 56; 1110; 57;
                       58; 59;   60; 61;  1120; 62;   63; 64;   65; 66;
                     1130; 67;   68; 69;    70; 71; 1140; 72;   73; 74;
                       75; 76; 1150; 77;    78; 79;   80; 81; 1160; 82;
                       83; 84;   85; 86;  1170; 87;   88; 89;   90; 91;
                     1180; 92;   93; 94;    95; 96; 1190; 97;   98; 99 |]
    
    let hResult = hostBulkInsert hA hI hB
    (hResult, answer) ||> Array.iter2 (fun h a -> Assert.AreEqual(h, a))

    let pfunct = pfuncts.BulkInsert()
    let bulkin = worker.LoadPModule(pfunct).Invoke

    let dResult = pcalc {
        let! dA = DArray.scatterInBlob worker hA
        let! dB = DArray.scatterInBlob worker hB
        let! dI = DArray.scatterInBlob worker hI
        let! dR = bulkin dA dI dB
        let! results = dR.Gather()
        return results } |> PCalc.run
    
    (hResult, dResult) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))


    printfn "%A" dResult



[<Test>]
let ``BulkInsert 3 value test`` () =
    testBulkInsert()

