module Test.Alea.CUDA.Extension.MGPU.BulkRemove

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.BulkRemove
open Test.Alea.CUDA.Extension.TestUtilities
//open Test.Alea.CUDA.Extension.TestUtilities.MGPU.BulkRemoveUtils
open NUnit.Framework


let bulkRemove =
    let br = worker.LoadPModule(MGPU.PArray.bulkRemove).Invoke
    fun (data:'TI[]) (indices:int[]) ->
        let calc = pcalc {
            let! data = DArray.scatterInBlob worker data
            let! indices = DArray.scatterInBlob worker indices
            let! result = br data indices
            return! result.Value }
        let dResult = PCalc.run calc
        dResult
//
//let bulkRemove = cuda {
//    let! api = BulkRemove.bulkRemove
//    
//    return PFunc(fun (m:Module) ->
//        let worker = m.Worker
//        let api = api.Apply m
//        
//


//[<Test>]
//let ``simple bulkRemove`` () =
//    let hValues = Array.init 1000 (fun i -> i)
//    printfn "Initial Array:  %A" hValues
//    let hIndices = [| 2; 3; 8; 11; 13; 14 |]
//    printfn "Indices to remove: %A" hIndices
//    let hResult = Set.difference (hValues |> Set.ofArray) (hIndices |> Set.ofArray) |> Set.toArray
//    printfn "Host Result After Removal:  %A" hResult
//    let dResult = bulkRemove hValues hIndices
//    printfn "Device Result After Removal:  %A" dResult
//
//    displayHandD hResult dResult
//


[<Test>]
let ``bulkRemove moderngpu web example`` () =
    let hValues = Array.init 100 (fun i -> i)
    //printfn "Initial Array:  %A" hValues
    let hRemoveIndices = [|  1;  4;  5;  7; 10; 14; 15; 16; 18; 19;
                            27; 29; 31; 32; 33; 36; 37; 39; 50; 59;
                            60; 61; 66; 78; 81; 83; 85; 90; 91; 96;
                            97; 98; 99 |]

    //printfn "Indices to remove: %A" hRemoveIndices
    let answer = [| 0;  2;  3;  6;  8;  9; 11; 12; 13; 17; 
                   20; 21; 22; 23; 24; 25; 26; 28; 30; 34;
                   35; 38; 40; 41; 42; 43; 44; 45; 46; 47;
                   48; 49; 51; 52; 53; 54; 55; 56; 57; 58;
                   62; 63; 64; 65; 67; 68; 69; 70; 71; 72;
                   73; 74; 75; 76; 77; 79; 80; 82; 84; 86;
                   87; 88; 89; 92; 93; 94; 95 |]


    let hResult = Set.difference (hValues |> Set.ofArray) (hRemoveIndices |> Set.ofArray) |> Set.toArray
    //printfn "Host Result After Removal:  %A" hResult
    let dResult = bulkRemove hValues hRemoveIndices
    //printfn "Device Result After Removal:  %A" dResult
    displayHandD hResult dResult

    printfn "hResult == answer?"
    verify hResult answer
    printfn "yes"
    printfn "dResult == hResult?"
    verify hResult dResult
    printfn "yes"

    
    
