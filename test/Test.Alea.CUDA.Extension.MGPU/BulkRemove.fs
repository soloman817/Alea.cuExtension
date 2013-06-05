module Test.Alea.CUDA.Extension.MGPU.BulkRemove

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.BulkRemove
open Test.Alea.CUDA.Extension.TestUtilities.General
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


[<Test>]
let ``simple bulkRemove`` () =
    let hValues = Array.init 20 (fun i -> i)
    printfn "Initial Array:  %A" hValues
    let hIndices = [| 2; 3; 8; 11; 13; 14 |]
    printfn "Indices to remove: %A" hIndices
    let hResult = Set.difference (hValues |> Set.ofArray) (hIndices |> Set.ofArray) |> Set.toArray
    printfn "Host Result After Removal:  %A" hResult
    let dResult = bulkRemove hValues hIndices
    printfn "Device Result After Removal:  %A" dResult

    displayHandD hResult dResult

    
    
