module Test.Alea.CUDA.Extension.MGPU.Search

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.PArray
open Alea.CUDA.Extension.MGPU.CTASearch
open Test.Alea.CUDA.Extension.TestUtilities.General
open Test.Alea.CUDA.Extension.TestUtilities.MGPU
open NUnit.Framework

let binarySearchPartitions (bounds:int) (compOp:CompType) =
    let bs = worker.LoadPModule(MGPU.PArray.binarySearchPartitions bounds compOp).Invoke
    let nv = 128 * 11
    fun (data:'TI[]) (indices:int[]) ->
        let calc = pcalc {
            let! data = DArray.scatterInBlob worker data
            let! indices = DArray.scatterInBlob worker indices
            let! result = bs data data.Length indices.Length nv
            return! result.Value }
        let dResult = PCalc.run calc
        dResult

[<Test>]
let ``binary search partitions test`` () =
    let data = Array.init 20 (fun i -> i)
    let indices = [| 1; 2; 3 |]
    let bsp = binarySearchPartitions MgpuBoundsLower CompTypeLess
    let result = bsp data indices
    printfn "BSP Result: %A" result

