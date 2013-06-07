module Test.Alea.CUDA.Extension.MGPU.Search

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.PArray
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.Search
open Test.Alea.CUDA.Extension.TestUtilities.General
open Test.Alea.CUDA.Extension.TestUtilities.MGPU
open NUnit.Framework


type Plan =
        {
            NT: int
            Bounds: int
        }


let binarySearchPartitions (bounds:int) (compOp:CompType) =
    let bs = worker.LoadPModule(MGPU.PArray.binarySearchPartitions bounds compOp).Invoke
    
    fun (count:int) (data:'TI[]) (numItems:int) (nv:int) ->
        let calc = pcalc {
            let! data = DArray.scatterInBlob worker data
            let! result = bs count data numItems nv
            return! result.Value }
        let dResult = PCalc.run calc
        dResult

[<Test>]
let ``bsp direct kernel test`` () =
    let pfunct = cuda {
        //let plan : Alea.CUDA.Extension.MGPU.Search.Plan = {NT = 64; Bounds = MgpuBoundsLower}
        //let! kbsp = (kernelBinarySearch plan CompTypeLess) |> defineKernelFunc
        let binarySearch = (binarySearchFun MgpuBoundsLower CompTypeLess).DBinarySearch
        let! kbsp =
            <@ fun (count:int) (data_global:DevicePtr<'TI>) (numItems:int) (nv:int) (partitions_global:DevicePtr<int>) (numSearches:int) ->
                let binarySearch = %binarySearch
          
                let gid = 64 * blockIdx.x + threadIdx.x
                if (gid < numSearches) then
                    let p = binarySearch data_global numItems (min (nv * gid) count)
                    partitions_global.[gid] <- p @>
            |> defineKernelFunc

        return PFunc(fun (m:Module) (indices_global:int[]) ->
            let worker = m.Worker
            let kbsp = kbsp.Apply m
            let numBlocks = divup 100 1408
            let numPartBlocks = divup (numBlocks + 1) 64
            use parts = worker.Malloc<int>(numBlocks + 1)
            use data = worker.Malloc(indices_global)
            let lp = LaunchParam(numPartBlocks, 64)
            kbsp.Launch lp 100 data.Ptr 33 1408 parts.Ptr (numBlocks + 1)
            parts.ToHost() ) }

    let pfuncm = Engine.workers.DefaultWorker.LoadPModule(pfunct)

    let indices_global = [|  1;  4;  5;  7; 10; 14; 15; 16; 18; 19;
                            27; 29; 31; 32; 33; 36; 37; 39; 50; 59;
                            60; 61; 66; 78; 81; 83; 85; 90; 91; 96;
                            97; 98; 99 |] 

    let output = pfuncm.Invoke indices_global
    printfn "%A" output

    printfn "HOST STUFF"
    let hbs = 
        fun (count:int) (data:int[]) (numItems:int) (nv:int) (parts:int List) (numSearches:int) ->
            let mutable parts = parts
            let bs = (binarySearchFun MgpuBoundsLower CompTypeLess).HBinarySearch
            for gid = 0 to 1 do
                let p = bs data numItems (min (nv * gid) count)
                parts <- parts @ [p]
            parts
    
    let mutable rp = []
    printfn "HOST PARTS:  %A" (hbs 100 indices_global 33 1408 rp 2)


[<Test>]
let ``bsp test - mgpu example`` () =
    let nv = 1408 // 11 * 128 (plan for kernelBinarySearch
    let source_global = Array.init 100 (fun i -> i)
    //printfn "Initial Values:  %A" source_global
    let sCount = source_global.Length
    printfn "Source Count: ( %d )" sCount
    printfn "*******************************************************"
    let indices_global = [|  1;  4;  5;  7; 10; 14; 15; 16; 18; 19;
                            27; 29; 31; 32; 33; 36; 37; 39; 50; 59;
                            60; 61; 66; 78; 81; 83; 85; 90; 91; 96;
                            97; 98; 99 |]    
    printfn "Indices to remove: %A" indices_global
    let nItems = indices_global.Length
    printfn "Num Items: ( %d )" nItems
    let bsp = binarySearchPartitions MgpuBoundsLower CompTypeLess
    let dResult = (bsp sCount indices_global nItems nv)

//    let hResult = Array.init 100 (fun _ -> 1)
//    for i = 0 to hRemoveIndices.Length - 1 do
//        hResult.[hRemoveIndices.[i]] <- 0
    
      
    printfn "BSP Result: %A" dResult

    //printfn "Should be:  %A" hResult