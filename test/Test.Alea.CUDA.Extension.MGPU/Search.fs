module Test.Alea.CUDA.Extension.MGPU.Search

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.PArray
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.CTAMerge
open Alea.CUDA.Extension.MGPU.Search
open Test.Alea.CUDA.Extension.MGPU.Util
open NUnit.Framework


type Plan =
        {
            NT: int
            Bounds: int
        }


type IMergePathPartitions<'T> =
    {
        Action : ActionHint -> DevicePtr<'T> -> DevicePtr<'T> -> DevicePtr<int> -> unit                
    }


let worker = getDefaultWorker()

let binarySearchPartitions (bounds:int) (compOp:IComp<int>) =
    
    let bs = worker.LoadPModule(MGPU.PArray.binarySearchPartitions bounds compOp).Invoke
    
    fun (count:int) (data:int[]) (numItems:int) (nv:int) ->
        let calc = pcalc {
            let! data = DArray.scatterInBlob worker data
            let! result = bs count data numItems nv
            return! result.Value }
        let dResult = PCalc.run calc
        dResult

[<Test>]
let ``bsp direct kernel test`` () =
    let pfunct = cuda {
        let binarySearch = (binarySearchFun MgpuBoundsLower (comp CompTypeLess 0)).DBinarySearch
        let! kbsp =
            <@ fun (count:int) (data_global:DevicePtr<int>) (numItems:int) (nv:int) (partitions_global:DevicePtr<int>) (numSearches:int) ->
                let binarySearch = %binarySearch
          
                let gid = 64 * blockIdx.x + threadIdx.x
                if (gid < numSearches) then
                    let p = binarySearch (data_global.Reinterpret<int>()) numItems (min (nv * gid) count)
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
            let bs = (binarySearchFun MgpuBoundsLower (comp CompTypeLess 0)).HBinarySearch
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
    let bsp = binarySearchPartitions MgpuBoundsLower (comp CompTypeLess 0)
    let dResult = (bsp sCount indices_global nItems nv)

//    let hResult = Array.init 100 (fun _ -> 1)
//    for i = 0 to hRemoveIndices.Length - 1 do
//        hResult.[hRemoveIndices.[i]] <- 0
    
      
    printfn "BSP Result: %A" dResult

    //printfn "Should be:  %A" hResult


[<Test>]
let ``merge path partitions for bulk insert`` () =
    let kernelMergePartition (plan:Plan) (compOp:IComp<'TI>) = 
        let NT = plan.NT
        let bounds = plan.Bounds
        let mergePath = (mergeSearch bounds compOp).DMergePath
        let findMergesortFrame = findMergesortFrame.Device

        <@ fun (a_global:DevicePtr<'TI>) (aCount:int) (b_global:DevicePtr<'TI>) (bCount:int) (nv:int) (coop:int) (mp_global:DevicePtr<int>) (numSearches:int) ->
            let mergePath = %mergePath
            let findMergesortFrame = %findMergesortFrame
                        
            let mutable aCount = aCount
            let mutable bCount = bCount
            let mutable a0 = 0
            let mutable b0 = 0
            
            let partition = NT * blockIdx.x * threadIdx.x
            if partition < numSearches then                
                let mutable gid = nv * partition
                // coop always 0 for bulk insert so I deleted that part for testing
                let mp = mergePath (a_global + a0) aCount (b_global + b0) bCount (min gid (aCount + bCount))
                mp_global.[partition] <- mp @>

                  

    let mergePathPartitions (bounds:int) (compOp:IComp<'T>) = cuda {
        let plan = { NT = 64; Bounds = bounds }
        let! kernelMergePartition = (kernelMergePartition plan compOp) |> defineKernelFuncWithName "mpp"

        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let kernelMergePartition = kernelMergePartition.Apply m

            fun (aCount:int) (bCount:int) (nv:int) (coop:int) ->
                let numPartitions = divup (aCount + bCount) nv
                let numPartitionBlocks = divup (numPartitions + 1) plan.NT
                let lp = LaunchParam(numPartitionBlocks, plan.NT)
            
                let action (hint:ActionHint) (a_global:DevicePtr<'T>) (b_global:DevicePtr<'T>) (partitionsDevice:DevicePtr<int>) =
                    let lp = lp |> hint.ModifyLaunchParam
                    kernelMergePartition.Launch lp a_global aCount b_global bCount nv coop partitionsDevice (numPartitions + 1)
                
                { Action = action } ) }
    
        
    let mPP (bounds:int) (compOp:IComp<int>) = cuda {
        let! api = mergePathPartitions bounds compOp

        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m

            fun (aCount:int) (bCount:int) (nv:int) (coop:int) (aGlobal:DArray<int>) (bGlobal:DArray<int>) ->
                                                              (*aka indices_global *) (*aka 0 counting itr*)
                pcalc {
                    let api = api aCount bCount nv coop
                    let NT = 64
                    let numPartitions = divup (aCount + bCount) nv
                    let numPartitionBlocks = divup (numPartitions + 1) NT

                    let! parts = DArray.createInBlob<int> worker (numPartitions + 1)
                    do! PCalc.action (fun hint -> api.Action hint aGlobal.Ptr bGlobal.Ptr parts.Ptr)

                    return parts } ) }


    let pfunct = mPP MgpuBoundsLower (comp CompTypeLess 0)
    let mpp = worker.LoadPModule(pfunct).Invoke

    let hDataSource = Array.init 100 int
    let hIndices = [|2..5..100|]
    let hDataToInsert = [|1000..10..((hIndices.Length*10+1000)-10)|]

    let dResult = pcalc {
        let! dIndices = DArray.scatterInBlob worker hIndices
        let! zeroCountingItr = DArray.scatterInBlob worker hIndices
        
        let aCount = hDataToInsert.Length
        let bCount = hDataSource.Length
        let nv = 128 * 7
        
        let! partitions = mpp aCount bCount nv 0 dIndices zeroCountingItr
        let! results = partitions.Gather()

        return results } |> PCalc.run

    printfn "%A" dResult