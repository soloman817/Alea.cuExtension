module Test.Alea.CUDA.Extension.MGPU.Search

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.PArray
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.CTAMerge
open Alea.CUDA.Extension.MGPU.CTAScan
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


let bCount, aCount = 400,100
let hI = Array.init aCount (fun _ -> rng.Next(300)) |> Array.sort
let NT = 64
let nv = NT * 2 + 1
let numPartitions = divup (aCount + bCount) nv
let numPartitionBlocks = divup (numPartitions + 1) NT      
let hZ = [|0..bCount|]          

let mP (a:int[]) (aCount:int) (b:int[]) (bCount:int) (diag:int) =
        let mutable begin' = max 0 (diag - bCount)
        let mutable end' = min diag aCount
        
        while begin' < end' do
            let mid = (begin' + end') >>> 1
            let aKey = a.[mid]
            let bKey = b.[diag - 1 - mid]

            let pred = not (bKey < aKey)
            if pred then 
                begin' <- mid + 1
            else
                end' <- mid
        begin'
    
let hP = Array.init (numPartitions + 1) (fun _ -> 0)
let kmPP (lp:int*int) (a_global:int[]) (aCount:int) (b_global:int[]) (bCount:int) (nv:int) (coop:int) (numSearches:int) =
    for i = 0 to (fst lp) - 1 do
        for j = 0 to (snd lp) - 1 do
            let partition = NT * i * j
                
            if partition < numSearches then                
                let a0 = 0
                let b0 = 0
                let gid = nv * partition
                let mp = mP a_global aCount b_global bCount (min gid (aCount + bCount))
                Array.set hP partition mp

kmPP (numPartitionBlocks, NT) hI aCount hZ bCount nv 0 (numPartitions + 1)

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
    let mergeSearch (compOp:IComp<int>) =
        let comp = compOp.Device
        <@ fun (a:RWPtr<int>) (aCount:int) (b:RWPtr<int>) (bCount:int) (diag:int) ->
            let comp = %comp
            let mutable begin' = max 0 (diag - bCount)
            let mutable end' = min diag aCount

            while begin' < end' do
                let mid = (begin' + end') >>> 1
                let aKey = a.[mid]
                let bKey = b.[diag - 1 - mid]

                let pred = not (bKey < aKey)
                if pred then 
                    begin' <- mid + 1
                else
                    end' <- mid
            begin' @>

    let pfunct (plan:Plan) (compOp:IComp<int>) = cuda {
        let NT = plan.NT
        let bounds = plan.Bounds
        let mergePath = mergeSearch compOp
        let! kernel =
            <@ fun (a_global:DevicePtr<int>) (aCount:int) (b_global:DevicePtr<int>) (bCount:int) (nv:int) (coop:int) (mp_global:DevicePtr<int>) (numSearches:int) ->
                let mergePath = %mergePath         
                        
                let partition = NT * blockIdx.x * threadIdx.x
                if partition < numSearches then                
                    let a0 = 0
                    let b0 = 0
                    let gid = nv * partition
                    let mp = mergePath (a_global + a0) aCount (b_global + b0) bCount (min gid (aCount + bCount))
                    mp_global.[partition] <- mp 
                @> |> defineKernelFunc

        return PFunc(fun (m:Module) ->
            let N = 400
            
            let hDataSource = Array.init N int
            let hInsertIndices = Array.init 100 (fun _ -> rng.Next(300)) |> Array.sort
            let InsertCount = hInsertIndices.Length
            let NT = 64
            let nv = NT * 2 + 1
            let numPartitions = divup (InsertCount + N) nv
            //printfn "nv = %d; numParts = %d" nv numPartitions
            let numPartitionBlocks = divup (numPartitions + 1) NT
            use partitionsDevice = m.Worker.Malloc(numPartitions + 1)
            use dIndices = m.Worker.Malloc(hInsertIndices)
            use zeroItr = m.Worker.Malloc([|0..N|])
            let lp = LaunchParam(numPartitionBlocks, NT)
            printfn "nv = %d\nnumPartitions = %d\nnumPartitionBlocks = %d\nInsertCount = %d\nN = %d"
                nv
                numPartitions
                numPartitionBlocks
                InsertCount
                N
            kernel.Launch m lp dIndices.Ptr InsertCount zeroItr.Ptr N nv 0 partitionsDevice.Ptr (numPartitions + 1)
            partitionsDevice.ToHost() ) }

            

    let pfunct = pfunct ({NT = 64; Bounds = MgpuBoundsLower}) (comp CompTypeLess 0)

    let pfuncm = Engine.workers.DefaultWorker.LoadPModule(pfunct)

    let output = pfuncm.Invoke
    pfuncm.Dispose()
    for i = 0 to output.Length - 1 do
        printfn "idx = %d; ( %d )" i output.[i]



[<Test>]
let ``merge path partitions for bulk insert # 2`` () =
    let mergeSearch (compOp:IComp<int>) =
        let comp = compOp.Device
        <@ fun (a:RWPtr<int>) (aCount:int) (b:RWPtr<int>) (bCount:int) (diag:int) ->
            let comp = %comp
            let mutable begin' = max 0 (diag - bCount)
            let mutable end' = min diag aCount

            while begin' < end' do
                let mid = (begin' + end') >>> 1
                let aKey = a.[mid]
                let bKey = b.[diag - 1 - mid]

                let pred = not (bKey < aKey)
                if pred then 
                    begin' <- mid + 1
                else
                    end' <- mid
            begin' @>

    let pfunct (plan:Plan) (compOp:IComp<int>) = cuda {
        let NT = plan.NT
        let bounds = plan.Bounds
        let mergePath = mergeSearch compOp
        let! kernel =
            <@ fun (a_global:DevicePtr<int>) (aCount:int) (b_global:DevicePtr<int>) (bCount:int) (nv:int) (coop:int) (mp_global:DevicePtr<int>) (numSearches:int) ->
                let mergePath = %mergePath         
                        
                let partition = NT * blockIdx.x * threadIdx.x
                if partition < numSearches then                
                    let a0 = 0
                    let b0 = 0
                    let gid = nv * partition
                    let mp = mergePath (a_global + a0) aCount (b_global + b0) bCount (min gid (aCount + bCount))
                    mp_global.[partition] <- mp 
                @> |> defineKernelFunc

        return PFunc(fun (m:Module) ->
            let bCount = 400
            let aCount = 100

            let hI = Array.init aCount (fun _ -> rng.Next(300)) |> Array.sort
            
            let NT = 64
            let nv = NT * 2 + 1
            
            let numPartitions = divup (aCount + bCount) nv
            let numPartitionBlocks = divup (numPartitions + 1) NT
            
            use partitionsDevice = m.Worker.Malloc(numPartitions + 1)
            use dI = m.Worker.Malloc(hI)
            use dZ = m.Worker.Malloc([|0..bCount|])
            
            let lp = LaunchParam(numPartitionBlocks, NT)
            
            printfn "nv = %d\nnumPartitions = %d\nnumPartitionBlocks = %d\nInsertCount = %d\nN = %d"
                nv
                numPartitions
                numPartitionBlocks
                aCount
                bCount
            
            kernel.Launch m lp dI.Ptr aCount dZ.Ptr bCount nv 0 partitionsDevice.Ptr (numPartitions + 1)
            
            partitionsDevice.ToHost() ) }

            

    let pfunct = pfunct ({NT = 64; Bounds = MgpuBoundsLower}) (comp CompTypeLess 0)

    let pfuncm = Engine.workers.DefaultWorker.LoadPModule(pfunct)

    let output = pfuncm.Invoke
    pfuncm.Dispose()
    for i = 0 to output.Length - 1 do
        printfn "idx = %d; ( %d )" i output.[i]


[<Test>]
let ``mpp for bulk insert, follows benchmark`` () =
    let mergeSearch (compOp:IComp<int>) =
        let comp = compOp.Device
        <@ fun (a:RWPtr<int>) (aCount:int) (b:RWPtr<int>) (bCount:int) (diag:int) ->
            let comp = %comp
            let mutable begin' = max 0 (diag - bCount)
            let mutable end' = min diag aCount

            while begin' < end' do
                let mid = (begin' + end') >>> 1
                let aKey = a.[mid]
                let bKey = b.[diag - 1 - mid]

                let pred = not (bKey < aKey)
                if pred then 
                    begin' <- mid + 1
                else
                    end' <- mid
            begin' @>

    let pfunct (plan:Plan) (compOp:IComp<int>) = cuda {
        let NT = plan.NT
        let bounds = plan.Bounds
        let mergePath = mergeSearch compOp
        let! kernel =
            <@ fun (a_global:DevicePtr<int>) (aCount:int) (b_global:DevicePtr<int>) (bCount:int) (nv:int) (coop:int) (mp_global:DevicePtr<int>) (numSearches:int) ->
                let mergePath = %mergePath         
                        
                let partition = NT * blockIdx.x * threadIdx.x
                if partition < numSearches then                
                    let a0 = 0
                    let b0 = 0
                    let gid = nv * partition
                    let mp = mergePath (a_global + a0) aCount (b_global + b0) bCount (min gid (aCount + bCount))
                    mp_global.[partition] <- mp 
                @> |> defineKernelFunc

        return PFunc(fun (m:Module) ->
            let bCount = 400
            let aCount = 100 
                       
            let hI = Array.init aCount (fun _ -> rng.Next(300)) |> Array.sort
            let iCt = hI.Length
            
            let NT = 64
            let nv = NT * 2 + 1
            let numPartitions = divup (aCount + bCount) nv
            let numPartitionBlocks = divup (numPartitions + 1) NT

            use dI = m.Worker.Malloc(hI)
            use dZ = m.Worker.Malloc([|0..bCount|])
            use partitionsDevice = m.Worker.Malloc(numPartitions + 1)
            
            let lp = LaunchParam(numPartitionBlocks, NT)
            printfn "nv = %d\nnumPartitions = %d\nnumPartitionBlocks = %d\nInsertCount = %d\nN = %d"
                nv
                numPartitions
                numPartitionBlocks
                aCount
                bCount
            kernel.Launch m lp dI.Ptr iCt dZ.Ptr bCount nv 0 partitionsDevice.Ptr (numPartitions + 1)
            partitionsDevice.ToHost() ) }

      
    let pfunct = pfunct ({NT = 64; Bounds = MgpuBoundsLower}) (comp CompTypeLess 0)

    let pfuncm = Engine.workers.DefaultWorker.LoadPModule(pfunct)

    let dOutput = pfuncm.Invoke
    pfuncm.Dispose()

    printfn "\nDevice Output:"
    for i = 0 to dOutput.Length - 1 do
        printfn "idx = %d; ( %d )" i dOutput.[i]

    printfn "\nHost Output:"
    for i = 0 to hP.Length - 1 do
        printfn "idx = %d; ( %d )" i hP.[i]