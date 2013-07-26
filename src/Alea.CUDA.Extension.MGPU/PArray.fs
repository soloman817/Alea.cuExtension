module Alea.CUDA.Extension.MGPU.PArray

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.CTAScan


// one wrapper for reduce, in this wrapper, the memory is created on the
// fly.
let reduce (op:IScanOp<'TI, 'TV, 'TR>) = cuda {
    let! api = Reduce.reduce op

    return PFunc(fun (m:Module) ->
        
        let worker = m.Worker
        let api = api.Apply m
        fun (data:DArray<'TI>) ->
            pcalc {
                let count = data.Length
                let api = api count
                let! reduction = DArray.createInBlob worker api.NumBlocks
                do! PCalc.action (fun hint -> api.Action hint data.Ptr reduction.Ptr)
                let result =
                    fun () ->
                        pcalc {
                            let! reduction = reduction.Gather()
                            return api.Result reduction }
                    |> Lazy.Create
                return result} ) }


let scan (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) = cuda {
    let! api = Scan.scan mgpuScanType op totalAtEnd
    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m
        fun (data:DArray<'TI>) ->
            pcalc {
                let count = data.Length
                let api = api count
                let! scanned = DArray.createInBlob worker count
                let! total = DArray.createInBlob worker 1
                let scanner =
                    fun () ->
                        pcalc {do! PCalc.action (fun hint -> api.Action hint data.Ptr total.Ptr scanned.Ptr )}
                    |> Lazy.Create
                return scanner, scanned} ) }



let binarySearchPartitions (bounds:int) (compOp:IComp<int>) = cuda {
    let! api = Search.binarySearchPartitions bounds compOp

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m
        
        fun (count:int) (data_global:DArray<int>) (numItems:int) (nv:int) ->
            pcalc {                
                let api = api count numItems nv
                let n = ((divup count nv) + 1)
                let! partData = DArray.createInBlob<int> worker n
                do! PCalc.action (fun hint -> api.Action hint data_global.Ptr partData.Ptr)
                                    
                let result =
                    fun () ->
                        pcalc { 
                            let! parts = partData.Gather()
                            return parts }
                    |> Lazy.Create
                return result} ) }



let bulkRemove() = cuda {
    let! api = BulkRemove.bulkRemove()
    
    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m
        fun (data:DArray<'TI>) (indices:DArray<int>) ->
            let sourceCount = data.Length
            let api = api sourceCount
            let indicesCount = indices.Length
            pcalc {
                // @COMMENTS@ : you need prepare paration memory for internal usage
                let! partition = DArray.createInBlob<int> worker api.NumPartitions

                // @COMMENTS@ : you can just add the action, and return the DArray
                // the DArray is lazy, you just enqueued action, but not executed
                // only when you call removed.Gather(), it will trigger all actions
                let! removed = DArray.createInBlob<'TI> worker (sourceCount - indicesCount)
                do! PCalc.action (fun hint -> api.Action hint indicesCount partition.Ptr data.Ptr indices.Ptr removed.Ptr)
                return removed } ) }


let bulkInsert() = cuda {
    let! api = BulkInsert.bulkInsert() 

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m
        fun (data_A:DArray<'TI>) (indices:DArray<int>) (data_B:DArray<'TI>) ->
            let aCount = data_A.Length
            let bCount = data_B.Length
            let api = api aCount bCount
            let sequence = Array.init bCount (fun i -> i)
            pcalc {
                let! partition = DArray.createInBlob<int> worker api.NumPartitions
                let! counter = DArray.scatterInBlob worker sequence
                let! inserted = DArray.createInBlob<'TI> worker (aCount + bCount)
                do! PCalc.action (fun hint -> api.Action hint data_A.Ptr indices.Ptr counter.Ptr data_B.Ptr partition.Ptr inserted.Ptr)
                return inserted } ) }


type PSortedSearch() =
    member pss.SortedSearch(bounds:int, typeA:MgpuSearchType, typeB:MgpuSearchType, compOp:IComp<'TI>) = cuda {
        let! api = SortedSearch.sortedSearch bounds typeA typeB compOp (scanOp ScanOpTypeAdd compOp.Identity)
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (aCount:int) (bCount:int) ->                
                pcalc {
                    let api = api aCount bCount
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions
                    let sortedSearch (aData:DArray<'TI>) (bData:DArray<'TI>) (aIndices:DArray<int>) (bIndices:DArray<int>) = 
                        pcalc { do! PCalc.action (fun hint -> api.Action hint aData.Ptr bData.Ptr partition.Ptr aIndices.Ptr bIndices.Ptr) }
                    return sortedSearch } ) }  

    member pss.SortedSearch(bounds:int, compOp:IComp<'TI>) = cuda {
        let! api = SortedSearch.sortedSearch bounds MgpuSearchTypeIndex MgpuSearchTypeNone compOp (scanOp ScanOpTypeAdd compOp.Identity)
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (aCount:int) (bCount:int) ->                
                pcalc {
                    let api = api aCount bCount
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions
                    let sortedSearch (aData:DArray<'TI>) (bData:DArray<'TI>) (aIndices:DArray<int>) = 
                        pcalc { do! PCalc.action (fun hint -> api.Action hint aData.Ptr bData.Ptr partition.Ptr aIndices.Ptr (DevicePtr<int>(0n)) ) }
                    return sortedSearch } ) } 
                     
    member pss.SortedSearch(bounds:int, ident:'TI) = pss.SortedSearch(bounds, (comp CompTypeLess ident))
        

let mergeKeys (compOp:IComp<'TK>) = cuda {
    let! api = Merge.mergeKeys compOp

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m
        fun (aCount:int) (bCount:int) -> 
            let api = api aCount bCount           
            pcalc {
                let! partition = DArray.createInBlob<int> worker api.NumPartitions
                let merger (aData:DArray<'TK>) (bData:DArray<'TK>) (cData:DArray<'TK>) =
                    pcalc { do! PCalc.action (fun hint -> api.Action hint aData.Ptr bData.Ptr partition.Ptr cData.Ptr) }
                return merger } ) }

let mergePairs (compOp:IComp<'TK>) = cuda {
    let! api = Merge.mergePairs compOp

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m
        fun (aCount:int) (bCount:int) -> 
            let api = api aCount bCount           
            pcalc {
                let! partition = DArray.createInBlob<int> worker api.NumPartitions
                let merger (aKeys:DArray<'TK>) (aVals:DArray<'TV>) (bKeys:DArray<'TK>) (bVals:DArray<'TV>) (cKeys:DArray<'TK>) (cVals:DArray<'TV>) =
                    pcalc { do! PCalc.action (fun hint -> api.Action hint aKeys.Ptr aVals.Ptr bKeys.Ptr bVals.Ptr partition.Ptr cKeys.Ptr cVals.Ptr) }
                return merger } ) }


let scanInPlace (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) = cuda {
    let! api = Scan.scan mgpuScanType op totalAtEnd

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m
        fun (numElements:int) ->
            let api = api numElements
            pcalc {
                let! total = DArray.createInBlob worker 1
                let scanner (data:DArray<'TI>) (scanned:DArray<'TR>) =
                    pcalc { do! PCalc.action (fun hint -> api.Action hint data.Ptr total.Ptr scanned.Ptr) }
                return scanner } ) }


// @COMMENTS@ : for benchmark test, we need wrap it with in-place pattern, so it is just different
// memory usage, that is also why we need a raw api and then wrap them and separate memory management
// so strictly
let bulkRemoveInPlace() = cuda {
    let! api = BulkRemove.bulkRemove()
    
    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m

        // for a in-place remover, first we need create it, such as internal partition memories which
        // could be reused, and for that we need know the source count
        fun (sourceCount:int) ->
            let api = api sourceCount 
            pcalc {
                // first, we need create partition memory for internal use
                let! partition = DArray.createInBlob<int> worker api.NumPartitions

                // now we return a function, which is in-place remover
                let remove (data:DArray<'T>) (indices:DArray<int>) (removed:DArray<'T>) =
                    pcalc { do! PCalc.action (fun hint -> api.Action hint indices.Length partition.Ptr data.Ptr indices.Ptr removed.Ptr) }

                return remove } ) }


let bulkInsertInPlace() = cuda {
    let! api = BulkInsert.bulkInsert()
    
    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m

        fun (aCount:int) (bCount:int) ->
            let sequence = Array.init bCount (fun i -> i)
            pcalc {
                let api = api aCount bCount 
                
                let! counter = DArray.scatterInBlob worker sequence
                let! partition = DArray.createInBlob<int> worker api.NumPartitions

                let insert (data_A:DArray<'T>) (indices:DArray<int>) (data_B:DArray<'T>) (inserted:DArray<'T>) =    
                    pcalc { do! PCalc.action (fun hint -> api.Action hint data_A.Ptr indices.Ptr counter.Ptr data_B.Ptr partition.Ptr inserted.Ptr) }

                return insert } ) }


