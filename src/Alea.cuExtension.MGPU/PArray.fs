[<AutoOpen>]
module Alea.cuExtension.MGPU.PArray

open Alea.CUDA
open Alea.cuExtension
open Alea.cuExtension.Util
open Alea.cuExtension.MGPU.DeviceUtil
open Alea.cuExtension.MGPU.CTAScan


//////////////////////////////////////////////////////////////////////////////////////////////
//// Bulk Insert
/// <summary></summary>
type PBulkInsert() =    
    member bi.BulkInsert() = cuda {
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


    member bi.BulkInsertFunc() = cuda {
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


//////////////////////////////////////////////////////////////////////////////////////////////
//// Bulk Remove
/// <summary></summary>
type PBulkRemove(?plan) =
    let p =
        match plan with
        | Some plan -> plan
        | None -> defaultPlan BulkRemove
    
    member br.BulkRemove() = cuda {
        let! api = BulkRemove.bulkRemove(p)    
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

    // @COMMENTS@ : for benchmark test, we need wrap it with in-place pattern, so it is just different
    // memory usage, that is also why we need a raw api and then wrap them and separate memory management
    // so strictly
    member br.BulkRemoveFunc() = cuda {
        let! api = BulkRemove.bulkRemove(p)    
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


//////////////////////////////////////////////////////////////////////////////////////////////
//// Interval Move
/// <summary></summary>
type PIntervalMove(?plan) =
    let p =
        match plan with
        | Some plan -> plan
        | None -> defaultPlan IntervalMove

    // IntervalExpand duplicates intervalCount items in values_global.
    // indices_global is an intervalCount-sized array filled with the scan of item
    // expand counts. moveCount is the total number of outputs (sum of expand 
    // counts).

    // Eg:
    //		values  =  0,  1,  2,  3,  4,  5,  6,  7,  8
    //		counts  =  1,  2,  1,  0,  4,  2,  3,  0,  2
    //		indices =  0,  1,  3,  4,  4,  8, 10, 13, 13 (moveCount = 15).
    // Expand values[i] by counts[i]:
    // output  =  0, 1, 1, 2, 4, 4, 4, 4, 5, 5, 6, 6, 6, 8, 8 
    member im.IntervalExpand() = cuda {
            let! api = IntervalMove.intervalExpand(p)            
            return PFunc(fun (m:Module) ->
                let worker = m.Worker
                let api = api.Apply m
                fun (moveCount:int) (indices:DArray<int>) (values:DArray<'T>) ->
                    let intervalCount = indices.Length
                    let api = api moveCount intervalCount
                    let itr = [|0..moveCount|]
                    pcalc {                    
                        let! partition = DArray.createInBlob<int> worker api.NumPartitions
                        let! countingItr_global = DArray.scatterInBlob worker itr
                        let! output = DArray.createInBlob<'T> worker moveCount
                        do! PCalc.action (fun hint -> api.Action hint indices.Ptr values.Ptr countingItr_global.Ptr partition.Ptr output.Ptr)
                        return output } ) }

    member im.IntervalExpandFunc() = cuda {
            let! api = IntervalMove.intervalExpand(p)            
            return PFunc(fun (m:Module) ->
                let worker = m.Worker
                let api = api.Apply m
                fun (moveCount:int) (intervalCount:int) ->
                    let itr = [|0..moveCount|]
                    pcalc {
                        let api = api moveCount intervalCount
                        let! partition = DArray.createInBlob<int> worker api.NumPartitions
                        let! countingItr_global = DArray.scatterInBlob worker itr
                        let expand (indices:DArray<int>) (values:DArray<'T>) (output:DArray<'T>) =
                            pcalc { do! PCalc.action (fun hint -> api.Action hint indices.Ptr values.Ptr countingItr_global.Ptr partition.Ptr output.Ptr) }
                        return expand } ) }

    member im.IntervalGather = ()
    member im.IntervalScatter = ()

    member im.IntervalMove() = cuda {
        let! api = IntervalMove.intervalMove(p)
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m            
            fun (total:int) (gather:DArray<int>) (scatter:DArray<int>) (counts:DArray<int>) (input:DArray<'T>) ->
                let numInputs = counts.Length
                let api = api total numInputs
                let sequence = Array.init total (fun i -> i)
                pcalc {                    
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions
                    let! countingItr = DArray.scatterInBlob worker sequence
                    let! output = DArray.createInBlob<'T> worker total
                    do! PCalc.action (fun hint -> api.Action hint gather.Ptr scatter.Ptr counts.Ptr input.Ptr countingItr.Ptr partition.Ptr output.Ptr)
                    return output } ) }

    member im.IntervalMoveFunc() = cuda {
        let! api = IntervalMove.intervalMove(p)
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m            
            fun (total:int) (numInputs:int) ->                
                let api = api total numInputs
                let sequence = Array.init total (fun i -> i)
                pcalc {                    
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions
                    let! countingItr = DArray.scatterInBlob worker sequence
                    let move (input:DArray<'T>) (gather:DArray<int>) (scatter:DArray<int>) (counts:DArray<int>) (output:DArray<'T>) = 
                        pcalc { do! PCalc.action (fun hint -> api.Action hint gather.Ptr scatter.Ptr counts.Ptr input.Ptr countingItr.Ptr partition.Ptr output.Ptr) }
                    return move } ) }


//////////////////////////////////////////////////////////////////////////////////////////////
//// Join
/// <summary></summary>
type PJoin() =
    member x.x = ()


//////////////////////////////////////////////////////////////////////////////////////////////
//// Load Balance Search
/// <summary></summary>
type PLoadBalanceSearch() =
    member plbs.Search() = cuda {
        let! api = LoadBalance.loadBalanceSearch()
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (aCount:int) (b_global:DArray<int>) ->
                let bCount = b_global.Length
                let api = api aCount bCount
                let itr = Array.init bCount (fun i -> i)
                pcalc {
                    //let! ctaCountingItr = DArray.scatterInBlob worker itr
                    //let! mpCountingItr = DArray.scatterInBlob worker itr
                    let! countingItr_global = DArray.scatterInBlob worker itr
                    let! mp_global = DArray.createInBlob<int> worker api.NumPartitions
                    let! indices_global = DArray.createInBlob<int> worker aCount                    
                    do! PCalc.action (fun hint -> api.Action hint b_global.Ptr indices_global.Ptr countingItr_global.Ptr mp_global.Ptr)                     
                    return indices_global } ) }
    

    member plbs.SearchFunc() = cuda {
        let! api = LoadBalance.loadBalanceSearch()
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (aCount:int) (bCount:int) ->
                let api = api aCount bCount
                let itr = Array.init aCount (fun i -> i)
                pcalc {
                    let! mp_global = DArray.createInBlob<int> worker api.NumPartitions
//                    let! mpCountingItr = DArray.scatterInBlob worker itr
//                    let! ctaCountingItr = DArray.scatterInBlob worker itr
                    let! countingItr_global = DArray.scatterInBlob worker itr
                    let search (b_global:DArray<int>) (indices_global:DArray<int>) =
                        pcalc { do! PCalc.action (fun hint -> api.Action hint b_global.Ptr indices_global.Ptr countingItr_global.Ptr mp_global.Ptr) }
                    return search } ) }


//////////////////////////////////////////////////////////////////////////////////////////////
//// Locality Sort
/// <summary></summary>
type PLocalitySort() =
    member x.x = ()


//////////////////////////////////////////////////////////////////////////////////////////////
//// Merge
/// <summary></summary>
type PMerge() =
    member m.MergeKeys (compOp:IComp<'TV>) = cuda {
        let! api = Merge.mergeKeys compOp
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (aCount:int) (bCount:int) -> 
                let api = api aCount bCount           
                pcalc {
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions
                    let merger (aData:DArray<'TV>) (bData:DArray<'TV>) (cData:DArray<'TV>) =
                        pcalc { do! PCalc.action (fun hint -> api.Action hint aData.Ptr bData.Ptr partition.Ptr cData.Ptr) }
                    return merger } ) }

    member m.MergePairs (compOp:IComp<'TV>) = cuda {
        let! api = Merge.mergePairs compOp
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (aCount:int) (bCount:int) -> 
                let api = api aCount bCount           
                pcalc {
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions
                    let merger (aKeys:DArray<'TV>) (aVals:DArray<'TV>) (bKeys:DArray<'TV>) (bVals:DArray<'TV>) (cKeys:DArray<'TV>) (cVals:DArray<'TV>) =
                        pcalc { do! PCalc.action (fun hint -> api.Action hint aKeys.Ptr aVals.Ptr bKeys.Ptr bVals.Ptr partition.Ptr cKeys.Ptr cVals.Ptr) }
                    return merger } ) }


//////////////////////////////////////////////////////////////////////////////////////////////
//// Mergesort
/// <summary></summary>
type PMergesort() =
    member ms.MergesortKeys() = cuda {
        let! api = Mergesort.mergesortKeys (comp CompTypeLess 0)
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (source:DArray<int>) ->
                let api = api source.Length
                pcalc {
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions                
                    let! dest = DArray.createInBlob<int> worker source.Length
                    do! PCalc.action (fun hint -> api.Action hint source.Ptr dest.Ptr partition.Ptr)
                    return dest } ) }

    member ms.MergesortKeys(compOp:IComp<'TV>) = cuda {
        let! api = Mergesort.mergesortKeys compOp
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (source:DArray<'TV>) ->
                let api = api source.Length
                pcalc {
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions                
                    let! dest = DArray.createInBlob<'TV> worker source.Length
                    do! PCalc.action (fun hint -> api.Action hint source.Ptr dest.Ptr partition.Ptr)
                    return dest } ) }

    member ms.MergesortKeysFunc(compOp:IComp<'TV>) = cuda {
        let! api = Mergesort.mergesortKeys compOp
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (count:int) ->
                let api = api count
                pcalc {
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions                
                    let merger (source:DArray<'TV>) (dest:DArray<'TV>) = 
                        pcalc { do! PCalc.action (fun hint -> api.Action hint source.Ptr dest.Ptr partition.Ptr) }
                    return merger } ) }

    member ms.MergesortPairs(keyType:'TV) = cuda {
        let! api = Mergesort.mergesortPairs (comp CompTypeLess keyType)
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (keysSource:DArray<'TV>) (valsSource:DArray<'TV>) ->
                let count = keysSource.Length
                let api = api count
                pcalc {
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions                
                    let! keysDest = DArray.createInBlob<'TV> worker count
                    let! valsDest = DArray.createInBlob<'TV> worker count
                    do! PCalc.action (fun hint -> api.Action hint keysSource.Ptr valsSource.Ptr keysDest.Ptr valsDest.Ptr partition.Ptr)
                    let! sortedKeys = keysDest.Gather()
                    let! sortedVals = valsDest.Gather()
                    return sortedKeys, sortedVals } ) }

    member ms.MergesortPairs(compOp:IComp<'TV>) = cuda {
        let! api = Mergesort.mergesortPairs compOp
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (keysSource:DArray<'TV>) (valsSource:DArray<'TV>) ->
                let count = keysSource.Length
                let api = api count
                pcalc {
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions                
                    let! keysDest = DArray.createInBlob<'TV> worker count
                    let! valsDest = DArray.createInBlob<'TV> worker count
                    do! PCalc.action (fun hint -> api.Action hint keysSource.Ptr valsSource.Ptr keysDest.Ptr valsDest.Ptr partition.Ptr)
                    return keysDest, valsDest } ) }

    member ms.MergesortPairsFunc(compOp:IComp<'TV>) = cuda {
        let! api = Mergesort.mergesortPairs compOp
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (count:int) ->
                let api = api count
                pcalc {
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions                
                    let merger (keysSource:DArray<'TV>) (valsSource:DArray<'TV>) (keysDest:DArray<'TV>) (valsDest:DArray<'TV>) = 
                        pcalc { do! PCalc.action (fun hint -> api.Action hint keysSource.Ptr valsSource.Ptr keysDest.Ptr valsDest.Ptr partition.Ptr) }
                    return merger } ) }

    member ms.MergesortIndices() = cuda {
        let! api = Mergesort.mergesortIndices (comp CompTypeLess 0)
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (keysSource:DArray<int>) (valsSource:DArray<int>) ->
                let count = keysSource.Length
                let api = api count
                let sequence = Array.init count (fun i -> i)
                pcalc {
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions
                    let! countingItr = DArray.scatterInBlob worker sequence
                    let! keysDest = DArray.createInBlob<int> worker count
                    let! valsDest = DArray.createInBlob<int> worker count
                    do! PCalc.action (fun hint -> api.Action hint keysSource.Ptr countingItr.Ptr valsSource.Ptr keysDest.Ptr valsDest.Ptr partition.Ptr)
                    return keysDest, valsDest } ) }

    member ms.MergesortIndices(compOp:IComp<int>) = cuda {
        let! api = Mergesort.mergesortIndices compOp
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (keysSource:DArray<int>) (valsSource:DArray<int>) ->
                let count = keysSource.Length
                let api = api count
                let sequence = Array.init count (fun i -> i)
                pcalc {
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions
                    let! countingItr = DArray.scatterInBlob worker sequence
                    let! keysDest = DArray.createInBlob<int> worker count
                    let! valsDest = DArray.createInBlob<int> worker count
                    do! PCalc.action (fun hint -> api.Action hint keysSource.Ptr countingItr.Ptr valsSource.Ptr keysDest.Ptr valsDest.Ptr partition.Ptr)
                    return keysDest, valsDest } ) }

    member ms.MergesortIndicesFunc(compOp:IComp<int>) = cuda {
        let! api = Mergesort.mergesortIndices compOp
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (count:int) ->
                let api = api count
                let sequence = Array.init count (fun i -> i)
                pcalc {
                    let! partition = DArray.createInBlob<int> worker api.NumPartitions
                    let! countingItr = DArray.scatterInBlob worker sequence
                    let merger (keysSource:DArray<int>) (valsSource:DArray<int>) (keysDest:DArray<int>) (valsDest:DArray<int>) = 
                        pcalc { do! PCalc.action (fun hint -> api.Action hint keysSource.Ptr countingItr.Ptr valsSource.Ptr keysDest.Ptr valsDest.Ptr partition.Ptr) }
                    return merger } ) }


//////////////////////////////////////////////////////////////////////////////////////////////
//// Reduce
/// <summary></summary>
type PReduce() =
    member r.Reduce(op:IScanOp<'TI, 'TV, 'TR>) = cuda {
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


//////////////////////////////////////////////////////////////////////////////////////////////
//// Scan
/// <summary></summary>
type PScan() =
    member ps.Scan(mgpuScanType:int, op:IScanOp<'TI, 'TV, 'TR>, totalAtEnd:int) = cuda {
        let! api = Scan.scan mgpuScanType op totalAtEnd
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (data:DArray<'TI>) ->
                let count = data.Length
                let api = api count
                pcalc {
                    let! scanned = DArray.createInBlob worker count
                    let! total = DArray.createInBlob worker 1
                    let! reductionDevice = DArray.createInBlob worker (api.NumBlocks + 1)                    
                    do! PCalc.action (fun hint -> api.Action hint data.Ptr total.Ptr reductionDevice.Ptr scanned.Ptr)
                    return total, scanned } ) }

    member ps.Scan(mgpuScanType:int, op:IScanOp<'TI, 'TV, 'TR>) = cuda {
        let! api = Scan.scan mgpuScanType op 0
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (data:DArray<'TI>) ->
                let count = data.Length
                let api = api count
                pcalc {
                    let! scanned = DArray.createInBlob worker count
                    let! total = DArray.createInBlob worker 1
                    let! reductionDevice = DArray.createInBlob worker (api.NumBlocks + 1)                    
                    do! PCalc.action (fun hint -> api.Action hint data.Ptr total.Ptr reductionDevice.Ptr scanned.Ptr)
                    return total, scanned } ) }

    member ps.Scan() = cuda {
        let! api = Scan.scan ExclusiveScan (scanOp ScanOpTypeAdd 0) 0
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (data:DArray<int>) ->
                let count = data.Length
                let api = api count
                pcalc {
                    //let! scanned = DArray.createInBlob worker count
                    let! total = DArray.createInBlob worker 1
                    let! reductionDevice = DArray.createInBlob worker (api.NumBlocks + 1)
                    do! PCalc.action (fun hint -> api.Action hint data.Ptr total.Ptr reductionDevice.Ptr data.Ptr)
                    let! scanned = data.Gather()
                    let! total = total.Gather()
                    return total.[0], scanned } ) }

    member ps.Scan(op:IScanOp<'TI,'TV,'TR>) = ps.Scan(ExclusiveScan, op)       

    member ps.ScanFunc(mgpuScanType:int, op:IScanOp<'TI, 'TV, 'TR>, totalAtEnd:int) = cuda {
        let! api = Scan.scan mgpuScanType op totalAtEnd
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (numElements:int) ->
                let api = api numElements
                pcalc {
                    let! reductionDevice = DArray.createInBlob worker (api.NumBlocks + 1)                    
                    let scanner (data:DArray<'TI>) (scanned:DArray<'TR>) (total:DArray<'TV>) =
                        pcalc { do! PCalc.action (fun hint -> api.Action hint data.Ptr total.Ptr reductionDevice.Ptr scanned.Ptr)}
                    return scanner } ) }
                    
    member ps.ScanFunc(op:IScanOp<'TI, 'TV, 'TR>) = ps.ScanFunc(ExclusiveScan, op, 0)    

    member ps.ScanFuncReturnTotal(op:IScanOp<'TI, 'TV, 'TR>) = cuda {
        let! api = Scan.scan ExclusiveScan op 0
        return PFunc(fun (m:Module) ->
            let worker = m.Worker
            let api = api.Apply m
            fun (numElements:int) ->
                let api = api numElements
                pcalc {
                    let! total = DArray.createInBlob worker 1
                    let! reductionDevice = DArray.createInBlob worker (api.NumBlocks + 1)                    
                    let scanner (data:DArray<'TI>) (scanned:DArray<'TR>) =
                        pcalc { 
                                do! PCalc.action (fun hint -> api.Action hint data.Ptr total.Ptr reductionDevice.Ptr scanned.Ptr)
                                let! total = total.Gather()
                                return total.[0] }                        
                    return scanner } ) }


//////////////////////////////////////////////////////////////////////////////////////////////
//// Search
/// <summary></summary>
type PSearch() =
    member s.BinarySearchPartitions(bounds:int, compOp:IComp<int>) = cuda {
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


//////////////////////////////////////////////////////////////////////////////////////////////
//// Segmented Sort
/// <summary></summary>
type PSegmentedSort() =
    member x.x = ()


//////////////////////////////////////////////////////////////////////////////////////////////
//// Sets
/// <summary></summary>
type PSets() =
    member x.x = ()


//////////////////////////////////////////////////////////////////////////////////////////////
//// Sorted Search
/// <summary></summary>
type PSortedSearch() =
    member pss.SortedSearch(bounds:int, typeA:MgpuSearchType, typeB:MgpuSearchType, compOp:IComp<'TI>) = cuda {
        let! api = SortedSearch.sortedSearch bounds typeA typeB compOp
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
        let! api = SortedSearch.sortedSearch bounds MgpuSearchTypeIndex MgpuSearchTypeNone compOp
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

