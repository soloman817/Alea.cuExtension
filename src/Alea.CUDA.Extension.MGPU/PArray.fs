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
        fun (data:DArray<int>) ->
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
        
        fun (count:int) (data_global:DArray<'TI>) (numItems:int) (nv:int) ->
            pcalc {                
                let api = api count numItems nv
                let n = ((divup count nv) + 1)
                let! partData = DArray.createInBlob<'TI> worker n
                do! PCalc.action (fun hint -> api.Action hint data_global.Ptr partData.Ptr)
                                    
                let result =
                    fun () ->
                        pcalc { 
                            let! parts = partData.Gather()
                            return parts }
                    |> Lazy.Create
                return result} ) }



//let bulkRemove() = cuda {
//    let! api = BulkRemove.bulkRemove()
//    
//    return PFunc(fun (m:Module) ->
//        let worker = m.Worker
//        let api = api.Apply m
//        fun (data:DArray<'TI>) (indices:DArray<int>) ->
//            let sourceCount = data.Length
//            let indicesCount = indices.Length
//            let api = api sourceCount indicesCount
//            pcalc {
//                let! removed = DArray.createInBlob<'TI> worker (sourceCount - indicesCount)
//                let remover =
//                    fun () ->
//                        pcalc {do! PCalc.action (fun hint -> api.Action hint data.Ptr indices.Ptr removed.Ptr)}
//                    |> Lazy.Create
//                return remover, removed } ) }

let bulkRemove() = cuda {
    let! api = BulkRemove.bulkRemove()
    
    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m
        fun (data:DArray<'TI>) (indices:DArray<int>) ->
            let sourceCount = data.Length
            let indicesCount = indices.Length
            let api = api sourceCount indicesCount
            pcalc {
                // @COMMENTS@ : you can just add the action, and return the DArray
                // the DArray is lazy, you just enqueued action, but not executed
                // only when you call removed.Gather(), it will trigger all actions
                let! removed = DArray.createInBlob<'TI> worker (sourceCount - indicesCount)
                do! PCalc.action (fun hint -> api.Action hint data.Ptr indices.Ptr removed.Ptr)
                return removed } ) }

