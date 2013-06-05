module Alea.CUDA.Extension.MGPU.PArray

open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.CTAScan
//open Alea.CUDA.Extension.MGPU.BuklRemove

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
                //printfn "COUNT: %d" count
                let! scanned = DArray.createInBlob worker count
                let! total = DArray.createInBlob worker 1
                
                do! PCalc.action (fun hint -> api.Action hint data.Ptr total.Ptr scanned.Ptr )
                let result =
                    fun () ->
                        pcalc {
                            let! scanned = scanned.Gather()
                            let! total = total.Gather()
                            //printfn "Scanned - from PFunc in PArray: %A" scanned
                            //printfn "Total - from PFunc in PArray: %A" total
                            return scanned }
                    |> Lazy.Create
                return result} ) }


let binarySearchPartitions (bounds:int) (compOp:CompType) = cuda {
    let! api = Search.binarySearchPartitions bounds compOp

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m

        fun (source_global:DevicePtr<'TI>) (count:int) (indicesCount:int) (nv:int) ->
            pcalc {
                
                let api = api count indicesCount nv
                //printfn "COUNT: %d" count
                //let! data = DArray.createInBlob worker count
                                
                do! PCalc.action (fun hint -> api.Action hint source_global)
                let result =
                    fun () ->
                        pcalc {
                            //printfn "DATA - from PFunc in PArray: %A" derp
                            let partitionsDevice = api.Partitions
                            //printfn "Partitions Device - from PFunc in PArray: %A" partitionsDevice                            
                            return partitionsDevice }
                    |> Lazy.Create
                return result} ) }

let bulkRemove = cuda {
    let! api = BulkRemove.bulkRemove

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m

        fun (data:DArray<'TI>) (indices:DArray<int>) ->
            pcalc {
                let sourceCount = data.Length
                let indicesCount = indices.Length
                let api = api sourceCount indicesCount

                let! dest = DArray.createInBlob worker (sourceCount - indicesCount)

                do! PCalc.action (fun hint -> api.Action hint data.Ptr indices.Ptr dest.Ptr)
                let result =
                    fun () ->
                        pcalc {
                            let! indRemoved = dest.Gather()
                            return indRemoved }
                    |> Lazy.Create
                return result } ) }