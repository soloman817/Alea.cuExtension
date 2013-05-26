module Alea.CUDA.Extension.MGPU.PArray

open Alea.CUDA
open Alea.CUDA.Extension
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


let scan (op:IScanOp<'TI, 'TV, 'TR>) = cuda {
    let! api = Scan.scan op
    

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m
        
        
        fun (data:DArray<'TI>) ->
            pcalc {
                let count = data.Length
                let api = api count
                let! reduction = DArray.createInBlob worker api.NumBlocks
                let! scanned = DArray.createInBlob worker count
                let! total = DArray.createInBlob worker 1
                let! end' = DArray.createInBlob worker 1
                do! PCalc.action (fun hint -> api.Action hint data.Ptr reduction.Ptr total.Ptr end'.Ptr scanned.Ptr)
                let result =
                    fun () ->
                        pcalc {
                            let! scanned = scanned.Gather()
                            let! total = total.Gather()
                            return api.Result total.[0] scanned }
                    |> Lazy.Create
                return result} ) }