module Test.Alea.CUDA.Extension.MGPU.Debug

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Intrinsics
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTAScan
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.CTAMerge
open Alea.CUDA.Extension.MGPU.CTASortedSearch
open Alea.CUDA.Extension.MGPU.MergeSort

open NUnit.Framework


let worker = getDefaultWorker()

let sortKeys (compOp:IComp<'TV>) = cuda {
    let! api = MergeSort.mergesortKeys compOp

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let api = api.Apply m
        fun (count:int) ->
            let api = api count
            pcalc {
                let! partition = DArray.createInBlob<int> worker api.NumPartitions
                let merger (source:DArray<'TV>) (dest:DArray<'TV>) =
                    pcalc { do! PCalc.action (fun hint -> api.Action hint source.Ptr partition.Ptr dest.Ptr) }
                return merger } ) }



[<Test>]
let `` simple MergeSort Keys test`` () =
    let compOp = (comp CompTypeLess 0)
    let pfunct = worker.LoadPModule(sortKeys compOp).Invoke
    let count = 100
    let data = Array.init count (fun _ -> rng.Next(count - 1))

    let dResult = pcalc {
        let! dSource = DArray.scatterInBlob worker data
        let! dDest = DArray.createInBlob worker count
        let! mergesort = pfunct count
        do! mergesort dSource dDest
        let! results = dDest.Gather()
        return results } |> PCalc.runInWorker worker

    printfn "%A" dResult

