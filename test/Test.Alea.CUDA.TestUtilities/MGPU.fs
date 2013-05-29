module Test.Alea.CUDA.Extension.TestUtilities.MGPU
open System
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Scan
open Alea.CUDA.Extension.MGPU.CTAScan
open NUnit.Framework
open Test.Alea.CUDA.Extension.TestUtilities.General


let testScanStats (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) =
        let scan = worker.LoadPModule(PArray.scan op totalAtEnd).Invoke
        
        fun (data:'TI[]) ->
            let calc = pcalc {
                let! data = DArray.scatterInBlob worker data
                let! result = scan data
                return! result.Value }
            runForStats calc |> ignore
            

let testScanVerify (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) (displayOutput:bool) =
    let scan = worker.LoadPModule(PArray.scan op totalAtEnd).Invoke
    fun (gold:'TI[] -> 'TV[]) (verify: 'TV[] -> 'TV[] -> unit) (data:'TI[]) ->
        let calc = pcalc {
            let! data = DArray.scatterInBlob worker data
            let! result = scan data
            return! result.Value }
        let hOutput = gold data
        let dOutput = PCalc.run calc
        if displayOutput then displayHandD hOutput dOutput
        verify hOutput dOutput
