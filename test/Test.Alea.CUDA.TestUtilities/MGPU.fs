module Test.Alea.CUDA.Extension.TestUtilities.MGPU
open System
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Scan
open Alea.CUDA.Extension.MGPU.CTAScan
open NUnit.Framework
open Test.Alea.CUDA.Extension.TestUtilities.General

/////////////////////////////////////////////////////////////////////////////////////////////
//                                  SCAN
module ScanUtils =
    let testScanStats (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) =
            let scan = worker.LoadPModule(PArray.scan mgpuScanType op totalAtEnd).Invoke
        
            fun (data:'TI[]) ->
                let calc = pcalc {
                    let! data = DArray.scatterInBlob worker data
                    let! result = scan data
                    return! result.Value }
                runForStats calc |> ignore            

    let testScanVerify (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) (displayOutput:bool) =
        let scan = worker.LoadPModule(PArray.scan mgpuScanType op totalAtEnd).Invoke
        fun (gold:'TI[] -> 'TV[]) (verify: 'TV[] -> 'TV[] -> unit) (data:'TI[]) ->
            let calc = pcalc {
                let! data = DArray.scatterInBlob worker data
                let! result = scan data
                return! result.Value }
            let hOutput = gold data
            let dOutput = PCalc.run calc
            if displayOutput then displayHandD hOutput dOutput
            verify hOutput dOutput

    let getScanResult (mgpuScanType:int) (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) = 
        let scan = worker.LoadPModule(PArray.scan mgpuScanType op totalAtEnd).Invoke
        fun (data:'TI[]) ->
            let calc = pcalc {
                let! data = DArray.scatterInBlob worker data
                let! result = scan data
                return! result.Value }        
            let dResult = PCalc.run calc
            dResult

    let getExclusiveAndInclusiveResults (op:IScanOp<'TI, 'TV, 'TR>) (totalAtEnd:int) =
        let excScan = worker.LoadPModule(PArray.scan ExclusiveScan op totalAtEnd).Invoke
        let incScan = worker.LoadPModule(PArray.scan InclusiveScan op totalAtEnd).Invoke
        fun (data:'TI[]) ->
            let excCalc = pcalc {
                let! excData = DArray.scatterInBlob worker data
                let! excResult = excScan excData
                return! excResult.Value }
            let incCalc = pcalc {
                let! incData = DArray.scatterInBlob worker data
                let! incResult = incScan incData
                return! incResult.Value }
            let excResult = PCalc.run excCalc
            let incResult = PCalc.run incCalc

            excResult, incResult

    let exclusiveScanResults (n:int) =
        fun (scannedData:'TI[]) ->
            let esr = Array.sub scannedData 0 n
            esr

    let inclusiveScanResults (n:int) =
        fun (scannedData:'TI[]) ->
            let isr = Array.sub scannedData 1 n
            isr

    let getHostScanResult (mgpuScanType:int) (n:int) =
        fun (scannedData:'TI[]) ->
            let sr = 
                if mgpuScanType = ExclusiveScan then
                    exclusiveScanResults n scannedData
                else
                    inclusiveScanResults n scannedData
            sr

    let scanTypeString (mgpuScanType:int) =
        if mgpuScanType = ExclusiveScan then
            "Exclusive"
        else
            "Inclusive"

    let printScanType (mgpuScanType:int) =
        let sts = scanTypeString mgpuScanType
        printfn "Scan Type: %s" sts

    let getHostExcAndIncScanResults (n:int) =
        fun (scannedData:'TI[]) ->
            let hostExcScanResult = exclusiveScanResults n scannedData
            let hostIncScanResult = inclusiveScanResults n scannedData
            hostExcScanResult, hostIncScanResult


/////////////////////////////////////////////////////////////////////////////////////////////
//                               BULK REMOVE
module BulkRemoveUtils = 
    let bulkRemove =
        let br = worker.LoadPModule(MGPU.PArray.bulkRemove).Invoke
        fun (data:'TI[]) (indices:int[]) ->
            let calc = pcalc {
                let! data = DArray.scatterInBlob worker data
                let! indices = DArray.scatterInBlob worker indices
                let! result = br data indices
                return! result.Value }
            let dResult = PCalc.run calc
            dResult