[<AutoOpen>]
module Test.Alea.CUDA.Extension.TestUtilities.General

open System
open Alea.CUDA
open Alea.CUDA.Extension
open NUnit.Framework


let worker = Engine.workers.DefaultWorker

let eps = 1e-8


let displayHandD (h:'T[]) (d:'T[]) =
    printfn "*********HOST************"
    printfn "COUNT = ( %d )" h.Length
    printfn "DATA = (%A)" h
    printfn "*************************"
    printfn ""
    printfn "********DEVICE***********"
    printfn "COUNT = ( %d )" d.Length
    printfn "DATA = (%A)" d
    printfn "*************************"



// need to figure out why device array for scan is 1 smaller than host (hence the -2 here)
let verify (h:int[]) (d:int[]) = 
        for i = 0 to h.Length - 2 do
            Assert.That(d.[i], Is.EqualTo(h.[i]).Within(eps))



let runForStats (pc:PCalc<'T[]>) =
    let _, loggers = pc |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = pc |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    pc |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
