[<AutoOpen>]
module Test.Alea.CUDA.Extension.MGPU.Util

open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

let rng = System.Random()


//////////////////////////////
// see http://stackoverflow.com/questions/17002632/how-to-generate-an-array-with-a-dynamic-type-in-f
type RngOverloads = RngOverloads with
    static member ($) (RngOverloads, fake:int) = fun (x:int) -> int x
    static member ($) (RngOverloads, fake:float32) = fun (x:int) -> float32 x
    static member ($) (RngOverloads, fake:float) = fun (x:int) -> float x
    static member ($) (RngOverloads, fake:int64) = fun (x:int) -> int64 x

// generate an array of random 'T values
let inline rngGenericArray sCount : 'T[] =
    let convert = (RngOverloads $ Unchecked.defaultof<'T>)
    let genValue() = rng.Next() |> convert
    let source = Array.init sCount (fun _ -> genValue())
    source

// generate an array of random 'T values along with a sorted array of random indices
// that are within the bounds of the source array
// example: let (r : float[] * _) = rngGenericArray 10 10
let inline rngGenericArrayI sCount iCount : 'T[] * int[] =
    let convert = (RngOverloads $ Unchecked.defaultof<'T>)
    let genValue() = rng.Next() |> convert
    let source = Array.init sCount (fun _ -> genValue())
    let indices = Array.init iCount (fun _ -> rng.Next sCount) |> Seq.distinct |> Seq.toArray |> Array.sort
    source, indices

let getDefaultWorker() =
    if Device.Count = 0 then Assert.Inconclusive("We need at least one device of compute capability 2.0 or greater.")
    Engine.workers.DefaultWorker

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

let inline verify (h:'T[]) (d:'T[]) = 
        for i = 0 to h.Length - 1 do
            Assert.That(d.[i], Is.EqualTo(h.[i]).Within(eps))

let runForStats (pc:PCalc<'T[]>) =
    let _, loggers = pc |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = pc |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    pc |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))


//type Stats(numTests:int) =
//    val TheirStats : float * float list
//    val mutable MyStats : List.Empty()
//    member s.Count i = int(s.MyStats.[i].[0])