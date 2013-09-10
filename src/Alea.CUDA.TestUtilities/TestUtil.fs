module Alea.CUDA.TestUtilities.TestUtil

open System
open System.Runtime.InteropServices
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

let rng = Random(2)

let genRandomBool _ = let n = rng.Next(0, 2) in if n = 0 then false else true
let genRandomSInt8 minv maxv _ = rng.Next(minv, maxv) |> int8
let genRandomUInt8 minv maxv _ = if minv < 0 || maxv < 0 then failwith "minv or maxv < 0" else rng.Next(minv, maxv) |> uint8
let genRandomSInt16 minv maxv _ = rng.Next(minv, maxv) |> int16
let genRandomUInt16 minv maxv _ = if minv < 0 || maxv < 0 then failwith "minv or maxv < 0" else rng.Next(minv, maxv) |> uint16
let genRandomSInt32 minv maxv _ = rng.Next(minv, maxv)
let genRandomUInt32 minv maxv _ = if minv < 0 || maxv < 0 then failwith "minv or maxv < 0" else rng.Next(minv, maxv) |> uint32
let genRandomSInt64 minv maxv _ = rng.Next(minv, maxv) |> int64
let genRandomUInt64 minv maxv _ = if minv < 0 || maxv < 0 then failwith "minv or maxv < 0" else rng.Next(minv, maxv) |> uint64
let genRandomDouble minv maxv _ = rng.NextDouble() * (maxv - minv) + minv
let genRandomSingle minv maxv _ = (rng.NextDouble() * (maxv - minv) + minv) |> float32

let assertArrayEqual (eps:float option) (A:'T[]) (B:'T[]) =
    (A, B) ||> Array.iter2 (fun a b -> eps |> function
        | None -> Assert.AreEqual(a, b)
        | Some eps -> Assert.That(b, Is.EqualTo(a).Within(eps)))

let testFunction1 (funcD:Expr<'T -> 'U>) (funcH:'T -> 'U) (eps:float option) (n:int) (gen:int -> 'T) =
    let template = cuda {
        let! kernel =
            <@ fun (output:deviceptr<'U>) (input:deviceptr<'T>) (n:int) ->
                let start = blockIdx.x * blockDim.x + threadIdx.x
                let stride = gridDim.x * blockDim.x
                let mutable i = start
                while i < n do
                    output.[i] <- (%funcD) input.[i]
                    i <- i + stride @>
            |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply(kernel)

            let calcH (input:'T[]) = input |> Array.map funcH

            let calcD (input:'T[]) =
                let n = input.Length
                use input = worker.Malloc(input)
                use output = worker.Malloc<'U>(n)
                let lp = LaunchParam(16, 64)
                kernel.Launch lp output.Ptr input.Ptr n
                output.Gather()

            let test input =
                let hOutput = calcH input
                let dOutput = calcD input
                if Util.debug then
                    for i = 0 to (min input.Length 30) - 1 do
                        printfn "#.%02d: I(%A) -> H(%A) D(%A)" i input.[i] hOutput.[i] dOutput.[i]
                assertArrayEqual eps hOutput dOutput

            test ) }

    let program = template |> Util.load Worker.Default
    Array.init n gen |> program.Run

let testFunction2 (funcD:Expr<'T1 -> 'T2 -> 'U>) (funcH:'T1 ->'T2 -> 'U) (eps:float option) (n:int) (gen1:int -> 'T1) (gen2:int -> 'T2) =
    let template = cuda {
        let! kernel =
            <@ fun (output:deviceptr<'U>) (input1:deviceptr<'T1>) (input2:deviceptr<'T2>) (n:int) ->
                let start = blockIdx.x * blockDim.x + threadIdx.x
                let stride = gridDim.x * blockDim.x
                let mutable i = start
                while i < n do
                    output.[i] <- (%funcD) input1.[i] input2.[i]
                    i <- i + stride @>
            |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply(kernel)

            let calcH (input1:'T1[]) (input2:'T2[]) = (input1, input2) ||> Array.map2 funcH

            let calcD (input1:'T1[]) (input2:'T2[]) =
                let n = input1.Length
                use input1 = worker.Malloc(input1)
                use input2 = worker.Malloc(input2)
                use output = worker.Malloc<'U>(n)
                let lp = LaunchParam(16, 64)
                kernel.Launch lp output.Ptr input1.Ptr input2.Ptr n
                output.Gather()

            let test input1 input2 =
                let hOutput = calcH input1 input2
                let dOutput = calcD input1 input2
                if Util.debug then
                    for i = 0 to (min input1.Length 30) - 1 do
                        printfn "#.%02d: I1(%A) I2(%A) -> H(%A) D(%A)" i input1.[i] input2.[i] hOutput.[i] dOutput.[i]
                (hOutput, dOutput) ||> Array.iter2 (fun h d ->
                    match eps with
                    | None -> Assert.AreEqual(h, d)
                    | Some(eps) -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            test ) }

    let program = template |> Util.load Worker.Default
    (Array.init n gen1, Array.init n gen2) ||> program.Run
