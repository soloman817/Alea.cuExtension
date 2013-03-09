module Test.Alea.CUDA.Extension.Heat2dAdi

open System
open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Heat2dAdi

let rng = Random(2)
 
let inline maxErr (b:'T[]) (b':'T[]) =
    Array.map2 (fun bi bi' -> abs (bi - bi')) b b' |> Array.max

let [<ReflectedDefinition>] pi = System.Math.PI

[<Test>]
let ``exp(-t) * sin(pi*x) * cos(pi*y) initial cond`` () =

    let uexact t x y =
        exp(-t) * sin(pi*x) * cos(pi*y)
  
    let initialCondExpr = <@ fun t x y -> exp(-t) * sin(pi*x) * cos(pi*y) @>
    let boundaryExpr = <@ fun t x y -> exp(-t) * sin(pi*x) * cos(pi*y) @>
    let sourceFunctionExpr = <@ fun t x y -> exp(-t) * sin(pi*x) * cos(pi*y) * (2.0*pi*pi - 1.0) @>

    printfn "******** 1"
    let worker = getDefaultWorker()
    printfn "******** 2"
    let solve = worker.LoadPModule(adiSolver initialCondExpr boundaryExpr sourceFunctionExpr)
    printfn "******** 3"
    let solve = solve.Invoke
    printfn "******** 4"

    let k = 1.0
    let tstart = 0.0
    let tstop = 0.0
    let Lx = 1.0
    let Ly = 1.0
    let dt = 0.01

    let nx = 127
    let ny = 127

    printfn "******** 5"

    let x, y, u = solve k tstart tstop Lx Ly nx ny dt

    let ue = Array.zeroCreate (x.Length*y.Length)
    let mstride = ny+1
    for i = 0 to x.Length-1 do
        for j = 0 to y.Length-1 do
            ue.[i*mstride+j] <- uexact tstop x.[i] y.[j]
            
    let uErr = maxErr u ue    

    printfn "uErr = %e" uErr

    
[<Test>]
let ``exp(-t) * sin(pi*x) * cos(pi*y)`` () =

    let uexact t x y =
        exp(-t) * sin(pi*x) * cos(pi*y)
  
    let initialCondExpr = <@ fun t x y -> exp(-t) * sin(pi*x) * cos(pi*y) @>
    let boundaryExpr = <@ fun t x y -> exp(-t) * sin(pi*x) * cos(pi*y) @>
    let sourceFunctionExpr = <@ fun t x y -> exp(-t) * sin(pi*x) * cos(pi*y) * (2.0*pi*pi - 1.0) @>

    printfn "******** 1"
    let worker = getDefaultWorker()
    printfn "******** 2"
    let solve = worker.LoadPModule(adiSolver initialCondExpr boundaryExpr sourceFunctionExpr)
    printfn "******** 3"
    let solve = solve.Invoke
    printfn "******** 4"

    let k = 1.0
    let tstart = 0.0
    let tstop = 1.0
    let Lx = 1.0
    let Ly = 1.0
    let dt = 0.01

    let nx = 127
    let ny = 127

    printfn "******** 5"

    let x, y, u = solve k tstart tstop Lx Ly nx ny dt

    let ue = Array.zeroCreate (x.Length*y.Length)
    let mstride = ny+1
    for i = 0 to x.Length-1 do
        for j = 0 to y.Length-1 do
            ue.[i*mstride+j] <- uexact tstop x.[i] y.[j]
            
    let uErr = maxErr u ue    

    printfn "uErr = %e" uErr

    

