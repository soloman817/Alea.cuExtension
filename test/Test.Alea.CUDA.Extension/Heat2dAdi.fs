module Test.Alea.CUDA.Extension.Heat2dAdi

open System
open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension

//open Alea.Matlab.Plot

let inline maxErr (b:'T[]) (b':'T[]) =
    Array.map2 (fun bi bi' -> abs (bi - bi')) b b' |> Array.max

let [<ReflectedDefinition>] pi = System.Math.PI
    
[<Test>]
let ``exp(-t) * sin(pi*x) * cos(pi*y)`` () =
    let uexact t x y = exp(-t) * sin(pi*x) * cos(pi*y)
    let initialCondExpr = <@ fun t x y -> exp(-t) * sin(pi*x) * cos(pi*y) @>
    let boundaryExpr = <@ fun t x y -> exp(-t) * sin(pi*x) * cos(pi*y) @>
    let sourceFunctionExpr = <@ fun t x y -> exp(-t) * sin(pi*x) * cos(pi*y) * (2.0*pi*pi - 1.0) @>

    let k = 1.0
    let tstart = 0.0
    let tstop = 1.0
    let Lx = 1.0
    let Ly = 1.0
    let dt = 0.01

    let nx = 128
    let ny = 128

    let worker = getDefaultWorker()
    let solve = worker.LoadPModule(Heat2dAdi.solve initialCondExpr boundaryExpr sourceFunctionExpr).Invoke
    let calc = pcalc {
        let! x, y, u = solve k tstart tstop Lx Ly nx ny dt
        let! u = u.Gather()
        return x, y, u }
       
    let x, y, u = calc |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

    //plotSurfaceOfArray x y u "x" "y" "heat" "Heat 2d ADI" ([400.; 200.; 750.; 700.] |> Seq.ofList |> Some)

    printfn "Here3 %d %d %d" x.Length y.Length u.Length

    let ue = Array.zeroCreate (x.Length*y.Length)
    //let mstride = ny+1 // Why?
    let mstride = ny
    for i = 0 to x.Length-1 do
        for j = 0 to y.Length-1 do
            ue.[i*mstride+j] <- uexact tstop x.[i] y.[j]
            
    let uErr = maxErr u ue    

    printfn "uErr = %e" uErr

    //Assert.IsTrue(uErr < 1.2e-4) // Why?
    Assert.IsTrue(uErr < 3e-4)

[<Test>]
let ``heat box (instable solution)`` () =

    let initialCondExpr = <@ fun t x y -> if x >= 0.4 && x <= 0.6 && y >= 0.4 && y <= 0.6 then 1.0 else 0.0 @>
    let boundaryExpr = <@ fun t x y -> 0.0 @>
    let sourceFunctionExpr = <@ fun t x y -> 0.0 @>

    let k = 1.0
    let tstart = 0.0
    let tstop = 1.0
    let Lx = 1.0
    let Ly = 1.0
    let dt = 0.01

    let nx = 128
    let ny = 128

    let worker = getDefaultWorker()
    let solve = worker.LoadPModule(Heat2dAdi.solve initialCondExpr boundaryExpr sourceFunctionExpr)
    let calc = pcalc {
        let! x, y, u = solve.Invoke k tstart tstop Lx Ly nx ny dt
        let! u = u.Gather()
        return x, y, u }

    let x, y, u = calc |> PCalc.run

    //plotSurfaceOfArray x y u "x" "y" "heat" "Heat 2d ADI" ([400.; 200.; 750.; 700.] |> Seq.ofList |> Some)

    printfn ""

[<Test>]
let ``heat gauss`` () =

    let sigma1 = 0.04
    let sigma2 = 0.04
    let sigma3 = 0.04
    let initialCondExpr = <@ fun t x y -> 1.0/3.0*exp (-((x-0.2)*(x-0.2) + (y-0.2)*(y-0.2))/(2.0*sigma1*sigma1)) / (sigma1*sigma1*2.0*pi) +
                                          1.0/3.0*exp (-((x-0.8)*(x-0.8) + (y-0.8)*(y-0.8))/(2.0*sigma2*sigma2)) / (sigma2*sigma2*2.0*pi) +
                                          1.0/3.0*exp (-((x-0.8)*(x-0.8) + (y-0.2)*(y-0.2))/(2.0*sigma3*sigma3)) / (sigma3*sigma3*2.0*pi) 
                           @>
    let boundaryExpr = <@ fun t x y -> 0.0 @>
    let sourceFunctionExpr = <@ fun t x y -> 0.0 @>

    let worker = getDefaultWorker()
    let solve = worker.LoadPModule(Heat2dAdi.solve initialCondExpr boundaryExpr sourceFunctionExpr).Invoke
    let calc k tstart tstop Lx Ly nx ny dt = pcalc {
        let! x, y, u = solve k tstart tstop Lx Ly nx ny dt
        let! u = u.Gather()
        return x, y, u }

    let heatdist tstop =
        let k = 1.0
        let tstart = 0.0
        let Lx = 1.0
        let Ly = 1.0
        let dt = 0.0001

        let nx = 512
        let ny = 512

        let x, y, u = calc k tstart tstop Lx Ly nx ny dt |> PCalc.run
        //plotSurfaceOfArray x y u "x" "y" "heat" (sprintf "Heat 2d ADI t=%f" tstop) ([400.; 200.; 750.; 700.] |> Seq.ofList |> Some)
        ()

    heatdist 0.0
    heatdist 0.005
    heatdist 0.01
    heatdist 0.02
    heatdist 0.03
    heatdist 0.04
                            

    printfn ""

[<Test>]
let ``time grid with initial condensing`` () =

    let dt = 0.02
    let tstop = 0.652

    let n = int(ceil tstop/dt)
    let dt' = tstop / float(n)

    let n1 = 5
    let ddt = dt' / float(1<<<(n1+1))
    let tg1 = [0..n1] |> Seq.map (fun n -> ddt * float(1<<<(n)))

    let tg2 = [1..n] |> Seq.map (fun n -> float(n)*dt')

    let tg = Seq.concat [Seq.singleton 0.0; tg1; tg2] |> Seq.toArray

    ()
    
    
