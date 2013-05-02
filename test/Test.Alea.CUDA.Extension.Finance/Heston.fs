module Test.Alea.CUDA.Extension.Finance.Heston

open System
open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Finance.Heston
open Test.Alea.CUDA.Extension

    
[<Test>]
let ``finite difference weights`` () =
      
    let worker = getDefaultWorker()
    let fdWeights = worker.LoadPModule(fdWeights).Invoke

    let n = 10
    let x = Array.init n (fun i -> float(i)/float(n))

    let delta, alpha0, alphaM1, alphaM2, beta0, betaP, betaM, gamma0, gammaP1, gammaP2, delta0, deltaP, deltaM = fdWeights x
    
    printfn "x = %A" x
    printfn "delta = %A" delta
    printfn "alpha0 = %A" alpha0
    printfn "alphaM1 = %A" alphaM1
    printfn "alphaM2 = %A" alphaM2
    printfn "beta0 = %A" beta0
    printfn "betaP = %A" betaP
    printfn "betaM = %A" betaM
    printfn "gamma0 = %A" gamma0
    printfn "gammaP1 = %A" gammaP1
    printfn "gammaP2 = %A" gammaP2
    printfn "delta0 = %A" delta0
    printfn "deltaP = %A" deltaP
    printfn "deltaM = %A" deltaM

    Assert.IsTrue(true)

