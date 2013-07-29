module Alea.CUDA.Extension.Finance.LiborMarketModel

open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util

// shorthands 
let [<ReflectedDefinition>] get (u:RMatrixRowMajor ref) (si:int) (vi:int) =
    RMatrixRowMajor.Get(u, si, vi)

let [<ReflectedDefinition>] set (u:RMatrixRowMajor ref) (si:int) (vi:int) (value:float) =
    RMatrixRowMajor.Set(u, si, vi, value)

let [<ReflectedDefinition>] inline pathCalc (N:int) (nMat:int) (delta:'T) (z:DevicePtr<'T>) (l:LocalPtr<'T>) (lambda:ConstantArrayResource<'T>) =
    let mutable n = 0
    let mutable zi = 0
    while n < nMat do
        let sqez = sqrt(delta)*z.[zi]
        zi <- zi + blockDim.x

        let mutable v = 0.0
        let mutable i = n + 1
        while i < N do
            let lam = lambda.[i-n-1]
            let con1 = delta*lam
            v <- v + con1*l.[i]/(1G + delta*l.[i])
            let vrat = exp(con1*v + lam*(sqez - con1/NumericLiteralG.FromInt32(2)))
            l.[i] <- l.[i]*vrat
            i <- i + 1

        n <- n + 1

/// Forward path calculation storing data for subsequent reverse path calculation 
let [<ReflectedDefinition>] pathCalcB1 (N:int) (nMat:int) (delta:'T) (z:DevicePtr<'T>) (l:LocalPtr<'T>) (l2:LocalPtr<'T>) (lambda:ConstantArrayResource<'T>) =
    let mutable i = 0
    while i < N do
        l2.[i] <- l.[i]

    let mutable n = 0
    let mutable zi = 0
    while n < nMat do
        let sqez = sqrt(delta)*z.[zi]
        zi <- zi + blockDim.x

        let mutable v = 0.0
        let mutable i = n + 1

        while i < N do
            let lam = lambda.[i-n-1]
            let con1 = delta*lam
            v <- v + con1*l.[i]/(1G + delta*l.[i])
            let vrat = exp(con1*v + lam*(sqez - con1/NumericLiteralG.FromInt32(2)))
            l.[i] <- l.[i]*vrat
            i <- i + 1

            // store these values for reverse path 
            l2.[i+(n+1)*N] <- l.[i]

        n <- n + 1

/// Reverse path calculation of deltas using stored data  
let [<ReflectedDefinition>] pathCalcB2 (N:int) (nMat:int) (delta:'T) (lb:LocalPtr<'T>) (l2:LocalPtr<'T>) (lambda:ConstantArrayResource<'T>) =
    let mutable n = nMat - 1
    while n >= 0 do
        let mutable v1 = 0.0

        let mutable i = N - 1
        while i > n do
            v1 <- v1 + lambda.[i-n-1]*l2.[i+(n+1)*N]*lb.[i]
            let faci = delta/(1G + delta*l2.[i+n*N])
            lb.[i] <- lb.[i]*l2.[i+(n+1)*N]/l2.[i+n*N] + v1*lambda.[i-n-1]*faci*faci
            i <- i - 1

        n <- n - 1

/// Calculate the portfolio value v  
let [<ReflectedDefinition>] portfolio (N:int) (nOpt:int) (nMat:int) (delta:'T) (l:LocalPtr<'T>) (B:LocalPtr<'T>) (S:LocalPtr<'T>) 
                                      (maturities:ConstantArrayResource<int>) (swapRates:ConstantArrayResource<'T>) =        
    let mutable b = 1G
    let mutable s = 0G

    let mutable n = nMat
    while nMat < N do 
        b <- b/(1G + delta*l.[n])
        s <- s + delta*b
        B.[n-nMat] <- b
        S.[n-nMat] <- s
        n <- n + 1

    let mutable v = 0.0
    let mutable i = 0
    while i < nOpt do
        let m = maturities.[i] - 1
        let swapVal = B.[m] + swapRates.[i]*S.[m] - 1G
        if swapVal < 0G then 
            v <- v + NumericLiteralG.FromInt32(100)*swapVal 
        i <- i + 1

    // apply discount  
    let mutable n = 0
    while n < nMat do
        v <- v/(1G + delta*l.[n])
        n <- n + 1

    v

/// Calculate the portfolio value v, and its sensitivity to L, hand-coded reverse mode sensitivity 
let [<ReflectedDefinition>] portfolioB (N:int) (nOpt:int) (nMat:int) (delta:'T) (l:LocalPtr<'T>) (lb:LocalPtr<'T>) 
                                       (B:LocalPtr<'T>) (S:LocalPtr<'T>) (Bb:LocalPtr<'T>) (Sb:LocalPtr<'T>)
                                       (maturities:ConstantArrayResource<int>) (swapRates:ConstantArrayResource<'T>) =        
    let mutable b = 1G
    let mutable s = 0G

    let mutable m = 0
    while m < N - nMat do
        let n = m + nMat
        b <- b/(1G + delta*l.[n])
        s <- s + delta*b
        B.[m] <- b
        S.[m] <- s
        m <- m + 1

    let mutable v = 0G
    let mutable m = 0
    while m < N - nMat do
        Bb.[m] <- 0G
        Sb.[m] <- 0G
        m <- m + 1

    let mutable n = 0
    while n < nOpt do
        let m = maturities.[n] - 1
        let swapVal = B.[m] + swapRates.[n]*S.[m] - 1G
        if swapVal < 0G then 
            v <- v - NumericLiteralG.FromInt32(100)*swapVal
            Sb.[m] <- Sb.[m] - NumericLiteralG.FromInt32(100)*swapRates.[n]
            Bb.[m] <- Bb.[m] - NumericLiteralG.FromInt32(100)
        n <- n + 1

    let mutable m = N - nMat - 1
    while m >= 0 do
        let n = m + nMat
        Bb.[m] <-  Bb.[m] + delta*Sb.[m]
        lb.[n] <- -Bb.[m]*B.[m]*delta/(1G + delta*l.[n])
        if m > 0 then
          Sb.[m-1] <- Sb.[m-1] + Sb.[m]
          Bb.[m-1] <- Bb.[m-1] + Bb.[m]/(1G + delta*l.[n])   
        m <- m - 1  

    // apply discount 
    let mutable b = 1G
    let mutable n = 0
    while n < nMat do
        b <- b/(1G + delta*l.[n])
        n <- n + 1

    v <- b*v

    let mutable n = 0
    while n < nMat do
        lb.[n] <- -v*delta/(1G + delta*l.[n])
        n <- n + 1

    let mutable n = nMat
    while n < N do
        lb.[n] <- b*lb.[n]
        n <- n + 1

    v

/// Standard setting for size parameters N = 80, nMat = 40, l2Size = 3280, nOpt = 15
let liborMarketModel N nMat nOpt = cuda {
    let l2Size = N*(nMat + 1)

    let! maturities = defineConstantArray<int>(nOpt)
    let! lambda = defineConstantArray<'T>(N)
    let! l0 = defineConstantArray<'T>(N)
    let! swapRates = defineConstantArray<'T>(nOpt)

    let! pathCalcPortfolioKernel =
            
        <@ fun (delta:'T) (dZ:DevicePtr<'T>) (dV:DevicePtr<'T>) (dLb:DevicePtr<'T>)d ->
            let tid = threadIdx.x + blockIdx.x*blockDim.x

            let l = __local__<'T>(N)
            let B = __local__<'T>(nMat)
            let S = __local__<'T>(nMat)
            let Bb = __local__<'T>(nMat)
            let Sb = __local__<'T>(nMat)

            let mutable i = 0
            while i < N do
                l.[i] <- l0.[i]
                i <- i + 1

            let z = dZ + threadIdx.x + nMat*blockIdx.x*blockDim.x

            pathCalc N nMat delta z (l.Ptr(0)) lambda

            dV.[tid] <- portfolio N nOpt nMat delta (l.Ptr(0)) (B.Ptr(0)) (S.Ptr(0)) maturities swapRates @> |> defineKernelFuncWithName "pathCalcPortfolio"

    let! pathCalcPortfolioBKernel =
            
        <@ fun (delta:'T) (dZ:DevicePtr<'T>) (dV:DevicePtr<'T>) (dLb:DevicePtr<'T>)d ->
            let tid = threadIdx.x + blockIdx.x*blockDim.x

            let l = __local__<'T>(N)
            let l2 = __local__<'T>(l2Size)
            let B = __local__<'T>(nMat)
            let S = __local__<'T>(nMat)
            let Bb = __local__<'T>(nMat)
            let Sb = __local__<'T>(nMat)

            let mutable i = 0
            while i < N do
                l.[i] <- l0.[i]
                i <- i + 1

            let z = dZ + threadIdx.x + nMat*blockIdx.x*blockDim.x

            pathCalcB1 N nMat delta z (l.Ptr(0)) (l2.Ptr(0)) lambda

            dV.[tid] <- portfolioB N nOpt nMat delta (l.Ptr(0)) (l.Ptr(0)) (B.Ptr(0)) (S.Ptr(0)) (Bb.Ptr(0)) (Sb.Ptr(0)) maturities swapRates

            pathCalcB2 N nMat delta (l.Ptr(0)) (l2.Ptr(0)) lambda 
            
            dLb.[tid] <- l.[N - 1] @> |> defineKernelFuncWithName "pathCalcPortfolioB"

    let launchParam (m:Module) (hint:ActionHint) (n:int) =
        let worker = m.Worker
        let blockSize = 256 
        let gridSize = min worker.Device.NumSm (Util.divup n blockSize)
        LaunchParam(gridSize, blockSize) |> hint.ModifyLaunchParam

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let pathCalcPortfolioKernel = pathCalcPortfolioKernel.Apply m
        let pathCalcPortfolioBKernel = pathCalcPortfolioBKernel.Apply m
        let maturities = maturities.Apply m
        let lambda = lambda.Apply m 
        let l0 = l0.Apply m 
        let swapRates = swapRates.Apply m 
        let launchParam = launchParam m
        fun (hint:ActionHint) (n:int) (nMat':int) (maturities':int[]) (delta':float) (lambda':float[]) (l0':float[]) (swapRates':float[]) ->
            let lp = launchParam hint n
            fun () ->
                maturities.Scatter(maturities')
                lambda.Scatter(lambda')
                l0.Scatter(l0')
                swapRates.Scatter(swapRates')
                //kernel.Launch lp n data
            |> worker.Eval) 
            
            }




