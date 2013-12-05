module Sample.LiborMM.GPU

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities

type Param =
    {
        NOPT : int
        NN : int
        NMAT : int
    }

    member this.L2_SIZE = this.NN * (this.NMAT + 1)

let inline template (real:RealConverter<'T>) (param:Param) = cuda {
    let! cMaturities = Compiler.DefineConstantArray<int>(param.NOPT)
    let! cDelta = Compiler.DefineConstantVariable<'T>(0G)
    let! cLambda = Compiler.DefineConstantArray<'T>(param.NN)
    let! cL0 = Compiler.DefineConstantArray<'T>(param.NN)
    let! cSwaprates = Compiler.DefineConstantArray<'T>(param.NOPT)

    let! pathCalc =
        <@ fun (z:deviceptr<'T>) (L:deviceptr<'T>) ->
            let cDelta = cDelta.Value
            let mutable z = z

            for n = 0 to param.NMAT - 1 do
                let sqez = sqrt(cDelta) * z.[0]
                z <- z + blockDim.x

                let mutable v = 0G
                for i = n + 1 to param.NN - 1 do
                    let lam = cLambda.[i - n - 1]
                    let con1 = cDelta * lam
                    v <- v + con1 * L.[i] / (1G + cDelta * L.[i])
                    let vrat = exp(con1 * v + lam * (sqez - (real.Of 0.5) * con1))
                    L.[i] <- L.[i] * vrat @>
        |> Compiler.DefineFunction

    let! pathCalcB1 =
        <@ fun (z:deviceptr<'T>) (L:deviceptr<'T>) (L2:deviceptr<'T>) ->
            let cDelta = cDelta.Value
            let mutable z = z

            for i = 0 to param.NN - 1 do L2.[i] <- L.[i]

            for n = 0 to param.NMAT - 1 do
                let sqez = sqrt(cDelta) * z.[0]
                z <- z + blockDim.x

                let mutable v = 0G
                for i = n + 1 to param.NN - 1 do
                    let lam = cLambda.[i - n - 1]
                    let con1 = cDelta * lam
                    v <- v + con1 * L.[i] / (1G + cDelta * L.[i])
                    let vrat = exp(con1 * v + lam * (sqez - (real.Of 0.5) * con1))
                    L.[i] <- L.[i] * vrat
                    L2.[i + (n + 1) * param.NN] <- L.[i] @>
        |> Compiler.DefineFunction

    let! pathCalcB2 =
        <@ fun (Lb:deviceptr<'T>) (L2:deviceptr<'T>) ->
            let cDelta = cDelta.Value
            let mutable n = param.NMAT - 1
            while n >= 0 do
                let mutable v1 = 0G
                let mutable i = param.NN - 1
                while i > n do
                    v1 <- v1 + cLambda.[i - n - 1] * L2.[i + (n + 1) * param.NN] * Lb.[i]
                    let faci = cDelta / (1G + cDelta * L2.[i + n * param.NN])
                    Lb.[i] <- Lb.[i] * L2.[i + (n + 1) * param.NN] / L2.[i + n * param.NN] + v1 * cLambda.[i - n - 1] * faci * faci
                    i <- i - 1                
                n <- n - 1 @>
        |> Compiler.DefineFunction

    let! portfolio =
        <@ fun (L:deviceptr<'T>) ->
            let B = __local__.Array<'T>(param.NMAT)
            let S = __local__.Array<'T>(param.NMAT)

            let cDelta = cDelta.Value
            let mutable b = 1G
            let mutable s = 0G

            for n = param.NMAT to param.NN - 1 do
                b <- b / (1G + cDelta * L.[n])
                s <- s + cDelta * b
                B.[n - param.NMAT] <- b
                S.[n - param.NMAT] <- s

            let mutable v = 0G

            for i = 0 to param.NOPT - 1 do
                let m = cMaturities.[i] - 1
                let swapval = B.[m] + cSwaprates.[i] * S.[m] - 1G
                if swapval < 0G then
                    v <- v + (real.Of -100) * swapval

            for n = 0 to param.NMAT - 1 do
                v <- v / (1G + cDelta * L.[n])

            v @>
        |> Compiler.DefineFunction

    let! portfolioB =
        <@ fun (L:deviceptr<'T>) (Lb:deviceptr<'T>) ->
            let B = __local__.Array<'T>(param.NMAT)
            let S = __local__.Array<'T>(param.NMAT)
            let Bb = __local__.Array<'T>(param.NMAT)
            let Sb = __local__.Array<'T>(param.NMAT)
            
            let cDelta = cDelta.Value
            let mutable b = 1G
            let mutable s = 0G
            
            for m = 0 to param.NN - param.NMAT - 1 do
                let n = m + param.NMAT
                b <- b / (1G + cDelta * L.[n])
                s <- s + cDelta * b
                B.[m] <- b
                S.[m] <- s
                
            let mutable v = 0G
            
            for m = 0 to param.NN - param.NMAT - 1 do
                Bb.[m] <- 0G
                Sb.[m] <- 0G
            
            for n = 0 to param.NOPT - 1 do
                let m = cMaturities.[n] - 1
                let swapval = B.[m] + cSwaprates.[n] * S.[m] - 1G
                if swapval < 0G then
                    v <- v + (real.Of -100) * swapval
                    Sb.[m] <- Sb.[m] + (real.Of -100) * cSwaprates.[n]
                    Bb.[m] <- Bb.[m] + (real.Of -100)
                    
            let mutable m = param.NN - param.NMAT - 1
            while m >= 0 do
                let n = m + param.NMAT
                Bb.[m] <- Bb.[m] + cDelta * Sb.[m]
                Lb.[n] <- -Bb.[m] * B.[m] * cDelta / (1G + cDelta * L.[n])
                if m > 0 then
                    Sb.[m - 1] <- Sb.[m - 1] + Sb.[m]
                    Bb.[m - 1] <- Bb.[m - 1] + Bb.[m] / (1G + cDelta * L.[n])
                m <- m - 1    

            let mutable b = 1G
            for n = 0 to param.NMAT - 1 do b <- b / (1G + cDelta * L.[n])

            let v = b * v

            for n = 0 to param.NMAT - 1 do
                Lb.[n] <- -v * cDelta / (1G + cDelta * L.[n])

            for n = param.NMAT to param.NN - 1 do
                Lb.[n] <- b * Lb.[n]

            v @>
        |> Compiler.DefineFunction

    let! kernelPathcalcPortfolio1 =
        <@ fun (dz:deviceptr<'T>) (dv:deviceptr<'T>) (dLb:deviceptr<'T>) ->
            let tid = threadIdx.x + blockIdx.x * blockDim.x
            let L = __local__.Array<'T>(param.NN) |> __ptrofarray
            let L2 = __local__.Array<'T>(param.L2_SIZE) |> __ptrofarray
            let Lb = L

            for i = 0 to param.NN - 1 do L.[i] <- cL0.[i]

            let dz = dz + threadIdx.x + param.NMAT * blockIdx.x * blockDim.x

            pathCalcB1.Invoke dz L L2
            dv.[tid] <- portfolioB.Invoke L Lb
            pathCalcB2.Invoke Lb L2
            dLb.[tid] <- Lb.[param.NN - 1] @>
        |> Compiler.DefineKernel

    let! kernelPathcalcPortfolio2 =
        <@ fun (dz:deviceptr<'T>) (dv:deviceptr<'T>) ->
            let tid = threadIdx.x + blockIdx.x * blockDim.x
            let L = __local__.Array<'T>(param.NN) |> __ptrofarray
            
            for i = 0 to param.NN - 1 do L.[i] <- cL0.[i]           
            
            let dz = dz + threadIdx.x + param.NMAT * blockIdx.x * blockDim.x
            
            pathCalc.Invoke dz L
            dv.[tid] <- portfolio.Invoke L @>
        |> Compiler.DefineKernel     

    let init (program:Program) =
        let cMaturities = program.Apply(cMaturities)
        let cDelta = program.Apply(cDelta)
        let cLambda = program.Apply(cLambda)
        let cL0 = program.Apply(cL0)
        let cSwaprates = program.Apply(cSwaprates)
        
        let init (maturities:int[]) (delta:'T) (lambda:'T[]) (L0:'T[]) (swaprates:'T[]) =
            cMaturities.Scatter(maturities)
            cDelta.Scatter(delta)
            cLambda.Scatter(lambda)
            cL0.Scatter(L0)
            cSwaprates.Scatter(swaprates)

        init

    return init, kernelPathcalcPortfolio1, kernelPathcalcPortfolio2 }
