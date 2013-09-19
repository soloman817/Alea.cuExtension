module Sample.Mrg32k3a.CPU

open Alea.CUDA.Utilities

let mrg32k3a_m1         = Common.mrg32k3a_m1 |> uint64
let mrg32k3a_m1c        = Common.mrg32k3a_m1c |> uint64
let mrg32k3a_m2         = Common.mrg32k3a_m2 |> uint64
let mrg32k3a_m2c        = Common.mrg32k3a_m2c |> uint64

let mrg32k3a_a12        = Common.mrg32k3a_a12 |> uint64
let mrg32k3a_a13        = Common.mrg32k3a_a13 |> uint64
let mrg32k3a_a13n       = Common.mrg32k3a_a13n |> uint64

let mrg32k3a_a21        = Common.mrg32k3a_a21 |> uint64
let mrg32k3a_a23        = Common.mrg32k3a_a23 |> uint64
let mrg32k3a_a23n       = Common.mrg32k3a_a23n |> uint64

type Generator private (v1:uint64[], v2:uint64[]) =

    new(v1:uint32[], v2:uint32[], offset:int) =
        let v1 = v1 |> Array.map uint64
        let v2 = v2 |> Array.map uint64
        let A1 = Array.zeroCreate<uint64> 9
        let A2 = Array.zeroCreate<uint64> 9
        let A1b = Array.zeroCreate<uint64> 9
        let A2b = Array.zeroCreate<uint64> 9
        let v1b = Array.zeroCreate<uint64> 3
        let v2b = Array.zeroCreate<uint64> 3

        // initialise skip-ahead matrices
        A1.[0] <- 0UL;          A1.[3] <- 1UL;          A1.[6] <- 0UL
        A1.[1] <- 0UL;          A1.[4] <- 0UL;          A1.[7] <- 1UL
        A1.[2] <- mrg32k3a_a13; A1.[5] <- mrg32k3a_a12; A1.[8] <- 0UL

        A2.[0] <- 0UL;          A2.[3] <- 1UL;          A2.[6] <- 0UL
        A2.[1] <- 0UL;          A2.[4] <- 0UL;          A2.[7] <- 1UL
        A2.[2] <- mrg32k3a_a23; A2.[5] <- 0UL;          A2.[8] <- mrg32k3a_a21

        let mutable offset = offset
        while offset > 0 do
            // apply offset to seed vectors
            if offset % 2 = 1 then
                for i = 0 to 2 do
                    v1b.[i] <- ( (A1.[i+3*0]*v1.[0])%mrg32k3a_m1
                               + (A1.[i+3*1]*v1.[1])%mrg32k3a_m1
                               + (A1.[i+3*2]*v1.[2])%mrg32k3a_m1 ) % mrg32k3a_m1
                    v2b.[i] <- ( (A2.[i+3*0]*v2.[0])%mrg32k3a_m2
                               + (A2.[i+3*1]*v2.[1])%mrg32k3a_m2
                               + (A2.[i+3*2]*v2.[2])%mrg32k3a_m2 ) % mrg32k3a_m2
                for i = 0 to 2 do
                    v1.[i] <- v1b.[i]
                    v2.[i] <- v2b.[i]
            offset <- offset / 2

            // square skip-ahead matrices
            for i = 0 to 2 do
                for j = 0 to 2 do
                    let mutable a1 = 0UL
                    let mutable a2 = 0UL
                    for k = 0 to 2 do
                        a1 <- a1 + (A1.[i+3*k]*A1.[k+3*j])%mrg32k3a_m1
                        a2 <- a2 + (A2.[i+3*k]*A2.[k+3*j])%mrg32k3a_m2
                    A1b.[i+3*j] <- a1%mrg32k3a_m1
                    A2b.[i+3*j] <- a2%mrg32k3a_m2
            for i = 0 to 8 do
                A1.[i] <- A1b.[i]
                A2.[i] <- A2b.[i]

        Generator(v1, v2)

    member this.NextStep() =
        let p = mrg32k3a_a12 * v1.[1] + mrg32k3a_a13n * (mrg32k3a_m1 - v1.[0])
        let p = p % mrg32k3a_m1
        v1.[0] <- v1.[1]; v1.[1] <- v1.[2]; v1.[2] <- p
        let p = mrg32k3a_a21 * v2.[2] + mrg32k3a_a23n * (mrg32k3a_m2 - v2.[0])
        let p = p % mrg32k3a_m2
        v2.[0] <- v2.[1]; v2.[1] <- v2.[2]; v2.[2] <- p
        let mutable p = v1.[2] - v2.[2]
        if v1.[2] < v2.[2] then p <- p + mrg32k3a_m1
        p

let raw (generator:Generator) (points:uint32[]) =
    let n = points.Length
    for i = 0 to n - 1 do
        let p = generator.NextStep() |> uint32
        points.[i] <- p

let inline uniform (real:RealConverter<'T>) (generator:Generator) (numbers:'T[]) =
    let mrg32k3a_norm = Common.mrg32k3a_norm |> real.Of
    let n = numbers.Length
    for i = 0 to n - 1 do
        let p = generator.NextStep() |> real.Of
        numbers.[i] <- mrg32k3a_norm * p

let inline exponential (real:RealConverter<'T>) (generator:Generator) (numbers:'T[]) =
    let mrg32k3a_norm = Common.mrg32k3a_norm |> real.Of
    let n = numbers.Length
    for i = 0 to n - 1 do
        let p = generator.NextStep() |> real.Of
        numbers.[i] <- log(mrg32k3a_norm * p)

let inline normal (real:RealConverter<'T>) (generator:Generator) (numbers:'T[]) =
    let mrg32k3a_norm = Common.mrg32k3a_norm |> real.Of
    let mrg32k3a_pi = Common.mrg32k3a_pi |> real.Of
    let n = numbers.Length
    let mutable x2 = 0G
    for i = 0 to n - 1 do
        let p = generator.NextStep() |> real.Of
        if i % 2 = 0 then
            x2 <- sqrt((real.Of -2) * log(mrg32k3a_norm * p))
        else
            numbers.[i - 1] <- x2 * sin(2G * mrg32k3a_pi * mrg32k3a_norm * p)
            if i < n then
                numbers.[i] <- x2 * cos(2G * mrg32k3a_pi * mrg32k3a_norm * p)

let inline gamma (real:RealConverter<'T>) (generator:Generator) (alpha:'T) (numbers:'T[]) =
    let mrg32k3a_norm = Common.mrg32k3a_norm |> real.Of
    let mrg32k3a_pi = Common.mrg32k3a_pi |> real.Of
    
    let nextUniform () =
        let p = generator.NextStep() |> real.Of
        mrg32k3a_norm * p

    let nextNormal (x2:'T ref) =
        if __isnan(!x2) then
            let p = generator.NextStep() |> real.Of
            let x3 = sqrt((real.Of -2) * log(mrg32k3a_norm * p))
            let p = generator.NextStep() |> real.Of
            x2 := x3 * cos(2G * mrg32k3a_pi * mrg32k3a_norm * p)
            x3 * sin(2G * mrg32k3a_pi * mrg32k3a_norm * p)
        else
            let x3 = !x2
            x2 := __nan()
            x3

    let nextGamma (x2:'T ref) =
        let mutable prefix = 0G
        let mutable alpha = alpha

        if alpha <= 1G then
            prefix <- __pow (nextUniform()) (1G/alpha)
            alpha <- alpha + 1G
        else
            prefix <- 1G

        let d = alpha - 1G/3G
        let c = 1G/sqrt(9G*d)

        let mutable x = 0G
        let mutable u = 0G
        let mutable v = real.Of -1
        while (v <= 0G) || (log u >= (real.Of 0.5) * x * x + d * (1G - v + log(v))) do
            x <- nextNormal(x2)
            v <- 1G + c * x
            v <- v * v * v
            u <- nextUniform()

        x
        //prefix * d * v

    let x2 : 'T ref = ref (__nan())
    let n = numbers.Length
    for i = 0 to n - 1 do
        numbers.[i] <- nextGamma(x2)


