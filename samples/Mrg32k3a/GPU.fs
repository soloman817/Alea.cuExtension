module Sample.Mrg32k3a.GPU

open System
open System.IO
open System.Reflection
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Sample.Mrg32k3a.Common

type Core =
    {
        // StreamInit v1 v2 np
        StreamInit : Resources.Function<deviceptr<uint32> -> deviceptr<uint32> -> int -> unit>

        // NextStep v1 v2 p
        NextStep : Resources.Function<deviceptr<uint32> -> deviceptr<uint32> -> uint32 ref -> unit>

        // host init function : program v1 v2 offset
        Init : Program -> uint32[] -> uint32[] -> int -> unit
    }

let core : Template<Core> = cuda {
    let! mrg32k3a_v1 = Compiler.DefineConstantArray<uint32>(3)
    let! mrg32k3a_A1 = Compiler.DefineConstantArray<uint32>(9 * 148)
    let! mrg32k3a_v2 = Compiler.DefineConstantArray<uint32>(3)
    let! mrg32k3a_A2 = Compiler.DefineConstantArray<uint32>(9 * 148)

    let! streamInit =
        <@ fun (v1:deviceptr<uint32>) (v2:deviceptr<uint32>) (np:int) ->
            let mutable n = 0
            let mutable m = 0
            let mutable vlo = 0u
            let mutable vhi = 0u
            let mutable vt0 = 0u
            let mutable vt1 = 0u
            let mutable vt2 = 0u
            let mutable vt = 0UL
            let mutable off : uint64 = uint64(np) * uint64(threadIdx.x + blockDim.x * blockIdx.x)

            for m = 0 to 2 do
                v1.[m] <- mrg32k3a_v1.[m]
                v2.[m] <- mrg32k3a_v2.[m]

            while off > 0UL do
                m <- int(off % 8UL) - 1
                if m >= 0 then
                    vt0 <- v1.[0]
                    vt1 <- v1.[1]
                    vt2 <- v1.[2]

                    for i = 0 to 2 do
                        vlo <-             mrg32k3a_A1.[i + 3 * 0 + 9 * m + 63 * n] * vt0
                        vhi <- __nv_umulhi mrg32k3a_A1.[i + 3 * 0 + 9 * m + 63 * n]   vt0
                        vt  <- uint64(vlo) + uint64(vhi) * uint64(mrg32k3a_m1c)

                        vlo <-             mrg32k3a_A1.[i + 3 * 1 + 9 * m + 63 * n] * vt1
                        vhi <- __nv_umulhi mrg32k3a_A1.[i + 3 * 1 + 9 * m + 63 * n]   vt1
                        vt  <- vt + uint64(vlo) + uint64(vhi) * uint64(mrg32k3a_m1c)
                    
                        vlo <-             mrg32k3a_A1.[i + 3 * 2 + 9 * m + 63 * n] * vt2
                        vhi <- __nv_umulhi mrg32k3a_A1.[i + 3 * 2 + 9 * m + 63 * n]   vt2
                        vt  <- vt + uint64(vlo) + uint64(vhi) * uint64(mrg32k3a_m1c)

                        vt <- (vt &&& uint64(mrg32k3a_2pow32m1)) + (vt >>> 32) * uint64(mrg32k3a_m1c)
                        if vt >= uint64(mrg32k3a_m1) then vt <- vt - uint64(mrg32k3a_m1)

                        v1.[i] <- uint32(vt)

                    vt0 <- v2.[0]
                    vt1 <- v2.[1]
                    vt2 <- v2.[2]

                    for i = 0 to 2 do
                        vlo <-             mrg32k3a_A2.[i + 3 * 0 + 9 * m + 63 * n] * vt0
                        vhi <- __nv_umulhi mrg32k3a_A2.[i + 3 * 0 + 9 * m + 63 * n]   vt0
                        vt  <- uint64(vlo) + uint64(vhi) * uint64(mrg32k3a_m2c)

                        vlo <-             mrg32k3a_A2.[i + 3 * 1 + 9 * m + 63 * n] * vt1
                        vhi <- __nv_umulhi mrg32k3a_A2.[i + 3 * 1 + 9 * m + 63 * n]   vt1
                        vt  <- vt + uint64(vlo) + uint64(vhi) * uint64(mrg32k3a_m2c)
                    
                        vlo <-             mrg32k3a_A2.[i + 3 * 2 + 9 * m + 63 * n] * vt2
                        vhi <- __nv_umulhi mrg32k3a_A2.[i + 3 * 2 + 9 * m + 63 * n]   vt2
                        vt  <- vt + uint64(vlo) + uint64(vhi) * uint64(mrg32k3a_m2c)

                        vt <- (vt &&& uint64(mrg32k3a_2pow32m1)) + (vt >>> 32) * uint64(mrg32k3a_m2c)
                        if vt >= uint64(mrg32k3a_m2) then vt <- vt - uint64(mrg32k3a_m2)

                        v2.[i] <- uint32(vt)

                off <- off / 8UL
                n <- n + 1 @>
        |> Compiler.DefineFunction

    let! nextStep =
        <@ fun (v1:deviceptr<uint32>) (v2:deviceptr<uint32>) (p:uint32 ref) -> 
            let mutable vs0 = 0u
            let mutable vs1 = 0u
            let mutable vs2 = 0u
            let mutable plo = 0u
            let mutable phi = 0u
            let mutable pl2 = 0u
            let mutable prod = 0UL

            vs1 <- v1.[1]; vs0 <- mrg32k3a_m1 - v1.[0]
            prod <- uint64(mrg32k3a_a12) * uint64(vs1) + uint64(mrg32k3a_a13n) * uint64(vs0)
            phi <- prod >>> 32 |> uint32; plo <- prod &&& uint64(mrg32k3a_2pow32m1) |> uint32

            pl2 <- plo + __nv_umul24 phi mrg32k3a_m1c
            if pl2 >= mrg32k3a_m1 || pl2 < plo then pl2 <- pl2 - mrg32k3a_m1

            v1.[0] <- v1.[1]; v1.[1] <- v1.[2]; v1.[2] <- pl2

            vs2 <- v2.[2]; vs0 <- mrg32k3a_m2 - v2.[0]
            prod <- uint64(mrg32k3a_a21) * uint64(vs2) + uint64(mrg32k3a_a23n) * uint64(vs0)
            phi <- prod >>> 32 |> uint32; plo <- prod &&& uint64(mrg32k3a_2pow32m1) |> uint32

            prod <- uint64(plo) + uint64(phi) * uint64(mrg32k3a_m2c)
            phi <- prod >>> 32 |> uint32; pl2 <- prod &&& uint64(mrg32k3a_2pow32m1) |> uint32

            plo <- pl2 + __nv_umul24 phi mrg32k3a_m2c
            if plo >= mrg32k3a_m2 || plo < pl2 then plo <- plo - mrg32k3a_m2

            v2.[0] <- v2.[1]; v2.[1] <- v2.[2]; v2.[2] <- plo

            p := v1.[2] - v2.[2]
            if v1.[2] <= v2.[2] then p := !p + mrg32k3a_m1 @>
        |> Compiler.DefineFunction

    let init (program:Program) =
        let mrg32k3a_v1 = program.Apply(mrg32k3a_v1)
        let mrg32k3a_A1 = program.Apply(mrg32k3a_A1)
        let mrg32k3a_v2 = program.Apply(mrg32k3a_v2)
        let mrg32k3a_A2 = program.Apply(mrg32k3a_A2)

        let init (v1o:uint32[]) (v2o:uint32[]) (offset:int) =
            if offset < 0 then failwithf "offset(%d) < 0" offset
            let mutable offset = offset
            let v1b = Array.zeroCreate<uint64>(3)
            let v2b = Array.zeroCreate<uint64>(3)
            let A1 = Array.zeroCreate<uint32>(9 * 148)
            let A2 = Array.zeroCreate<uint32>(9 * 148)
            let v1 = Array.zeroCreate<uint32>(3)
            let v2 = Array.zeroCreate<uint32>(3)

            for i = 0 to 2 do
                v1.[i] <- v1o.[i]
                v2.[i] <- v2o.[i]

            A1.[0] <- 0u;           A1.[3] <- 1u;           A1.[6] <- 0u
            A1.[1] <- 0u;           A1.[4] <- 0u;           A1.[7] <- 1u
            A1.[2] <- mrg32k3a_a13; A1.[5] <- mrg32k3a_a12; A1.[8] <- 0u

            A2.[0] <- 0u;           A2.[3] <- 1u;           A2.[6] <- 0u
            A2.[1] <- 0u;           A2.[4] <- 0u;           A2.[7] <- 1u
            A2.[2] <- mrg32k3a_a23; A2.[5] <- 0u;           A2.[8] <- mrg32k3a_a21

            for n = 0 to 20 do
                for m = 0 to 6 do
                    for i = 0 to 2 do
                        for j = 0 to 2 do
                            let mutable a1 = 0UL
                            let mutable a2 = 0UL
                            for k = 0 to 2 do
                                a1 <- a1 + ( (uint64(A1.[i + 3 * k + 63 * n]))
                                           * (uint64(A1.[k + 3 * j + 9 * m + 63 * n])) ) % uint64(mrg32k3a_m1)
                                a2 <- a2 + ( (uint64(A2.[i + 3 * k + 63 * n]))
                                           * (uint64(A2.[k + 3 * j + 9 * m + 63 * n])) ) % uint64(mrg32k3a_m2)
                            A1.[i + 3 * j + 9 * (m + 1) + 63 * n] <- uint32(a1 % uint64(mrg32k3a_m1))
                            A2.[i + 3 * j + 9 * (m + 1) + 63 * n] <- uint32(a2 % uint64(mrg32k3a_m2))

                let m = offset % 8 - 1
                if m >= 0 then
                    for i = 0 to 2 do
                        v1b.[i] <- v1.[i] |> uint64
                        v2b.[i] <- v2.[i] |> uint64

                    for i = 0 to 2 do
                        v1.[i] <- ( (uint64(A1.[i + 3 * 0 + 9 * m + 63 * n]) * v1b.[0]) % uint64(mrg32k3a_m1)
                                  + (uint64(A1.[i + 3 * 1 + 9 * m + 63 * n]) * v1b.[1]) % uint64(mrg32k3a_m1)
                                  + (uint64(A1.[i + 3 * 2 + 9 * m + 63 * n]) * v1b.[2]) % uint64(mrg32k3a_m1) ) % uint64(mrg32k3a_m1) |> uint32
                        v2.[i] <- ( (uint64(A2.[i + 3 * 0 + 9 * m + 63 * n]) * v2b.[0]) % uint64(mrg32k3a_m2)
                                  + (uint64(A2.[i + 3 * 1 + 9 * m + 63 * n]) * v2b.[1]) % uint64(mrg32k3a_m2)
                                  + (uint64(A2.[i + 3 * 2 + 9 * m + 63 * n]) * v2b.[2]) % uint64(mrg32k3a_m2) ) % uint64(mrg32k3a_m2) |> uint32
                offset <- offset / 8

            mrg32k3a_v1.Scatter(v1)
            mrg32k3a_A1.Scatter(A1)
            mrg32k3a_v2.Scatter(v2)
            mrg32k3a_A2.Scatter(A2)

        init

    return { StreamInit = streamInit; NextStep = nextStep; Init = init } }

type Raw =
    {
        Kernel : Resources.Kernel<int -> deviceptr<uint32> -> unit>
        Init : Program -> uint32[] -> uint32[] -> int -> unit
    }

let raw (core:Core) : Template<Raw> = cuda {
    let! kernel =
        <@ fun (np:int) (points:deviceptr<uint32>) ->
            let v1 = __local__.Array<uint32>(3) |> __ptrofarray
            let v2 = __local__.Array<uint32>(3) |> __ptrofarray
            let p = ref 0u
            core.StreamInit.Invoke v1 v2 np
            let mutable i = threadIdx.x + np * blockDim.x * blockIdx.x
            for n = 0 to np - 1 do
                core.NextStep.Invoke v1 v2 p
                points.[i] <- !p
                i <- i + blockDim.x @>
        |> Compiler.DefineKernel

    return { Kernel = kernel; Init = core.Init } }

type Uniform<'T> =
    {
        Kernel : Resources.Kernel<int -> deviceptr<'T> -> unit>
        Init : Program -> uint32[] -> uint32[] -> int -> unit
    }

let inline uniform (real:RealConverter<'T>) (core:Core) : Template<Uniform<'T>> = cuda {
    let! next =
        <@ fun (v1:deviceptr<uint32>) (v2:deviceptr<uint32>) (x:'T ref) ->
            let p = ref 0u
            core.NextStep.Invoke v1 v2 p
            x := (real.Of mrg32k3a_norm) * (real.Of !p) @>
        |> Compiler.DefineFunction

    let! kernel =
        <@ fun (np:int) (numbers:deviceptr<'T>) ->
            let v1 = __local__.Array<uint32>(3) |> __ptrofarray
            let v2 = __local__.Array<uint32>(3) |> __ptrofarray
            let x : 'T ref = ref (__nan())
            core.StreamInit.Invoke v1 v2 np
            let mutable i = threadIdx.x + np * blockDim.x * blockIdx.x
            for n = 0 to np - 1 do
                next.Invoke v1 v2 x
                numbers.[i] <- !x
                i <- i + blockDim.x @>
        |> Compiler.DefineKernel

    return { Kernel = kernel; Init = core.Init } }
           
type Exponential<'T> =
    {
        Kernel : Resources.Kernel<int -> deviceptr<'T> -> unit>
        Init : Program -> uint32[] -> uint32[] -> int -> unit
    }

let inline exponential (real:RealConverter<'T>) (core:Core) : Template<Exponential<'T>> = cuda {
    let! next =
        <@ fun (v1:deviceptr<uint32>) (v2:deviceptr<uint32>) (x:'T ref) ->
            let p = ref 0u
            core.NextStep.Invoke v1 v2 p
            x := log ((real.Of mrg32k3a_norm) * (real.Of !p)) @>
        |> Compiler.DefineFunction

    let! kernel =
        <@ fun (np:int) (numbers:deviceptr<'T>) ->
            let v1 = __local__.Array<uint32>(3) |> __ptrofarray
            let v2 = __local__.Array<uint32>(3) |> __ptrofarray
            let x : 'T ref = ref (__nan())
            core.StreamInit.Invoke v1 v2 np
            let mutable i = threadIdx.x + np * blockDim.x * blockIdx.x
            for n = 0 to np - 1 do
                next.Invoke v1 v2 x
                numbers.[i] <- !x
                i <- i + blockDim.x @>
        |> Compiler.DefineKernel

    return { Kernel = kernel; Init = core.Init } }
           
type Normal<'T> =
    {
        Kernel : Resources.Kernel<int -> deviceptr<'T> -> unit>
        Init : Program -> uint32[] -> uint32[] -> int -> unit
    }

let inline normal (real:RealConverter<'T>) (core:Core) : Template<Normal<'T>> = cuda {
    let! next =
        <@ fun (v1:deviceptr<uint32>) (v2:deviceptr<uint32>) (x0:'T ref) (x1:'T ref) ->
            if __isnan(!x1) then
                let p = ref 0u

                core.NextStep.Invoke v1 v2 p
                let x2 = sqrt ((real.Of -2) * log((real.Of mrg32k3a_norm) * (real.Of !p)))

                core.NextStep.Invoke v1 v2 p
                __sincos ((real.Of 2) * (real.Of mrg32k3a_pi) * (real.Of mrg32k3a_norm) * (real.Of !p)) x0 x1

                x0 := !x0 * x2
                x1 := !x1 * x2
            else
                x0 := !x1
                x1 := __nan() @>
        |> Compiler.DefineFunction

    let! kernel =
        <@ fun (np:int) (numbers:deviceptr<'T>) ->
            let v1 = __local__.Array<uint32>(3) |> __ptrofarray
            let v2 = __local__.Array<uint32>(3) |> __ptrofarray
            let x0 : 'T ref = ref (__nan())
            let x1 : 'T ref = ref (__nan())
            core.StreamInit.Invoke v1 v2 np
            let mutable i = threadIdx.x + np * blockDim.x * blockIdx.x
            for n = 0 to np - 1 do
                next.Invoke v1 v2 x0 x1
                numbers.[i] <- !x0
                i <- i + blockDim.x @>
        |> Compiler.DefineKernel

    return { Kernel = kernel; Init = core.Init } }

type Gamma<'T> =
    {
        Kernel : Resources.Kernel<int -> 'T -> deviceptr<'T> -> unit>
        Init : Program -> uint32[] -> uint32[] -> int -> unit
    }

let inline gamma (real:RealConverter<'T>) (core:Core) : Template<Gamma<'T>> = cuda {
    let! nextUniform =
        <@ fun (v1:deviceptr<uint32>) (v2:deviceptr<uint32>) (x:'T ref) ->
            let p = ref 0u
            core.NextStep.Invoke v1 v2 p
            x := (real.Of mrg32k3a_norm) * (real.Of !p) @>
        |> Compiler.DefineFunction

    let! nextNormal =
        <@ fun (v1:deviceptr<uint32>) (v2:deviceptr<uint32>) (x0:'T ref) (x1:'T ref) ->
            if __isnan(!x1) then
                let p = ref 0u
                core.NextStep.Invoke v1 v2 p
                let x2 = sqrt((real.Of -2) * log((real.Of mrg32k3a_norm) * (real.Of !p)))
                core.NextStep.Invoke v1 v2 p
                __sincos (2G * (real.Of mrg32k3a_pi) * (real.Of mrg32k3a_norm) * (real.Of !p)) x0 x1
                x0 := !x0 * x2
                x1 := !x1 * x2
            else
                x0 := !x1
                x1 := __nan() @>
        |> Compiler.DefineFunction

    let! nextGamma =
        <@ fun (v1:deviceptr<uint32>) (v2:deviceptr<uint32>) (alpha:'T) (x:'T ref) (x1:'T ref) ->
            let mutable alpha = alpha
            let mutable prefix = 0G
            let u = ref 0G

            if alpha <= 1G then
                nextUniform.Invoke v1 v2 u
                prefix <- __pow !u (1G/alpha)
                alpha <- alpha + 1G
            else
                prefix <- 1G

            let d : 'T = alpha - 1G/3G
            let c : 'T = __rsqrt(9G*d)

            //x := 0G
            //u := 0G
            let mutable v = (real.Of -1)

            while (v <= 0G) || (log(!u) >= (real.Of 0.5) * !x * !x + d * (1G - v + log(v))) do
                nextNormal.Invoke v1 v2 x x1
                v <- 1G + c * !x
                v <- v * v * v
                nextUniform.Invoke v1 v2 u

            x := !x @>
            //x := prefix * d * v @>
        |> Compiler.DefineFunction

    let! kernel =
        <@ fun (np:int) (alpha:'T) (numbers:deviceptr<'T>) ->
            let v1 = __local__.Array<uint32>(3) |> __ptrofarray
            let v2 = __local__.Array<uint32>(3) |> __ptrofarray
            let x : 'T ref = ref (__nan())
            let x2 : 'T ref = ref (__nan())
            core.StreamInit.Invoke v1 v2 (np*8)
            //core.StreamInit.Invoke v1 v2 np
            let mutable i = threadIdx.x + np * blockDim.x * blockIdx.x
            for n = 0 to np - 1 do
                nextGamma.Invoke v1 v2 alpha x x2
                numbers.[i] <- !x
                i <- i + blockDim.x @>
        |> Compiler.DefineKernel

    return { Kernel = kernel; Init = core.Init } }

let reorder (nb:int) (nt:int) (np:int) (numbers:'T[]) =
    let n = numbers.Length
    let numbers' = Array.zeroCreate<'T> n
    for ib = 0 to nb - 1 do
        for it = 0 to nt - 1 do
            let istream = ib * nt + it
            for ip = 0 to np - 1 do
                numbers'.[ib * nt * np + ip * nt + it] <- numbers.[istream * np + ip]
    numbers'

