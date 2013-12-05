module Sample.TriDiag.Common

open Alea.CUDA.TestUtilities

type TriDiag = float[] * float[] * float[] // l, d, u
type Generator = int -> TriDiag * float[] * float // n -> A, b, tol
type Solver = TriDiag -> float[] -> float[] // A -> b -> x

let apply (A:TriDiag) (x:float[]) =
    let l, d, u = A
    let n = d.Length
    let b = Array.zeroCreate n
    b.[0] <- d.[0] * x.[0] + u.[0] * x.[1]
    for i = 1 to n - 2 do
        b.[i] <- l.[i] * x.[i - 1] + d.[i] * x.[i] + u.[i] * x.[i + 1]
    b.[n - 1] <- l.[n - 1] * x.[n - 2] + d.[n - 1] * x.[n - 1]
    b

let maxError (b:float[]) (b':float[]) =
    (b, b') ||> Array.map2 ( - ) |> Array.map abs |> Array.max

let generate1 sigma h n =
    let l = Array.init n (fun i -> if i = 0 then 0.0 else sigma / (h * h))
    let d = Array.init n (fun i -> -2.0 * sigma / (h * h))
    let u = Array.init n (fun i -> if i = n - 1 then 0.0 else sigma / (h * h))
    let A = l, d, u
    let b = Array.init n (fun _ -> 1.0)
    let tol = if n > 1024 then 0.005 else 1e-9
    A, b, tol

let generate2 n =
    let l = Array.init n (TestUtil.genRandomDouble -100.0 100.0)
    let d = Array.init n (TestUtil.genRandomDouble -100.0 100.0)
    let u = Array.init n (TestUtil.genRandomDouble -100.0 100.0)
    let x = Array.init n (TestUtil.genRandomDouble -20.0 20.0)
    let A = l, d, u
    let b = apply A x
    let tol = 0.0007
    A, b, tol
