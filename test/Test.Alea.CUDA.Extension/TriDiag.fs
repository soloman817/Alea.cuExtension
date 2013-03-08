module Test.Alea.CUDA.Extension.TriDiag

open System
open NUnit.Framework
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.TriDiag

module ForsytheMoler =

    /// Solves tridiagonal system according to   
    /// G.E. Forsythe and C.B. Moler: Computer solutions to linear algebraic systems, Prentice-Hall, 1967
    let solve (e:float[]) (d:float[]) (f:float[]) (b:float[]) =
        let n = e.Length
        let m = Array.zeroCreate n
        let u = Array.zeroCreate n
        let x = Array.zeroCreate n
        u.[0] <- d.[0]
        for i = 1 to n-1 do
            m.[i] <- e.[i]/u.[i-1]
            u.[i] <- d.[i] - m.[i]*f.[i-1]
        
        x.[0] <- b.[0]
        for i = 1 to n-1 do
            x.[i] <- b.[i] - m.[i]*x.[i-1]

        x.[n-1] <- x.[n-1]/u.[n-1]
        for i = n-2 downto 0 do
            x.[i] <- (x.[i] - f.[i]*x.[i+1])/u.[i] 
            
        x   

type TriDiag =
    { lower:float[]; diag:float[]; upper:float[] }
    
    member this.apply (x:float[]) =
        let n = this.diag.Length
        let b = Array.zeroCreate n
        b.[0] <- this.diag.[0]*x.[0] + this.upper.[0]*x.[1]
        for i = 1 to n-2 do
            b.[i] <- this.lower.[i]*x.[i-1] + this.diag.[i]*x.[i] + this.upper.[i]*x.[i+1]
        b.[n-1] <- this.lower.[n-1]*x.[n-2] + this.diag.[n-1]*x.[n-1]
        b

    member this.solve b =
        ForsytheMoler.solve this.lower this.diag this.upper b

let rng = Random(2)

let createSystem n sigma h =
    let l = Array.init n (fun i -> if i = 0 then 0.0 else sigma/(h*h))
    let d = Array.init n (fun i -> -2.0*sigma/(h*h))
    let u = Array.init n (fun i -> if i = n-1 then 0.0 else sigma/(h*h))
    {lower = l; diag = d; upper = u}
 
let inline maxErr (b:'T[]) (b':'T[]) =
    Array.map2 (fun bi bi' -> abs (bi - bi')) b b' |> Array.max

[<Test>]
let ``tridiag ForsytheMoler cpu`` () =

    let test n tol =
        let A = createSystem n 0.1 0.01
        let b = Array.init n (fun _ -> 1.0)
        let x = A.solve b
        let b' = A.apply x

        let err = maxErr b b'       
        Assert.IsTrue(err < tol)

        //printfn "dim = %d, err = %f" n err

        ()

    [4; 8; 16; 18; 20; 22] |> List.iter (fun e -> test (2<<<e) 0.005)


[<Test>]
let ``tridiag pcr single block gpu`` () =

    let worker = getDefaultWorker()
    let triPcrM = worker.LoadPModule(triDiag ())
    let triPcr = triPcrM.Invoke

    let test n tol =
        let A = createSystem n 0.1 0.01
        let b = Array.init n (fun _ -> 1.0)
        let x = A.solve b
        let x' = triPcr A.lower A.diag A.upper b

        let b1 = A.apply x
        let b1' = A.apply x'

        let xErr = maxErr x x'
        let bErrCpu = maxErr b b1
        let bErrGpu = maxErr b b1'

        printfn "n = %d, xErr = %e, bErrGpu = %e, bErrCpu = %e" n xErr bErrGpu bErrCpu 
        
        //Assert.IsTrue(err < tol)

        //printfn "dim = %d, err = %f" n err

        ()

    [32; 64; 128; 512; 1024] |> List.iter (fun n -> test n 0.005)



