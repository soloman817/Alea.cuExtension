open System
open System.IO
open Alea.CUDA.Extension.Random.SobolJoeKuo_6_21201

let file = new StreamWriter(@"..\..\..\Alea.CUDA.Extension.Random\SobolJoeKuo_6_21201.fs")

fprintfn file "module Alea.CUDA.Extension.Random.SobolJoeKuo_6_21201"
fprintfn file ""

fprintfn file "let d = [|"
primitivePolynomials |> Array.iter (fun (_, d, _, _) -> fprintfn file "    %d" d)
fprintfn file "    |]"
fprintfn file ""

fprintfn file "let a = [|"
primitivePolynomials |> Array.iter (fun (_, _, a, _) -> fprintfn file "    %du" a)
fprintfn file "    |]"
fprintfn file ""

fprintfn file "let o = [|"
primitivePolynomials
|> Array.scan (fun s (_, _, _, m) -> s + m.Length) 0
|> Array.iter (fun x -> fprintfn file "    %d" x)
fprintfn file "    |]"
fprintfn file ""

fprintfn file "let m = [|"
primitivePolynomials
|> Array.iter (fun (_, _, _, m) ->
    fprintf file "    "
    m |> Array.iter (fun x -> fprintf file "%du;" x)
    fprintfn file "")
fprintfn file "    |]"
fprintfn file ""

file.Dispose()
