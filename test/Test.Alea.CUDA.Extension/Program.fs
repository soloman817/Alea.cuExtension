//Test.Alea.CUDA.Extension.Sobol.``32 x 256 1``()
printfn "for test"

let foo a b c =
    printfn "foo: a=%A" a
    printfn "foo: b=%A" b
    printfn "foo: c=%A" c
    a + b + c

let bar = 
    printfn "bar"
    foo 10.0


bar 1.0 2.0 |> ignore