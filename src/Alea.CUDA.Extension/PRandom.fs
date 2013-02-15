module Alea.CUDA.Extension.PRandom

open Alea.CUDA

let sobol converter = cuda {
    let! generator = Sobol.generator converter

    return PFunc(fun (m:Module) dimensions vectors offset (output:PArray<'T>) ->
        let generator = generator.Invoke m dimensions
        use directions = PArray.Create(m.Worker, generator.Directions)
        generator.Generate vectors offset directions output) }

let sobol' converter = Sobol.generator converter

