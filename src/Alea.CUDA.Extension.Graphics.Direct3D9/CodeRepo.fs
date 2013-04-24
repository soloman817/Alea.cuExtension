module Alea.CUDA.Extension.Graphics.Direct3D9.CodeRepo

open System.IO

let genSurfacePlotter(path:string) =
    let filename = "SurfacePlotter.pcr"
    printfn "Generating %s..." filename
    let filename = Path.Combine(path, filename)
    SurfacePlotter.genCodeRepo(filename)

let gen(path:string) =
    genSurfacePlotter(path)
