#I "../../FSharp.Formatting/bin"
#load "../../FSharp.Formatting/literate/literate.fsx"
open FSharp.Literate
open System.IO

//let source = "../../src/Alea.CUDA.Extension.MGPU"
let source = __SOURCE_DIRECTORY__
//printfn "%s" source
let template = Path.Combine(__SOURCE_DIRECTORY__, "template-file.html")

let script = Path.Combine(source, "../../src/Alea.CUDA.Extension.MGPU/Script1.fsx")
Literate.ProcessScriptFile(script, template)
