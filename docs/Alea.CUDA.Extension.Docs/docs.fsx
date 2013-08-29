#I "../../packages/FSharp.Formatting.1.0.15/lib/net40"
#load "../../packages/FSharp.Formatting.1.0.15/literate/literate.fsx"
open FSharp.Literate
open System.IO

let source = __SOURCE_DIRECTORY__
printfn "%s" source
let template = Path.Combine(source, "template-project.html")

let script = Path.Combine(source, "Script1.fsx")
Literate.ProcessScriptFile(script, template)
