module Alea.CUDA.Utilities.Util

open Alea.CUDA

#if DEBUG
let debug = true
#else
let debug = false
#endif

let compileWithOptions (options:CompileOptions) (template:Template<'T>) =
    Compiler.Compile(template, options) |> function
    | CompileResult.Success(irm, warnings) ->
        warnings |> Array.iter (fun (src, warning) -> printfn "%A: %A" src warning)
        if debug then
            printfn "==== IRMODULE ===="
            irm.Dump()
            printfn "=================="
        irm
    | CompileResult.Fail(src, err, ex) ->
        printfn "%A" err
        printfn "Location: %s" (src.ToString())
        failwith "compile fail"

let compile (template:Template<'T>) = template |> compileWithOptions CompileOptions.Default

let linkWithLibraries (libraries:IRModule list) (irm:IRModule<'T>) =
    Compiler.Link(irm, libraries) |> function
    | LinkResult.Success(ptxm, log) ->
        if log.Length > 0 then
            printfn "===== Link Log ====="
            printfn "%s" log
            printfn "===================="
        if debug then
            printfn "==== PTXMODULE ===="
            ptxm.Dump()
            printfn "==================="
        ptxm
    | LinkResult.Fail(err) ->
        printfn "%A" err
        failwith "link fail"

let link (irm:IRModule<'T>) = irm |> linkWithLibraries List.empty

let load (worker:Worker) (template:Template<Entry<'T>>) =
    if not debug then worker.LoadProgram(template)
    else template |> compile |> link |> worker.LoadProgram

let loadWithLibraries (worker:Worker) (libraries:IRModule list) (template:Template<Entry<'T>>) =
    if not debug then worker.LoadProgram(template, libraries)
    else template |> compile |> linkWithLibraries libraries |> worker.LoadProgram

