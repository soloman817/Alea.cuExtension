module Test.Alea.CUDA.Extension.SegmentedScan

open System
open Microsoft.FSharp.Quotations
open NUnit.Framework
open Alea.Interop.CUDA
open Alea.CUDA
open Alea.CUDA.Extension

let worker = getDefaultWorker()
let sizes = [12; 2597152; 128; 4002931; 511; 1024; 8191; 1200; 4096; 5000; 8192; 12; 8193; 9000; 10000; 2097152]

let zeroCreate : int -> PCalc<DArray<int>> = worker.LoadPModule(PArray.zeroCreate()).Invoke

let createFlags (n:int) (headIndices:int[]) = pcalc {
    let! flags = zeroCreate n
    let setIndices (hint:ActionHint) =
        fun () ->
            for i = 0 to headIndices.Length - 2 do // bypass the last one becuase it is the cpu scan algorithm
                let ptr = flags.Ptr + headIndices.[i]
                cuSafeCall(cuMemsetD32Async(ptr.Handle, 1u, 1n, hint.Stream.Handle))
        |> worker.Eval
    do! PCalc.action setIndices
    return flags }

[<Test>]
let ``sumsegscan: int``() =
    let scan = worker.LoadPModule(PArray.sumsegscan()).Invoke
    let test verify (hValuess:int[][]) = pcalc {
        let headIndices = hValuess |> Array.map (fun x -> x.Length) |> Array.scan (+) 0

        let hValues = hValuess |> Array.concat
        let! dValues = DArray.scatterInBlob worker hValues
        let! dFlags = createFlags dValues.Length headIndices
        let! dResultsIncl = scan true dValues dFlags
        let! dResultsExcl = scan false dValues dFlags
        
        match verify with
        | true ->
            let hResultss = hValuess |> Array.map (fun x -> x |> Array.scan (+) 0)

            // check inclusive
            let hResults = hResultss |> Array.map (fun x -> Array.sub x 1 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsIncl.Gather()
            (hResults, dResults) ||> Array.iteri2 (fun i h d -> Assert.AreEqual(d, h))

            // check exclusive
            let hResults = hResultss |> Array.map (fun x -> Array.sub x 0 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsExcl.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.AreEqual(d, h))
        | false -> do! PCalc.force() }

    let valuess1() = sizes |> Seq.map (fun n -> Array.init n (fun _ -> 1)) |> Array.ofSeq
    let valuess2() = let rng = Random(2) in sizes |> Seq.map (fun n -> Array.init n (fun _ -> rng.Next(-100, 100))) |> Array.ofSeq

    test true (valuess1()) |> PCalc.run
    test true (valuess2()) |> PCalc.run

    let test = test false (valuess2())
    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``sumsegscan: float32``() =
    let scan = worker.LoadPModule(PArray.sumsegscan()).Invoke
    let test (verify:float option) (hValuess:float32[][]) = pcalc {
        let headIndices = hValuess |> Array.map (fun x -> x.Length) |> Array.scan (+) 0

        let hValues = hValuess |> Array.concat
        let! dValues = DArray.scatterInBlob worker hValues
        let! dFlags = createFlags dValues.Length headIndices
        let! dResultsIncl = scan true dValues dFlags
        let! dResultsExcl = scan false dValues dFlags
        
        match verify with
        | Some(eps) ->
            let hResultss = hValuess |> Array.map (fun x -> x |> Array.scan (+) 0.0f)

            // check inclusive
            let hResults = hResultss |> Array.map (fun x -> Array.sub x 1 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsIncl.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            // check exclusive
            let hResults = hResultss |> Array.map (fun x -> Array.sub x 0 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsExcl.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
        | None -> do! PCalc.force() }

    let eps = Some 1e-1
    let valuess1() = sizes |> Seq.map (fun n -> Array.init n (fun _ -> 1.0f)) |> Array.ofSeq
    let valuess2() = let rng = Random(2) in sizes |> Seq.map (fun n -> Array.init n (fun _ -> (rng.NextDouble() - 0.5) |> float32)) |> Array.ofSeq

    test eps (valuess1()) |> PCalc.run
    test eps (valuess2()) |> PCalc.run

    let test = test None (valuess2())
    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``sumsegscan: float``() =
    let scan = worker.LoadPModule(PArray.sumsegscan()).Invoke
    let test (verify:float option) (hValuess:float[][]) = pcalc {
        let headIndices = hValuess |> Array.map (fun x -> x.Length) |> Array.scan (+) 0

        let hValues = hValuess |> Array.concat
        let! dValues = DArray.scatterInBlob worker hValues
        let! dFlags = createFlags dValues.Length headIndices
        let! dResultsIncl = scan true dValues dFlags
        let! dResultsExcl = scan false dValues dFlags
        
        match verify with
        | Some(eps) ->
            let hResultss = hValuess |> Array.map (fun x -> x |> Array.scan (+) 0.0)

            // check inclusive
            let hResults = hResultss |> Array.map (fun x -> Array.sub x 1 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsIncl.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

            // check exclusive
            let hResults = hResultss |> Array.map (fun x -> Array.sub x 0 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsExcl.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
        | None -> do! PCalc.force() }

    let eps = Some 1e-10
    let valuess1() = sizes |> Seq.map (fun n -> Array.init n (fun _ -> 1.0)) |> Array.ofSeq
    let valuess2() = let rng = Random(2) in sizes |> Seq.map (fun n -> Array.init n (fun _ -> rng.NextDouble() - 0.5)) |> Array.ofSeq

    test eps (valuess1()) |> PCalc.run
    test eps (valuess2()) |> PCalc.run

    let test = test None (valuess2())
    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))

[<Test>]
let ``sumsegscan: float32 debug``() =
    let scan = worker.LoadPModule(PArray.sumsegscan()).Invoke
    let test (verify:float option) (hValuess:float32[][]) = pcalc {
        let headIndices = hValuess |> Array.map (fun x -> x.Length) |> Array.scan (+) 0

        let hValues = hValuess |> Array.concat
        let! dValues = DArray.scatterInBlob worker hValues
        let! dFlags = createFlags dValues.Length headIndices
        let! dResultsIncl = scan true dValues dFlags
//        let! dResultsExcl = scan false dValues dFlags
        
        match verify with
        | Some(eps) ->
            let hResultss = hValuess |> Array.map (fun x -> x |> Array.scan (+) 0.0f)

            // check inclusive
            let hResults = hResultss |> Array.map (fun x -> Array.sub x 1 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsIncl.Gather()
            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))

//            // check exclusive
//            let hResults = hResultss |> Array.map (fun x -> Array.sub x 0 (x.Length - 1)) |> Array.concat
//            let! dResults = dResultsExcl.Gather()
//            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
        | None -> do! PCalc.force() }

    let sizes = [4096*2]
    let eps = Some 1e-1
    let valuess1() = sizes |> Seq.map (fun n -> Array.init n (fun _ -> 1.0f)) |> Array.ofSeq
    let valuess2() = let rng = Random(2) in sizes |> Seq.map (fun n -> Array.init n (fun _ -> (rng.NextDouble() - 0.5) |> float32)) |> Array.ofSeq

    test eps (valuess1()) |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(2))

[<Test>]
let ``sumsegscan: float debug``() =
    let scan = worker.LoadPModule(PArray.sumsegscan()).Invoke
    let test (verify:float option) (hValuess:float[][]) = pcalc {
        let headIndices = hValuess |> Array.map (fun x -> x.Length) |> Array.scan (+) 0

        let hValues = hValuess |> Array.concat
        let! dValues = DArray.scatterInBlob worker hValues
        let! dFlags = createFlags dValues.Length headIndices
        let! dResultsIncl = scan true dValues dFlags
        //let! dResultsExcl = scan false dValues dFlags
        
        match verify with
        | Some(eps) ->
            let hResultss = hValuess |> Array.map (fun x -> x |> Array.scan (+) 0.0)

            // check inclusive
            let hResults = hResultss |> Array.map (fun x -> Array.sub x 1 (x.Length - 1)) |> Array.concat
            let! dResults = dResultsIncl.Gather()
            (hResults, dResults) ||> Array.iteri2 (fun i h d ->
                if h <> d then
                    printfn "#.%d d(%f) h(%f)" (i-2) dResults.[i-2] hResults.[i-2]
                    printfn "#.%d d(%f) h(%f)" (i-1) dResults.[i-1] hResults.[i-1]
                    printfn "#.%d d(%f) h(%f)" i d h
                    printfn "#.%d d(%f) h(%f)" (i+1) dResults.[i+1] hResults.[i+1]
                    printfn "#.%d d(%f) h(%f)" (i+2) dResults.[i+2] hResults.[i+2]
                Assert.That(d, Is.EqualTo(h).Within(eps)))

//            // check exclusive
//            let hResults = hResultss |> Array.map (fun x -> Array.sub x 0 (x.Length - 1)) |> Array.concat
//            let! dResults = dResultsExcl.Gather()
//            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.That(d, Is.EqualTo(h).Within(eps)))
        | None -> do! PCalc.force() }

    let sizes = [12; 2597152; 128; 4002931; 511; 1024; 8191; 1200; 4096; 5000; 8192; 12; 8193; 9000; 10000; 2097152]
    let sizes = [2597152]
    let eps = Some 1e-10
    let valuess1() = sizes |> Seq.map (fun n -> Array.init n (fun _ -> 1.0)) |> Array.ofSeq
    let valuess2() = let rng = Random(2) in sizes |> Seq.map (fun n -> Array.init n (fun _ -> rng.NextDouble() - 0.5)) |> Array.ofSeq

    test eps (valuess1()) |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(2))


//
//[<Test>]  
//let ``segmented scan reduce test max<int>`` () =
//    let worker = getDefaultWorker()   
//    let test = worker.LoadPModule(Sum.reduceTest <@(fun () -> -10)@> <@(max)@>).Invoke
//
//    let n = plan32.NumThreads
//    let v = Array.init n (fun _ -> rng.Next(-5, 5))
//    let d = test v
//    let expected = Array.max v
//
//    printfn "v = %A" v
//    printfn "d = %A" d
//    printfn "expected = %A" expected
//
//[<Test>]
//let ``segmented scan reduce test sum<int>`` () =
//    let worker = getDefaultWorker()
//    let test = worker.LoadPModule(Sum.reduceTest <@(fun () -> 0)@> <@(+)@>).Invoke
//
//    let n = plan32.NumThreads
//    let v = Array.init n (fun _ -> rng.Next(-5, 5))
//    let d = test v
//    let expected = Array.sum v
//
//    printfn "v = %A" v
//    printfn "d = %A" d
//    printfn "expected = %A" expected
//
//[<Test>]
//let ``segmented scan sum<int>`` () =
//    let worker = getDefaultWorker()
//    let scan = worker.LoadPModule(segScan ()).Invoke
//
//    let n = 200
//    let values = Array.init n (fun _ -> 1)
//    let flags = Array.zeroCreate n
//    flags.[0] <- 1
//    flags.[50] <- 1
//    flags.[100] <- 1
//    flags.[150] <- 1
//
//    let segScan = scan values flags false
//
//    printfn "segScan = %A" segScan
//
//[<Test>]
//let ``debug`` () =
//    let worker = getDefaultWorker()
//    let pfunc, irm = genirm (SegmentedScan.segScan())
//    let pfunc, ptxm = genptxm (2, 0) (pfunc, irm)
//    //ptxm.Dump()
//    //let scan = worker.LoadPModule(SegmentedScan.segScan()).Invoke
//    let scan = worker.LoadPModule(pfunc, ptxm).Invoke
//
//    let n = 20*1024 + 1
//    let values = Array.init n (fun _ -> 1.0)
//    let flags = Array.zeroCreate n
//    flags.[0] <- 1
//    flags.[10] <- 1
//    flags.[22] <- 1
//    flags.[512] <- 1
//    flags.[1024] <- 1
//    flags.[2000] <- 1
//    flags.[3000] <- 1
//    flags.[5000] <- 1
//    flags.[7000] <- 1
//
//    let segScan = scan values flags false
//
//    printfn "segScan = %A" segScan
//
//


