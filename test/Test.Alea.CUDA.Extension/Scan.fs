module Test.Alea.CUDA.Extension.Scan

//open System
//open NUnit.Framework
//open Alea.CUDA
//open Alea.CUDA.Extension
//open Alea.CUDA.Extension.Reduce
//open Alea.CUDA.Extension.Scan
//
//let rng = System.Random()
//
//let sizes = [12; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608; 33554432]
//
//[<Test>]
//let ``scan sum<int>`` () =
//    let worker = getDefaultWorker()
//    let scan = worker.LoadPModule(scan plan32).Invoke
//    let scan values incl = scan.Scan(values, incl)
//
//    let test init n =
//        let values = Array.init n (init)        
//        printfn "size = %d, values = %A" n values
//        let scanValues = values |> Array.scan (+) 0
//        let expectedExcl = Array.sub scanValues 0 values.Length
//        let scanExcl = scan values false
//        let scanIncl = scan values true
//        printfn "expectedExcl = %A" expectedExcl
//        printfn "scanExcl = %A" scanExcl
//        printfn "scanIncl = %A" scanIncl
//        expectedExcl |> Array.iteri (fun i e -> Assert.AreEqual(e, scanExcl.[i]))
//
//    sizes |> Seq.iter (test (fun _ -> 1))
//    sizes |> Seq.iter (test (fun _ -> rng.Next(-100, 100)))
//
//[<Test>]
//let ``scan generic sum<int>`` () =
//    let worker = getDefaultWorker()
//    let scan = worker.LoadPModule(genericScan plan32 <@ fun () -> 0 @> <@ (+) @> <@ fun x -> x @>).Invoke
//    let scan values incl = scan.Scan(values, incl)
//
//    let test init n =
//        let values = Array.init n (init)        
//        printfn "size = %d, values = %A" n values
//        let scanValues = values |> Array.scan (+) 0
//        let expectedExcl = Array.sub scanValues 0 values.Length
//        let scanExcl = scan values false
//        let scanIncl = scan values true
//        printfn "expectedExcl = %A" expectedExcl
//        printfn "scanExcl = %A" scanExcl
//        printfn "scanIncl = %A" scanIncl
//        expectedExcl |> Array.iteri (fun i e -> Assert.AreEqual(e, scanExcl.[i]))
//
//    sizes |> Seq.iter (test (fun _ -> 1))
//    sizes |> Seq.iter (test (fun _ -> rng.Next(-100, 100)))
//
//[<Test>]
//let ``scan generic sum<float>`` () =
//    let worker = getDefaultWorker()
//    let scan = worker.LoadPModule(genericScan plan64 <@ fun () -> 0.0 @> <@ (+) @> <@ fun x -> x @>).Invoke
//    let scan values incl = scan.Scan(values, incl)
//
//    let test init n =
//        let values = Array.init n (init)        
//        printfn "size = %d, values = %A" n values
//        let scanValues = values |> Array.scan (+) 0.0
//        let expectedExcl = Array.sub scanValues 0 values.Length
//        let scanExcl = scan values false
//        let scanIncl = scan values true
//        printfn "expectedExcl = %A" expectedExcl
//        printfn "scanExcl = %A" scanExcl
//        printfn "scanIncl = %A" scanIncl
//        expectedExcl |> Array.iteri (fun i e -> let err = abs (e - scanExcl.[i]) / (e+1.0) in Assert.That(err < 1e-8))
//
//    sizes |> Seq.iter (test (fun _ -> 1.0))
//    sizes |> Seq.iter (test (fun _ -> rng.NextDouble()))
//
//[<Test>]
//let ``scan generic sum squared<float>`` () =
//    let worker = getDefaultWorker()
//    let scan = worker.LoadPModule(genericScan plan64 <@ fun () -> 0.0 @> <@ (+) @> <@ fun x -> x*x @>).Invoke
//    let scan values incl = scan.Scan(values, incl)
//
//    let test init n =
//        let values = Array.init n (init)        
//        printfn "size = %d, values = %A" n values
//        let scanValues = values |> Array.map (fun x -> x*x) |> Array.scan (+) 0.0
//        let expectedExcl = Array.sub scanValues 0 values.Length
//        let scanExcl = scan values false
//        let scanIncl = scan values true
//        printfn "expectedExcl = %A" expectedExcl
//        printfn "scanExcl = %A" scanExcl
//        printfn "scanIncl = %A" scanIncl
//        expectedExcl |> Array.iteri (fun i e -> let err = abs (e - scanExcl.[i]) / (e+1.0) in Assert.That(err < 1e-8))
//
//    sizes |> Seq.iter (test (fun _ -> rng.NextDouble()))
//
//[<Test>]
//let ``scan generic max<float>`` () =
//    let worker = getDefaultWorker()
//    let scan = worker.LoadPModule(genericScan plan64 <@ fun () -> Double.NegativeInfinity @> <@ max @> <@ fun x -> x @>).Invoke
//    let scan values incl = scan.Scan(values, incl)
//
//    let test init n =
//        let values = Array.init n (init)        
//        printfn "size = %d, values = %A" n values
//        let scanValues = values |> Array.scan (max) Double.NegativeInfinity
//        let expectedExcl = Array.sub scanValues 0 values.Length
//        let scanExcl = scan values false
//        let scanIncl = scan values true
//        printfn "expectedExcl = %A" expectedExcl
//        printfn "scanExcl = %A" scanExcl
//        printfn "scanIncl = %A" scanIncl
//        expectedExcl |> Array.iteri (fun i e -> if i > 0 then let err = abs (e - scanExcl.[i]) / (e+1.0) in Assert.That(err < 1e-8))
//
//    sizes |> Seq.iter (test (fun _ -> rng.NextDouble()))
//
//[<Test>]
//let ``scan generic min<float>`` () =
//    let worker = getDefaultWorker()
//    let scan = worker.LoadPModule(genericScan plan64 <@ fun () -> Double.PositiveInfinity @> <@ min @> <@ fun x -> x @>).Invoke
//    let scan values incl = scan.Scan(values, incl)
//
//    let test init n =
//        let values = Array.init n (init)        
//        printfn "size = %d, values = %A" n values
//        let scanValues = values |> Array.scan (min) Double.PositiveInfinity
//        let expectedExcl = Array.sub scanValues 0 values.Length
//        let scanExcl = scan values false
//        let scanIncl = scan values true
//        printfn "expectedExcl = %A" expectedExcl
//        printfn "scanExcl = %A" scanExcl
//        printfn "scanIncl = %A" scanIncl
//        expectedExcl |> Array.iteri (fun i e -> if i > 0 then let err = abs (e - scanExcl.[i]) / (e+1.0) in Assert.That(err < 1e-8))
//
//    sizes |> Seq.iter (test (fun _ -> rng.NextDouble()))
//
//
//    