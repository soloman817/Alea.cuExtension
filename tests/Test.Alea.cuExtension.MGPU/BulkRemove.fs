module Test.Alea.cuBase.MGPU.BulkRemove
//
//open System
//open System.Collections.Generic
//open Alea.CUDA
////open Alea.cuBase
////open Alea.cuBase.MGPU
////open Alea.cuBase.IO.Util
//open NUnit.Framework
//
//
//let worker = getDefaultWorker()
////let pfuncts = new PBulkRemove()
//let rng = System.Random()
//
//let sourceCounts = [512; 1024; 2048; 3000; 6000; 12000; 24000; 100000; 1000000]
//let removeAmount = 2
//let removeCount c = c / removeAmount
//let removeCounts = sourceCounts |> List.map (fun x -> removeCount x)
//
//let verifyCount = 3
//
//let percentDiff x y = abs(x - y) / ((x + y) / 2.0)
//
//let hostBulkRemove (data:'T[]) (indices:int[]) =
//    let result = List<'T>()
//    let indices = indices |> Set.ofArray
//    data |> Array.iteri (fun i x -> if not (indices |> Set.contains i) then result.Add(x))
//    result.ToArray()
//
//// @COMMENTS@: index we assume always be int type
//let testBulkRemove() =
//    let test verify eps (data:'T[]) (indices:int[]) = pcalc {
//        let bulkrem = worker.LoadPModule(pfuncts.BulkRemove()).Invoke
//    
//        let n = data.Length
//        printfn "Testing size %d..." n
//
//        let! dSource = DArray.scatterInBlob worker data
//        let! dRemoveIndices = DArray.scatterInBlob worker indices
//        let! dRemoved = bulkrem dSource dRemoveIndices
//
//        if verify then
//            let hResults = hostBulkRemove data indices
//            let! dResults = dRemoved.Gather()
//            // @COMMENTS@ : cause we don't change the data (we just removed something), so it should equal
//            (hResults, dResults) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
//                    
//        else 
//            do! PCalc.force() }
//
//    let eps = 1e-10
//    let values1 n r = 
//        //let source = Array.init n (fun i -> i)
//        let (r:int[]*_) = rngGenericArrayI n r
//        let source, indices = (fst r), (snd r)
//        source, indices
//
//    let values2 n r = 
//        //let source = Array.init n (fun i -> -i)
//        let (r:int[]*_) = rngGenericArrayI n r
//        let source, indices = (fst r), (snd r)
//        source, indices
//
//    let values3 n r = 
//        let source = let rng = Random(2) in Array.init n (fun _ -> rng.NextDouble() - 0.5)
//        let (r:float[]*_) = rngGenericArrayI n r
//        let indices = (snd r)
//        source, indices  
//        
//
//    (sourceCounts, removeCounts) ||> Seq.iter2 (fun ns nr -> let test = test true eps  
//                                                             values1 ns nr ||> test |> PCalc.run)
//    (sourceCounts, removeCounts) ||> Seq.iter2 (fun ns nr -> let test = test true eps  
//                                                             values2 ns nr ||> test |> PCalc.run)
//    (sourceCounts, removeCounts) ||> Seq.iter2 (fun ns nr -> let test = test true eps  
//                                                             values3 ns nr ||> test |> PCalc.run)
//         
//    let n = 2097152
//    let test = values1 n (removeCount n) ||> test false eps
//
//    let _, loggers = test |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
//    let _, ktc = test |> PCalc.runWithKernelTiming 10 in ktc.Dump()
//    test |> PCalc.runWithDiagnoser(PCalcDiagnoser.All(1))
//   
//
//let inline verifyAll (h:'T[] list) (d:'T[] list) =
//    (h, d) ||> List.iteri2 (fun i hi di -> if i < verifyCount then Util.verify hi di)
//
// 
// 
//[<Test>]
//let ``bulkRemove moderngpu web example : float`` () =
//    let hValues = Array.init 100 float
//    let hIndices = [|  1;  4;  5;  7; 10; 14; 15; 16; 18; 19;
//                      27; 29; 31; 32; 33; 36; 37; 39; 50; 59;
//                      60; 61; 66; 78; 81; 83; 85; 90; 91; 96;
//                      97; 98; 99 |]
//    let answer = [| 0;  2;  3;  6;  8;  9; 11; 12; 13; 17; 
//                   20; 21; 22; 23; 24; 25; 26; 28; 30; 34;
//                   35; 38; 40; 41; 42; 43; 44; 45; 46; 47;
//                   48; 49; 51; 52; 53; 54; 55; 56; 57; 58;
//                   62; 63; 64; 65; 67; 68; 69; 70; 71; 72;
//                   73; 74; 75; 76; 77; 79; 80; 82; 84; 86;
//                   87; 88; 89; 92; 93; 94; 95 |] |> Array.map float
//
//    let hResult = hostBulkRemove hValues hIndices
//    (hResult, answer) ||> Array.iter2 (fun h a -> Assert.AreEqual(h, a))
//
//    let pfunct = pfuncts.BulkRemove()
//    let br = worker.LoadPModule(pfunct).Invoke
//
//    let dResult = pcalc {
//        let! data = DArray.scatterInBlob worker hValues
//        let! indices = DArray.scatterInBlob worker hIndices
//        let! removed = br data indices
//        let! results = removed.Gather()
//        return results } |> PCalc.run
//    (hResult, dResult) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
//
//[<Test>]
//let ``bulkRemove moderngpu web example : int`` () =
//    let hValues = Array.init 100 int
//    let hIndices = [|  1;  4;  5;  7; 10; 14; 15; 16; 18; 19;
//                      27; 29; 31; 32; 33; 36; 37; 39; 50; 59;
//                      60; 61; 66; 78; 81; 83; 85; 90; 91; 96;
//                      97; 98; 99 |]
//    let answer = [| 0;  2;  3;  6;  8;  9; 11; 12; 13; 17; 
//                   20; 21; 22; 23; 24; 25; 26; 28; 30; 34;
//                   35; 38; 40; 41; 42; 43; 44; 45; 46; 47;
//                   48; 49; 51; 52; 53; 54; 55; 56; 57; 58;
//                   62; 63; 64; 65; 67; 68; 69; 70; 71; 72;
//                   73; 74; 75; 76; 77; 79; 80; 82; 84; 86;
//                   87; 88; 89; 92; 93; 94; 95 |] |> Array.map int
//
//    let hResult = hostBulkRemove hValues hIndices
//    (hResult, answer) ||> Array.iter2 (fun h a -> Assert.AreEqual(h, a))
//
//    let pfunct = pfuncts.BulkRemove()
//    let br = worker.LoadPModule(pfunct).Invoke
//
//    let dResult = pcalc {
//        let! data = DArray.scatterInBlob worker hValues
//        let! indices = DArray.scatterInBlob worker hIndices
//        let! removed = br data indices
//        let! results = removed.Gather()
//        return results } |> PCalc.run
//    (hResult, dResult) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
//
//
//[<Test>]
//let ``bulkRemove moderngpu web example : float32`` () =
//    let hValues = Array.init 100 float32
//    let hIndices = [|  1;  4;  5;  7; 10; 14; 15; 16; 18; 19;
//                      27; 29; 31; 32; 33; 36; 37; 39; 50; 59;
//                      60; 61; 66; 78; 81; 83; 85; 90; 91; 96;
//                      97; 98; 99 |]
//    let answer = [| 0;  2;  3;  6;  8;  9; 11; 12; 13; 17; 
//                   20; 21; 22; 23; 24; 25; 26; 28; 30; 34;
//                   35; 38; 40; 41; 42; 43; 44; 45; 46; 47;
//                   48; 49; 51; 52; 53; 54; 55; 56; 57; 58;
//                   62; 63; 64; 65; 67; 68; 69; 70; 71; 72;
//                   73; 74; 75; 76; 77; 79; 80; 82; 84; 86;
//                   87; 88; 89; 92; 93; 94; 95 |] |> Array.map float32
//
//    let hResult = hostBulkRemove hValues hIndices
//    (hResult, answer) ||> Array.iter2 (fun h a -> Assert.AreEqual(h, a))
//
//    let pfunct = pfuncts.BulkRemove()
//    let br = worker.LoadPModule(pfunct).Invoke
//
//    let dResult = pcalc {
//        let! data = DArray.scatterInBlob worker hValues
//        let! indices = DArray.scatterInBlob worker hIndices
//        let! removed = br data indices
//        let! results = removed.Gather()
//        return results } |> PCalc.run
//    (hResult, dResult) ||> Array.iter2 (fun h d -> Assert.AreEqual(h, d))
//
//
//[<Test>]
//let ``BulkRemove 3 value test`` () =
//    testBulkRemove()