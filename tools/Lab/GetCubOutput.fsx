﻿open System
open System.Diagnostics
open System.IO

module CppOutput =
    let lineSequence(file) = 
        let reader = File.OpenText(file) 
        Seq.unfold(fun line -> 
            if line = null then 
                reader.Close() 
                None 
            else 
                Some(line,reader.ReadLine())) (reader.ReadLine())


      
    let working_dir = @"C:\Users\Aaron\Documents"
    

    module BlockScan =
        //open Alea.cuExtension.CUB.Block

        module example_block_scan =
            let txt = @"example_block_scan_output.txt"
            let exe = @"X:\repo\git\cub\Release\example_block_scan"
        
        type ScanTimingParams =
            {
                items               : int
                iterations          : int
                blocks              : int
                threads             : int
                items_per_thread    : int
                sm_occupancy        : int
            }
            static member Init(items, iterations, blocks, threads, items_per_thread, sm_occupancy) =
                { items = items; iterations = iterations; blocks = blocks; threads = threads; items_per_thread = items_per_thread; sm_occupancy = sm_occupancy }
        //     0    |    1    |   2  |3 |  4    |  5  |      6      |   7   |   8      |    9   |  10   |    11   |   12   |  13 |  14 | 15 |  16  |   17   | 18|  19  
        // blockscan algorithm <algo> on <items> items (<iterations> timing iterations, <blocks> blocks, <threads> threads, <ipt> items per thread, <sm_occ> SM occupancy
        module TokenId =
            let algo = 2
            let items = 4
            let iterations = 6
            let blocks = 9
            let threads = 11
            let items_per_thread = 13
            let sm_occ = 17

        let processLines (lines:seq<string>) = 
            let avgSeen = ref 0
            lines |> Seq.fold(
                fun (deviceInfo:string, algorithms:string list, timingParams:ScanTimingParams list, outputPass:bool list, aggrPass:bool list, algAverages:float list, allAverages:(float list) list) line ->
                    line |> function
                    | _ when line.StartsWith("Using") ->
                        (line, algorithms, timingParams, outputPass, aggrPass, algAverages, allAverages)
                    | _ when line.StartsWith("BlockScan algorithm") -> 
                        let tokens = line.Split(' ')
                        //printfn "%A" tokens
                        let alg = tokens.[TokenId.algo]
                        let items = tokens.[TokenId.items] |> int
                        let iterations = (tokens.[TokenId.iterations]).Substring(1) |> int
                        let blocks = tokens.[TokenId.blocks] |> int
                        let threads = tokens.[TokenId.threads] |> int
                        let items_per_thread = tokens.[TokenId.items_per_thread] |> int
                        let sm_occ = tokens.[TokenId.sm_occ] |> int
                        let stp = ScanTimingParams.Init(items, iterations, blocks, threads, items_per_thread, sm_occ)
                        (line, algorithms @ [alg], timingParams @ [stp], outputPass, aggrPass, [], allAverages @ [algAverages])
                    | _ when line.Contains("Output items:") ->
                        let tokens = line.Split(' ')
                        let r = if tokens.[2] = "PASS" then true else false
                        (line, algorithms, timingParams, outputPass @ [r], aggrPass, algAverages, allAverages)
                    | _ when line.Contains("Aggregate:") ->
                        let tokens = line.Split(' ')
                        let r = if tokens.[1] = "PASS" then true else false
                        (line, algorithms, timingParams, outputPass, aggrPass @ [r], algAverages, allAverages)

                    | _ when line.Contains("Average") ->
                        let tokens = line.Split(' ')
                        let value = 
                            line |> function
                            | _ when line.Contains("clocks:") ->            tokens.[3] |> float
                            | _ when line.Contains("clocks per item:") ->   tokens.[5] |> float
                            | _ when line.Contains("kernel millis") ->      tokens.[3] |> float
                            | _ when line.Contains("million items") ->      tokens.[5] |> float
                            | _ -> 0.8434340031
                        avgSeen := !avgSeen + 1
                        (line, algorithms, timingParams, outputPass, aggrPass, algAverages @ [value], allAverages)
                    | _ ->
                        //if !avgSeen >= 4 then avgSeen := 0
                        (line, algorithms, timingParams, outputPass, aggrPass, algAverages, allAverages)
            ) ("", List.empty<string>, List.empty<ScanTimingParams>, List.empty<bool>, List.empty<bool>, List.empty<float>, List.empty<List<float>>)


        type ScanTimingResults =
            {
                DeviceInfo              : string
                ALGORITHM               : string
                TimingParams            : ScanTimingParams
                OutputPass              : bool
                AggregatePass           : bool
                AvgClocks               : float
                AvgClocksPerItem        : float
                AvgKernelMillis         : float
                AvgMillionItemsPerSec   : float
            }

            static member Init(di, algo, tp, op, ap, avgClk, avgClkPI, avgKms, avgMIPS) =
                {   DeviceInfo = di; ALGORITHM = algo; TimingParams = tp; OutputPass = op; AggregatePass = ap; 
                    AvgClocks = avgClk; AvgClocksPerItem = avgClkPI; AvgKernelMillis = avgKms; AvgMillionItemsPerSec = avgMIPS} 
                

        let getScanTimingResults() =
            let deviceinfo, alg, tparam, op, ap, algAverages, allAverages =
                (working_dir + @"\" + example_block_scan.txt) |> lineSequence |> processLines
            
            //let algsAndParams = (alg, tparam) ||> List.zip
            //let passResults = (op, ap) ||> List.zip
            printfn "alg names%A" alg.Length
            printfn "%A" alg
            printfn "timing params%A" tparam.Length
            printfn "%A" tparam 
            printfn "output pass%A" op.Length
            printfn "%A" op
            printfn "agg pass%A" ap.Length
            printfn "%A" ap
            printfn "all avgs%A" allAverages.Length
            printfn "%A" allAverages
            [for i = 1 to alg.Length - 1 do
                yield ScanTimingResults.Init(deviceinfo, alg.[i], tparam.[i], op.[i], ap.[i], (allAverages.[i]).[0], (allAverages.[i]).[1], (allAverages.[i]).[2], (allAverages.[i]).[3])]
                



        let getScanResults() =
            let scan = new Process()
            let scanInfo = new ProcessStartInfo(example_block_scan.exe)
            scanInfo.WorkingDirectory <- working_dir
            scanInfo.UseShellExecute <- false
            scanInfo.RedirectStandardOutput <- true
            scanInfo.CreateNoWindow <- true
            scan.StartInfo <- scanInfo
            scan.Start() |> ignore

            let reader = scan.StandardOutput
            let ao = reader.ReadToEnd()

            scan.WaitForExit()
            scan.Close()

            File.AppendAllText(working_dir + @"\" + example_block_scan.txt, ao)

            //printfn "%s" ao

            //getScanTimingResults(example_block_scan.txt)

        
CppOutput.BlockScan.getScanResults()
let sct = CppOutput.BlockScan.getScanTimingResults()
printfn "%A" sct

//sct |> List.iter (fun e -> )