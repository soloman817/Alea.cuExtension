open System
open System.IO

let lineSequence(file) = 
        let reader = File.OpenText(file) 
        Seq.unfold(fun line -> 
            if line = null then 
                reader.Close() 
                None 
            else 
                Some(line,reader.ReadLine())) (reader.ReadLine())

let processLines (lines:seq<string>) = 
    lines |> Seq.fold(
        fun (keep) line ->
            let dontKeep = keep
            match line with
            | _ when line.StartsWith(@"/")          -> (dontKeep)
            | _ when line.StartsWith(@"  //")       -> (dontKeep)
            | _ when line.StartsWith(@"    /")      -> (dontKeep)
            | _ when line.StartsWith(@"        /")  -> (dontKeep)
            | _ when line.StartsWith(@"  /*")       -> (dontKeep)
            | _ when line.StartsWith(@"   *")       -> (dontKeep)
            | _ when line.StartsWith(@"     *")     -> (dontKeep)
            | _ when line.StartsWith(@" *")         -> (dontKeep)
            | _ when line.Contains(@"///") ->
                (keep @ [(0, (line.LastIndexOf(@"///"))) |> line.Substring ])
            | _ when line.Length <= 1 -> (dontKeep)
            | _ when (line.Contains(@"{") || line.Contains(@"}")) && line.Trim().Length <= 2 -> (dontKeep)
            | _ -> 
                (keep @ [line])
    ) (List.empty<string>)


let processDirectory (inputPath:string) (outputPath:string) =
    let dirs = (inputPath, "*", SearchOption.AllDirectories) |> Directory.GetDirectories
    let files dir = dir |> Directory.GetFiles
    let filename (fp:string) = fp.LastIndexOf(@"\") |> fp.Substring

    for f in inputPath |> files do
        let filename = f |> filename
        let od = outputPath //(outputPath + (inputPath.LastIndexOf(@"\") |> inputPath.Substring))
        od |> Directory.CreateDirectory |> ignore
        //printfn "%A" (od + filename)
        (od + filename, (f |> lineSequence |> processLines))
        |> File.WriteAllLines

    //let od = outputPath |> Directory.CreateDirectory
    for d in dirs do
        let files = d |> files
        printfn "%A" d
        //if outputPath |> Directory.Exists then (outputPath, true) |> Directory.Delete
        
        let od = outputPath + (d.LastIndexOf(@"cub") |> (+) 3 |> d.Substring)
        od |> Directory.CreateDirectory |> ignore  
        
        printfn "%A" od
        for f in files do
            let filename = f |> filename
            //printfn "%A" (od + filename)
            (od + filename, (f |> lineSequence |> processLines)) 
            |> File.WriteAllLines





let outpath = @"C:\Users\Aaron\Desktop\Temp"
let inpath = @"X:\repo\git\cub\cub"

(inpath, outpath) ||> processDirectory