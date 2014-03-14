//open System
//open System.IO

//let src = __SOURCE_DIRECTORY__
//
//
//let lineSequence(file) = 
//        let reader = File.OpenText(file) 
//        Seq.unfold(fun line -> 
//            if line = null then 
//                reader.Close() 
//                None 
//            else 
//                Some(line,reader.ReadLine())) (reader.ReadLine())
//
//let processLines (lines:seq<string>) = 
//    lines |> Seq.fold(
//        fun (size,edges,order) line -> 
//            let tokens = line.Split(' ')
//            match tokens with
//            | _ when tokens.Length < 2 ->
//                (size, edges, tokens.[0] |> int)
//            | _ when tokens.Length = 2 ->
//                let u,v = ((tokens.[0] |> int), (tokens.[1] |> int))
//                (size+1, edges |> Seq.append [(u,v)], order)
//            | _ -> 
//                (size, edges, order)
//    ) (0,Seq.empty,0)
//
////let utilfile = File.OpenWrite(Path.Combine(src, "Utilities.fs"))
//let oldutilfiles = Directory.GetFiles(src)
//oldutilfiles |> Array.iter (fun f ->
//    if f.Contains("util_") then        
//        File.AppendAllLines(Path.Combine(src, "Utilities2.fs"), File.ReadAllLines(f))
//    )
