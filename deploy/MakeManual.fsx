#I @"..\packages\FSPowerPack.Core.Community.3.0.0.0\Lib\Net40"
#I @"..\lib\fsformatting"
#r "FSharp.PowerPack.dll"
#r "FSharp.PowerPack.Metadata.dll"
#r "RazorEngine.dll"
#r "FSharp.CodeFormat.dll"
#r "FSharp.Literate.dll"
#r "FSharp.Markdown.dll"
#r "FSharp.MetadataFormat.dll"

open System
open System.IO
open System.Text
open System.Text.RegularExpressions
open System.Reflection
open System.Diagnostics
open System.Collections.Generic
open Microsoft.FSharp.Metadata
open FSharp.Literate
open FSharp.MetadataFormat

type Options private () =
    static let instance = Options()
    let mutable solutionFolder : string option = @"..\" |> Some
    let mutable destinationFolder : string option = None
    let mutable assemblyFolder : string option = None
    let mutable sampleListFile : string = @"x"
    let mutable noReference : bool = false

    member this.SampleListFile
        with get() = sampleListFile
        and set value = sampleListFile <- value

    member this.SolutionFolder 
        with get() = match solutionFolder with Some folder -> folder | None -> failwith "Solution folder not set."
        and set folder = solutionFolder <- Some folder

    member this.DestinationFolder 
        with get() = match destinationFolder with Some folder -> folder | None -> failwith "Destination folder not set."
        and set folder = destinationFolder <- Some folder

    member this.AssemblyFolder 
        with get() = match assemblyFolder with Some folder -> folder | None -> failwith "Assembly folder not set."
        and set folder = assemblyFolder <- Some folder

    member this.TryAssemblyFolder = assemblyFolder

    member this.NoReference 
        with get() = noReference
        and set v = noReference <- v

    static member Instance = instance

let (@@) a b = Path.Combine(a, b)

let projectName = "Alea.ProjectScaffold"

let productVersion() =
    if Options.Instance.NoReference then "x.x.x"
    else
        let assemblyFolder = Options.Instance.TryAssemblyFolder |> function
            | Some folder -> folder
            | None -> Options.Instance.SolutionFolder @@ "src" @@ "Alea.ProjectTemplate" @@ "bin" @@ "Release"

        let assemblyPath = assemblyFolder @@ "Alea.ProjectTemplate.dll"

        let fvi = FileVersionInfo.GetVersionInfo(assemblyPath)

        sprintf "%d.%d.%d" fvi.ProductMajorPart fvi.ProductMinorPart fvi.ProductBuildPart

let emptyFolder (folder:string) =
    try Directory.Delete(folder, true)
    with _ -> ()
    Directory.CreateDirectory(folder) |> ignore

let rec copyFolder (src:string) (dst:string) =
    if not (Directory.Exists(dst)) then
        Directory.CreateDirectory(dst) |> ignore

    for file in (Directory.GetFiles(src)) do
        let dst = Path.Combine(dst, Path.GetFileName(file))
        File.Copy(file, dst)
        while not (File.Exists(dst)) do
            System.Threading.Thread.Sleep(100)

    for folder in (Directory.GetDirectories(src)) do
        match (Path.GetFileName(folder)) with
        | "bin" | "obj" -> ()
        | _ ->
            let dst = Path.Combine(dst, Path.GetFileName(folder))
            copyFolder folder dst

let normalizeDocumentName (name:string) =   
    let idx = name.IndexOf('.')
    let order = name.Substring(0, idx)
    let order = Int32.Parse(order)
    let name = name.Substring(idx + 1, name.Length - idx - 1)
    let filename = name
    let filename = filename.Replace(' ', '_')
    let filename = filename.Replace('.', '_')
    let filename = filename.Replace(",", "")
    let filename = filename.Replace("#", "sharp")
    let filename = filename.ToLower()
    order, name, filename

type [<AbstractClass>] Document(parent:Document option, srcPath:string) =
    member this.IsRoot = parent.IsNone
    member this.Parent = match parent with Some parent -> parent | None -> failwith "This is root doc"
    member this.SrcPath = srcPath
    abstract DstPath : string
    abstract Prefix : string
    abstract Name : string
    abstract UrlName : string
    abstract Order : int
    default this.DstPath = failwith "DstPath not set"
    default this.Prefix = failwith "Prefix not set"
    default this.Order = failwith "Order not set"
    abstract Dump : unit -> unit
    abstract BuildContent : unit -> unit

type Folder(parent:Document option, srcPath:string) =
    inherit Document(parent, srcPath)

    let prefix, name, order = parent |> function
        | None -> "", "Manual", 0
        | Some(parent) ->
            let order, name, filename = Path.GetFileName(srcPath) |> normalizeDocumentName
            let prefix = sprintf "%s%s-" parent.Prefix filename
            prefix, name, order

    let urlname = sprintf "%sindex.html" prefix

    let documents = List<Document>()

    member this.AddDocument(doc) = documents.Add(doc)
    member this.Documents = documents |> Seq.toArray |> Array.sortBy (fun doc -> doc.Order)

    override this.Prefix = prefix
    override this.Order = order
    override this.Name = name
    override this.UrlName = urlname
    override this.Dump() = this.Documents |> Array.iter (fun doc -> doc.Dump())
    override this.BuildContent() = documents |> Seq.iter (fun doc -> doc.BuildContent())   

    member this.GenNavList(urlroot:string, child:Document) =
        let strs = List<string>()

        if not this.IsRoot then
            let parent = this.Parent :?> Folder
            strs.Add(parent.GenNavList(urlroot, this))

        strs.Add(sprintf "<li class=\"nav-header\">%s</li>" this.Name)

        this.Documents |> Array.iter (fun doc -> doc.Name |> function
            | "Index" | "index" -> ()
            | name when name = child.Name -> strs.Add(sprintf "<li class=\"active\"><a href=\"%s%s\">%s</a></li>" urlroot doc.UrlName doc.Name)
            | name -> strs.Add(sprintf "<li><a href=\"%s%s\">%s</a></li>" urlroot doc.UrlName doc.Name))

        strs |> String.concat "\n"

    member this.GenIndex(urlroot:string, child:Document) =
        let strs = List<string>()

        if child.Order = 0 then
            strs.Add("<ul>")
            this.Documents |> Array.iter (fun doc -> doc.Name |> function
                | "Index" | "index" -> ()
                | name -> strs.Add(sprintf "<li><a href=\"%s%s\">%s</a></li>" urlroot doc.UrlName doc.Name))
            strs.Add("</ul>")

        strs |> String.concat "\n"

type [<AbstractClass>] Page(parent:Document option, srcPath:string) =
    inherit Document(parent, srcPath)

    let ext = Path.GetExtension(srcPath)
    
    let order, name, filename = Path.GetFileNameWithoutExtension(srcPath) |> normalizeDocumentName
    
    let dstPath = parent |> function
        | None -> Options.Instance.DestinationFolder @@ (sprintf "%s.html" filename)
        | Some(parent) -> Options.Instance.DestinationFolder @@ (sprintf "%s%s.html" parent.Prefix filename)

    let urlname = Path.GetFileName(dstPath)

    override this.DstPath = dstPath
    override this.Order = order    
    override this.Name = name
    override this.UrlName = urlname
    override this.Dump() = printfn "%s -> %s" srcPath dstPath

type MarkdownPage(parent:Document option, srcPath:string) =
    inherit Page(parent, srcPath)

    let templatePath = Options.Instance.SolutionFolder @@ "doc" @@ "Manual" @@ "Templates" @@ "Manual" @@ "template.html"

    override this.BuildContent() =
        printfn "Generating %s ..." this.UrlName
        let projectInfo =
            [ "project-name", projectName
              "product-version", productVersion()
              "nav-list", ((this.Parent :?> Folder).GenNavList("", this))
              "index", ((this.Parent :?> Folder).GenIndex("", this)) ]
        Literate.ProcessMarkdown(this.SrcPath, templatePath, this.DstPath, OutputKind.Html, replacements = projectInfo, lineNumbers=false)

type ReferencePage(parent:Document option, srcPath:string, assemblyPath) as this =
    inherit Page(parent, srcPath)

    let templatePath = Options.Instance.SolutionFolder @@ "doc" @@ "Manual" @@ "Templates" @@ "Reference"

    let urlname =
        let _, _, filename = sprintf "0.%s" this.Name |> normalizeDocumentName
        sprintf "reference/%s/index.html" filename

    let outputFolder =
        let _, _, filename = sprintf "0.%s" this.Name |> normalizeDocumentName
        Options.Instance.DestinationFolder @@ "reference" @@ filename

    do emptyFolder outputFolder

    override this.UrlName = urlname
    override this.Dump() = printfn "%s -> Reference pages for assembly %s" srcPath this.Name

    override this.BuildContent() =
        if Options.Instance.NoReference then
            printfn "Skipping refernce generating for %s.dll ..." this.Name
        else
            let projectInfo =
                [ "project-name", projectName
                  "product-version", productVersion()
                  "assembly-name", this.Name
                  "nav-list", ((this.Parent :?> Folder).GenNavList("../../", this))
                  "index", ((this.Parent :?> Folder).GenIndex("../../", this)) ]

            let binpath = Path.GetFullPath(Path.GetDirectoryName(assemblyPath))
            AppDomain.CurrentDomain.add_AssemblyResolve(ResolveEventHandler(fun o e ->
                printfn "Resolving assembly: %s" e.Name
                let asmName = AssemblyName(e.Name)
                let file = binpath @@ (sprintf "%s.dll" asmName.Name)
                printfn "Filename: %s" file
                if File.Exists(file) then printfn "Resolved."; Assembly.LoadFile(file)
                else printfn "Not resolved."; null))

            MetadataFormat.Generate(assemblyPath, outputFolder, [ templatePath ], parameters = projectInfo)

let createDocument() =
    let rec create (parent:Document option) (srcDir:string) =
        let folder = Folder(parent, srcDir)
        let parent = folder :> Document |> Some

        for file in (Directory.GetFiles(srcDir)) do
            let srcPath = file
            let ext = Path.GetExtension(srcPath)
            let order, name, _ = Path.GetFileNameWithoutExtension srcPath |> normalizeDocumentName
            match name, ext with
            | _, ".md" ->
                folder.AddDocument(MarkdownPage(parent, srcPath))

            | "SAMPLES", ".reference" ->
                File.ReadAllLines(Options.Instance.SampleListFile)
                |> Array.filter (fun sample -> sample.Trim() <> "")
                |> Array.iteri (fun i sample ->
                    let assemblyFolder = Options.Instance.SolutionFolder @@ "samples" @@ sample @@ "bin" @@ "Release"
                    let assemblyFile = assemblyFolder @@ (sprintf "%s.exe" sample)
                    let srcPath = srcDir @@ (sprintf "%d.%s.reference" (i+100) sample)
                    folder.AddDocument(ReferencePage(parent, srcPath, assemblyFile)))

            | _, ".reference" ->
                let assemblyFolder = Options.Instance.TryAssemblyFolder |> function
                    | Some folder -> folder
                    | None -> Options.Instance.SolutionFolder @@ "src" @@ name @@ "bin" @@ "Release"
                let assemblyFile = assemblyFolder @@ (sprintf "%s.dll" name)
                folder.AddDocument(ReferencePage(parent, srcPath, assemblyFile))

            | _ -> failwithf "Unkown file: %s" srcPath

        for dir in (Directory.GetDirectories(srcDir)) do
            let srcPath = dir
            let doc = create parent srcPath
            folder.AddDocument(doc)

        folder :> Document

    create None (Options.Instance.SolutionFolder @@ "doc" @@ "Manual" @@ "Source")

let build() =
    emptyFolder(Options.Instance.DestinationFolder)
    emptyFolder(Options.Instance.DestinationFolder @@ "style")
    emptyFolder(Options.Instance.DestinationFolder @@ "images")
    emptyFolder(Options.Instance.DestinationFolder @@ "reference")
    File.Copy(Options.Instance.SolutionFolder @@ "doc" @@ "Manual" @@ "Templates" @@ "Favicon" @@ "favicon.ico", Options.Instance.DestinationFolder @@ "favicon.ico")
    copyFolder (Options.Instance.SolutionFolder @@ "doc" @@ "Manual" @@ "Templates" @@ "Style") (Options.Instance.DestinationFolder @@ "style")
    copyFolder (Options.Instance.SolutionFolder @@ "doc" @@ "Manual" @@ "Images") (Options.Instance.DestinationFolder @@ "images")
    let doc = createDocument()
    //doc.Dump()
    doc.BuildContent()

let specs =
    [
        "-a", ArgType.String(fun folder -> Options.Instance.AssemblyFolder <- folder), "set assembly folder"
        "-s", ArgType.String(fun folder -> Options.Instance.SolutionFolder <- folder), "set solution folder"
        "-d", ArgType.String(fun folder -> Options.Instance.DestinationFolder <- folder), "set destination folder"
        "-noref", ArgType.Unit(fun () -> Options.Instance.NoReference <- true), "not generate reference (for fast view doc only)"
        "-samples", ArgType.String(fun file -> Options.Instance.SampleListFile <- file), "set sample list file (default is all list)"
    ] |> List.map (fun (sh, ty, desc) -> ArgInfo(sh, ty, desc))

ArgParser.Parse(specs)

build()




