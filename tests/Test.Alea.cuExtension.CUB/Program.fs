// Learn more about F# at http://fsharp.net
// See the 'F# Tutorial' project for more help.
open Test.Alea.cuExtension.CUB.Block

[<EntryPoint>]
let main argv = 
    printfn "%A" argv
    
    Test.Alea.cuExtension.CUB.Block.Specializations.BlockScanWarpScans.``BlockScanWarpScans exclusive sum``()
    
    0 // return an integer exit code

