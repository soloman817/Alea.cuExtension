// Learn more about F# at http://fsharp.net
// See the 'F# Tutorial' project for more help.

[<EntryPoint>]
let main argv = 

    Test.Sample.LaunchingTime.``test on int``()
    Test.Sample.LaunchingTime.``test on float``()
    Test.Sample.LaunchingTime.``test on Int3A8``()

    0 // return an integer exit code
