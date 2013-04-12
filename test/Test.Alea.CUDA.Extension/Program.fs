let segscanPerformance() =
    printfn "====> Segmented Scan By Flag"
    Test.Alea.CUDA.Extension.SegmentedScanByFlag.``performance: compare with thrust``()
    printfn "====> Segmented Scan By Key"
    Test.Alea.CUDA.Extension.SegmentedScanByKey.``performance: compare with thrust``()

let segscanVsMGPUByFlags() =
    Test.Alea.CUDA.Extension.SegmentedScanByFlag.``performance: compare with mgpu``()

let segscanVsMGPUByKeys() =
    Test.Alea.CUDA.Extension.SegmentedScanByKey.``performance: compare with mgpu``()

segscanVsMGPUByKeys()

//for i = 1 to 3 do
//    segscanPerformance()
//    printfn ""