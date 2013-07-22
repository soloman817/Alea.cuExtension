module Alea.CUDA.Extension.Output.Excel

open System.IO
open Microsoft.Office.Interop.Excel
open Alea.CUDA.Extension.Output.Util


let benchmarkExcelOutput (bms:BenchmarkStats) =
    let app = new ApplicationClass(Visible = true)
    let workbook = app.Workbooks.Add(XlWBATemplate.xlWBATWorksheet)
    let worksheet = (workbook.Worksheets.[1] :?> Worksheet)
    
    let cr = ref 1 // current row
    let crLtr = char(!cr + 64).ToString() // number to nth letter of alphabet

    let rangeStr1 (nHeaders:int) = 
        let c = !cr
        let x = c + nHeaders
        let rstr = 
            if nHeaders > 1 then
                crLtr + c.ToString() + ":" + char((x - c) + 64).ToString() + c.ToString()
            else
                crLtr + c.ToString()
        cr := !cr + 1
        rstr

    let rangeStrCol (col:int) (numTests:int) = 
        let colChar = char(col + 64).ToString()
        let rstr = colChar + (!cr).ToString() + ":" + colChar + ((numTests - 1) + !cr).ToString()
        rstr
        
    // build data
    let nItrs, counts = bms.NumIterations, bms.SourceCounts
    let nTests = counts.Length
    
    let nItrs, counts = 
        nItrs |> List.map (fun x -> float(x)), 
        counts |> List.map (fun x -> float(x))        

    let nItrs = Array2D.init nTests 1 (fun i _ -> nItrs.[i])
    let counts = Array2D.init nTests 1 (fun i _ -> counts.[i])
        
    let mytp = Array2D.init nTests 1 (fun i _ -> bms.MyThroughput.[i].Value)
    let otp = Array2D.init nTests 1 (fun i _ -> bms.OpponentThroughput.[i].Value)
    let tpDataCols = [| nItrs; counts; mytp; otp |]

    let mybw = Array2D.init nTests 1 (fun i _ -> bms.MyBandwidth.[i].Value)
    let obw = Array2D.init nTests 1 (fun i _ -> bms.OpponentBandwidth.[i].Value)
    let bwDataCols = [| nItrs; counts; mybw; obw |]
        
    // begin output
    let mainHeaders = [| "Algorithm"; "Tested Type"; "Device Used"; "Compared Against" |]
    let mainInfo = [| bms.AlgorithmName; bms.TestedType; bms.DeviceName; bms.Opponent |]

    worksheet.Range(rangeStr1 mainHeaders.Length).Value2 <- mainHeaders
    worksheet.Range(rangeStr1 mainInfo.Length).Value2 <- mainInfo
    cr := !cr + 1 // skip line
    
    
    let tpTitle = [| "Throughput" |]    
    let tpHeaders = [| "Iterations"; "Elements"; "Alea.cuBase"; bms.Opponent |]
    worksheet.Range(rangeStr1 1).Value2 <- tpTitle
    worksheet.Range(rangeStr1 tpHeaders.Length).Value2 <- tpHeaders
    for i = 0 to tpDataCols.Length - 1 do
        worksheet.Range(rangeStrCol (i + 1) nTests).Value2 <- tpDataCols.[i]
    // throughput chart
    // just a quick try at this because there's a tutorial at
    // http://msdn.microsoft.com/en-us/library/vstudio/hh297098(v=vs.100).aspx
    let chartobjects = (worksheet.ChartObjects() :?> ChartObjects)
    let chartobject = chartobjects.Add(300.0, 20.0, 700.0, 300.0)
    let range1 = "B" + (!cr - 1).ToString() + ":" + "B" + (!cr + nTests - 1).ToString()
    let range2 = "C" + (!cr - 1).ToString() + ":" + "D" + (!cr + nTests - 1).ToString() 
    chartobject.Chart.ChartWizard
        ( Title = "Throughput (M/s)",
          Source = worksheet.Range(range1 + "," + range2),
          Gallery = XlChartType.xlXYScatterSmooth,
          PlotBy = XlRowCol.xlColumns,
          CategoryLabels = 1,
          SeriesLabels = 1)
    chartobject.Chart.ChartStyle <- 20
    cr := !cr + nTests + 1
    
    let bwTitle = [| "Bandwidth" |]
    let bwHeaders = tpHeaders
    worksheet.Range(rangeStr1 1).Value2 <- bwTitle
    worksheet.Range(rangeStr1 tpHeaders.Length).Value2 <- bwHeaders
    for i = 0 to bwDataCols.Length - 1 do
        worksheet.Range(rangeStrCol (i + 1) nTests).Value2 <- bwDataCols.[i]
    // bandwidth chart
    let chartobjects = (worksheet.ChartObjects() :?> ChartObjects)
    let chartobject = chartobjects.Add(300.0, 330.0, 700.0, 300.0)
    let range1 = "B" + (!cr - 1).ToString() + ":" + "B" + (!cr + nTests - 1).ToString()
    let range2 = "C" + (!cr - 1).ToString() + ":" + "D" + (!cr + nTests - 1).ToString()
    chartobject.Chart.ChartWizard
        ( Title = "Bandwidth (GB/s)",
          Source = worksheet.Range(range1, range2),
          Gallery = XlChartType.xlXYScatterSmooth,
          PlotBy = XlRowCol.xlColumns,
          CategoryLabels = 1,
          SeriesLabels = 1)
    chartobject.Chart.ChartStyle <- 7
    cr := !cr + nTests + 1

    let kernelTimingTitles = bms.KernelsUsed
    let kernelTimingHeaders = [| "Elements"; "Alea.cuBase"; bms.Opponent; "Difference" |]
    let ktZeros = Array2D.init nTests 1 (fun i _ -> 0.0)

    
    let ktDataCols = [| counts; ktZeros; ktZeros; ktZeros |]
    for i = 0 to kernelTimingTitles.Length - 1 do
        worksheet.Range(rangeStr1 1).Value2 <- [| kernelTimingTitles.[i] |]
        worksheet.Range(rangeStr1 kernelTimingHeaders.Length).Value2 <- kernelTimingHeaders
        for j = 0 to ktDataCols.Length - 1 do
            worksheet.Range(rangeStrCol (j + 1) nTests).Value2 <- ktDataCols.[j]
        cr := !cr + nTests + 1
    
//
//let csvToExcel (csvPath:string) =
//    let app = new ApplicationClass(Visible = true)
//    let workbook = app.Workbooks.Add(XlWBATemplate.xlWBATWorksheet)
//    let worksheet = (workbook.Worksheets.[1] :?> Worksheet)
//    app.Workbooks.Open(csvPath)