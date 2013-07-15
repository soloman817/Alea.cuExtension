module Test.Alea.CUDA.Extension.Output.Excel

open Microsoft.Office.Interop.Excel
open Test.Alea.CUDA.Extension.Output.Util



let benchmarkExcelOutput (bms:BenchmarkStats) =
    let app = new ApplicationClass(Visible = true)

    let workbook = app.Workbooks.Add(XlWBATemplate.xlWBATWorksheet)
    let worksheet = (workbook.Worksheets.[1] :?> Worksheet)

    let tpTitle = [| "Throughput" |]    
    let tpHeaders = [| "Iterations"; "Elements"; "Alea.cuBase"; bms.Opponent |]

    let bwTitle = [| "Bandwidth" |]
    let bwHeaders = tpHeaders

    let nItrs, counts = bms.NumIterations, bms.SourceCounts

    let nTests = counts.Length

    let lastRow colChar startRow = sprintf "%c%d" colChar (startRow + (nTests - 1))
    
    let topRow = [| "Algorithm"; "Tested Type"; "Device Used"; "Compared Against" |]
    let bmsInfo = [| bms.AlgorithmName; bms.TestedType; bms.DeviceName; bms.Opponent |]
    
    worksheet.Range("A1","D1").Value2 <- topRow
    worksheet.Range("A2","D2").Value2 <- bmsInfo
    
    worksheet.Range("A3").Value2 <- tpTitle
    worksheet.Range("A4","D4").Value2 <- tpHeaders
    
    let nItrs = Array2D.init nTests 1 (fun i _ -> nItrs.[i])
    let counts = Array2D.init nTests 1 (fun i _ -> counts.[i])
    
    let mytp = Array2D.init nTests 1 (fun i _ -> bms.MyThroughput.[i].Value)
    let otp = Array2D.init nTests 1 (fun i _ -> bms.OpponentThroughput.[i].Value)
    
    let mybw = Array2D.init nTests 1 (fun i _ -> bms.MyBandwidth.[i].Value)
    let obw = Array2D.init nTests 1 (fun i _ -> bms.OpponentBandwidth.[i].Value)

    // throughput headers are on row 4 so these columns start on row 5
    worksheet.Range("A5",(lastRow 'A' 5)).Value2 <- nItrs
    worksheet.Range("B5",(lastRow 'B' 5)).Value2 <- counts
    worksheet.Range("C5",(lastRow 'C' 5)).Value2 <- mytp
    worksheet.Range("D5",(lastRow 'D' 5)).Value2 <- otp

    // this is location of 'Bandwidth' header
    // throughput starts on 5, we add number of rows we had (nTests), then skip a line
    let bwTitleRow = (5 + nTests + 1).ToString()
    let bwHeaderRow = (5 + nTests + 2).ToString()
    let r0 = (5 + nTests + 3) // row 0, i.e. row where data starts
    let r0s = r0.ToString()
    let ri c = (lastRow c r0)
    worksheet.Range(("A" + bwTitleRow)).Value2 <- bwTitle
    worksheet.Range(("A" + bwHeaderRow), ("D" + bwHeaderRow)).Value2 <- bwHeaders
    worksheet.Range(("A" + r0s), (ri 'A')).Value2 <- nItrs
    worksheet.Range(("B" + r0s), (ri 'B')).Value2 <- counts
    worksheet.Range(("C" + r0s), (ri 'C')).Value2 <- mybw
    worksheet.Range(("D" + r0s), (ri 'D')).Value2 <- obw


    // all of this (above) looks kind of ugly... need to clean up later

    // just a quick try at this because there's a tutorial at
    // http://msdn.microsoft.com/en-us/library/vstudio/hh297098(v=vs.100).aspx
    let chartobjects = (worksheet.ChartObjects() :?> ChartObjects)
    let chartobject = chartobjects.Add(400.0, 20.0, 550.0, 350.0)

    chartobject.Chart.ChartWizard
        ( Title = "Throughput",
          Source = worksheet.Range("B5:B14", ("C4:D" + (4 + nTests).ToString())),
          //Source = worksheet.Range("B5:B14","C4:D14"),
          Gallery = XlChartType.xlXYScatterSmooth,
          PlotBy = XlRowCol.xlColumns) 
          //CategoryLabels = 2)

    chartobject.Chart.ChartStyle <- 6