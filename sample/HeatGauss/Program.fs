open System
open System.Collections.Generic
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.Matlab.Plot

//open MathWorks.MATLAB.NET.Arrays
//
//let plotter = new AleaMatlabWrapper.MatlabPlotter()
//
///// Plot a surface with row values in x and column values in y
//let plotSurface (x:float[]) (y:float[]) (v:float[]) (xl:string) (yl:string) (zl:string) (title:string) (position:float[]) =          
//    let n = x.Length
//    let m = y.Length
//
//    let xMW = new MWNumericArray(1, n, x, false)
//    let yMW = new MWNumericArray(1, m, y, false)
//    let vMW = new MWNumericArray(n, m, v, false)
//
//    if position.Length <> 4 then failwith "position array must be of length 4"
//
//    let info = List<MWArray>()
//    info.Add(new MWCharArray("xlabel"))
//    info.Add(new MWCharArray(yl))
//    info.Add(new MWCharArray("ylabel"))
//    info.Add(new MWCharArray(xl))
//    info.Add(new MWCharArray("zlabel"))
//    info.Add(new MWCharArray(zl))
//    info.Add(new MWCharArray("title"))
//    info.Add(new MWCharArray(title))
//    info.Add(new MWCharArray("position"))
//    info.Add(new MWNumericArray(1, 4, position, false))
//    let info = info.ToArray()
//
//    plotter.plotsurface(yMW, xMW, vMW, info)
//    
//
///// Plot a surface with row values in x and column values in y
//let plotCurve (x:float[]) (y:float[]) =
//    let xMW = new MWNumericArray(1, x.Length, x, false)
//    let yMW = new MWNumericArray(1, x.Length, y, false)
//
//    plotter.plotcurve(xMW, yMW)        
//            
//
///// Plot a surface with row values in x and column values in y
//let plotMultiCurves (sampledCurves:seq<float[] * float[] * string>) =
//    let mwArrays = sampledCurves |> Seq.map (fun (x, y, fmt) -> 
//        [
//            new MWNumericArray(1, x.Length, x, false) :> MWArray
//            new MWNumericArray(1, y.Length, y, false) :> MWArray
//            new MWCharArray(fmt) :> MWArray
//        ] |> List.toSeq) |> Seq.concat |> Seq.toArray
//
//    plotter.plotmulticurve(mwArrays)
//
//
///// Convergence plot 
//let convergencePlot (price:float) (sampleIds:float[]) (meanHist:float[]) (varHist:float[]) (errHist:float[]) (alpha:float) (title:string) (filename:string) = 
//    let priceMW = new MWNumericArray(price)
//    let alphaMW = new MWNumericArray(alpha)
//    let sampleIdsMW = new MWNumericArray(1, sampleIds.Length, sampleIds, false)
//    let meanHistMW = new MWNumericArray(1, meanHist.Length, meanHist, false)
//    let varHistMW = new MWNumericArray(1, varHist.Length, varHist, false)
//    let errHistMW = new MWNumericArray(1, errHist.Length, errHist, false)            
//    let titleMW = new MWCharArray(title)
//    let filenameMW = new MWCharArray(filename)
//
//    plotter.plotconvergence(priceMW, sampleIdsMW, meanHistMW, varHistMW, errHistMW, alphaMW, titleMW, filenameMW)


let [<ReflectedDefinition>] pi = System.Math.PI
let [<ReflectedDefinition>] sigma1 = 0.04
let [<ReflectedDefinition>] sigma2 = 0.04
let [<ReflectedDefinition>] sigma3 = 0.04
let initialCondExpr =
    <@ fun t x y -> 1.0/3.0*exp (-((x-0.2)*(x-0.2) + (y-0.2)*(y-0.2))/(2.0*sigma1*sigma1)) / (sigma1*sigma1*2.0*pi) +
                    1.0/3.0*exp (-((x-0.8)*(x-0.8) + (y-0.8)*(y-0.8))/(2.0*sigma2*sigma2)) / (sigma2*sigma2*2.0*pi) +
                    1.0/3.0*exp (-((x-0.8)*(x-0.8) + (y-0.2)*(y-0.2))/(2.0*sigma3*sigma3)) / (sigma3*sigma3*2.0*pi) @>
let boundaryExpr = <@ fun t x y -> 0.0 @>
let sourceFunctionExpr = <@ fun t x y -> 0.0 @>

let worker = Engine.workers.DefaultWorker
let solve = worker.LoadPModule(Heat2dAdi.solve initialCondExpr boundaryExpr sourceFunctionExpr).Invoke
let calc k tstart tstop Lx Ly nx ny dt = pcalc {
    let! x, y, u = solve k tstart tstop Lx Ly nx ny dt
    let! u = u.Gather()
    return x, y, u }

let heatdist tstop =
    let k = 1.0
    let tstart = 0.0
    let Lx = 1.0
    let Ly = 1.0
    let dt = 0.0001
    //let dt = 0.01

    let nx = 512
    let ny = 512

    let x, y, u =
        let calc = calc k tstart tstop Lx Ly nx ny dt
        //calc k tstart tstop Lx Ly nx ny dt |> PCalc.run
        let (x, y, u), ktc = calc |> PCalc.runWithKernelTiming 1
        ktc.Dump()
        x, y, u

    x, y, u

    //let x, y, u = calc k tstart tstop Lx Ly nx ny dt |> PCalc.run
    //plotSurfaceOfArray x y u "x" "y" "heat" (sprintf "Heat 2d ADI t=%f" tstop) ([400.; 200.; 750.; 700.] |> Seq.ofList |> Some)
    
    //let _, loggers = calc k tstart tstop Lx Ly nx ny dt |> PCalc.runWithTimingLogger in loggers.["default"].DumpLogs()
    //let _, ktc = calc k tstart tstop Lx Ly nx ny dt |> PCalc.runWithKernelTiming 1 in ktc.Dump()
    //calc k tstart tstop Lx Ly nx ny dt |> PCalc.runWithDiagnoser({ PCalcDiagnoser.None with DebugLevel = 1}) |> ignore
    //calc k tstart tstop Lx Ly nx ny dt |> PCalc.run |> ignore

// to use matlab plot, you should install Matlab Compiler Runtime:
// http://www.mathworks.com/products/compiler/mcr/index.html
// choose version R2012a (7.17) 32bit (because we are in 32bit for this program)
// uhmm, actually, we need 7.15 version
let plotWithMatlab (results:float * float[] * float[] * float[]) =
    let tstop, x, y, u = results
    plotSurfaceOfArray x y u "x" "y" "heat" (sprintf "Heat 2d ADI t=%f" tstop) ([400.; 200.; 750.; 700.] |> Seq.ofList |> Some)

let results =
    [| 0.0; 0.005; 0.01; 0.02; 0.03; 0.04 |]
    //[| 0.01 |]
    |> Array.map (fun tstop -> let x, y, u = heatdist tstop in tstop, x, y, u)

//results |> Array.iter plotWithMatlab

//printf "Press Enter to quit..."
//System.Console.ReadKey(true) |> ignore
