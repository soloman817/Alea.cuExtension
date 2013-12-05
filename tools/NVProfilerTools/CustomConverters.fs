module NVProfilerTools.CustomConverters

open System
open System.IO
open System.Text.RegularExpressions
open FileHelpers

let kNameRegex = Regex(@"(Kernel[\w]+)")

type KernelNameConverter() =
    inherit ConverterBase()    
    override snc.StringToField(from) =
            let m = kNameRegex.Match(from)
            let from = m.Groups.[1].Captures.[0].Value
            from :> obj
    override snc.FieldToString(fieldValue:obj) =
            fieldValue.ToString()


type SciNotationConverter() =
    inherit ConverterBase()
    let sciReg = Regex(@"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)")
    override snc.StringToField(from) =
            let m = sciReg.Match(from)
            let from = m.Groups.[1].Captures.[0].Value
            Convert.ToDouble(Double.Parse(from)) :> obj
    override snc.FieldToString(fieldValue:obj) =            
            fieldValue.ToString()

