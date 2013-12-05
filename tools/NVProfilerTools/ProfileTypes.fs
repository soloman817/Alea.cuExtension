module NVProfilerTools.ProfileTypes

open FileHelpers
open NVProfilerTools.CustomConverters

[< DelimitedRecord(",") >] [< IgnoreEmptyLines >] [<IgnoreFirst(6)>]
type ProfiledKernelLaunch_summary () =
    [<DefaultValue>] val mutable Time_percent : float
    [<DefaultValue>] [<FieldConverter(typeof<SciNotationConverter>)>] val mutable Time : float
    [<DefaultValue>] val mutable Calls : int
    [<DefaultValue>] [<FieldConverter(typeof<SciNotationConverter>)>] val mutable Avg : float
    [<DefaultValue>] [<FieldConverter(typeof<SciNotationConverter>)>] val mutable Min : float
    [<DefaultValue>] [<FieldConverter(typeof<SciNotationConverter>)>] val mutable Max : float
    [<DefaultValue>] [<FieldQuoted()>] [<FieldConverter(typeof<KernelNameConverter>)>] val mutable Name : string

// TODO    
//[< DelimitedRecord(",") >] [< IgnoreEmptyLines >]
//type ProfiledKernelLaunch_apiTrace () =

[<ConditionalRecord(RecordCondition.IncludeIfMatchRegex, @"(Kernel[\w]+)")>]
[< DelimitedRecord(",") >] [< IgnoreEmptyLines >]
type ProfiledKernelLaunch_gpuTrace () =
    [<DefaultValue>] [<FieldConverter(typeof<SciNotationConverter>)>] val mutable Start : float
    [<DefaultValue>] [<FieldConverter(typeof<SciNotationConverter>)>] val mutable Duration : float
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable GridX : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable GridY : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable GridZ : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable BlockX : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable BlockY : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable BlockZ : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable RegPerThread : int
    [<DefaultValue>] [<FieldNullValue(typeof<float>, "0.0")>] val mutable StaticSMem : float
    [<DefaultValue>] [<FieldNullValue(typeof<float>, "0.0")>] val mutable DynamicSMem : float
    [<DefaultValue>] [<FieldNullValue(typeof<float>, "0.0")>] val mutable Size : float
    [<DefaultValue>] [<FieldNullValue(typeof<float>, "0.0")>] val mutable Throughput : float
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable Device : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable Contex : int
    [<DefaultValue>] [<FieldNullValue(typeof<int>, "0")>] val mutable Stream : int
    [<DefaultValue>] [<FieldQuoted()>] [<FieldConverter(typeof<KernelNameConverter>)>] val mutable Name : string

