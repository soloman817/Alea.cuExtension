Alea.cuExtension
================

Alea.cuExtension is an extension of Alea.cuBase, which is a complete
solution to develop CUDA accelerated GPU applications on the .NET
framework. It relies on Microsoft's new functional language F# to 
generate highly optimized CUDA code.

While Alea.cuBase provides a way to code CUDA kernel in F# language,
Alea.cuExtension provides more advanced runtime GPU resource 
management, which is called the PCalc monad. With this monad, you 
can easily manage GPU memories, launching diagnosing and other
things in a unify manner, which makes the composition of various
parallel algorithm more easily.

Alea.cuExtension also tends to provide parallel algorithm primitives,
such as transfomation, scan, random number generation, etc. And with
the PCalc monad, you can combine these kernels and reuse them to build
up your own application.

Alea.cuExtension is now under developing period, you are free to clone
it or fork it and check the code. It is also a good example showing
the usage of Alea.cuBase.

To get started, clone this project and check the unit tests code and
sample (coming soon) code.

Resources
---------

- Alea.cuBase Introduction: http://www.quantalea.net/products/introduction/
- Alea.cuBase Resources : http://www.quantalea.net/products/resources/
- QuantAlea Blog : http://blog.quantalea.net
- Google Groups : https://groups.google.com/d/forum/aleacubase
- Contact : support@quantalea.net
