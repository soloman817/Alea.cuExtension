Alea.cuExtension
================

Provide basic parallel algorithms extension for Alea.cuBase.

Developing
----------

1. The main assembly is Alea.CUDA.Extension.dll, and it has
   namespace Alea.CUDA.Extension.
   
2. A type called PArray and PArray<'T> are under namespace:
   Alea.CUDA.Extension.
   
3. The main types such as PArray will be declared in Common.fs,
   it is a namespace, not a module.
   
4. Other files, such as Util.fs (represent a Util module), and
   PArray.fs (represent a PArray module), each file contains
   one module named by its filename.
   
5. DONOT use nested modules, this will have compilcated situation
   when interop with Python. So try to keep the module design
   simple. If there do need multiple sub-modules, then just
   create a new assembly, with deeper namespace. The example
   would be the Alea.CUDA.Extension.NormalDistribution, which
   is namespace, and it contains sub-modules under that namespace.
   NOTICE, later when deploy Alea.cuExtension, we will pack
   all those assemblies with ILMerge, So, if you need a new
   namespace, just create a new assembly for it.
   
6. The Test.Alea.CUDA.Extension is an EXECUTABLE assembly now. In
   fact, it is unit tests, but I need create it as executable because
   sometime, I would like to use Nsight to profile, so I can just
   call one unit test in the program entry point function.