#!/Applications/Mathematica.app/Contents/MacOS/MathematicaScript -script

LaunchKernels[4];

file1 = ToString[$ScriptCommandLine[[2]]];
file2 = ToString[$ScriptCommandLine[[3]]];
ksize = 50;
samplecount = 2000; 

data1 = Import[file1, "Table"];
data2 = Import[file2, "Table"];

computeDistThread[d_] := 
  MultinormalDistribution[{\[Mu]0, \[Mu]1, \[Mu]2, \[Mu]3}, \
{{\[Sigma]00, \[Sigma]01, \[Sigma]02, \[Sigma]03}, {\[Sigma]10, \
\[Sigma]11, \[Sigma]12, \[Sigma]13}, {\[Sigma]20, \[Sigma]21, \
\[Sigma]22, \[Sigma]23}, {\[Sigma]30, \[Sigma]31, \[Sigma]32, \
\[Sigma]33}}] /. 
   Thread[{\[Mu]0, \[Mu]1, \[Mu]2, \[Mu]3, \[Sigma]00, \[Sigma]01, \
\[Sigma]02, \[Sigma]03, \[Sigma]10, \[Sigma]11, \[Sigma]12, \
\[Sigma]13, \[Sigma]20, \[Sigma]21, \[Sigma]22, \[Sigma]23, \
\[Sigma]30, \[Sigma]31, \[Sigma]32, \[Sigma]33} -> d];

compList1 = ParallelMap[computeDistThread, data1];
compList2 = ParallelMap[computeDistThread, data2];

weights = ConstantArray[1./ksize, ksize];
gmm0 = MixtureDistribution[weights, compList1];
gmm1 = MixtureDistribution[weights, compList2];

p0 = Compile[{{x1, _Real}, {x2, _Real}, {x3, _Real}, {x4, _Real}}, 
   PDF[gmm0, {x1, x2, x3, x4}], 
   RuntimeAttributes -> {Listable}, 
   CompilationOptions -> {"InlineExternalDefinitions" -> True}, 
   CompilationTarget -> "C"];
p1 = Compile[{{x1, _Real}, {x2, _Real}, {x3, _Real}, {x4, _Real}}, 
   PDF[gmm1, {x1, x2, x3, x4}], 
   RuntimeAttributes -> {Listable}, 
   CompilationOptions -> {"InlineExternalDefinitions" -> True}, 
   CompilationTarget -> "C"];

sample = RandomVariate[gmm0, samplecount];

ps0 = p0 @@@ sample;
ps1 = p1 @@@ sample;

innerKL[x_, y_] := Log[2, Abs[x / If[y != 0, y, 1.*^-16]]];
kl = Sum[innerKL[ps0[[i]], ps1[[i]]], {i, 1, samplecount}] / 
   samplecount;
Print[kl]
