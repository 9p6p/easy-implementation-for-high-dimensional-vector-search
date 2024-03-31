# implementation-for-high-dimensional-vector-search
an easy C++ source code with KNN and NSG method for ANNS，without any third party or any other dependency.

modified from https://github.com/ZJULearning/nsg and https://github.com/ZJULearning/efanna_graph.
and the principle of the KNN and NSG method are mentioned below:
KNN:https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=103ac7f316bf8cdad3133b4ce2bbd28d091e7974
NSG:https://arxiv.org/pdf/1707.00143.pdf

## goal
make it easy for beginners to experience similarity searches.
the code doesn't conclude SIMD or multithread, if you want to use, please see the ZJU original project to modify. 

## using way
the main.cpp accept outside input to test the program, and can be easily modified.
with cmake tools, it's easy to run testfile.cpp to test the program.
for more information read the problem file will help.

## performance
used for Huawei Algorithm Elite Practical Camp Phase 5 - Approximate Retrieval of High-Dimensional Vector Data.
and in the case of 50000 vector datas and 1600 querys with total processing time 50s, the code get a recall over 0.8725.
although in my case with total 50s, the recall is over 0.99.

## future work
the product quantization method and other methods are on the way, but maybe I will create a new repository for it.

## other
if you have any problem or any thing want to ask, welcome to email me at 12332491@mail.sustech.edu.cn.
I would love to be contacted for any internship opportunities!
