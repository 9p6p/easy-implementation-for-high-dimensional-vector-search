# Implementation-for-approximate-nearest-neighbor-search
## An easy C++ source code without any third party or any other dependency.

with KNN and NSG method for ANNS, for test data, see the right release v1.

modified from https://github.com/ZJULearning/nsg and https://github.com/ZJULearning/efanna_graph.

and the principle of the KNN and NSG method are mentioned below:

KNN: Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures

NSG: Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph

## Goal
make it easy for beginners to experience similarity searches.

the code doesn't conclude SIMD or multithread, if you want to use, please see the ZJU original project to modify. 

if you find it useful, give me a star!

## Using
the main.cpp accept outside input to test the program, and can be easily modified.

with cmake tools, it's easy to run testfile.cpp to test the program.

for more information read the problem file will help.

## Performance
used for Huawei Algorithm Elite Practical Camp Phase 5 - Approximate Retrieval of High-Dimensional Vector Data.

and in the case of 50000 vector datas and 1600 querys with total processing time 50s, the code get a recall over 0.8725.

although in my case with total 50s, the recall is over 0.99.

## Future work
the product quantization method and other methods are on the way, but maybe I will create a new repository for it.

## others
if you have any problem or any thing want to ask, welcome to email me at 12332491@mail.sustech.edu.cn.

I would love to be contacted for any internship opportunities!
