# AI-Memory1
New learning rule of neural networks.  
Similar to the learning rule in the brain, completely different with gradient descent.  
The foundation and the key of AI memory, will open a huge growth potential for Artificial Intelligence.

# Paper
[Ultimate AI-Memory1.pdf](https://jiurl.github.io/files/Ultimate_AI-Memory1.pdf)

# Contact
Email: jiurl@outlook.com  
Homepage: [https://jiurl.github.io](https://jiurl.github.io)

# Requirements
Windows 7 or later  
Visual Studio 2015 or later  
libtorch  
mnist dataset

At present, TensorObserver(observation tool in AI-Memory1) use native win32 gdi APIs.  
So AI-Memory1 can only run in windows, at present.  
In the future, TensorObserver will switch to QT.  
Then AI-Memory1 could run in linux too.

# Building

1 Preparation

1.1 libtorch  
download libtorch 1.3.0 cpu (greater version need Visual Studio version greater than 2015)  
https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.3.0.zip

extract  
copy libtorch-win-shared-with-deps-1.3.0\libtorch\* to AI-Memory1\libtorch\
 
1.2 mnist dataset  
download mnist dataset  
http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz  
http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz  
http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz  
http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

extract
copy files: train-images.idx3-ubyte, train-labels.idx1-ubyte, t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte to AI-Memory1\mnist\

2 Build and Run
Open project with Visual Studio.  
build and run in Release mode.
