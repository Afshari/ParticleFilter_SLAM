
## ParticleFilter SLAM on NVidia GPU (CUDA)

- [Project Summary](#1-project-summary)
- [Project Structure]
- [Google Test]
- [Legacy Test]
- [References]




### 1. Project Summary

#### This project is implementation of ParticleFilter SLAM on NVidia GPU, the original implementation is the same algorithm with python programming language. I ran the python implementation on my PC (with 3.3 GHz and 16 GB of RAM) and it took about 50 minutes to creating the Map of the environment. But the CUDA implementation (on RTX 3060) tooks about 59 seconds. It shows that the CUDA implementation takes about 50 times faster than the Python implementation.
