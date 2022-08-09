
## ParticleFilter SLAM on NVidia GPU (CUDA)

- [ParticleFilter SLAM on NVidia GPU (CUDA)](#particlefilter-slam-on-nvidia-gpu-cuda)
  - [1. Project Summary](#1-project-summary)
  - [2. Hints](#2-hints)
  - [3. Project Structure](#3-project-structure)
  - [4. Tools & Libraries](#4-tools--libraries)
  - [5. Keyboard Usage](#5-keyboard-usage)
  - [6. Google Tests](#6-google-tests)
  - [7. Legacy Tests](#7-legacy-tests)
  - [8. Demonstration Video](#8-demonstration-video)
  - [9. References](#9-references)




### 1. Project Summary

This project is an implementation of ParticleFilter SLAM on NVidia GPU, the original implementation is the same algorithm with the python programming language (Python Implementation Link on [Github](https://github.com/PenroseWang/SLAM-Based-on-Particle-Filters)). I ran the python implementation on my PC (with CPU Core i7 3.3 GHz and 16 GB of RAM) and it took about 50 minutes to create the Map of the environment. But the CUDA implementation (on RTX 3060) took about 59 seconds. It shows that the CUDA implementation takes about 50 times faster than the Python implementation.
<br />

### 2. Hints
1. Before building the project and Run, make sure to choose "Release" as the build type and x64 for the Platform.
2. Before building the project for the "Legacy Test," make sure to choose "Debug" as the build type.
3. If you lost Camera Direction, press "R" button on keyboard to reset Camera Heading and Position.


### 3. Project Structure

    ├── data_meas                   # Measurement Data
    ├── kernels                     # CUDA Kernels
    ├── device                      # Functions that Handle CUDA Kernels
    ├── host                        # Host Functions (used in Legacy Tests)
    ├── gl                          # OpenGL Render Classes and Functions
    ├── Shaders                     # OpenGL Shader for 3D Rendering
    ├── tests_legacy                # Simple Tests for testing CUDA Kernels Functionality
    ├── tests_google                # Unit-Tests of Project Functionality with Google-Tests Library
    ├── External_Libs               # OpenGL related Library Files


### 4. Tools & Libraries
~~~
Processor : x86 & NVidia GPU
IDE       : Microsoft Visual 2019
Platform  : Windows
Language  : C++ 14
Library   : OpenGL Related Libraries (GLEW, GLFW, GLM)
Library   : C++ Classes for reading python numpy files (cnpy)
Test      : Google Test
~~~

### 5. Keyboard Usage
~~~
A     :  Left
D     :  Right
W     :  Forward
S     :  Backward
U     :  Up
R     :  Reset Camera to the beggining Location
Esc   :  Close UI Window
Mouse :  Camera Heading
~~~

### 6. Google Tests
For running "Google Tests", in the "Solution Explorer" right-click on the "tests_google" project and choose "Set as Startup Project," then run the project.

### 7. Legacy Tests
For using "Legacy Tests" first you have to download the data file ([Download Link](https://drive.google.com/file/d/14LSzWpw70DIyk2ylUonQl43srUwkdQxE/view?usp=sharing)), then extract it in the "Solution Directory." After that, you have to open the "main.cu" file and uncomment "#define TEST_LEGACY." Now you can build the project and see the results of the Tests. Before building the project make sure to choose "Debug" as the build type.

### 8. Demonstration Video
You can watch the [Demonstration Video](https://www.youtube.com/watch?v=LYMZJeQxGHw) on the Youtube.

### 9. References
~~~
1. Python implementation of "Particle Filter SLAM"
    https://github.com/PenroseWang/SLAM-Based-on-Particle-Filters
~~~
