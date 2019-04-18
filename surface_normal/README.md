# surface-normal
This is a tool.code for CVPR2019 paper 1899: Deep Surface Normal Guided Depth Prediction for Outdoor Secene from Sparce Lidar Data and Single Color Image.
The toolbox consists of some tools you might need for preparing the training data .  

## calplane-normal  
The most important one is the  Mat calplanenormal(Mat &src);  

This function calculate the surface normal of the sparce lidar input, it will return a Mat res as the surface normal.
You should set fcxcy ,windowsize, threshold before use the function.
The following function is used in Mat calplanenormal(Mat &src):  
void cvFitPlane(const CvMat* points, float* plane);  
void CallFitPlane(const Mat& depth,int * points,int i,int j,float *plane12);  
void search_plane_neighbor(Mat &img,int i,int j ,float threhold,int* result);  
int telldirection(float * abc,int i,int j,float d);  
  
  
  
  
So if you need a clean code , you can download the clean.hpp. 
There is also a demo to show how to use the function.
If you get any bugs in the clean.hpp,please check the original tool.cpp or pull an issue. 

## The suggest environment is g++ 4.1/linux16.04, opencv2.4.9.
