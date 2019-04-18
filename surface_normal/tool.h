#ifndef TOOL
#define TOOL
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <vector>
#include <sys/types.h>  
#include <sys/stat.h>  
#include <fstream>
#include <iomanip>
#include <cassert>
using namespace cv;
using namespace std;

template <class Type>  
    Type stringToNum(const std::string& str){  
            std::istringstream iss(str);  
            Type num;  
            iss >> num;  
            return num;      
    }  
// string Int_to_String(int n)

// {

// ostringstream stream;

// stream<<n; //n为int类型

// return stream.str();

//}
//-- basic toolbox------
//----------------------
void MkoneDir(char * dirname);
void MkoneDir(string stdirname);
void Mat2CvMat(Mat* Input,CvMat * out);
void CvMat2Mat(CvMat* Input,Mat * out);
vector<string> ReadDir(string path);
vector<string> ReadDir(char* fpath);
void MkDir(string stdirname);
void MkDir(char * char_stdirname);
//根据特定后缀获得文件
std::vector<std::string> get_specific_files(std::string path, std::string suffix);

//temp test function
Mat closecheck( Mat &raw);
Mat rawdepth2normal(Mat & rawdepth,float * paras);
//-- sparcenormal toolbox------
//-----------------------------
extern float fcxcy[3];
extern int  WINDOWSIZE;
extern float Tthrehold;
//生成一个球形物体的深度图用来求normal标称方向
Mat GetaSphere();
Mat sample(Mat input);
void readTxt(string file,float* fcxcy);
void cvFitPlane(const CvMat* points, float* plane);
void CallFitPlane(const Mat& depth,int * points,int i,int j,float *plane12);
void search_plane_neighbor(Mat &img,int i,int j ,float threhold,int* result);
int telldirection(float * abc,int i,int j,float d);
vector<string> search_working_dir(string inputdir);
void get_dir_para(string inputdir,float * fcxcy);
Mat calplanenormal(Mat &src);
Mat caldensenormal(Mat & rawdept);
//------------------lidar-combine------------------------
//最近邻插值同时生成一张显示最近邻的欧式距离的可信度图
void nearneigbor(Mat& src,int windowsize, Mat * ress);
int search_neighbor_x(Mat &img,int i,int j,int circle_size );
int search_neighbor_y(Mat &img,int i,int j,int circle_size );
Mat sparce_depth2normal( Mat &input,float * paras,int circle_size);
//误差欧式距离权重法
void os(Mat& src,int windowsize, Mat * ress);
#endif
