#include "tool.h"
float fcxcy[3]={0,0,0};
//15 0.1 for kitti lidar, 7 0.1 for nyu
int  WINDOWSIZE =15;
float Tthrehold = 0.1;

void 
Mat2CvMat(Mat* Input,CvMat * out){*out = *Input;}
void 
CvMat2Mat(CvMat* Input,Mat * out){*out = Mat(Input, true);}
void 
MkoneDir(char * dirname){
     if(access(dirname, F_OK)==-1){
        bool success_flag=mkdir(dirname, S_IRWXU|S_IRWXG|S_IRWXO);  
        if(success_flag!=0){
           string Dir_p(dirname);
           std::cout<<"E: Error raise in make dir: "<<Dir_p<<std::endl;
           system("pause");
           exit(1);
           }
       }
     else if((access(dirname, F_OK)==0)){ string Dir_p(dirname);
            // std::cout<<"W: Dir already exists: "<<Dir_p<<std::endl;
            }
       
}
void 
MkoneDir(string stdirname){
     
    const char *dirname=stdirname.c_str();
     if(access(dirname, F_OK)==-1){
        bool success_flag=mkdir(dirname, S_IRWXU|S_IRWXG|S_IRWXO);  
        if(success_flag!=0){
           string Dir_p(dirname);
           std::cout<<"E: Error raise in make dir: "<<Dir_p<<std::endl;
           system("pause");
           exit(1);
           }
       }
     else if((access(dirname, F_OK)==0)){ string Dir_p(dirname);
            // std::cout<<"W: Dir already exists: "<<Dir_p<<std::endl;
            }
       
}
void
MkDir(string stdirname){
	if(stdirname.at(stdirname.size()-1)!='/')
	stdirname.append(1,'/');
    bool flag=true;
	vector<string>caps;
	while(flag){
		string firstname=stdirname.substr(0,stdirname.find_first_of("/")+1);
		stdirname=stdirname.substr(stdirname.find_first_of("/")+1,stdirname.size()-1);
		caps.push_back(firstname);
		if(stdirname.size()<=1)
		flag=false;
	}
	int cengji=caps.size();
	for (int i= 0;i<cengji;i++){
	string dir_name;
	for(int j=0;j<=i;j++)
	dir_name.append(caps[j]);
	MkoneDir(dir_name);
	// cout<<dir_name<<endl;
	}
}
void
MkDir(char * char_stdirname){
	string stdirname(char_stdirname);
	if(stdirname.at(stdirname.size()-1)!='/')
	stdirname.append(1,'/');
    bool flag=true;
	vector<string>caps;
	while(flag){
		string firstname=stdirname.substr(0,stdirname.find_first_of("/")+1);
		stdirname=stdirname.substr(stdirname.find_first_of("/")+1,stdirname.size()-1);
		caps.push_back(firstname);
		if(stdirname.size()<=1)
		flag=false;
	}
	int cengji=caps.size();
	for (int i= 0;i<cengji;i++){
	string dir_name;
	for(int j=0;j<=i;j++)
	dir_name.append(caps[j]);
	MkoneDir(dir_name);
	// cout<<dir_name<<endl;
	}
}
std::vector<std::string> 
get_specific_files(std::string path, std::string suffix){
  cout<<path<<endl;
  DIR* dir;
 struct dirent* ptr;
 vector<string> fileList; 
 dir = opendir(path.c_str());
 while((ptr = readdir(dir)) != NULL)
 {
    if(strcmp(ptr->d_name,".") != 0 && strcmp(ptr->d_name,"..") != 0)
    {
       
        string name=path+ptr->d_name;
        // cout<<"filename:"<<name<<endl;
        if( (name.compare((name.size() - 4), 4, suffix) == 0)  ||   (name.compare((name.size() - 3), 3, suffix) == 0) )
        fileList.push_back(name);    
     
    } 
 }
 closedir(dir); 

 sort(fileList.begin(),fileList.end());
 return fileList;
}




vector<string>
ReadDir(string path){
 DIR* dir;
 struct dirent* ptr;
 vector<string> fileList; 
 dir = opendir(path.c_str());
 while((ptr = readdir(dir)) != NULL)
 {
    if(strcmp(ptr->d_name,".") != 0 && strcmp(ptr->d_name,"..") != 0)
    {
       
        string name=path+ptr->d_name;
        //cout<<"filename:"<<name<<endl;
        fileList.push_back(name);    
     
    } 
 }
 closedir(dir); 
 sort(fileList.begin(),fileList.end());
 return fileList;
}
vector<string>
ReadDir(char* fpath){
 string path(fpath);
 DIR* dir;
 struct dirent* ptr;
 vector<string> fileList; 
 dir = opendir(path.c_str());
 while((ptr = readdir(dir)) != NULL)
 {
    if(strcmp(ptr->d_name,".") != 0 && strcmp(ptr->d_name,"..") != 0)
    {
       
        string name=path+ptr->d_name;
        //cout<<"filename:"<<name<<endl;
        fileList.push_back(name);    
     
    } 
 }
 closedir(dir); 
 sort(fileList.begin(),fileList.end());
 return fileList;
}
//--------------------------sp2normal tool----


Mat sample(Mat input){

	int h=input.rows;
	int w=input.cols;
	srand((unsigned )time(NULL));

	if(input.channels()==1){

	input.convertTo(input, CV_16UC1);
	Mat res = Mat::zeros(input.size(),CV_16UC1);
     for(int i =0;i<500;i++){
		  int ph=rand()%h;int pw=rand()%w;
		//for (int idx=0; idx<1;idx++)
			//for (int idy=0; idy<1;idy++){
			   if((  (ph>0)&&(ph<h)&&(pw>0)&&(pw<w) )) 
				res.at<unsigned short >(ph,pw)=input.at<unsigned short >(ph,pw);
				}return res;
	 }
	
	
	else{
		  input.convertTo(input, CV_8UC3);
		  Mat res = Mat::zeros(input.size(),CV_8UC3);
		   for(int i =0;i<500;i++){
		  int ph=rand()%h;int pw=rand()%w;
		// for (int idx=0; idx<1;idx++)
			// for (int idy=0; idy<1;idy++)
			{
			    if((  (ph>0)&&(ph<h)&&(pw>0)&&(pw<w) )) 
				res.at<Vec3b >(ph,pw)=input.at<Vec3b>(ph,pw);
				}
	 }
	  return res;
	 }

	 
   
}
void
search_plane_neighbor(Mat &img,int i,int j ,float threhold,int* result){
	 int cols =img.cols;
	 int rows =img.rows; 
	 for (int ii=0; ii<WINDOWSIZE*WINDOWSIZE;ii++)
	 result[ii]=0;
	 float center_depth = img.at<float>(i,j);
     for (int idx=0; idx<WINDOWSIZE;idx++)
	  for (int idy=0; idy<WINDOWSIZE;idy++){
		  int rx= i-int(WINDOWSIZE/2)+idx;
		  int ry= j-int(WINDOWSIZE/2)+idy;
		 if(  rx>= rows || ry>=cols )continue;
		 if( img.at<float>(rx,ry)==0.0)continue;
		 if( abs(img.at<float>(rx,ry)-center_depth)<=Tthrehold*center_depth )
             result[idx*WINDOWSIZE+idy]=1;
		}
}

// Ax+by+cz=D
void
CallFitPlane(const Mat& depth,int * points,int i,int j,float *plane12) {
	float f =fcxcy[0];
	float cx=fcxcy[1];
	float cy=fcxcy[2];
	vector<float>X_vector;
	vector<float>Y_vector;
	vector<float>Z_vector;
	for(int num_point=0; num_point<WINDOWSIZE*WINDOWSIZE;num_point++ )
		if (points[num_point]==1) {//search 已经处理了边界,此处不需要再处理了
		int point_i,point_j;
		point_i=floor(num_point/WINDOWSIZE);
		point_j=num_point-(point_i*WINDOWSIZE);
		point_i+=i-int(WINDOWSIZE/2);point_j+=j-int(WINDOWSIZE/2);
		float x = (point_j - cx) * depth.at<float>(point_i, point_j ) * 1.0 / f;
		float y = (point_i - cy) * depth.at<float>(point_i, point_j )* 1.0 / f;
		float z = depth.at<float>(point_i,point_j);
		X_vector.push_back(x);
		Y_vector.push_back(y);
		Z_vector.push_back(z);
		}
    CvMat*points_mat = cvCreateMat(X_vector.size(), 3, CV_32FC1);//定义用来存储需要拟合点的矩阵 
	if(X_vector.size()<3){ plane12[0]=-1;plane12[1]=-1;plane12[2]=-1;plane12[3]=-1;return;}
	for (int ii=0;ii < X_vector.size(); ++ii){
			points_mat->data.fl[ii * 3 + 0] = X_vector[ii];//矩阵的值进行初始化   X的坐标值
			points_mat->data.fl[ii * 3 + 1] = Y_vector[ii];//  Y的坐标值
			points_mat->data.fl[ii * 3 + 2] = Z_vector[ii];//
		}
		// float plane12[4] = { 0 };//定义用来储存平面参数的数组 
		cvFitPlane(points_mat, plane12);//调用方程 
		if( telldirection(plane12,i,j,depth.at<float>(i,j))  ){
		plane12[0]=-plane12[0];
		plane12[1]=-plane12[1];
		plane12[2]=-plane12[2];}
		X_vector.clear();
		Y_vector.clear();
		Z_vector.clear();
		cvReleaseMat(&points_mat);
}

void 
cvFitPlane(const CvMat* points, float* plane){
	// Estimate geometric centroid.
	int nrows = points->rows;
	int ncols = points->cols;
	int type = points->type;
	CvMat* centroid = cvCreateMat(1, ncols, type);
	cvSet(centroid, cvScalar(0));
	for (int c = 0; c<ncols; c++){
		for (int r = 0; r < nrows; r++)
		{
			centroid->data.fl[c] += points->data.fl[ncols*r + c];
		}   
		centroid->data.fl[c] /= nrows;
	}
	// Subtract geometric centroid from each point.
	CvMat* points2 = cvCreateMat(nrows, ncols, type);
	for (int r = 0; r<nrows; r++)
		for (int c = 0; c<ncols; c++)
			points2->data.fl[ncols*r + c] = points->data.fl[ncols*r + c] - centroid->data.fl[c];
	// Evaluate SVD of covariance matrix.
	CvMat* A = cvCreateMat(ncols, ncols, type);
	CvMat* W = cvCreateMat(ncols, ncols, type);
	CvMat* V = cvCreateMat(ncols, ncols, type);
	cvGEMM(points2, points, 1, NULL, 0, A, CV_GEMM_A_T);
	cvSVD(A, W, NULL, V, CV_SVD_V_T);
	// Assign plane coefficients by singular vector corresponding to smallest singular value.
	plane[ncols] = 0;
	for (int c = 0; c<ncols; c++){
		plane[c] = V->data.fl[ncols*(ncols - 1) + c];
		plane[ncols] += plane[c] * centroid->data.fl[c];
	}
	// Release allocated resources.
	cvReleaseMat(&centroid);
	cvReleaseMat(&points2);
	cvReleaseMat(&A);
	cvReleaseMat(&W);
	cvReleaseMat(&V);
}

int 
telldirection(float * abc,int i,int j,float d){
	float f =fcxcy[0];
	float cx=fcxcy[1];
	float cy=fcxcy[2];
	float x = (j - cx) *d * 1.0 / f;
    float y = (i - cy) *d * 1.0 / f;
    float z = d;
	// Vec3f camera_center=Vec3f(cx,cy,0);
	Vec3f cor = Vec3f(0-x, 0-y, 0-z);
	Vec3f abcline = Vec3f(abc[0],abc[1],abc[2]);
	float corner = cor.dot(abcline);
	//  float corner =(cx-x)*abc[0]+(cy-y) *abc[1]+(0-z)*abc[2];
	if (corner>=0)
	   return 1;
	else return 0;
 
}
void 
readTxt(string file,float* fcxcy){
    ifstream infile;  
    cout.setf(ios::fixed);
    infile.open(file.data());   //将文件流对象与文件连接起来 
    assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 
    float f = 0.0 ,cx= 0.0,cy = 0.0,null=0.0;
   
     infile>>f;
     infile>>null;
     infile>>cx;
     infile>>null;
     infile>>null;
     infile>>cy;
    
    // float fcxcy[3];
     fcxcy[0] =f;
     fcxcy[1] =cx;
     fcxcy[2] =cy;
    //  return fcxcy;
    //  cout<<fixed <<setprecision(20)<<f<<endl;
    //  cout<< setprecision(4)<<cx<<endl;
    //  cout<< setprecision(4)<<cy<<endl;

}
vector<string>
search_working_dir(string inputdir){ 
	   string prefix ="_sync";
       vector<string> input_filepathList= ReadDir(inputdir);
	   vector<string>  workingdirlist ;
	   int filenum=input_filepathList.size();
	   for(int m=0;m<filenum; m++){
			string filename=input_filepathList[m];
			string filename_prefix =filename.substr(filename.size()-5,filename.size()-1) ;
			if(filename_prefix==prefix)
			  workingdirlist.push_back(filename);
           
   		 }
    return workingdirlist;
}
void
get_dir_para(string inputdir,float * fcxcy){ 
	 string data = inputdir.substr(inputdir.find("2011_"), 10);
	 if(data=="2011_09_26"){fcxcy[0]=721.5377;fcxcy[1]=596.5593;fcxcy[2]=149.854;}
	 else if(data=="2011_09_28"){fcxcy[0]=707.0493;fcxcy[1]=604.0814;fcxcy[2]=162.5066;}
	 else if(data=="2011_09_29"){fcxcy[0]=718.3351 ;fcxcy[1]=600.3891;fcxcy[2]=159.5122;}
	 else if(data=="2011_09_30"){fcxcy[0]=707.0912;fcxcy[1]=601.8873;fcxcy[2]=165.1104;}
	 else if(data=="2011_10_03"){fcxcy[0]=718.856 ;fcxcy[1]= 607.1928;fcxcy[2]= 161.2157;}
     else{cout<<"Dir data:"<<data<<" is unexpected. please check."<<endl;exit(1);}
	//  
}
//平面拟合求normal的方法，再此处备份，应该不会再用了。
Mat 
calplanenormal(Mat  &src){
	 float f =fcxcy[0];
	 float cx=fcxcy[1];
	 float cy=fcxcy[2];
     Mat normals = Mat::zeros(src.size(),CV_32FC3);
	 src.convertTo(src,CV_32FC1);
	 src*=1.0;
	 int cols =src.cols;
	 int rows =src.rows;
    //  int plane_points[WINDOWSIZE*WINDOWSIZE]={0};
	 int * plane_points = new int[WINDOWSIZE*WINDOWSIZE];
	 float * plane12 = new float[4];
	 for (int i=0;i< rows;i++)
				for (int j=0;j< cols;j++){
                    //for kitti and nyud test
					if(src.at<float>(i,j)==0.0)continue;
                    //for:nyud train
                    //  if(src.at<float>(i,j)<=4000.0)continue;   

					search_plane_neighbor(src,i,j,15.0,plane_points);
					CallFitPlane(src,plane_points,i,j,plane12);
					Vec3f d = Vec3f(plane12[0],plane12[1],plane12[2]);
					Vec3f n = normalize(d);
					normals.at<Vec3f>(i, j) = n;
			}
	 Mat res = Mat::zeros(src.size(),CV_32FC3);
     for (int i=0;i<rows;i++)
      for (int j=0;j<cols;j++){
        res.at<Vec3f>(i, j)[0] = -1.0 * normals.at<Vec3f>(i, j)[0];
        res.at<Vec3f>(i, j)[2] = -1.0 * normals.at<Vec3f>(i, j)[1];
        res.at<Vec3f>(i, j)[1] = -1.0 * normals.at<Vec3f>(i, j)[2];
     }

	 delete[] plane12;
	 delete[] plane_points;
	 normals.release();
     for (int i=0;i<rows;i++)
      for (int j=0;j<cols;j++){
		if(!(res.at<Vec3f>(i, j)[0]==0&&res.at<Vec3f>(i, j)[1]==0&&res.at<Vec3f>(i, j)[2]==0)){
			res.at<Vec3f>(i, j)[0] += 1.0 ;
			res.at<Vec3f>(i, j)[2] += 1.0 ;
			res.at<Vec3f>(i, j)[1] += 1.0;
		 }
      }
	 
     res =res * 127.5;
     res.convertTo(res, CV_8UC3);
     cvtColor(res, res, COLOR_BGR2RGB);
	 return res;
}


Mat 
caldensenormal(Mat & depth){
     float f =fcxcy[0];
	 float cx=fcxcy[1];
	 float cy=fcxcy[2];
    int rows=depth.rows;
    int cols=depth.cols;
    Mat normals = Mat::zeros(depth.size(),CV_32FC3);
   for (int i=0;i<rows;i++)
    for (int j=0;j<cols;j++){
        float x,y,z,x1,y1,z1,x2,y2,z2;
        Vec3f x3,y3;
        if(i == 0 || i == rows-1 || j == 0 || j == cols-1){
            x = (j - cx) * depth.at<float>(i, j ) * 1.0 / f;
            y = (i - cy) * depth.at<float>(i, j )* 1.0 / f;
            z = depth.at<float>(i, j);
            if (j == 0  && i == 0){
                x1 = (j + 1 - cx) *depth.at<float>(i, j+1 ) * 1.0 / f;
                y1 = (i - cy) * depth.at<float>(i, j+1 ) * 1.0 / f;
                z1 = depth.at<float>(i, j+1 );
                x2 = (j - cx) * depth.at<float>(i+1, j ) * 1.0 / f;
                y2 = (i + 1 - cy) * depth.at<float>(i+1, j ) * 1.0 / f;
                z2 = depth.at<float>(i+1, j);
                x3 = Vec3f(x1-x, y1-y, z1-z);
                y3 = Vec3f(x2-x, y2-y, z2-z); 
            }
            else if (j == 0 && i == rows- 1){
                x1 = (j + 1 - cx) *depth.at<float>(i, j+1 ) * 1.0 / f;
                y1 = (i - cy) * depth.at<float>(i, j+1 ) * 1.0 / f;
                z1 = depth.at<float>(i, j+1 );
                x2 = (j - cx) * depth.at<float>(i-1, j ) * 1.0 / f;
                y2 = (i - 1 - cy) * depth.at<float>(i-1, j ) * 1.0 / f;
                z2 = depth.at<float>(i-1, j);
                x3 = Vec3f(x1-x, y1-y, z1-z);
                y3 = -1.0 * Vec3f(x2-x, y2-y, z2-z); 
            }
            else if(i == 0 && j ==cols - 1) {
                x1 = (j - 1 - cx) *depth.at<float>(i, j-1 ) * 1.0 / f;
                y1 = (i - cy) * depth.at<float>(i, j-1 ) * 1.0 / f;
                z1 = depth.at<float>(i, j-1 );
                x2 = (j - cx) * depth.at<float>(i+1, j ) * 1.0 / f;
                y2 = (i + 1 - cy) * depth.at<float>(i+1, j ) * 1.0 / f;
                z2 = depth.at<float>(i+1, j);
                x3 = -1.0 * Vec3f(x1-x, y1-y, z1-z);
                y3 = Vec3f(x2-x, y2-y, z2-z); 
            }
            else if (j == cols - 1 && i == rows - 1){
                x1 = (j - 1 - cx) *depth.at<float>(i, j-1 ) * 1.0 / f;
                y1 = (i - cy) * depth.at<float>(i, j-1 ) * 1.0 / f;
                z1 = depth.at<float>(i, j-1 );
                x2 = (j - cx) * depth.at<float>(i-1, j ) * 1.0 / f;
                y2 = (i - 1 - cy) * depth.at<float>(i-1, j ) * 1.0 / f;
                z2 = depth.at<float>(i-1, j);
                x3 = -1.0 * Vec3f(x1-x, y1-y, z1-z);
                y3 = -1.0 * Vec3f(x2-x, y2-y, z2-z); 
            }
            else if (j == 0 || i==0){
                x1 = (j + 1 - cx) *depth.at<float>(i, j+1 ) * 1.0 / f;
                y1 = (i - cy) * depth.at<float>(i, j+1 ) * 1.0 / f;
                z1 = depth.at<float>(i, j+1 );
                x2 = (j - cx) * depth.at<float>(i+1, j ) * 1.0 / f;
                y2 = (i + 1 - cy) * depth.at<float>(i+1, j ) * 1.0 / f;
                z2 = depth.at<float>(i+1, j);
                x3 = Vec3f(x1-x, y1-y, z1-z);
                y3 = Vec3f(x2-x, y2-y, z2-z); 
            }
            else if (j == cols- 1){
                x1 = (j - 1 - cx) *depth.at<double>(i, j-1 ) * 1.0 / f;
                y1 = (i - cy) * depth.at<double>(i, j-1 ) * 1.0 / f;
                z1 = depth.at<double>(i, j-1 );
                x2 = (j - cx) * depth.at<double>(i+1, j ) * 1.0 / f;
                y2 = (i + 1 - cy) * depth.at<double>(i+1, j ) * 1.0 / f;
                z2 = depth.at<double>(i+1, j);
                x3 = -1.0 * Vec3d(x1-x, y1-y, z1-z);
                y3 = Vec3d(x2-x, y2-y, z2-z); 
            }
            else if(i ==  rows- 1){
                x1 = (j + 1 - cx) *depth.at<float>(i, j+1 ) * 1.0 / f;
                y1 = (i - cy) * depth.at<float>(i, j+1 ) * 1.0 / f;
                z1 = depth.at<float>(i, j+1 );
                x2 = (j - cx) * depth.at<float>(i-1, j ) * 1.0 / f;
                y2 = (i - 1 - cy) * depth.at<float>(i-1, j ) * 1.0 / f;
                z2 = depth.at<float>(i-1, j);
                x3 = Vec3f(x1-x, y1-y, z1-z);
                y3 = -1.0 * Vec3f(x2-x, y2-y, z2-z); 
            }
            Vec3f d = x3.cross(y3);
            Vec3f n = normalize(d);
            normals.at<Vec3f>(i, j) = n;
            }
        else{
            x = (j - cx) * depth.at<float>(i, j ) * 1.0 / f;
            y = (i - cy) * depth.at<float>(i, j )* 1.0 / f;
            z = depth.at<float>(i, j);
            x1 = (j + 1 - cx) *depth.at<float>(i, j+1 ) * 1.0 / f;
            y1 = (i - cy) * depth.at<float>(i, j+1 ) * 1.0 / f;
            z1 = depth.at<float>(i, j+1 );
            x2 = (j - cx) * depth.at<float>(i+1, j ) * 1.0 / f;
            y2 = (i + 1 - cy) * depth.at<float>(i+1, j ) * 1.0 / f;
            z2 = depth.at<float>(i+1, j);
            x3 = Vec3f(x1-x, y1-y, z1-z);
            y3 = Vec3f(x2-x, y2-y, z2-z); 
            Vec3d d = x3.cross(y3);
            Vec3d n = normalize(d);
            normals.at<Vec3f>(i, j) = n;
        }               
    }
    Mat res = Mat::zeros(depth.size(),CV_32FC3);
    for (int i=0;i<rows;i++)
     for (int j=0;j<cols;j++){
        res.at<Vec3f>(i, j)[0] = -1.0 * normals.at<Vec3f>(i, j)[0];
        res.at<Vec3f>(i, j)[2] = -1.0 * normals.at<Vec3f>(i, j)[1];
        res.at<Vec3f>(i, j)[1] = -1.0 * normals.at<Vec3f>(i, j)[2];
    }
    //---------------------------------------
    for (int i=0;i<rows;i++)
     for (int j=0;j<cols;j++)
          if (depth.at<float>(i,j)!=0)             
          {
        res.at<Vec3f>(i, j)[0] += 1.0 ;
        res.at<Vec3f>(i, j)[2] += 1.0 ;
        res.at<Vec3f>(i, j)[1] += 1.0;
    }
    res =res * 127.5;
    res.convertTo(res, CV_8UC3);
    cvtColor(res, res, COLOR_BGR2RGB);
    return res;

    

}
Mat 
rawdepth2normal(Mat & rawdepth,float* paras){
    float f=paras[0];
	float cx=paras[1];
	float cy=paras[2];
    Mat depth;
    rawdepth.convertTo(rawdepth,CV_32FC3);
    std::vector<cv::Mat> channels(3);
    split(rawdepth, channels); 
    depth=65536.0*channels[0]+ 256.0*channels[1]+1.0*channels[2];
    Mat normals = Mat::zeros(depth.size(),CV_32FC3);
    int rows=depth.rows;
    int cols=depth.cols;
    for (int i=0;i<rows;i++)
    for (int j=0;j<cols;j++){
        float x,y,z,x1,y1,z1,x2,y2,z2;
        Vec3f x3,y3;
        if(i == 0 || i == rows-1 || j == 0 || j == cols-1){
            x = (j - cx) * depth.at<float>(i, j ) * 1.0 / f;
            y = (i - cy) * depth.at<float>(i, j )* 1.0 / f;
            z = depth.at<float>(i, j);
            if (j == 0  && i == 0){
                x1 = (j + 1 - cx) *depth.at<float>(i, j+1 ) * 1.0 / f;
                y1 = (i - cy) * depth.at<float>(i, j+1 ) * 1.0 / f;
                z1 = depth.at<float>(i, j+1 );
                x2 = (j - cx) * depth.at<float>(i+1, j ) * 1.0 / f;
                y2 = (i + 1 - cy) * depth.at<float>(i+1, j ) * 1.0 / f;
                z2 = depth.at<float>(i+1, j);
                x3 = Vec3f(x1-x, y1-y, z1-z);
                y3 = Vec3f(x2-x, y2-y, z2-z); 
            }
            else if (j == 0 && i == rows- 1){
                x1 = (j + 1 - cx) *depth.at<float>(i, j+1 ) * 1.0 / f;
                y1 = (i - cy) * depth.at<float>(i, j+1 ) * 1.0 / f;
                z1 = depth.at<float>(i, j+1 );
                x2 = (j - cx) * depth.at<float>(i-1, j ) * 1.0 / f;
                y2 = (i - 1 - cy) * depth.at<float>(i-1, j ) * 1.0 / f;
                z2 = depth.at<float>(i-1, j);
                x3 = Vec3f(x1-x, y1-y, z1-z);
                y3 = -1.0 * Vec3f(x2-x, y2-y, z2-z); 
            }
            else if(i == 0 && j ==cols - 1) {
                x1 = (j - 1 - cx) *depth.at<float>(i, j-1 ) * 1.0 / f;
                y1 = (i - cy) * depth.at<float>(i, j-1 ) * 1.0 / f;
                z1 = depth.at<float>(i, j-1 );
                x2 = (j - cx) * depth.at<float>(i+1, j ) * 1.0 / f;
                y2 = (i + 1 - cy) * depth.at<float>(i+1, j ) * 1.0 / f;
                z2 = depth.at<float>(i+1, j);
                x3 = -1.0 * Vec3f(x1-x, y1-y, z1-z);
                y3 = Vec3f(x2-x, y2-y, z2-z); 
            }
            else if (j == cols - 1 && i == rows - 1){
                x1 = (j - 1 - cx) *depth.at<float>(i, j-1 ) * 1.0 / f;
                y1 = (i - cy) * depth.at<float>(i, j-1 ) * 1.0 / f;
                z1 = depth.at<float>(i, j-1 );
                x2 = (j - cx) * depth.at<float>(i-1, j ) * 1.0 / f;
                y2 = (i - 1 - cy) * depth.at<float>(i-1, j ) * 1.0 / f;
                z2 = depth.at<float>(i-1, j);
                x3 = -1.0 * Vec3f(x1-x, y1-y, z1-z);
                y3 = -1.0 * Vec3f(x2-x, y2-y, z2-z); 
            }
            else if (j == 0 || i==0){
                x1 = (j + 1 - cx) *depth.at<float>(i, j+1 ) * 1.0 / f;
                y1 = (i - cy) * depth.at<float>(i, j+1 ) * 1.0 / f;
                z1 = depth.at<float>(i, j+1 );
                x2 = (j - cx) * depth.at<float>(i+1, j ) * 1.0 / f;
                y2 = (i + 1 - cy) * depth.at<float>(i+1, j ) * 1.0 / f;
                z2 = depth.at<float>(i+1, j);
                x3 = Vec3f(x1-x, y1-y, z1-z);
                y3 = Vec3f(x2-x, y2-y, z2-z); 
            }
            else if (j == cols- 1){
                x1 = (j - 1 - cx) *depth.at<double>(i, j-1 ) * 1.0 / f;
                y1 = (i - cy) * depth.at<double>(i, j-1 ) * 1.0 / f;
                z1 = depth.at<double>(i, j-1 );
                x2 = (j - cx) * depth.at<double>(i+1, j ) * 1.0 / f;
                y2 = (i + 1 - cy) * depth.at<double>(i+1, j ) * 1.0 / f;
                z2 = depth.at<double>(i+1, j);
                x3 = -1.0 * Vec3d(x1-x, y1-y, z1-z);
                y3 = Vec3d(x2-x, y2-y, z2-z); 
            }
            else if(i ==  rows- 1){
                x1 = (j + 1 - cx) *depth.at<float>(i, j+1 ) * 1.0 / f;
                y1 = (i - cy) * depth.at<float>(i, j+1 ) * 1.0 / f;
                z1 = depth.at<float>(i, j+1 );
                x2 = (j - cx) * depth.at<float>(i-1, j ) * 1.0 / f;
                y2 = (i - 1 - cy) * depth.at<float>(i-1, j ) * 1.0 / f;
                z2 = depth.at<float>(i-1, j);
                x3 = Vec3f(x1-x, y1-y, z1-z);
                y3 = -1.0 * Vec3f(x2-x, y2-y, z2-z); 
            }
            Vec3f d = x3.cross(y3);
            Vec3f n = normalize(d);
            normals.at<Vec3f>(i, j) = n;
            }
        else{
            x = (j - cx) * depth.at<float>(i, j ) * 1.0 / f;
            y = (i - cy) * depth.at<float>(i, j )* 1.0 / f;
            z = depth.at<float>(i, j);
            x1 = (j + 1 - cx) *depth.at<float>(i, j+1 ) * 1.0 / f;
            y1 = (i - cy) * depth.at<float>(i, j+1 ) * 1.0 / f;
            z1 = depth.at<float>(i, j+1 );
            x2 = (j - cx) * depth.at<float>(i+1, j ) * 1.0 / f;
            y2 = (i + 1 - cy) * depth.at<float>(i+1, j ) * 1.0 / f;
            z2 = depth.at<float>(i+1, j);
            x3 = Vec3f(x1-x, y1-y, z1-z);
            y3 = Vec3f(x2-x, y2-y, z2-z); 
            Vec3d d = x3.cross(y3);
            Vec3d n = normalize(d);
            normals.at<Vec3f>(i, j) = n;
        }               
    }
    Mat res = Mat::zeros(depth.size(),CV_32FC3);
    for (int i=0;i<rows;i++)
     for (int j=0;j<cols;j++){
        res.at<Vec3f>(i, j)[0] = -1.0 * normals.at<Vec3f>(i, j)[0];
        res.at<Vec3f>(i, j)[2] = -1.0 * normals.at<Vec3f>(i, j)[1];
        res.at<Vec3f>(i, j)[1] = -1.0 * normals.at<Vec3f>(i, j)[2];
    }
    //-----------------test:天空变成朝下(屋顶),便于测试学习效果---------------------
    for (int i=0;i<rows;i++)
     for (int j=0;j<cols;j++){
         if(depth.at<float>(i, j)==16777215.0){
        res.at<Vec3f>(i, j)[0] = 0.0 ;
        res.at<Vec3f>(i, j)[1] = 0.0 ;
        res.at<Vec3f>(i, j)[2] = 1.0;
        }
    }
    //---------------------------------------
    for (int i=0;i<rows;i++)
     for (int j=0;j<cols;j++){
        res.at<Vec3f>(i, j)[0] += 1.0 ;
        res.at<Vec3f>(i, j)[2] += 1.0 ;
        res.at<Vec3f>(i, j)[1] += 1.0;
    }
    res =res * 127.5;
    res.convertTo(res, CV_8UC3);
    cvtColor(res, res, COLOR_BGR2RGB);
    return res;
}

Mat 
GetaSphere(){
   int x=512;
   int y=512;
   int r=250;
   int centerx=x/2;
   int centery=y/2;
   Mat res= Mat::zeros(x,y,CV_32FC1);
   for(int i=centerx-r;i<centerx+r ;i++)
     for(int j=centery-r;j<centery+r ;j++)
        {
            
             float distance2=(i-centerx)*(i-centerx)+(j-centery)*(j-centery);
             if (distance2>=r*r-5)
              res.at<float>(i,j)=0;
              else
              { float depth= sqrt(r*r-distance2);
               res.at<float>(i,j)=depth;}
             
        }
      return res;
}
void 
nearneigbor(Mat& src,int windowsize, Mat * ress){ 
    int cols =src.cols;
	int rows =src.rows; 
      src.convertTo(src,CV_16UC1);
      Mat res1 = Mat::zeros(src.size(),CV_16UC1);
      Mat res2 = Mat::zeros(src.size(),CV_8UC1);
      for(int i=0;i<rows;i++)
       for(int j=0;j<cols;j++){
             if(src.at<unsigned short>(i,j)!=0) {res1.at<unsigned short>(i,j)=src.at<unsigned short>(i,j);   continue;}
             int rdx=0,rdy=0;float depth=65535256;
             for(int m=0;m<windowsize;m++)
                 for(int n=0;n<windowsize;n++){
                       int idx=i-windowsize/2+m,idy=j-windowsize/2+n;
                       if(idx>=rows ||idy>=cols||idx<0||idy<0)continue;
                       if(src.at<unsigned short>(idx,idy)==0)continue;
                       float rd=sqrt(pow((m-windowsize/2),2)+pow((n-windowsize/2),2));
                       if( rd<depth ){
                             rdx=idx;rdy=idy;depth=rd;
                       }
            
                 }
             if(rdx*rdy!=0){
                    res1.at<unsigned short>(i,j)=src.at<unsigned short>(rdx,rdy);   
                   res2.at<uchar>(i,j)=(uchar)depth;    
             }
                      
       }
      ress[0]=res1; ress[1]=res2;
}
void 
os(Mat& src,int windowsize, Mat * ress){ 
    int cols =src.cols;
	int rows =src.rows; 
      src.convertTo(src,CV_16UC1);
      Mat res1 = Mat::zeros(src.size(),CV_8UC1);
      
      for(int i=0;i<rows;i++)
       for(int j=0;j<cols;j++){
             if(src.at<unsigned short>(i,j)==0) { continue;}
             int rdx=0,rdy=0;float depth=65535256;
             for(int m=0;m<windowsize;m++)
                 for(int n=0;n<windowsize;n++){
                       int idx=i-windowsize/2+m,idy=j-windowsize/2+n;
                       if(idx>=rows ||idy>=cols||idx<0||idy<0||(idx==i&&idy==j))continue;
                       if(src.at<unsigned short>(idx,idy)==0)continue;
                       float rd=sqrt(pow((m-windowsize/2),2)+pow((n-windowsize/2),2));
                       if( rd<depth ){
                             rdx=idx;rdy=idy;depth=rd;
                       }
            
                 }
             if(rdx*rdy!=0){
                   res1.at<uchar>(i,j)=(uchar)depth;    
             }
                      
       }
      ress[0]=res1; 
}
//###############################################################
//以前比较菜的求normal的方法，再此处备份，应该不会再用了。
int
search_neighbor_x(Mat &img,int i,int j,int circle_size ){
	 int cols =img.cols;
	 int rows =img.rows; 
	 int neighbor_x =-1;	
	 int idx = 0;
     for (int idx1=0; idx1<circle_size;idx1++){
         idx =idx1+ 1;
         if (i+idx >= rows)continue;
         else{
			 if(img.at<float>(i+idx , j )!= 0.0){
                neighbor_x =  i+ idx;
                break;
				 }
              else continue;
		 }
		}
	return neighbor_x;
}
int 
search_neighbor_y(Mat &img,int i,int j,int circle_size ){
	 int cols =img.cols;
	 int rows =img.rows; 
	 int neighbor_y = -1; 
     int idy = 0;
	for (int idy1=0; idy1<circle_size;idy1++){
         idy = idy1+1;
         if (j+idy >= cols)continue;
         else{
			 if(img.at<float>(i , j + idy)!= 0.0){
                neighbor_y =  j + idy;
                break;
				 }
              else continue;
		 }
		}
	return neighbor_y;
}
Mat 
sparce_depth2normal( Mat &input,float * paras,int circle_size){
	float f=paras[0];
	float cx=paras[1];
	float cy=paras[2];
    Mat normals = Mat::zeros(input.size(),CV_32FC3);
    Mat depth=input.clone();
	depth.convertTo(depth,CV_32FC1);
    
	int cols =depth.cols;
	int rows =depth.rows;
    
	for (int i=0;i< rows;i++)
    		for (int j=0;j< cols;j++){
             if (depth.at<float>(i,j)!=0.0){
				float x,y,z,x1,y1,z1,x2,y2,z2;
                Vec3f x3,y3;
				int neighborx = search_neighbor_x(depth,i,j,circle_size);
				int neighbory = search_neighbor_y(depth,i,j,circle_size);
                if(neighborx == -1||neighbory == -1)continue;
				else{
						// cout<<neighborx<<","<<neighbory<< endl;
						x = (j - cx) * depth.at<float>(i, j ) * 1.0 / f;
						y = (i - cy) * depth.at<float>(i, j )* 1.0 / f;
						z = depth.at<float>(i, j);
						x1 = (neighbory - cx) *depth.at<float>(i, neighbory) * 1.0 / f;
						y1 = (i - cy) * depth.at<float>(i,neighbory ) * 1.0 / f;
						z1 = depth.at<float>(i, neighbory);
						x2 = (j - cx) * depth.at<float>(neighborx, j ) * 1.0 / f;
						y2 = (i + 1 - cy) * depth.at<float>(neighborx, j ) * 1.0 / f;
						z2 = depth.at<float>(neighborx, j);
						x3 = Vec3f(x1-x, y1-y, z1-z);
						y3 = Vec3f(x2-x, y2-y, z2-z); 
						Vec3d d = x3.cross(y3);
						Vec3d n = normalize(d);
						normals.at<Vec3f>(i, j) = n;
					} 
			 }   
			}    
		   
	Mat res = Mat::zeros(depth.size(),CV_32FC3);
    for (int i=0;i<rows;i++)
     for (int j=0;j<cols;j++){
        res.at<Vec3f>(i, j)[0] = -1.0 * normals.at<Vec3f>(i, j)[0];
        res.at<Vec3f>(i, j)[2] = -1.0 * normals.at<Vec3f>(i, j)[1];
        res.at<Vec3f>(i, j)[1] = -1.0 * normals.at<Vec3f>(i, j)[2];
    }

    for (int i=0;i<rows;i++)
     for (int j=0;j<cols;j++){
		if(!(res.at<Vec3f>(i, j)[0]==0&&res.at<Vec3f>(i, j)[1]==0&&res.at<Vec3f>(i, j)[2]==0)){
			res.at<Vec3f>(i, j)[0] += 1.0 ;
			res.at<Vec3f>(i, j)[2] += 1.0 ;
			res.at<Vec3f>(i, j)[1] += 1.0;
		}
    }
    res =res * 127.5;
    res.convertTo(res, CV_8UC3);
    cvtColor(res, res, COLOR_BGR2RGB);
    return res;


}
Vec3b
Fill( int i, int j,Mat &src,int size){
	int w =src.cols;
	int h =src.rows;
	vector<int> idxs,idys,dsts;
	for (int ii = 0;ii<size;ii++)
	 for (int jj = 0;jj<size;jj++){
		  int idx= ii + i - int(size / 2);
		  int idy= jj + j - int(size / 2);
		  if( idx < h && idy< w && idx>= 0 &&idy >= 0) {
		    if(src.at<Vec3b>(idx,idy)[0]!=0 || src.at<Vec3b>(idx,idy)[1]!=0 ||src.at<Vec3b>(idx,idy)[2]!=0 ){
				int dst = abs(ii- int(size / 2))+abs(jj - int(size / 2));
				idxs.push_back(idx);
				idys.push_back(idy);
				dsts.push_back(dst);
			 }
			}
			else continue;
	 }
     int num =idxs.size();
	 if( num ==0 )
		 return Vec3b(0,0,0);
	 else{
		int resdix=0; 
		int min =size * 100;
        for(int g=0;g< num;g++){
			if (dsts[g]<min){
				min=dsts[g];
			    resdix = g;
			}	  
		}
       return src.at<Vec3b>(idxs[resdix],idys[resdix] );

	 }

	 




}
Mat 
Interpoletion(Mat & src,int size =8){
	int cols =src.cols;
	int rows =src.rows;
	Mat res = Mat::zeros(src.size(),CV_8UC3);
	for (int i=0;i< rows;i++)
    		for (int j=0;j< cols;j++){
				if(src.at<Vec3b>(i,j)[0]!=0 +src.at<Vec3b>(i,j)[1]!=0 +src.at<Vec3b>(i,j)[2]!=0)res.at<Vec3b>(i,j)=src.at<Vec3b>(i,j);
				else res.at<Vec3b>(i,j)=Fill(i,j,src,size);
			}
   return res;
}
void
get_pic_para(string name,float * fcxcy){ 
	 string data = name.substr(name.find("2011_"), 10);
	 if(data=="2011_09_26"){fcxcy[0]=721.5377;fcxcy[1]=596.5593;fcxcy[2]=149.854;}
	 else if(data=="2011_09_28"){fcxcy[0]=707.0493;fcxcy[1]=604.0814;fcxcy[2]=162.5066;}
	 else if(data=="2011_09_29"){fcxcy[0]=718.3351 ;fcxcy[1]=600.3891;fcxcy[2]=159.5122;}
	 else if(data=="2011_09_30"){fcxcy[0]=707.0912;fcxcy[1]=601.8873;fcxcy[2]=165.1104;}
	 else if(data=="2011_10_03"){fcxcy[0]=718.856 ;fcxcy[1]= 607.1928;fcxcy[2]= 161.2157;}
     else{cout<<"Dir data:"<<data<<" is unexpected. please check."<<endl;exit(1);}
	//  cout<<fcxcy[0]<<endl;
	//  
}
//通用的cmd参数获取模板
// static int 
// ParseArgs(int argc, char **argv){
//   argc--; argv++;
//   while (argc > 0) {
//     if ((*argv)[0] == '-') {
//       if (!strcmp(*argv, "-begin")) { argc--; argv++; begin= atof(*argv); }
// 	  else if(!strcmp(*argv, "-end")) { argc--; argv++; end= atof(*argv); }
// 	 else {
//             fprintf(stderr, "E: Invalid program argument: %s", *argv);
//             return 0;
//             } 
//  }
//  else {
//         fprintf(stderr, "E: Invalid program argument: %s", *argv);
//         return 0;
//       }
// 	 argv++; argc--; 
//  }
//  return 1;
// }