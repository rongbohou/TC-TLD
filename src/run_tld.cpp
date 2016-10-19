#include <opencv2/opencv.hpp>
#include <tld_utils.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <TLD.h>
#include <stdio.h>


using namespace cv;
using namespace std;
//Global variables
Rect box;

bool drawing_box = false;
bool MouseSle=true;
bool gotBB = false;
bool tl = true;
bool rep = false;
bool fromfile=false;
string video;
int CropImageCount=0;
char saveName[256];

string FilePath = "J:\\指尖跟踪项目相关\\指尖书写视频\\测试视频\\", SaveName = "I:\\指尖跟踪项目相关\\TLD(evaluate)\\TLD\\",
VideoName = "yuxun\\2";

//bounding box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param){
	switch( event ){
	case CV_EVENT_MOUSEMOVE:
		if (drawing_box){
			box.width = x-box.x;
			box.height = y-box.y;
		}
		break;
	case CV_EVENT_LBUTTONDOWN:
		drawing_box = true;
		box = Rect( x, y, 0, 0 );
		break;
	case CV_EVENT_LBUTTONUP:
		drawing_box = false;
		if( box.width < 0 ){
			box.x += box.width;
			box.width *= -1;
		}
		if( box.height < 0 ){
			box.y += box.height;
			box.height *= -1;
		}
		gotBB = true;
		break;
	}
}


int main(int argc, char * argv[])
{

	VideoCapture capture;
	capture.open(0);

	FilePath = FilePath + VideoName + ".avi";
	//capture.open("/home/bobo/code/TLD/data/2.avi");

	cout<<capture.get(CV_CAP_PROP_FRAME_COUNT)<<"frames"<<endl;
        capture.set(CV_CAP_PROP_FPS,30);
	FileStorage fs;
	fs.open("/home/bobo/code/TLD/parameters.yml", FileStorage::READ);
	//Init camera
	if (!capture.isOpened())
	{
		cout << "capture device failed to open!" << endl;
		return 1;
	}
	cvNamedWindow("TLD",CV_WINDOW_AUTOSIZE);
	//TLD framework
	TLD tld;
	//Read parameters file
	tld.read(fs.getFirstTopLevelNode());
	Point pt1, pt2,location;//two points
	vector<Point>LocationCenter;//location vector 
	vector<Rect>rect_vector;//box vector
	Mat frame;
	Mat last_gray;
	Mat first;
	if (fromfile)
	{
		capture >> frame;
		flip(frame,frame, 1); // flip by y axis
		cvtColor(frame, last_gray, CV_RGB2GRAY);
		frame.copyTo(first);
	}
	else
	{
		capture.set(CV_CAP_PROP_FRAME_WIDTH,640);
		capture.set(CV_CAP_PROP_FRAME_HEIGHT,480);
	}
 
	///Initialization
	if(MouseSle)
	{
		//Register mouse callback to draw the Fist box
		Rect fistbox;
		cvSetMouseCallback( "TLD", mouseHandler, NULL );

		GETBOUNDINGBOX:
		while(!gotBB)
		{
			if (!fromfile)
			{
				capture >> frame;
				flip(frame,frame, 1); // flip by y axis
			}
			else
			first.copyTo(frame);
			cvtColor(frame, last_gray, CV_RGB2GRAY);
			drawBox(frame,box,Scalar(0, 0, 255),2);
			imshow("TLD", frame);
			if (cvWaitKey(33) == 'q')
				return 0;
		}
		//Remove callback

		fistbox=box;
		cvSetMouseCallback( "TLD", NULL, NULL );
		/*
		//Register mouse callback to click the fingers' center
		cvSetMouseCallback( "TLD", mouseHandler, NULL );
		gotBB=false;
		while(!gotBB)
		{  
			if (!fromfile)
			{
				capture >> frame;
				flip(frame,frame, 1); // flip by y axis
			}
			else
			first.copyTo(frame);

			drawBox(frame,box,Scalar(0, 0, 255),2);
			imshow("TLD", frame);
			if (cvWaitKey(33) == 'q')
				return 0;
		}
		box.width=sqrt((double)fistbox.area()/18);
		box.height=box.width;
		box.x=box.x-0.5*box.width;
		box.y=box.y-0.5*box.height;
		drawBox(frame,box,Scalar(255, 0, 0),2);
		//Remove callback
		cvSetMouseCallback( "TLD", NULL, NULL );*/
		MouseSle=false;
	}
	else
	{
		box=Rect(400,75,47,32);
		rect_vector.push_back(box);
		drawBox(frame,box,Scalar(255, 0, 0),2);
	}
	printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);
	imshow("TLD",frame);
	/*while(1)
	{
		if (cvWaitKey(10) == 13)
			break;
	}*/
	//Output file
	FILE  *bb_file = fopen("bounding_boxes.txt","w");
	//TLD initialization
	tld.init(last_gray,box,bb_file);

	///Run-time
	Mat current_gray;
	BoundingBox pbox;
 
	vector<double>CostTime;
	double totaltime=0;
	Mat image = Mat::zeros( frame.rows,frame.cols, CV_8UC3);
	int i=1;
	vector<Point2f> pts1;
	vector<Point2f> pts2;
	bool status=true;
	int frames = 1;
	int detections = 1;
REPEAT:
	while(capture.read(frame))
	{ 
		flip(frame,frame, 1); // flip by y axis
		cvtColor(frame, current_gray, CV_RGB2GRAY);
		double t = (double)cvGetTickCount();
		//Process Frame
		tld.processFrame(last_gray,current_gray,pts1,pts2,pbox,status,tl,bb_file);
		//Draw Points
		if (tld.center!=Point(0,0))
		{
			//drawPoints(frame,pts1);
			//drawPoints(frame,pts2,Scalar(0,255,0));
			drawBox(frame,pbox,Scalar(255, 0, 0),2);
			detections++;
			location=tld.center;
			rect_vector.push_back(pbox);
		}
		else
		{
			location=Point(0,0);
			rect_vector.push_back(Rect(0,0,0,0));
			drawBox(frame,tld.cxtRegion_current,Scalar(0,255,0),3);
		}

		//Display
		stringstream buf;
		buf << frames;
		string num = buf.str();
		//putText(frame, num, Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);
		t = (double)cvGetTickCount() - t;
		t=t / ((double)cvGetTickFrequency()*1000.);
		cout << "tracking cost time: " <<t <<"ms" << endl;
		totaltime=t+totaltime;
		CostTime.push_back(t);
		if (location!=Point(0,0))
		{
			if(i%2==1)
			{
			 pt1=location;
		 
			}
			if(i%2==0)
			{
			 pt2=location;
			}
			if(i>1)
			{
			line(image,pt1,pt2,Scalar(0, 0, 255),3,8,0);
			}
			i++;
		}
		//swap points and images
		Mat image1=image+frame;
		imshow("TLD",image1);
		swap(last_gray,current_gray);
		pts1.clear();
		pts2.clear();
		frames++;
		printf("Detection rate: %d/%d\n",detections,frames);
		if (cvWaitKey(10) == 27)
		  break;
	}
	if (rep){
		rep = false;
		tl = false;
		//capture.set(CV_CAP_PROP_POS_AVI_RATIO,0);
		capture.release();
		capture.open(video);
		goto REPEAT;
	}
 	cout<<"错误跟踪"<<tld.n<<"帧"<<endl;	
	cout<<"Average FPS is:"<<frames/(totaltime/1000)<<endl;
	return 0;
}
