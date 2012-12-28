//
//  Created by Rebecca Bever and Anurag Kamasamudram on 11/8/12.
//
//

/*
 --------------------------------------------------------------------------------------------------------------
 Sources by:
 
 [1]Hand Gesture Detection
 Done By
 Mohafiz Raz.M.A
 Febin Jose
 C.Sanjay Arvind
 
 [2]Randah
 http://stackoverflow.com/questions/5746170/how-can-i-compute-number-of-finger-opencv-contour-detection
 
 [3]Aniket Tatipamula
 Btech electronics
 V.J.T.I Mumbai
 http://anikettatipamula.blogspot.com/2012/02/hand-gesture-using-opencv.html
 
 [4]jemmyjone
 http://opencv-users.1802565.n2.nabble.com/how-to-remove-the-background-by-use-background-substraction-td5606545.html
 
 [5]OpenCV Tutorials:
 camshiftdemo.cpp
 calc_HistDemo.cpp
 
 Steps:
 background removal
 convexities
 hulls
 find difference of movement from one frame to the next
 haar cascade for fist
 --------------------------------------------------------------------------------------------------------------
 */

#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/cxmisc.h>
#include <opencv2/highgui/highgui.hpp>
#include <ctype.h>

#include <iostream>
#include <ApplicationServices/ApplicationServices.h>

using namespace cv;
using namespace std;

void detect(IplImage* img_8uc1,IplImage* img_8uc3);
void detectFist(IplImage *img);
//void  histogram(IplImage* img_src);
//void  histogram(IplImage* img_src,Rect selection);

struct staticBoundBox {
    int x;
    int y;
    int width;
    int height;
};

struct prevBoundBox {
    int x;
    int y;
    int width;
    int height;
};

struct fistBoundBox {
    int x;
    int y;
};

staticBoundBox staticBox;
prevBoundBox prevBox;
fistBoundBox fistBox;

bool firstLoop = true;

//slide booleans
bool nextSlide = false;
bool prevSlide = false;
bool stopSlide = false;

int fistCounter = 0;

//haarcascade global variables
CvMemStorage* fistStorage = 0;
CvHaarClassifierCascade* cascade = 0;
const char* cascade_name = "fist.xml";

int main(int argc, char** argv)
{
    fistStorage = cvCreateMemStorage(0);
    CvCapture* capture = cvCaptureFromCAM(0);
    
    cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
	if( !cascade ){
        fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
        return -1;
	}
    
    int nframes=0;
    bool pause = false;
    fistBox.x = 0;
    fistBox.y = 0;
    
    IplImage* img_yuv  = cvCreateImage( cvSize(640,480), IPL_DEPTH_8U, 1 );
    IplImage* img_src  = cvCreateImage( cvGetSize(img_yuv), IPL_DEPTH_8U, 1 );
    IplImage* img_fore = cvCreateImage( cvGetSize(img_yuv), IPL_DEPTH_8U, 1 );
    
    IplImage* before;
    IplImage* after;
    IplImage* img_bw;
    
    while(true)
    {
        while(!pause)
        {
            img_src = cvQueryFrame(capture);
            ++nframes;
            if (nframes == 1)
            {
                img_yuv = cvCloneImage(img_src);
                cout<<"background Captured\n";
                pause = true;
                break;
            }
            cout<<nframes<<endl;
        }
        
        if( nframes == 1 && img_src )
        {
            before = cvCreateImage(cvGetSize(img_yuv), IPL_DEPTH_8U, 1 );
            cvCvtColor(img_yuv,before,6);
            cvFlip(before,before,4);
        }
        
        after = cvCreateImage(cvGetSize(img_yuv), IPL_DEPTH_8U, 1 );
        img_bw =cvCreateImage(cvGetSize(img_yuv), IPL_DEPTH_8U, 1 );
        
        if(img_src)
        {
            img_fore = cvQueryFrame(capture);
            cvFlip(img_fore,img_fore,4);
            cvCvtColor(img_fore, after, 6);
            cvAbsDiff(before,after,img_bw);
            cvThreshold(img_bw,img_bw,30,255,CV_THRESH_BINARY);
            cvErode(img_bw,img_bw,0, 3);
            cvSmooth(img_bw,img_bw,2,3,3,0,0);
            cvDilate(img_bw,img_bw,0,1);
        }
        
        //detect the hand
        detect(img_bw,img_src);
        
        if (waitKey(30) == 'q')
            break;
    }
    
    cvReleaseCapture( &capture );
    return 0;
}

//8uc1 is BW image with hand as white And 8uc3 is the original source image
void  detect(IplImage* img_8uc1, IplImage* img_8uc3)
{
    cvNamedWindow( "result", CV_WINDOW_AUTOSIZE );
    
    //cvThreshold( img_8uc1, img_edge, 128, 255, CV_THRESH_BINARY );
    CvMemStorage* storage = cvCreateMemStorage();
    CvSeq* first_contour = NULL;
    CvSeq* maxitem=NULL;
    double area=0,areamax=0;
    int maxn=0;
    
    //find contours
    int Nc = cvFindContours(
                            img_8uc1,
                            storage,
                            &first_contour,
                            sizeof(CvContour),
                            CV_RETR_LIST // Try all four values and see what happens
                            );
    int n=0;
    //printf( "Total Contours Detected: %d\n", Nc );
    
    //accounts for when there are no contours detected
    if (Nc == 0)
    {
        if ((prevBox.width != 0) || (prevBox.height != 0))
        {
            if (((staticBox.width) < prevBox.width+100) && ((staticBox.height) > prevBox.height+100))
            {
                nextSlide = true;
                prevBox.width = 0;
                prevBox.height = 0;
            }
            
            //compare previous values to current within 100 pixels
            if (((staticBox.width) > prevBox.width+100) && ((staticBox.height) < prevBox.height+100))
            {
                prevSlide = true;
                prevBox.width = 0;
                prevBox.height = 0;
            }
        }
        
        firstLoop = true;
        //frameCount = 0;
    }
    else if(Nc>0)//accounts for when contours are detected
    {
        CvSeq* c;
        for( c=first_contour; c!=NULL; c=c->h_next )
        {
            //cvCvtColor( img_8uc1, img_8uc3, CV_GRAY2BGR );
            
            area=cvContourArea(c, CV_WHOLE_SEQ );
            
            if(area>areamax)
            {areamax=area;
                maxitem=c;
                maxn=n;
            }
            
            n++;
        }
        
        if(areamax>5000)
        {
            //find convex hulls
            CvPoint pt0;
            
            CvMemStorage* storage1 = cvCreateMemStorage();
            CvSeq* ptseq = cvCreateSeq( CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvContour),
                                       sizeof(CvPoint), storage1 );
            CvSeq* hull;
            
            for(int i = 0; i < maxitem->total; i++ )
            {   CvPoint* p = CV_GET_SEQ_ELEM( CvPoint, maxitem, i );
                pt0.x = p->x;
                pt0.y = p->y;
                cvSeqPush( ptseq, &pt0 );
            }
            hull = cvConvexHull2( ptseq, 0, CV_CLOCKWISE, 0 );
            
            //create bounding box
            CvRect rect = cvBoundingRect( ptseq, 1 );
            //CvBox2D box = cvMinAreaRect2( ptseq, storage1 );
            
            cvRectangle( img_8uc3, cvPoint(rect.x, rect.y + rect.height), cvPoint(rect.x + rect.width, rect.y), CV_RGB(255, 255, 0), 1, 8, 0 );
            
            //handle motion vectors
            if(firstLoop)
            {
                //allows for you to get your hand in desired position
                sleep(1);
                staticBox.x = rect.x;
                staticBox.y = rect.y;
                staticBox.width = rect.width;
                staticBox.height = rect.height;
                firstLoop = false;
            }
            else
            {
                prevBox.x = rect.x;
                prevBox.y = rect.y;
                prevBox.width = rect.width;
                prevBox.height = rect.height;
            }
            
            //histogram
            //histogram(img_8uc3,rect);
            
            //create sequence array for convex hull points
            int hullcount = hull->total;
            
            pt0 = **CV_GET_SEQ_ELEM( CvPoint*, hull, hullcount - 1 );
            
            for(int i = 0; i < hullcount; i++ )
            {
                CvPoint pt = **CV_GET_SEQ_ELEM( CvPoint*, hull, i );
                cvLine( img_8uc3, pt0, pt, CV_RGB( 0, 255, 0 ), 1, CV_AA, 0 );
                pt0 = pt;
            }
            
            CvPoint* hullPoints;
            hullPoints = ( CvPoint *)malloc((hull->total)*sizeof(CvSeq));
            cvCvtSeqToArray(hull,hullPoints);
            
            //detect all convexity defects
            CvSeq* defect = cvConvexityDefects( ptseq, hull, storage1 );
            int numofdefect =defect->total;
            CvConvexityDefect* defectArray;
            defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*numofdefect);
            cvCvtSeqToArray(defect,defectArray);
            
            for(;defect;defect = defect->h_next)
            {
                int nomdef = defect->total;
                if(nomdef == 0)
                    continue;
                defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*nomdef);
                cvCvtSeqToArray (defect, defectArray, CV_WHOLE_SEQ);
                for(int i = 0; i<nomdef; i++)
                {
                    //cvCircle( img_8uc3, *(defectArray[i].end), 5, CV_RGB(255,0,0), -1, 8,0);
                    //cvCircle( img_8uc3, *(defectArray[i].start), 5, CV_RGB(0,0,255), -1, 8,0);
                    cvCircle( img_8uc3, *(defectArray[i].depth_point), 5, CV_RGB(0,255,255), -1, 8,0);
                }
                
                free(defectArray);
            }
            
            cvReleaseMemStorage( &storage );
            cvReleaseMemStorage( &storage1 );
            //return 0;
            
        }
        
        detectFist(img_8uc3);
        cvShowImage( "result", img_8uc3 );
    }
    
    //function that handles next slide set nextSlide = false
    CGEventRef a;
    if(nextSlide == true)
    {
        cout<<"NEXT SLIDE"<<endl;
        a = CGEventCreateKeyboardEvent(NULL, (CGKeyCode)124, true);
        CGEventPost(kCGHIDEventTap, a);
        nextSlide = false;
    }
    else if(prevSlide == true)
    {
        cout<<"PREV SLIDE"<<endl;
        a = CGEventCreateKeyboardEvent(NULL, (CGKeyCode)123, true);
        CGEventPost(kCGHIDEventTap, a);
        prevSlide = false;
    }
    else if(stopSlide == true)
    {
        cout<<"STOP SLIDE"<<endl;
        a = CGEventCreateKeyboardEvent(NULL, (CGKeyCode)53, true);
        CGEventPost(kCGHIDEventTap, a);
        exit(0);
    }

}

void detectFist(IplImage *img)
{
	double scale = 1.1;
	IplImage* temp = cvCreateImage( cvSize(img->width/scale,img->height/scale), 8, 3 );
	CvPoint pt1, pt2;
	int i;
    
    CvPoint origin;
    origin.x = 0;
    origin.y = 0;
    
	cvClearMemStorage( fistStorage );
	if(cascade){
		CvSeq* faces = cvHaarDetectObjects(
                                           img,
                                           cascade,
                                           fistStorage,
                                           scale, 2, CV_HAAR_DO_CANNY_PRUNING,
                                           cvSize(24, 24) );
        

		for( i = 0; i < (faces ? faces->total : 0); i++ )
		{
			CvRect* r = (CvRect*)cvGetSeqElem( faces, i );
			pt1.x = r->x*scale;
			pt2.x = (r->x+r->width)*scale;
			pt1.y = r->y*scale;
			pt2.y = (r->y+r->height)*scale;
            
            if(r->width > 150)
            {
                if(abs(pt1.x - fistBox.x) <= 15)
                {
                    cvRectangle( img, pt1, pt2, CV_RGB(200, 0, 0), 1, 8, 0 );
                    stopSlide = true;
                }
            }
            
		}
        
        fistBox.x = pt1.x;
        fistBox.y = pt1.y;
	}
	cvShowImage("result", img);
	cvReleaseImage( &temp );
}


/*void  histogram(IplImage* img_src, Rect selection)
 {
 Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
 Mat imgMat(img_src);
 
 //Transform the source image to HSV
 cvtColor(imgMat, hsv, CV_BGR2HSV);
 
 int vmin = 10, vmax = 256, smin = 30;
 int hsize = 16;
 float hranges[] = {0,180};
 const float* phranges = hranges;
 int _vmin = vmin, _vmax = vmax;
 
 inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
 Scalar(180, 256, MAX(_vmin, _vmax)), mask);
 
 /// Use only the Hue value
 int ch[] = {0, 0};
 hue.create(hsv.size(), hsv.depth());
 mixChannels(&hsv, 1, &hue, 1, ch, 1);
 
 //Region of interest
 Mat roi(hue, selection), maskroi(mask, selection);
 // Get the Histogram and normalize it
 calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
 normalize(hist, hist, 0, 255, CV_MINMAX);
 
 histimg = Scalar::all(0);
 int binW = histimg.cols / hsize;
 Mat buf(1, hsize, CV_8UC3);
 
 for( int i = 0; i < hsize; i++ )
 buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
 cvtColor(buf, buf, CV_HSV2BGR);
 
 //1-D Hue histogram of the image
 for( int i = 0; i < hsize; i++ )
 {
 int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
 rectangle( histimg, Point(i*binW,histimg.rows),
 Point((i+1)*binW,histimg.rows - val),
 Scalar(buf.at<Vec3b>(i)), -1, 8 );
 }
 
 namedWindow("Histogram", CV_WINDOW_AUTOSIZE );
 imshow( "Histogram", histimg );
 
 }*/