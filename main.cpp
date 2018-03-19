/*
 * David Ruiz García. *
*/

/*Step 1
   Sea aplica una mascara, usando la derivada de sobel.

  Step 2
   Se calcula el histograma de gradientes.
   Devuelve una matriz unidimencional.
   histogram_gradient(src,cell_TAM,bin_TAM);

 Step 4 se genera el vector caracteristico.
*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "iostream"

using namespace std;
using namespace cv;

void histogram(Mat src, int histSize,float* range,Mat* dst, Mat* dst_img){
  const float* histRange = { range };
  bool uniform = true; bool accumulate = false;
  Mat hist;
  vector<Mat> aux;
  split(src,aux);
  calcHist( &aux[0], 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
  *dst = hist.clone();
  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );
  Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
  normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  for( int i = 1; i < histSize; i++ )
  {

      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
  }
  *dst_img = histImage.clone();
}

Mat patch(int radio){
  int l = 2* radio;
  Mat result = Mat::zeros(l,l,CV_8UC1 );
  for(int y=0; y<l; y++){
    for(int x=0; x<l; x++){
      double distance = sqrt(pow((x-radio),2)+pow((y-radio),2));
      if(distance<radio){
        float value = 1-distance/radio;
        result.at<uchar>(Point(x,y)) = value*255;
      }
    }
  }
  imwrite("patch.png",result);
  return result;
}

void get_gradient(Mat src,Mat* angle, Mat* magnitude){
  Mat grad;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_32F;
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  /// Gradient X
  //Scharr( src, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  /// Gradient Y
  //Scharr( src, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  //Calculates an absolute value of each matrix element
  Sobel( src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  /// Total Gradient (approximate)
  cartToPolar(grad_x, grad_y, *magnitude,*angle,1);
}

Mat hgd(Mat src, int nreg, int ndiv, int norm){
  Mat angle,magnitude;
  get_gradient(src, &angle,&magnitude);
  //normalize(magnitude,magnitude,1,0, NORM_MINMAX);
  int rad_t = src.size().width/2;
  Mat regions_img = src.clone();
  Mat patch_img = patch(rad_t);
  Mat ref_angle, ref_magnitude;
  get_gradient(patch_img, &ref_angle,&ref_magnitude);

  Mat div_angle, div_magnitude;
  absdiff(angle, ref_angle, div_angle);
  absdiff(magnitude, ref_magnitude, div_magnitude);

  /*float range[] = { 0,360} ;
  Mat hist;
  histogram(angle, ndiv,range,&hist);*/

  vector <Mat> regions;

  double rad_int = 0;
  double rad_ext;
  double ang_bin = 360/ndiv;

  /*Seleccion de pixeles por region*/
  Mat region_chanel = Mat::zeros(src.size(), CV_32FC1);
  for(double reg=1; reg<=nreg; reg++){
    rad_ext = sqrt(pow(rad_t,2)*(reg/nreg));
    int npixels = cvRound((CV_PI * pow(rad_ext,2))-(CV_PI * pow(rad_int,2)));
    int ind = 0;
    for(int r=rad_int; r<rad_ext; r++){
      for(int a=0; a<360; a++){
        int x = r*cos(a * CV_PI/180.0)+rad_t;
        int y = r*sin(a * CV_PI/180.0)+rad_t;
        region_chanel.at<float>(Point(x,y)) = reg;

      }
    }
    circle(regions_img, Point(rad_t,rad_t), rad_ext, 255);
    rad_int=rad_ext;
  }
  for(int a=0; a<ndiv; a++){
    double angl = int(a*ang_bin)*CV_PI/180.0;
    int x = rad_t * cos(angl) + rad_t;
    int y = rad_t * sin(angl) + rad_t;
    line(regions_img, Point(rad_t,rad_t), Point(x,y), 255);
  }
  imshow("Regions Bins", regions_img);

  /*Calculo de vector caracteristico*/
  Mat vec = Mat::zeros(0,1,CV_32FC1);
  for(double reg=1; reg<=nreg; reg++){
    int npixels=0;
    for(int x =0; x<region_chanel.size().width; x++){
      for(int y =0; y<region_chanel.size().height; y++){
        if(region_chanel.at<float>(Point(x,y))==reg)
          npixels++;
      }
    }
    Mat bin_mat = Mat::zeros(npixels,1,CV_32FC1);
    int np = 0;
    for(int x =0; x<region_chanel.size().width; x++){
      for(int y =0; y<region_chanel.size().height; y++){
        if(region_chanel.at<float>(Point(x,y))==reg){
          bin_mat.at<float>(np) = angle.at<float>(Point(x,y));
          np++;
        }
      }
    }
    float range[] = { 0,360} ;
    Mat hist, hist_img;
    histogram(bin_mat, ndiv,range,&hist,&hist_img);
    vconcat(vec,hist,vec);
    ostringstream name;
    name << "R" << reg;
    imshow(name.str(), hist_img);
  }

  /*Retorno de vector*/
  switch (norm) {
    case 1:
      normalize(vec,vec,1,0, NORM_MINMAX);
      return vec;
    case 2:
      normalize(vec,vec,1,0, NORM_L2);
      return vec;
    default:
      return vec;
  }
}


int main( int argc, char** argv ){
 Mat src;
 src = imread( argv[1] ,CV_LOAD_IMAGE_GRAYSCALE);
 if(argc < 5){ cout << "./main img.jpg cell_tam bin_tam" <<endl; return -1;}
 if( !src.data ){ cout << "Image not found" <<endl;return -1; }

 int nreg=stoi (argv[2],nullptr,10), ndiv=stoi (argv[3],nullptr,10);
 Mat vec = hgd(src,nreg,ndiv,stoi (argv[4],nullptr,10));

 cout << vec << endl << endl;
 cout << "S: " <<  vec.size().height << endl;
 waitKey(0);
 return 0;
 }
