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

Mat calc_histogram(vector<double> angle, vector<double> magnitude, int nbin){
  Mat bin = Mat::zeros( nbin,1, CV_32FC1 );
  int ibin = 0;
  int tam_bin = 360/nbin;
  double max = magnitude.at(0);
  for(int i=0; i<magnitude.size(); i++){
    if(max < magnitude.at(i))
      max = magnitude.at(i);
  }
  for(int b=0; b<nbin; b++){
    int a = tam_bin*b;
    for(int i=0; i<angle.size(); i++){
      double angle_value = angle.at(i);
      if(angle_value>a && angle_value < a+tam_bin/2){
        bin.at<float>(ibin)+= magnitude.at(i)/max;
      }
      else if(ibin==nbin)
            bin.at<float>(ibin+1)+= magnitude.at(i)/max;
          else
            bin.at<float>(0)+= magnitude.at(i)/max;
    }
    ibin++;
  }
  return bin;
}

Mat hgd(Mat src, int nreg, int ndiv, int norm){
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
  /*Step 2*/
  Mat angle,magnitude;
  cartToPolar(grad_x, grad_y, magnitude,angle,1);

  normalize(magnitude,magnitude,1,0, NORM_MINMAX);


  Mat vec = Mat::zeros( nreg*ndiv,1, CV_32FC1 );
  int nfeature=0;
  Mat regions = src.clone();
  double ang_bin = 360/ndiv;
  double rad_t = src.size().width/2;
    double rad_int = 0;
    double rad_ext;
    for(double reg=1; reg<=nreg; reg++){
      vector<double> bin_magnitude;
      vector<double> bin_angle;
      rad_ext = sqrt(pow(rad_t,2)*(reg/nreg));
      circle(regions, Point(rad_t,rad_t), rad_ext, 255);
      for(int mag=rad_int; mag<rad_ext; mag++){
        for(int ang=0; ang<360; ang++){
          int x = mag * cos(ang) + rad_t;
          int y = mag * sin(ang) + rad_t;
          double ref_mag = sqrt(pow((x-rad_t),2)+pow((y-rad_t),2));
          double ref_ang = ref_mag * cos(ref_mag);
          double div_mag = magnitude.at<float>(Point(x,y));
          double div_angle = angle.at<float>(Point(x,y));
          bin_magnitude.push_back(div_mag);
          bin_angle.push_back(div_angle);
        }
      }
      Mat bin = calc_histogram(bin_angle,bin_magnitude,ndiv);
      Mat bin_normal;
      normalize(bin,bin_normal,rad_t,0, NORM_L2);
      Mat hist = Mat::zeros(regions.size(), CV_32FC1 );
      for(int f=0; f<ndiv; f++){
        float value = bin_normal.at<float>(f);
        float an = int(f*ang_bin)*CV_PI/180.0;
        float mg = rad_t * value ;
        int zx = mg * cos(an) + rad_t;
        int zy = mg * sin(an) + rad_t;
        line(hist, Point(rad_t,rad_t), Point(zx,zy), 255);
        value = bin.at<float>(f);
        vec.at<float>(nfeature) = value;
        nfeature++;
      }
      ostringstream name;
      name << "R" << reg;
      imshow(name.str(), hist);
      rad_int=rad_ext;
    }
  for(int a=0; a<ndiv; a++){
    double angl = int(a*ang_bin)*CV_PI/180.0;
    int x = rad_t * cos(angl) + rad_t;
    int y = rad_t * sin(angl) + rad_t;
    line(regions, Point(rad_t,rad_t), Point(x,y), 255);
  }
  imshow("Regions Bins", regions);
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
