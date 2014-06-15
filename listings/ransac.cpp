#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h" 
#include "boost/lexical_cast.hpp"
#include "test.h"
#include "printing.h"
#include "linfun.h"

using namespace std;
using namespace cv;
using namespace test;
using namespace printing;
using namespace linfun;

namespace ransac {

  static bool TEST = true;
  static int best_iteration = 0;

  vector<int> sample(int n, int lowest, int highest) 
  {
  	vector<int> random_integers(n);
  	int r;
  	bool exists = false;
  	int range=(highest-lowest)+1;
  	for(int i= 0; i < min(n, range);) {
  		exists = false;
  		r = lowest+int(range*rand()/(RAND_MAX + 1.0));
  		for(int j=0; j < i ; j++)
  			if (random_integers[j] == r)
  				exists = true;
  		if (!exists) {
  			random_integers[i] = r;
  			i++;
  		}
  	}
  	return random_integers;
  }

  static std::vector<bool> flag_choosen(MAXINT32);
  static std::vector<bool> flag_new(MAXINT32);

  bool belongs(std::vector<vector<int>> *already_choosen, 
  		vector<int> *sample) 
  {
  	vector<int> choosen_sample;
  	int max = 0;
  	for (int i = 0; i < (*already_choosen).size(); i++) {
  		choosen_sample = (*already_choosen)[i];		
  		for (int j = 0; j < (*sample).size(); j++) {
  			flag_new[(*sample)[j]] = true;
  			flag_choosen[choosen_sample[j]] = true;
  			if (max < (*sample)[j])
  				max = (*sample)[j];
  			if (max < choosen_sample[j])
  				max = choosen_sample[j];
  		}

  		for(int j = 0; j < max; j++) {
  			if(flag_new[j] != flag_choosen[j])
  				break;
  			if(j == max - 1)
  				return true;
  		}

  		for(int k = 0; k < max; k++) {
  			flag_choosen[k] = false;
  			flag_new[k] = false;
  		}
  	}
  	return false;
  }

  vector<int> nonrecurring_sample(int n, int lowest, 
    int highest, std::vector<vector<int>> *already_choosen) 
  {
  	int level = 0;
  	int r;
  	vector<int> sample1(0);
  	int i = 0;
  	int iters = fact(highest - lowest)/fact(n)*
  		    fact(n - highest + lowest) - 
  		    (*already_choosen).size();
  	do {
  		sample1 = sample(n, lowest, highest);
  		i++;
  	} while((belongs(already_choosen, &sample1)) &&
  		       	(i < iters));
  	return sample1;
  }

  pair<Mat, double> ransac(vector<Point2f> points1, 
  		vector<Point2f> points2, int maxiter) 
  {
  	if ((points1.size() < 8) || (points2.size() < 8)) 
  		return Mat::zeros(0,0, CV_32FC1);
  	if (points1.size() != points2.size())
  		return Mat::zeros(0,0, CV_32FC1);

  	std::vector<int> samples;
  	std::vector<std::vector<int>> already_choosen;
  	vector<int> outlinerst;
  	int n = 8;		
  	Mat matrix(n, 8, CV_32FC1);
  	Mat solution(n, 1, CV_32FC1);
  	Mat fund(3, 3, CV_32FC1);
  	Mat epipolar_line(3, 1, CV_32FC1);
  	Mat best_fund(3, 3, CV_32FC1);
  	Mat T1(3, 3, CV_32FC1), T2(3, 3, CV_32FC1);
  	float mindist = (float)MAXINT32;
  	float dist = (float)MAXINT32;
  	float bias;

  	if(normalization) {
  		points1 = normalizePts(points1, &T1);
  		points2 = normalizePts(points2, &T2);
  		threshold = T1.at<float>(0,0)*threshold;
  	}
  	for(int i = 0; i < maxiter; i++) {
  		bias = 0;
  		samples = nonrecurring_sample(n, 0, 
  			   points1.size() - 1, 
  			   &already_choosen);
  		if (samples.size() == 0) break;
  		already_choosen.push_back(samples);


  		if (i == 1) {
  			samples.clear();
  			for(int k = 0; k < 8; k++)
  				samples.push_back(k);
  		}
  		{
  		 float u; 
  		 float u_;
  		 float v;v
  		 float v_;
  		 sort(samples.begin(), samples.end());
  		 for(int j = 0; j < n; j++) {
  			float* mi = matrix.ptr<float>(j); 
  			u  = (points1[samples[j]]).x;
  			v  = (points1[samples[j]]).y;
  			u_ = (points2[samples[j]]).x;
  			v_ = (points2[samples[j]]).y;
  			mi[0] = u * u_;
  			mi[1] = u * v_;
  			mi[2] = u;
  			mi[3] = v * u_;
  			mi[4] = v * v_;
  			mi[5] = v;
  			mi[6] = u_;
  			mi[7] = v_;
  		 }
  	}
  	 Mat reverted = Mat::ones(n, 1, CV_32FC1)*(-1); 
  	 solution = (matrix.inv())* reverted;

  	 fund.at<float>(0,0) = solution.at<float>(0,0);
  	 fund.at<float>(0,1) = solution.at<float>(1,0);
  	 fund.at<float>(0,2) = solution.at<float>(2,0);
  	 fund.at<float>(1,0) = solution.at<float>(3,0);
  	 fund.at<float>(1,1) = solution.at<float>(4,0);
  	 fund.at<float>(1,2) = solution.at<float>(5,0);
  	 fund.at<float>(2,0) = solution.at<float>(6,0);
  	 fund.at<float>(2,1) = solution.at<float>(7,0);
   	 fund.at<float>(2,2) = 1.0;
  
  	 Mat s = Mat::zeros(3, 3, CV_32FC1); 
  	 Mat u;
  	 Mat w;
  	 Mat vt;
  	 cv::SVD::compute(fund, w, u, vt); 
  	 for (int j = 0; j < 3; j++){
  	 	s.at<float>(j,j) = w.at<float>(j,0);
  	 }
  	 s.at<float>(2,2) = 0; 
        	 fund = u*s*vt; 

  	 //find outliners
  		outlinerst.clear();
   	 for(int j = 0; j < points1.size(); j++) {
  		epipolar_line = get_epipole_line(fund.t(),
  			       	points1[j]); 
  		dist = distance(points2[j], epipolar_line); 
  		if(dist < threshold) {
  			bias += dist;
  		} else {
  			bias += threshold;				
  			outlinerst.push_back(j);
  		}
  	}
  	if (mindist > bias) {
  	 mindist = bias;
  	 fund.copyTo(best_fund); 
  	 (*outliners) = vector<int> (outlinerst.begin(), 
  			 outlinerst.end()); 
  	 best_iteration = i;
  	}
  	}
  	if (normalization) {
  		best_fund = T2.t()*best_fund*T1;
  	} 

  	return pair<Mat, double>(best_fund, mindist);
  }
}
