#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv/cv.h" 
#include "opencv2/features2d/features2d.hpp" 
#include "printing.h"
#include "linfun.h"
#include "drawing.h"
#include "test.h"

using namespace std;
using namespace cv;
using namespace printing;
using namespace linfun;
using namespace drawing;

namespace findcorrespondance {

  static double pi = 3.141592653589793;

  struct Match {
	Point2f first_pt;
	Point2f second_pt;
	double distance;

	Match() {
	}

	Match(Point2f fp, Point2f sp, double dist) {
  		first_pt = fp;
  		second_pt = sp;
  		distance = dist;
  	}
  };


  struct Strip {
  	Mat strip;
  	Mat mask;
  	double angle;
  	Point2f shift;

  	Strip() {
  	}

  	Strip(Mat nstrip, Mat nmask, double nangle, 
  		Point2f nshift) {
  		strip = nstrip;
  		angle = nangle;
  		shift = nshift;
  		mask = nmask;
  	}
  };

  pair<double, double> get_angles(Point2f epipole, Point2f point,
  			  Mat epipolar_line, Mat hline) {
  	double angle1, angle2;
  	angle1 = get_angle(epipole, point);
  	if (point.y < epipole.y)
  		angle2 = get_angle(epipolar_line, hline);
  	else 
  		angle2 = -get_angle(epipolar_line, hline);
  	return pair<double, double>(angle1, angle2);
  }

  void init_mat_with_border(Mat *mat, int bordercols, 
  			  int borderrows) {

  	for(int i = 0; i < bordercols; i++) {
  		(*mat).col(i) = (*mat).col(i)*0;
  		(*mat).col((*mat).cols - 1 - i) = 
  			(*mat).col((*mat).cols - 1 - i)*0;
  	}
  	for(int j = 0; j < borderrows; j++) {
  		for (int i = bordercols ; i < 
  		     (*mat).cols - bordercols; i++) {
  			(*mat).at<byte>(j, i) = 0;
  			(*mat).at<byte>((*mat).rows - 1 - j, i) = 0;
  		}
  	}
  }

  pair<Rect, Rect> get_cut_bounds(double bandwidth, pair<Point2f,
  			        Point2f> points) {
  	Rect rect1 = Rect(points.first.x, 
  			  points.first.y - bandwidth/2,
                                  points.first.x, 
                                  points.first.y + bandwidth/2);
  	Rect rect2 = Rect(points.second.x, 
  			  points.second.y - bandwidth/2, 
  			  points.second.x, 
  			  points.second.y + bandwidth/2);
  	return pair<Rect, Rect>(rect1, rect2);
  }

  pair<Strip, Strip> cutstrips(Mat *left_image, 
  			     Mat *right_image, 
  			     Point2f epipole_left, 
   			     Point2f epipole_right, 
  			     Mat epipolar_line, 
  			     Point2f direction_point,
  			     double bandwidth) { 
  	Mat strip1, strip2, mask1, mask2;
  	Mat hline = (Mat_<float>(3, 1) << 0, 1, 0); 
  	pair<Size, Size> sizes = 
  		pair<Size, Size>((*left_image).size(),
  			         (*right_image).size());
  	pair<Point2f, Point2f> centers = 
      		pair<Point2f, Point2f>(
  		 Point2f(sizes.first.width/2,
  			 sizes.first.height/2), 
  		 Point2f(sizes.second.width/2, 
  			 sizes.second.height/2));
  	pair<double, double>   angles = 
  		get_angles(epipole_left, 
  			   direction_point, 
  			   epipolar_line, 
  			   hline);
  	pair<Point2f, Point2f> shifts = 
  		pair<Point2f, Point2f>(
  		 get_rotation_shift(sizes.first.width, 
  				    sizes.first.height, 
  				    angles.first), 
  		 get_rotation_shift(sizes.second.width, 
  				    sizes.second.height, 
  				    angles.second));

  	Point2f new_epipole_left = 
  		rotatePoint(epipole_left, 
  			    angles.first, 
  			    centers.first) +  shifts.first;
  	Point2f new_epipole_right = 
  		rotatePoint(epipole_right, 
  			    angles.second, 
  			    centers.second) + shifts.second;

  	pair<Rect, Rect> bounds = 
  		get_cut_bounds(bandwidth, 
  			       pair<Point2f, Point2f>(
  				new_epipole_left, 
  			   	new_epipole_right));
  	pair<Point2f, Point2f> rshifts = 
  		pair<Point2f, Point2f>(
  		 new_epipole_left - epipole_left 
  		  - Point2f(0, bounds.first.y),
  		 new_epipole_right - epipole_right 
  		  - Point2f(0, bounds.second.y));


  	mask1 = Mat::ones(sizes.first, CV_8U);
  	mask2 = Mat::ones(sizes.first, CV_8U);

  	init_mat_with_border(&mask1, 10, 10);
  	init_mat_with_border(&mask2, 10, 10);

  	strip1 = rotateImage(*left_image, 
  			     -angles.first, 
  			     true, false, 
  			     bounds.first.y, 
  			     bounds.first.height);
  	strip2 = rotateImage(*right_image, 
  			     -angles.second, 
  			     true, false, 
  			     bounds.second.y, 
  			     bounds.second.height);
  	mask1  = rotateImage(mask1, 
  			     -angles.first, 
  			     true, false, 
  			     bounds.first.y, 
  			     bounds.first.height);
  	mask2  = rotateImage(mask2, 
  			     -angles.second, 
  			     true, false, 
  			     bounds.second.y, 
   	 		     bounds.second.height);


  	return pair<Strip, Strip>(
  		Strip(strip1, 
  		      mask1, 
  		      angles.first,  
  		      rshifts.first), 
  		Strip(strip2, 
  		      mask2, 
  		      angles.second,
  		      rshifts.second));
  }

  vector<Match> get_corresp_points(Mat *left_image, 
  				 Mat *right_image, 
  				 int cornercount = 12) {
  	Mat firstImg, secondImg;
  	vector<Point2f> resf;
  	vector<Match> res_matches;

  	vector<uchar> status;
  	vector<float> err;

  	cvtColor(*left_image, firstImg, CV_RGB2GRAY); 
  	cvtColor(*right_image, secondImg, CV_RGB2GRAY);

  	int width, height;
  	width = min(firstImg.cols,secondImg.cols);
  	height = min(firstImg.rows,secondImg.rows);
  	int thresh = 240;

  	firstImg = firstImg(Rect(0, 0, width, height)); 
  	secondImg = secondImg(Rect(0, 0, width, height));
  	mask = mask(Rect(0, 0, width, height));

  	blockSize = min(height/2, width/2) - 2;
  	if (blockSize < 1 ) return res_matches;

  	GoodFeaturesToTrackDetector detector(
  			cornercount,
  			qualitylevel,
  			minDistance,
  			blockSize,
  			false, k);
  	vector<KeyPoint> keypoints_1, keypoints_2;
  	detector.detect(firstImg, keypoints_1, mask);
  	detector.detect(secondImg, keypoints_2, mask);

  	SurfDescriptorExtractor extractor; 
  	Mat descriptors1, descriptors2;

  	extractor.compute(firstImg, 
  			  keypoints_1, 
  			  descriptors1);
  	extractor.compute(secondImg, 
  			  keypoints_2, 
  			  descriptors2);

  	BruteForceMatcher<L2<float>> matcher;
  	vector<DMatch> matches; 
  	matcher.match(descriptors1, descriptors2, matches);
  	double distance;
  	bool contains1 = false, contains2 = false;
  	int index;

  	for(int k = 0; k < matches.size(); k++) {
  		distance = distanceEuclidian(
  		 keypoints_1[matches[k].queryIdx].pt,
  		 keypoints_2[matches[k].trainIdx].pt) ;
  		if (distance < 50) {
  			for (int l = 0; 
  			     l < res_matches.size(); l++) {
  			  if (keypoints_1
  			      [matches[k].queryIdx].pt 
  				== res_matches[l].first_pt) {
  			    contains1 = true; 
  			    break;
  			  }
  			  if (keypoints_2
  			       [matches[k].trainIdx].pt 
  				 == res_matches[l].second_pt) {
  			    contains2 = true; 
  			    break;
  			  }
  			}
  			if (!contains1 && ! contains2) {
  			  res_matches.push_back(
  			   Match(keypoints_1
  				 [matches[k].queryIdx].pt,
  				 keypoints_2
  				 [matches[k].trainIdx].pt,
  				 distance)); 
  			}
  			contains1 = false; 
  			contains2 = false;
  		}
  	}
  	return res_matches;
  }

  int get_quarter(Point2f point, Point2f origin) {
  	if (point.x > origin.x) {
  		if (point.y > origin.y) {
  			return 4;
  		} else {
  			return 2;
  		}
  	} else {
  		if (point.y > origin.y) {
  			return 3;
  		} else {
  			return 1;
  		}
  	}
  }


  double get_max_distance(int quarter, 
  	                Point2f point, Size size) {		
  	Point2f p1 = Point2f(0, 0);
  	Point2f p2 = Point2f(size.width, 0);
  	Point2f p3 = Point2f(size.width, size.height);
  	Point2f p4 = Point2f(0, size.height);
  	switch(quarter) {
  	case 1: 
  		return distanceEuclidian(point, p3);
  	case 2: 
  		return distanceEuclidian(point, p4);
  	case 3:
  		return distanceEuclidian(point, p2);
  	case 4: 
  		return distanceEuclidian(point, p1);
  	}
  }

  double bias_coef = 2000; // ~ bias = 0.026  error ~ 10

  double get_delta(double bias, double max_distance) {
  	return bias/bias_coef*max_distance;
  }

  double get_bandwidth(double h, double distance) {
  	return 2 * distance * tan(h/2);
  }

  bool compare_matches_orig_dist(Match m1, Match m2) {
  	return sqrt(m1.first_pt.x*m1.first_pt.x +
		    m1.first_pt.y*m1.first_pt.y) <
	       sqrt(m2.first_pt.x*m2.first_pt.x +
		    m2.first_pt.y*m2.first_pt.y);
  }

  void find_correspondance(Mat *left_image, 
  		         Mat *right_image, 
  		         const Mat fund,
  			 vector<Point2f> *firstPts,
   		  	 vector<Point2f> *secondPts, 
  			 int cornercount = 12) {
  	Size imSize1 = (*left_image).size();
  	Size imSize2 = (*right_image).size();
  	pair<Point2f, Point2f> epipoles = 
  			get_epipole_SVD(fund);
  	Point2f epipole_left = epipoles.first;
  	Point2f epipole_right = epipoles.second;
  	Size imSize  = imSize1;
  	Point2f imgCenter1 = Point2f(imSize1.width/2, 
  				     imSize1.height/2);
  	Point2f imgCenter2 = Point2f(imSize2.width/2, 
  				     imSize2.height/2);

  	int quarter = get_quarter(epipole_left, imgCenter1);
  	int position = get_position(epipole_left, imSize);

  	pair<double, double> min_max_angle = 
  			get_min_max_angle(position,
  					  epipole_left,
  					  imSize);
  	double max_distance = get_max_distance(quarter, 
  					       epipole_left,
  					       imSize);
  	double dist_to_epipole;
  	vector<Match> min_dist_area_matches;

  	double delta = 2;
  	double bandwidth = 10

  	pair<Strip, Strip> strips; Mat im1, im2;
  	Point2f p1,  p2;
  	vector<Match> matches, res_matches;
  	Point2f direction_point;
  	Mat epipole_line;
  	double anglemin = min(min_max_angle.first, 
  			      min_max_angle.second);
  	double anglemax = max(min_max_angle.first, 
  			      min_max_angle.second);
  	for(double a = anglemin; a < anglemax; a+= delta) { 
  		direction_point = 
  		 rotatePoint(Point2f(100, 0) + epipole_left,
  			     a, epipole_left);
  		epipole_line = 
  		 (right)?get_epipole_line(fund.t(), 
  				          direction_point):
  			 get_epipole_line(fund, 
  					  direction_point);
  		strips = cutstrips(left_image, 
  				   right_image, 
  				   epipole_left, 
  				   epipole_right, 
  				   epipole_line, 
  				   direction_point, 
  				   bandwidth);
  		matches = 
  		 get_correspondent_points(
  			&strips.first.strip, 
  			&strips.second.strip, 
  			4, 
  			(right)?strips.second.mask:
  			        strips.first.mask);

  		for (int i = 0; i < matches.size(); i++) {
  			p1 = rotatePoint(
  			 matches[i].first_pt 
  			  - strips.first.shift,
  			 -strips.first.angle, 
  			 epipole_left);
  			p2 = rotatePoint(
  			 matches[i].second_pt 
  			  - strips.second.shift, 
  			 -strips.second.angle, 
  			 epipole_right);
  			res_matches.push_back(
  			 Match(p1, 
  			       p2, 
  			       matches[i].distance));
  		}
  	}
  	sort(res_matches.begin(), 
  	     res_matches.end(), 
  	     compare_matches_orig_dist);


  	double shift, shift_dist;
  	double shift_eps = 50, shift_koef = 1.2;
  	double min_dist, max_dist;
  	int i = 0, j = 1;
  	Point2f dist_p = Point2f(minDistance, minDistance);
  	while(!(j >= res_matches.size()) && 
  	      !((*firstPts).size() > cornercount)) {
  		p1 = res_matches[i].first_pt + dist_p;
  		p2 = res_matches[j].first_pt;
  		if ((p1.x > p2.x) && (p1.y > p2.y)) {
  		  j++;
  		}
  		else {
  		  (*firstPts).push_back(
  			res_matches[i].first_pt);
  		  (*secondPts).push_back(
  			res_matches[i].second_pt);				
  		  i = j;
  		  j++;
  		}
  	}

  	(*firstPts).push_back(res_matches[i].first_pt);
  	(*secondPts).push_back(res_matches[i].second_pt);
  }
}
