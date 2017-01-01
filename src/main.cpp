#include "iostream"
#include "opencv2\imgproc.hpp"
#include "highgui.h"
#include "opencv2\opencv.hpp"
#include "time.h"
using namespace std;
using namespace cv;

//求三个数的中位数
int mid(int a, int b, int c)
{
	int max = a;
	int min = b;
	if ((a <=b&&b<=c)|| (c <= b&&b <= a))
	{
		return b;
	}
	else if ((b <= a&&a <= c) || (c <= a&&a <= b))
	{
		return a;
	}
	else
	{
		return c;
	}
}
Mat MyDilate(Mat src, int KernelSize=3,int step=1)
{
	Mat dst(src.rows,src.cols,src.type()); 
	src.copyTo(dst);
	//半个核的大小，用以计算滤波中心
	int KernelHalfSize = (KernelSize - 1) / 2;
	if (0 == KernelHalfSize)
	{
		return dst;
	}
	int channels = src.channels();
	//膨胀是求最大值的滤波
	for (int c = 0; c < channels; c++)
	{
		for (int col = 0; col < src.cols; col+=step)
		{
			for (int row = 0; row < src.rows; row+=step)
			{
				int max = 0;
				for (int kc = col-KernelHalfSize; kc < col+KernelHalfSize; kc ++)
				{
					for (int kr = row- KernelHalfSize; kr < row+KernelHalfSize; kr ++)
					{
						if (src.at<Vec3b>(mid(kr, 0, src.rows-1), mid(kc, 0, src.cols-1))[c] > max)
						{
							//如果是彩色图
							if (3 == channels)
							{
								//这里求三个数的中位数是为了解决滤波边界的问题，
								//如果超出界限的就选边界那一点的值，因为是求最大值所以某个点多算几次也没有关系
								max = src.at<Vec3b>(mid(kr, 0, src.rows-1), mid(kc, 0, src.cols-1))[c];
							}
							//如果是灰度图
							else if (1 == channels)
							{
								max = src.at<uchar>(mid(kr, 0, src.rows-1), mid(kc, 0, src.cols-1));
							}
						}
					}
				}
				if (3 == channels)
				{
					dst.at<Vec3b>(row,col)[c] = max;
				}
				//如果是灰度图
				else if (1 == channels)
				{
					dst.at<uchar>(row,col) = max;
				}
			}
		}
	}
	return dst;
}
Mat MyErode(Mat src, int KernelSize = 3, int step = 1)
{
	Mat dst(src.rows, src.cols, src.type());
	src.copyTo(dst);
	//半个核的大小，用以计算滤波中心
	int KernelHalfSize = (KernelSize - 1) / 2;
	if (0 == KernelHalfSize)
	{
		return dst;
	}
	int channels = src.channels();
	//腐蚀是求最小值的滤波
	for (int c = 0; c < channels; c++)
	{
		for (int col = 0; col < src.cols; col += step)
		{
			for (int row = 0; row < src.rows; row += step)
			{
				int min = 255;
				for (int kc = col - KernelHalfSize; kc < col + KernelHalfSize; kc++)
				{
					for (int kr = row - KernelHalfSize; kr < row + KernelHalfSize; kr++)
					{
						if (src.at<Vec3b>(mid(kr, 0, src.rows - 1), mid(kc, 0, src.cols - 1))[c] < min)
						{
							//如果是彩色图
							if (3 == channels)
							{
								//这里求三个数的中位数是为了解决滤波边界的问题，
								//如果超出界限的就选边界那一点的值，因为是求最大值所以某个点多算几次也没有关系
								min = src.at<Vec3b>(mid(kr, 0, src.rows - 1), mid(kc, 0, src.cols - 1))[c];
							}
							//如果是灰度图
							else if (1 == channels)
							{
								min = src.at<uchar>(mid(kr, 0, src.rows - 1), mid(kc, 0, src.cols - 1));
							}
						}
					}
				}
				if (3 == channels)
				{
					dst.at<Vec3b>(row, col)[c] = min;
				}
				//如果是灰度图
				else if (1 == channels)
				{
					dst.at<uchar>(row, col) = min;
				}
			}
		}
	}
	return dst;
}
//开操作先腐蚀后膨胀
Mat MyOpen(Mat img, int KernelSize = 3, int step = 1)
{
	Mat ErodeImg = MyErode(img, KernelSize, step);
	Mat DilateImg = MyDilate(ErodeImg, KernelSize, step);
	return DilateImg;
}
//闭操作先膨胀后腐蚀
Mat MyClose(Mat img, int KernelSize = 3, int step = 1)
{
	Mat DilateImg = MyDilate(img, KernelSize, step);
	Mat ErodeImg = MyErode(DilateImg, KernelSize, step);
	return ErodeImg;
}
//background:背景图
//num需要分类的个数
bool kmeans(Mat &background, int num, int x_array[], int y_array[])
{
	//选择前num个点作为分类起点
	int *start_x = new int[num];
	int *start_y = new int[num];
	int class_array[20];//记录每个点的类别
	int *dist = new int[num];//记录每一类的距离
	for (int i = 0; i < num; i++)
	{
		start_x[i] = x_array[i];
		start_y[i] = y_array[i];
		class_array[i] = i;
		dist[i] = 0;
	}
	for (int i = 0; i < 20; i++)
	{
		int dist_min = -1;//寻找最近距离类别
		for (int j = 0; j < num; j++)
		{
			if (dist_min >= 0 && dist_min > (x_array[i] - start_x[j])*(x_array[i] - start_x[j]))
			{
				class_array[i] = j;
				dist_min = (x_array[i] - start_x[j])*(x_array[i] - start_x[j]);
			}
		}
		dist[class_array[i]] += dist_min;
	}
}
bool drawCircle(Mat &background, int x_array[],int y_array[])
{
	
	srand((int)time(NULL));
	//随机产生坐标
	for (int i = 0; i < 20; i++)
	{
		x_array[i] = rand() % background.cols;
		y_array[i] = rand() % background.rows;
		//防止有画重的圆圈
		for (int j = 0; j < i; j++)
		{
			while (x_array[i] == x_array[j] && y_array[i] == y_array[j])
			{
				x_array[i] = rand() % background.cols;
				y_array[i] = rand() % background.rows;
			}
		}
		//绘制圆形
		circle(background, Point(x_array[i], y_array[i]), 5, CV_RGB(255, 255, 255),-1);
	}
	return true;

}
void main()
{
	Mat background = Mat::zeros(Size(640, 320), 1);
	int x_array[20], y_array[20];
	drawCircle(background, x_array, y_array);
	imshow("background",background);
	waitKey(0);
}
