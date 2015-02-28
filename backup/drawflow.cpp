#include <cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>

#define SearchSize 4

static int width = 640;
static int height = 480;

static inline float cvtVec2Hue(int x, int y) {
	return (90.0 * atan2f((float)y, (float)x) / M_PI);
}

static inline float cvtVec2Value(int x, int y, float norm) {
	return (255.0 * (sqrtf((float)(y * y + x * x)) / norm));
}

static void drawFlow(cv::Mat& samples, const char *window, int idWrite)
{
	const float norm = sqrtf(2 * SearchSize * SearchSize);
	cv::Mat hsv = cv::Mat::zeros(height, width, CV_8UC3);
	for (int i = 0; i < (int)samples.rows; ++ i)
	{
		float *pf = samples.ptr<float>(i);
		int x = pf[0];
		int y = pf[1];
		int u = pf[2];
		int v = pf[3];

		uchar *ph = hsv.ptr<uchar>(y);
		ph[x * 3] = (uchar)cvtVec2Hue(u, v);
		ph[x * 3 + 1] = 100;
		ph[x * 3 + 2] = (uchar)cvtVec2Value(u, v, norm);
	}

	cv::Mat rgb;
	cv::cvtColor(hsv, rgb, CV_HSV2BGR);
	cv::imshow(window, rgb);
	cv::waitKey(0);
	
	if (idWrite > -1) {
		char fn[1024];
		sprintf(fn,"%s-%d.jpg",window,idWrite);
		cv::imwrite(fn,rgb);
	}
}

cv::Mat getSampleFlow(const char* filename)
{
	int rows, cols;
	FILE* input = fopen(filename, "r");
	fread(&rows, sizeof(int), 1, input);
	fread(&cols, sizeof(int), 1, input);

	width = cols;
	height = rows;

	cv::Mat sample = cv::Mat::zeros(rows * cols, 4, CV_32F);
	int x, y, sid = 0;
	float u, v;
	while (!feof(input))
	{
		fread(&x, sizeof(int), 1, input);
		fread(&y, sizeof(int), 1, input);
		fread(&u, sizeof(float), 1, input);
		fread(&v, sizeof(float), 1, input);

		float *ps = sample.ptr<float>(sid++);
		ps[0] = (float)x;
		ps[1] = (float)y;
		ps[2] = (float)(u > 0.001)? u: 0.0;
		ps[3] = (float)(v > 0.001)? v: 0.0;
	}

	return sample;
}

int main(int argc, char** argv)
{
	if (argc != 2) {
		printf("Use %s <input.flo>\n", argv[0]);
		exit(0);
	}

	cv::Mat flow = getSampleFlow(argv[1]);
	drawFlow(flow, "flow", -1);
}
