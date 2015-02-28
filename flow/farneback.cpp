#include <cv.hpp>
#include <opencv2/highgui/highgui.hpp>

void writeFlow(cv::Mat& flow, FILE* output)
{
  int ch = flow.channels();
  for (int y = 0; y < flow.rows; ++y)
  {
    float *pf = flow.ptr<float>(y);
    for (int x = 0; x < flow.cols; ++x)
    {
      float u = pf[x * ch + 0];
      float v = pf[x * ch + 1];
      fwrite(&x, sizeof(int), 1, output);
      fwrite(&y, sizeof(int), 1, output);
      fwrite(&u, sizeof(float), 1, output);
      fwrite(&v, sizeof(float), 1, output);
    }
  }
}

int main(int argc, char** argv)
{
  if (argc < 3) {
    printf("Use %s <output.flow> <images...>\n", argv[0]);
    return -1;
  }

  cv::Mat frame[2], flow;
  int frameId0 = 0;
  int frameId1 = 1;

  frame[frameId0] = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

  FILE *output = fopen(argv[1], "w");
  int size[2] = {frame[frameId0].rows, frame[frameId0].cols};
  fwrite(size, sizeof(int), 2, output);

  for (int idf = 3; idf < argc; ++idf) {
    frame[frameId1] = cv::imread(argv[idf], CV_LOAD_IMAGE_GRAYSCALE);
    printf("computing flow at frame %s...\n", argv[idf]);
    cv::calcOpticalFlowFarneback(frame[frameId0], frame[frameId1], flow,
        0.5, 2, 3, 3, 5, 1.2, cv::OPTFLOW_FARNEBACK_GAUSSIAN);
    writeFlow(flow, output);
    frameId0 = (frameId0 + 1) % 2;
    frameId1 = (frameId1 + 1) % 2;
  }
  printf("\n");

  return 0;
}
