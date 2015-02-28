#include <cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <cstring>

void writeFlow(cv::Mat& flow, FILE* output)
{
  for (int y = 0; y < flow.rows; ++y)
  {
    float *pf = flow.ptr<float>(y);
    for (int x = 0; x < flow.cols; ++x)
    {
      float u = pf[x * 2 + 0];
      float v = pf[x * 2 + 1];
      fwrite(&x, sizeof(int), 1, output);
      fwrite(&y, sizeof(int), 1, output);
      fwrite(&u, sizeof(float), 1, output);
      fwrite(&v, sizeof(float), 1, output);
    }
  }
}

cv::Mat segmentFlowFromDepth(cv::Mat& flow, cv::Mat& depth)
{
  const uint16_t mask_playerid = 0x07;
  // const uint16_t mask_depthdata = 0xf8;
  cv::Mat res = cv::Mat::zeros(flow.size(), CV_32FC2);

  for (int y = 0; y < flow.rows; ++y)
  {
    float *pf = flow.ptr<float>(y);
    uint16_t *pd = depth.ptr<uint16_t>(y);
    float *pr = res.ptr<float>(y);

    for (int x = 0; x < flow.cols; ++x)
    {
      uint16_t playerid = pd[x] & mask_playerid;
      // uint16_t depthdata = (pd[x] & mask_depthdata) >> 3;
      if (playerid) {
        memcpy(pr + x * 2, pf + x * 2, 2 * sizeof(float));
      }
    }
  }

  return res;
}

int main(int argc, char** argv)
{
  if (argc != 5) {
    printf("Use %s <output.flow> <src> <dst> <depth>\n", argv[0]);
    return -1;
  }

  cv::Mat src, dst, depth, flow;
  src = cv::imread(argv[2], cv::IMREAD_COLOR);
  dst = cv::imread(argv[3], cv::IMREAD_COLOR);
  depth = cv::imread(argv[4], cv::IMREAD_ANYDEPTH); // CV_16U

  FILE *output = fopen(argv[1], "w");
  int size[2] = {src.rows, src.cols};
  fwrite(size, sizeof(int), 2, output);

  int layers = 3;
  int blockSize = 2;
  int maxFlow = 4;
  cv::calcOpticalFlowSF(src, dst, flow, layers, blockSize, maxFlow);

  cv::Mat s_flow = segmentFlowFromDepth(flow, depth);
  writeFlow(s_flow, output);

  return 0;
}
