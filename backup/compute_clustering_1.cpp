/* Compute Clustering using K-Means algorithm and then normalize components
 * Input: [rows cols] [x y u v]+
 * Output: Gaussian Mixtures YML
 *
 * Rafael H Tibaes [ra.fael.nl]
 * IMAGO Research Group
 */

#include <cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <cmath>

#define NDIM 4

typedef struct Component {
  cv::Mat mean, cov;
  float weight;
} Component;

cv::Mat getSampleFlow(const char* filename) {
  int rows, cols;
  FILE* input = fopen(filename, "r");
  fread(&rows, sizeof(int), 1, input);
  fread(&cols, sizeof(int), 1, input);

  cv::Mat sample = cv::Mat::zeros(rows * cols, NDIM, CV_32F);
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

void saveMixtures(std::vector<Component>& vcomp, cv::FileStorage& output) {
  output << "GMM" << "[";
  for (int i = 0; i < vcomp.size(); ++i) {
    output << "{"
    << "mean" << vcomp[i].mean
    << "cov" << vcomp[i].cov
    << "weight" << vcomp[i].weight
    << "}";
  }
  output << "]";
}

// -PI .. PI
float angle(float x, float y) {
  float a = atanf(y / fabsf(x));
  if (x < 0.0f) {
    if (y < 0.0f) {
      a = -M_PI - a;
    }
    else {
      a = M_PI - a;
    }
  }
  return a;
}

std::vector<Component> components(int k, cv::Mat samples, cv::Mat labels)
{
  // (x,y,u,v) -> normalized(Angle_xy, Intensity_xy, Angle_uv, Intensity_uv)

  cv::Mat meanflow = cv::Mat::zeros(1, NDIM, CV_32F);
  for (int sid = 0; sid < samples.rows; ++sid) {
    meanflow += samples.row(sid);
  }
  meanflow /= (float)samples.rows;

  float max_xy = 0.0f;
  float max_uv = 0.0f;
  for (int sid = 0; sid < samples.rows; ++sid) {
    cv::Mat diff = samples.row(sid) - meanflow;
    float diff_x = diff.at<float>(0,0);
    float diff_y = diff.at<float>(0,1);
    float diff_u = diff.at<float>(0,2);
    float diff_v = diff.at<float>(0,3);
    float Angle_xy = angle(diff_x, diff_y);
    float Angle_uv = angle(diff_u, diff_v);
    float Intensity_xy = sqrtf(diff_x * diff_x + diff_y * diff_y);
    float Intensity_uv = sqrtf(diff_u * diff_u + diff_v * diff_v);
    if (Intensity_xy > max_xy) {
      max_xy = Intensity_xy;
    }
    if (Intensity_uv > max_uv) {
      max_uv = Intensity_uv;
    }
    samples.at<float>(sid,0) = Angle_xy / M_PI;
    samples.at<float>(sid,1) = Intensity_xy;
    samples.at<float>(sid,2) = Angle_uv / M_PI;
    samples.at<float>(sid,3) = Intensity_uv;
  }

  for (int sid = 0; sid < samples.rows; ++sid) {
    samples.at<float>(sid,1) /= max_xy;
    samples.at<float>(sid,3) /= max_uv;
  }

  // Split samples accordingly the cluster label

  std::vector<cv::Mat> clusterSamples;
  for (int kid = 0; kid < k; ++kid) {
    clusterSamples.push_back(cv::Mat::zeros(0, NDIM, CV_32F));
  }
  for (int sid = 0; sid < samples.rows; ++sid) {
    clusterSamples[labels.at<int>(sid)].push_back(samples.row(sid));
  }

  // Statistical measures, i.e. motion components

  std::vector<Component> vcomp;
  for (int kid = 0; kid < k; ++kid) {
    Component c;
    c.weight = (float)clusterSamples[kid].rows / (float)samples.rows;
    calcCovarMatrix(clusterSamples[kid], c.cov, c.mean,
      CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32F);

    float ixy = fabsf(c.mean.at<float>(1));
    float iuv = fabsf(c.mean.at<float>(3));
    if (ixy + iuv > 0.001f && fabsf(determinant(c.cov)) > 0.001) {
      vcomp.push_back(c);
    }
  }

  return vcomp;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    printf("Use %s <K-Clusters> <output.cluster.yml> [input.flo]+\n", argv[0]);
    return -1;
  }

  int k = atoi(argv[1]);
  cv::FileStorage output(argv[2], cv::FileStorage::WRITE);

  cv::Mat samples = cv::Mat::zeros(0, NDIM, CV_32F);
  for (int frameid = 3; frameid < argc; ++frameid) {
    cv::Mat s = getSampleFlow(argv[frameid]);
    samples.push_back(s);
  }

  // K-Means clustering over motion features
  cv::Mat labels, centers;
  kmeans(samples, k, labels, cv::TermCriteria(cv::TermCriteria::EPS
    + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

  // Normalized motion components
  std::vector<Component> vcomp = components(k, samples, labels);

  saveMixtures(vcomp, output);
  return 0;
}
