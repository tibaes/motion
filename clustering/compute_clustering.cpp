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

typedef unsigned int uint;

typedef struct Component {
  cv::Mat mean, cov;
  float weight;
} Component;

cv::Mat getSampleFlow(const char* filename)
{
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

void saveMixtures(std::vector<Component>& vcomp, cv::FileStorage& output)
{
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

void convFlowFeatures(cv::Mat samples, cv::Mat& conv_samples,
  cv::Mat& conv_samples_inv)
{
  conv_samples = cv::Mat::zeros(samples.size(), CV_32F);
  conv_samples_inv = cv::Mat::zeros(samples.size(), CV_32F);

  // Angular reference

  cv::Mat mean, cov;
  calcCovarMatrix(samples, cov, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32F);

  cv::Mat cov_xy = cv::Mat::zeros(2, 2, CV_32F);
  cov_xy.at<float>(0,0) = cov.at<float>(0,0);
  cov_xy.at<float>(0,1) = cov.at<float>(0,1);
  cov_xy.at<float>(1,0) = cov.at<float>(1,0);
  cov_xy.at<float>(1,1) = cov.at<float>(1,1);

  cv::Mat eigenvalues_xy, eigenvectors_xy;
  cv::eigen(cov_xy, true, eigenvalues_xy, eigenvectors_xy);

  float angle_ref = atan2f(eigenvectors_xy.at<float>(1,0),
    eigenvectors_xy.at<float>(0,0)) / M_PI;

  // Convert angle_xy

  float mean_x = mean.at<float>(0,0);
  float mean_y = mean.at<float>(0,1);

  for (uint sid = 0; sid < samples.rows; ++sid) {
    float x = samples.at<float>(sid,0);
    float y = samples.at<float>(sid,1);
    float angle_xy = atan2f(y - mean_y, x - mean_x) / M_PI;
    conv_samples.at<float>(sid,0) = angle_xy - angle_ref;
    conv_samples_inv.at<float>(sid,0) = angle_xy + angle_ref;
  }

  // Convert intensity_xy

  std::vector<float> dist_xy(samples.rows);

  float max_xy = 0.0f;
  for (uint sid = 0; sid < samples.rows; ++sid) {
    float x = samples.at<float>(sid,0);
    float y = samples.at<float>(sid,1);
    float dx = x - mean_x;
    float dy = y - mean_y;
    float dist = sqrtf(dx*dx + dy*dy);
    dist_xy[sid] = dist;
    if (dist > max_xy) max_xy = dist;
  }

  for (uint sid = 0; sid < samples.rows; ++sid) {
    float norm_dist = dist_xy[sid] / max_xy;
    conv_samples.at<float>(sid,1) = norm_dist;
    conv_samples_inv.at<float>(sid,1) = norm_dist;
  }

  // convert angle_uv

  for (uint sid = 0; sid < samples.rows; ++sid) {
    float u = samples.at<float>(sid,2);
    float v = samples.at<float>(sid,3);
    float angle_uv = atan2f(v, u) / M_PI;
    conv_samples.at<float>(sid,2) = angle_uv - angle_ref;
    conv_samples_inv.at<float>(sid,2) = angle_uv + angle_ref;
  }

  // convert intensity_uv

  std::vector<float> dist_uv(samples.rows);

  float max_uv = 0.0f;
  for (uint sid = 0; sid < samples.rows; ++sid) {
    float u = samples.at<float>(sid,2);
    float v = samples.at<float>(sid,3);
    float dist = sqrtf(u*u + v*v);
    dist_uv[sid] = dist;
    if (dist > max_uv) max_uv = dist;
  }

  for (uint sid = 0; sid < samples.rows; ++sid) {
    float norm_dist = dist_uv[sid] / max_uv;
    conv_samples.at<float>(sid,3) = norm_dist;
    conv_samples_inv.at<float>(sid,3) = norm_dist;
  }
}

std::vector<Component> components(int k, cv::Mat samples)
{
  // K-Means clustering over motion features
  cv::Mat labels, centers;
  kmeans(samples, k, labels, cv::TermCriteria(cv::TermCriteria::EPS
    + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

  // Split samples according the cluster label

  std::vector<cv::Mat> clusterSamples;
  for (int kid = 0; kid < k; ++kid) {
    clusterSamples.push_back(cv::Mat::zeros(0, NDIM, CV_32F));
  }
  for (uint sid = 0; sid < samples.rows; ++sid) {
    clusterSamples[labels.at<int>(sid)].push_back(samples.row(sid));
  }

  // Statistical measures, i.e. motion components

  std::vector<Component> vcomp;
  for (int kid = 0; kid < k; ++kid) {
    // TODO some component left behind modifies the weight of the others
    Component c;
    c.weight = (float)clusterSamples[kid].rows / (float)samples.rows;
    calcCovarMatrix(clusterSamples[kid], c.cov, c.mean,
      CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32F);

    // float ixy = fabsf(c.mean.at<float>(1));
    // float iuv = fabsf(c.mean.at<float>(3));
    if (/* ixy + iuv > 0.001f && */ fabsf(determinant(c.cov)) > 0.001f) {
      vcomp.push_back(c);
    }
  }

  return vcomp;
}

int main(int argc, char** argv)
{
  if (argc < 4) {
    printf("Use %s <K-Clusters> <output.yml> <output_inv.yml> [input.flo]+\n",
      argv[0]);
    return -1;
  }

  int k = atoi(argv[1]);
  cv::FileStorage output(argv[2], cv::FileStorage::WRITE);
  cv::FileStorage output_inv(argv[3], cv::FileStorage::WRITE);

  // read flow vectors
  cv::Mat samples = cv::Mat::zeros(0, NDIM, CV_32F);
  for (int frameid = 4; frameid < argc; ++frameid) {
    cv::Mat s = getSampleFlow(argv[frameid]);
    samples.push_back(s);
  }

  // convert flow vectors (invariance)
  cv::Mat conv_samples, conv_samples_inv;
  convFlowFeatures(samples, conv_samples, conv_samples_inv);

  // compute models
  std::vector<Component> vcomp = components(k, conv_samples);
  saveMixtures(vcomp, output);

  // compute inverted models
  std::vector<Component> vcomp_inv = components(k, conv_samples_inv);
  saveMixtures(vcomp_inv, output_inv);

  return 0;
}
