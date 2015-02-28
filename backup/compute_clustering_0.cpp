/* Compute Clustering using K-Means algorithm
 * Input: [rows cols] [x y u v]+
 * Output: Gaussian Mixtures YML
 *
 * Rafael H Tibaes [ra.fael.nl]
 * IMAGO Research Group
 */

#include <cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;

#define NDIM 4
#define F_EPS 0.0000001

Mat getSampleFlow(const char* filename) {
  int rows, cols;
  FILE* input = fopen(filename, "r");
  fread(&rows, sizeof(int), 1, input);
  fread(&cols, sizeof(int), 1, input);

  Mat sample = Mat::zeros(rows * cols, NDIM, CV_32F);
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

int main(int argc, char** argv) {
  if (argc < 4) {
    printf("Use %s <K-Clusters> <output.cluster.yml> [input.flo]+\n", argv[0]);
    return -1;
  }

  int k = atoi(argv[1]);
  FileStorage output(argv[2], FileStorage::WRITE);

  Mat samples = Mat::zeros(0, NDIM, CV_32F);
  for (int frameid = 3; frameid < argc; ++frameid) {
    Mat s = getSampleFlow(argv[frameid]);
    samples.push_back(s);
  }

  // K-Means clustering over motion features
  Mat labels, centers;
  kmeans(samples, k, labels,
         TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
         3, KMEANS_PP_CENTERS, centers);

  // Split samples accordingly the cluster label
  vector<Mat> clusterSamples;
  for (int kid = 0; kid < k; ++kid) {
    clusterSamples.push_back(Mat::zeros(0, NDIM, CV_32F));
  }
  for (int sid = 0; sid < samples.rows; ++sid) {
    clusterSamples[labels.at<int>(sid)].push_back(samples.row(sid));
  }

  // Store statistical measures when displacement is greater than zero
  output << "GMM" << "[";
  for (int kid = 0; kid < k; ++kid) {
    Mat cov, mean;
    calcCovarMatrix(clusterSamples[kid], cov, mean,
                    CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32F);


    float mu = fabsf(mean.at<float>(2));
    float mv = fabsf(mean.at<float>(3));
    Size count = Size(clusterSamples[kid].size());
    if (mu + mv > F_EPS && fabsf(determinant(cov)) > F_EPS) {
      output << "{"
      << "mean" << mean
      << "cov" << cov
      << "count" << count
      << "}";
    }
  }
  output << "]";

  printf("Done clustering flow to file %s.\n", argv[2]);
  return 0;
}
