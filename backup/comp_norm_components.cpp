/* Compute Normalized Mixtures
 * Input: GMM YML
 * Output: Normalized GMM YML
 *
 * Rafael H Tibaes [ra.fael.nl]
 * IMAGO Research Group
 */

#include <cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>
#include <cstdio>

typedef struct Component {
  cv::Mat mean, cov;
  int flowvector_count;
  float weight;
} Component;

void importComponentsFromFile(const char* filename,
  std::vector<Component>& vcomp)
{
  cv::FileStorage input(filename, cv::FileStorage::READ);
  cv::FileNode gmm = input["GMM"];
  cv::Size tmp;
  for(cv::FileNodeIterator it = gmm.begin(); it < gmm.end(); ++it) {
    Component c;
    (*it)["mean"] >> c.mean;
    (*it)["cov"] >> c.cov;
    (*it)["count"] >> tmp;
    c.flowvector_count = tmp.height; // TODO - fix Size
    c.weight = 0.0f;
    vcomp.push_back(c);
  }
}

void saveMixture(std::vector<Component>& vcomp, cv::FileStorage& output) {
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

void normalize(std::vector<Component>& vcomp) {
  // spatial translation + weights
  cv::Mat sum_comp = cv::Mat::zeros(vcomp[0].mean.size(), vcomp[0].mean.type());
  int sum_flowvector = 0;
  for (int i = 0; i < vcomp.size(); ++i) {
    sum_comp += vcomp[i].mean;
    sum_flowvector += (int)vcomp[i].flowvector_count;
  }

  cv::Mat meancomp = sum_comp / vcomp.size();
  for (int i = 0; i < vcomp.size(); ++i) {
    vcomp[i].mean -= meancomp;
    vcomp[i].weight = (float)vcomp[i].flowvector_count / (float)sum_flowvector;
  }

  // Center normalization
  cv::Mat max = cv::Mat::zeros(vcomp[0].mean.size(), vcomp[0].mean.type());
  for (int i = 0; i < vcomp.size(); ++i) {
    for (int d = 0; d < max.cols; ++d) {
      float comp_d = fabsf(vcomp[i].mean.at<float>(0,d));
      if (comp_d > max.at<float>(0,d)) {
        max.at<float>(0,d) = comp_d;
      }
    }
  }
  for (int i = 0; i < vcomp.size(); ++i) {
    vcomp[i].mean /= max;
  }

  // Covariance normalization - TODO, conferir!
  for (int i = 0; i < vcomp.size(); ++i) {
    for (int a = 0; a < max.cols; ++a) {
      for (int b = 0; b < max.cols; ++b) {
        vcomp[i].cov.at<float>(a,b) *= max.at<float>(0,a) * max.at<float>(0,b);
      }
    }
  }
}


int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Use %s <output.mixture.yml> [<input.cluster.yml>]+\n", argv[0]);
    return -1;
  }

  std::vector<Component> vcomp;
  for(int fileid = 2; fileid < argc; ++fileid) {
    importComponentsFromFile(argv[fileid], vcomp);
  }

  if (vcomp.size() >= 1) normalize(vcomp);

  cv::FileStorage output(argv[1], cv::FileStorage::WRITE);
  saveMixture(vcomp, output);

  return 0;
}
