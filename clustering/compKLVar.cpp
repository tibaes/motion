/* Convert YML GMM to Mathematica GMM
 *
 * Input: GMM1 and GMM2 YML
 * Output: Divergence using KL Variational Approx.
 *
 * Rafael H Tibaes [ra.fael.nl]
 * IMAGO Research Group
 */

#include <cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <cmath>

#define DSize 4

typedef struct Component {
  cv::Mat mean, cov;
  double weight;
} Component;

std::vector<Component> importGMMFromFile(cv::FileStorage& input) {
  cv::FileNode gmm = input["GMM"];
  std::vector<Component> vcomp;
  for(cv::FileNodeIterator it = gmm.begin(); it < gmm.end(); ++it) {
    Component c;
    (*it)["mean"] >> c.mean;
    (*it)["cov"] >> c.cov;
    c.weight = (double)(*it)["weight"];
    vcomp.push_back(c);
  }
  return vcomp;
}

double klc(Component& c0, Component& c1) {
  cv::Mat diff = c0.mean - c1.mean;
  double a = log(cv::determinant(c1.cov) / cv::determinant(c0.cov));
  double b = cv::trace(c1.cov.inv() * c0.cov)[0] - (double)DSize;
  cv::Mat c = diff * c1.cov.inv() * diff.t();
  return (0.5 * (a + b + c.at<float>(0, 0)));
}

double kl2(int a, std::vector<Component>& compList1,
  std::vector<Component>& compList2)
{
  double sum = 0.0;
  for(int b = 0; b < compList2.size(); ++b) {
    sum += compList2[b].weight * exp(-1 * klc(compList1[a], compList2[b]));
  }
  return sum;
}

double kl1(int a, std::vector<Component>& compList1) {
  double sum = 0.0;
  for(int b = 0; b < compList1.size(); ++b) {
    sum += compList1[b].weight * exp(-1 * klc(compList1[a], compList1[b]));
  }
  return sum;
}

double klv(std::vector<Component>& compList1, std::vector<Component>& compList2) {
  double sum = 0.0;
  for (int a = 0; a < compList1.size(); ++a) {
    sum += compList1[a].weight *
      log(kl1(a, compList1) / kl2(a, compList1, compList2));
  }
  return fabs(sum);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Use %s <input1.yml> <input1.yml>\n", argv[0]);
    exit(-1);
  }

  cv::FileStorage input1(argv[1], cv::FileStorage::READ);
  cv::FileStorage input2(argv[2], cv::FileStorage::READ);

  std::vector<Component> compList1 = importGMMFromFile(input1);
  std::vector<Component> compList2 = importGMMFromFile(input2);
  double divergence = klv(compList1, compList2);

  printf("%lf\n", divergence);

  return 0;
}
