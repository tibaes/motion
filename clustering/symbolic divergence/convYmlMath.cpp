/* Convert YML GMM to Mathematica GMM
 *
 * Input: GMM YML
 * Output: GMM Mathematica TXT
 *
 * Rafael H Tibaes [ra.fael.nl]
 * IMAGO Research Group
 */

#include <cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;

typedef struct Component {
  Mat mean, cov;
} Component;

vector<Component> importGMMFromFile(FileStorage& input) {
  FileNode gmm = input["GMM"];
  vector<Component> vcomp;
  for(FileNodeIterator it = gmm.begin(); it < gmm.end(); ++it) {
    Component c;
    (*it)["mean"] >> c.mean;
    (*it)["cov"] >> c.cov;
    vcomp.push_back(c);
  }
  return vcomp;
}

// Thread format: {m0, m1, m2, m3, cov00, cov01, ... , cov32, cov33}
void writeGmm4Mathematica(vector<Component>& gmm, FILE* output) {
  for (int i = 0; i < gmm.size(); ++i) {
    for (int col = 0; col < gmm[i].mean.cols; ++col) { // Write means
      fprintf(output, "%.20f ", gmm[i].mean.at<float>(0, col));
    }
    for (int lin = 0; lin < gmm[i].cov.rows; ++lin) { // Write cov matrix
      for (int col = 0; col < gmm[i].cov.cols; ++col) {
        fprintf(output, "%.20f ", gmm[i].cov.at<float>(lin, col));
      }
    }
    fprintf(output, "\n");
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("Use %s <input.yml> <output.txt>\n", argv[0]);
    exit(-1);
  }

  FileStorage input(argv[1], FileStorage::READ);
  FILE* output = fopen(argv[2], "w");

  vector<Component> gmm = importGMMFromFile(input);
  writeGmm4Mathematica(gmm, output);

  return 0;
}
