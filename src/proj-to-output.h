#ifndef PROJ_TO_OUTPUT_H_
#define PROJ_TO_OUTPUT_H_

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

class ProjToOutput {
 public:
  Expression hidden_to_output, hidden_to_output_bias;
  Parameters *phidden_to_output, *phidden_to_output_bias;

  ProjToOutput() {}

  ProjToOutput(const int& hidden_length, const int& output_length, Model *m);

  void AddParamsToCG(ComputationGraph* cg);

  void ProjectToOutput(const Expression& hidden, Expression* out) const;
};

#endif
