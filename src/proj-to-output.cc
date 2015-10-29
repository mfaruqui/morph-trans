#include "proj-to-output.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

ProjToOutput::ProjToOutput(const int& hidden_length, const int& output_length,
                           Model* m) {
  phidden_to_output = m->add_parameters({output_length, hidden_length});
  phidden_to_output_bias = m->add_parameters({output_length, 1});
}

void ProjToOutput::AddParamsToCG(ComputationGraph* cg) {
  hidden_to_output = parameter(*cg, phidden_to_output);
  hidden_to_output_bias = parameter(*cg, phidden_to_output_bias);
}

void ProjToOutput::ProjectToOutput(const Expression& hidden, Expression* out) const {
  *out = affine_transform({hidden_to_output_bias, hidden_to_output, hidden});
}
