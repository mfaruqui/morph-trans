#include "morph-trans.h"
#include "ensemble.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

template<typename T>
Ensemble<T>::Ensemble(const int& char_length, const int& hidden_length,
                      const int& vocab_length, const int& layers,
                      const int& ensmb_size, Model *m) {
  ensmb = ensmb_size;
  for (unsigned i = 0; i < ensmb; ++i) {
    models.push_back(T(char_length, hidden_length, vocab_length, layers, m));
  }
}

template<typename T> void
Ensemble<T>::AddParamsToCG(ComputationGraph* cg) {
  for (unsigned i = 0; i < ensmb; ++i) {
    models[i].AddParamsToCG(cg);
  }
}

template<typename T> float
Ensemble<T>::Train(const vector<unsigned>& inputs,
                   const vector<unsigned>& outputs,
                   AdadeltaTrainer* ada_gd) {
  float loss = 0;
  for (unsigned i = 0; i < ensmb; ++i) {
    loss += models[i].Train(inputs, outputs, ada_gd);
  }
  return loss / ensmb;
}

template class Ensemble<MorphTrans>;
