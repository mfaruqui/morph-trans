#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"

#include "utils.h"
#include "proj-to-output.h"

#include <unordered_map>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

void Decode(const Expression& encoded_word_vec,
            unordered_map<string, unsigned>& char_to_id,
            const vector<unsigned>& input_ids,
            ProjToOutput& proj_to_vocab, LSTMBuilder* decoder,
            LookupParameters* char_vecs, Expression& EPS,
            vector<unsigned>* pred_target_ids, ComputationGraph* cg);
