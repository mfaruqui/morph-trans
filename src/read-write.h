#ifndef READ_WRITE_H_
#define READ_WRITE_H_

#include "morph-trans.h"

#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

void Serialize(vector<MorphTrans>& models, vector<Model*>& cnn_models,
               string& filename);

void Read(vector<MorphTrans>* models, vector<Model*>* cnn_models,
          string& filename);

#endif
