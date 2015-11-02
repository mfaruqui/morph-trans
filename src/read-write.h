#ifndef READ_WRITE_H_
#define READ_WRITE_H_

#include "sep-morph.h"

#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

void Serialize(string& filename, SepMorph& model, vector<Model*>* cnn_model);

void Read(string& filename, SepMorph* model, vector<Model*>* cnn_model);

#endif
