
#include <unordered_set>
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <memory>
#include <cstdlib>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/platform/load_library.h"
//#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/protobuf/tensor_bundle.pb.h"

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"

/*
 * 1.   schema.yaml    样本数据每一列的特征类
 * 2.   samples.txt   测试样本数据
 *
 */

std::vector<std::vector<std::string>> samples;

// 样本中下标->特征类名称
std::unordered_map<int, std::string> idx_feat_map;

// 样本中特征类名称->下标
std::unordered_map<std::string, int> feat_idx_map;


// 需要抽取的特征所在样本的下标
std::vector<int> feature_ids;
std::unordered_map<std::string, std::string> feature_type_map;
// 一条样本包含的字段数
int field_num;


std::vector<std::string> Split(const std::string& s, char delim) {
  std::string item;
  std::istringstream is(s);
  std::vector<std::string> ret;
  while (std::getline(is, item, delim)) {
    ret.push_back(item);
  }
  return ret;
}

void Trim(std::string& s) {
    if (s.empty()) {
        return;
    }
    s.erase(0,s.find_first_not_of(" "));
    s.erase(s.find_last_not_of(" ") + 1);
    return;
}


int ParseSchema(const std::string& schema_f) {
    std::ifstream ifs(schema_f);
    if (schema_f.empty()) {
        std::cout << "schema file empty error" << std::endl;
        return -1;
    }
    std::string line;
    std::vector<std::string> line_vec;
    while (std::getline(ifs, line)) {
        line_vec.clear();
        if (line.find('#') != std::string::npos) {
            continue;
        }
        if (line.empty()) { continue; }

        //std::cout << line << std::endl;
        line_vec = Split(line, ':');
        if (line_vec.size() != 2) {
            std::cout << "schema format error " << std::endl;
            return -1;
        }
        Trim(line_vec[0]);
        Trim(line_vec[1]);
        int idx = std::stoi(line_vec[0]) - 1 ;
        idx_feat_map[idx] = line_vec[1];
        feat_idx_map[line_vec[1]] = idx;
        std::cout << "feature " << line_vec[1].c_str() << ":" << idx << std::endl;
        field_num++;
    }
    ifs.close();
    return 0;

}

int ParseExample(const std::string& sample_f) {
    std::ifstream ifs(sample_f);
    if (sample_f.empty()) {
        std::cout << "sample file empty error" << std::endl;
        return -1;
    }
    std::string line;
    std::vector<std::string> line_vec;
    while (std::getline(ifs, line)) {
        line_vec.clear();
        line_vec = Split(line, '\t');
        if (line_vec.size() != field_num) {
            std::cout << "sample filted; line size:" << line_vec.size() << ";field_num:" << field_num << std::endl;
            continue;
        }
        samples.push_back(line_vec);
    }
    ifs.close();
    return 0;
}

int ParseFeature(const std::string& feature_f) {
    std::ifstream ifs(feature_f);
    if (feature_f.empty()) {
        std::cout << "feature file empty error" << std::endl;
        return -1;
    }
    std::string line;
    std::vector<std::string> line_vec;
    while(std::getline(ifs, line)) {
        if (line.empty() || line.find('#') != std::string::npos) {
            continue;
        }
        line_vec.clear();
        line_vec = Split(line, ',');
        if (line_vec.size() != 2) {
            continue;
        }
        std::string feat = line_vec[0];
        auto it = feat_idx_map.find(feat);
        if (it == feat_idx_map.end()) {
            std::cout << "error: feat: " << feat.c_str() << " not in schema" << std::endl;
        }
        feature_ids.push_back(it->second);
        feature_type_map[feat] = line_vec[1];
    }

}


class Model {
    public:
        Model() {

        }
        int Load(std::string model_dir) {
            tensorflow::SessionOptions options;
            tensorflow::RunOptions run_options;
            tensorflow::Status load_status = tensorflow::LoadSavedModel(
                    options, tensorflow::RunOptions(), model_dir, {"serve"}, &_bundle);
            if (!load_status.ok()) {
                std::cout << "load model error" << std::endl;
                return -1;
            } else {
                std::cout << "load model success"  << std::endl;
                return 0;
            }
        }

        
        int PrepareInput(tensorflow::SavedModelBundle& bundle,
                std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs, 
                int batch_size,
                std::map<std::string, int>& inputs_index) {
          // signature
          const auto& signature_map = bundle.meta_graph_def.signature_def();
          // signature predict 
          const auto& signature_predict = signature_map.at("predict");
          // tensorinfo
          auto& inputs_map = signature_predict.inputs();
          auto& outputs_map = signature_predict.outputs();
        
          int loc_index = 0;
          for (auto const& imap:inputs_map) {
            auto tensor_info = imap.second;
            //int feature_size = tensor_info.tensor_shape().dim(1).size();
            int feature_size = tensor_info.tensor_shape().dim(0).size();
                                                                                                                                                                                                
            auto x = tensorflow::Tensor(tensor_info.dtype(), tensorflow::TensorShape({batch_size}));
            inputs.push_back(std::pair<std::string, tensorflow::Tensor>(tensor_info.name(), x));
            inputs_index[imap.first] = loc_index;
            //std::cout << "input_feature_name: " << imap.first.c_str() << " feature_dtype:" << tensor_info.dtype() << std::endl;
            loc_index++;
          }
        
          return 0;
        }        

        // 一个batch样本的预估
        int Predict(std::vector<std::vector<std::string>>& samples) {
            int batch_size = samples.size();
            std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
            std::map<std::string, int> inputs_index;
            std::vector<tensorflow::Tensor> outputs;
            auto& signature_def_map = _bundle.meta_graph_def.signature_def();
            auto& signature_def = signature_def_map.at("predict");
            const std::string& scores_name = signature_def.outputs().at("logistic").name();
            int pre_status = PrepareInput(_bundle, inputs, batch_size, inputs_index);
            //for (auto it = inputs_index.begin(); it != inputs_index.end();++it) {
            //    std::cout << it->first << " : " << it->second << std::endl;
            //}
            std::cout << "======num_feature_used[" << feature_ids.size() << "] batch_size[" << batch_size << "] rawdata_field_num[" << field_num << "] ========"<<std::endl;
            for (auto& feat_idx: feature_ids) {
                std::string feat_name = idx_feat_map[feat_idx];
                // 特征在样本中的位置
                int sample_idx = feat_idx_map[feat_name];
                // 特征在inputs中的位置
                int input_idx = inputs_index[feat_name];
                for (int i=0; i < batch_size; ++i) {
                    // 拿到每个样本的第sample_idx个特征, 并循环push到inputs对应的位置
                    // 需要判定特征的数据类型进行类型转换
                    std::string& feat_val = samples[i][sample_idx];
                    tensorflow::Tensor& t = inputs[input_idx].second;
                    //std::cout << feat_name.c_str() << ":" << feat_val.c_str() <<";sample_idx:"<<sample_idx<< ";dtype: " << t.dtype() << std::endl;
                    if (t.dtype() == tensorflow::DT_STRING) {
                        t.flat<std::string>()(i) = feat_val;
                        // if (feat_val != "") {
                        //     t.flat<std::string>()(i) = feat_val;
                        //     //t.scalar<std::string>()(i) = feat_val;
                        // } else {
                        //     t.flat<std::string>()(i) = "-";
                        //     //if (feat_name == "industry_level2_id" || feat_name == "industry_level1_id" || 
                        //     //        feat_name == "idea_type") {
                        //     //    t.flat<std::string>()(i) = "0";
                        //     //} else {
                        //     //    t.flat<std::string>()(i) = "-";
                        //     //}
                        // }
                    } else if (t.dtype() == tensorflow::DT_INT64) {
                        try{
                            t.flat<tensorflow::int64>()(i) = std::stoll(feat_val);
                            //t.scalar<tensorflow::int64>()(i) = std::stoll(feat_val);
                        } catch (std::exception& sx) {
                            std::cout << "int64 throw exception" << std::endl;
                            t.flat<tensorflow::int64>()(i) = 0;
                            continue;
                        }
                    } else if (t.dtype() == tensorflow::DT_FLOAT) {
                        try {
                            t.flat<float>()(i) = std::stof(feat_val);
                            //t.scalar<float>()(i) = std::stof(feat_val);
                        } catch (std::exception& sx){
                            t.flat<float>()(i) = 0.0;
                            std::cout << "float throw exception" << std::endl;
                            continue;
                        }
                    } else {
                        std::cout << "feat type error" << std::endl;
                    }
                    //std::cout << "sample_idx:" << sample_idx << ";input_idx:" << input_idx << std::endl;
                }
            }
            
            
            tensorflow::Status run_status = _bundle.session->Run(inputs, {scores_name}, {}, &outputs);
            if (run_status.ok()) {
                std::cout << "run status ok" << std::endl;
            } else {
                std::cout << "run status error: " << run_status.error_message() << std::endl;
                return 0;
            }
            auto& tensor = outputs[0];
            auto scores = tensor.flat<float>();
            int out_batch_size = tensor.shape().dim_size(0);
            for (int i=0;i<out_batch_size;++i) {
                std::cout << "score: " << scores(i) << std::endl;
            }

        }




    private:
        tensorflow::SavedModelBundle _bundle;
};


int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "usege: ./predict <schema.txt> <samples.txt>" << std::endl;
        return -1;
    }
    std::string model_dir = argv[1];
    std::string schema_f = argv[2];
    std::string sample_f = argv[3];
    std::string feature_f = argv[4];
    int status1 = ParseSchema(schema_f);
    int status2 = ParseExample(sample_f);
    int status3 = ParseFeature(feature_f);
    std::cout << "sample size : " << samples.size() << std::endl;
    std::shared_ptr<Model> _model = std::make_shared<Model>();
    int lstatus = _model->Load(model_dir);
    _model->Predict(samples);
}

