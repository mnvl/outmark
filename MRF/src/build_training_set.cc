
#include <stdint.h>
#include <time.h>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <gflags/gflags.h>
#include <image/image.hpp>

DEFINE_string(image_dir, "/large/data/Abdomen/RawData/Training/img", "");
DEFINE_string(image_prefix, "img", "");
DEFINE_string(label_dir, "/large/data/Abdomen/RawData/Training/label", "");
DEFINE_string(label_prefix, "label", "");
DEFINE_string(training_set_file, "/large/data/Abdomen/training_set.csv", "");

#define EXPECT(x) if (!(x)) { throw std::runtime_error(std::string(#x " failed in ") + __PRETTY_FUNCTION__); }

class TrainingSetBuilder {
private:
  std::unordered_map<std::string, long> data_to_weights_;
 
  void process_image_and_labels(const image::basic_image<short, 3> &image_data,
                              const image::basic_image<short, 3> &label_data) {
    for (int x = 1; x < image_data.width() - 1; ++x) {
      for (int y = 1; y < image_data.height() - 1; ++y) {
        for (int z = 1; z < image_data.depth() - 1; ++z) {
          std::stringstream ss;
          ss << label_data.at(x, y, z) << ","
             << image_data.at(x, y, z) << ","
             << image_data.at(x - 1, y, z) << ","
             << image_data.at(x, y - 1, z) << ","
             << image_data.at(x, y, z - 1) << ","
             << image_data.at(x + 1, y, z) << ","
             << image_data.at(x, y + 1, z) << ","
             << image_data.at(x, y, z + 1) << "\n";
          ++data_to_weights_[ss.str()];
        }
      }
    }
  }

  std::vector<std::pair<std::string, std::string>> build_files_list() const {
      std::vector<std::pair<std::string, std::string>> result;

      for (auto &image_file : boost::make_iterator_range(
               boost::filesystem::directory_iterator(boost::filesystem::path(FLAGS_image_dir)), {})) {
          boost::filesystem::path label_file =
              boost::filesystem::path(FLAGS_label_dir) /
              boost::replace_all_copy<std::string>(
                  image_file.path().filename().string(), FLAGS_image_prefix, FLAGS_label_prefix);

          result.push_back(std::make_pair(image_file.path().string(), label_file.string()));
      }

      return result;
  }

public:
  TrainingSetBuilder() {
  }
          
  void build_training_set() {
    for (const std::pair<std::string, std::string> &files :
         build_files_list()) {
      if (files.first.find("img0001.nii") != std::string::npos) continue;

      std::cerr << "\n*** loading " << files.first << " with labels "
                << files.second << "...\n";

      image::basic_image<short, 3> image_data;
      EXPECT (image_data.load_from_file<image::io::nifti>(files.first.c_str()));

      image::basic_image<short, 3> label_data;
      EXPECT (label_data.load_from_file<image::io::nifti>(files.second.c_str()));

      std::cerr << "image dimensions are " << image_data.width() << "x"
                << image_data.height() << "x" << image_data.depth() << " ("
                << label_data.width() << "x" << label_data.height() << "x"
                << label_data.depth() << " for label)\n";

      EXPECT (image_data.width() == label_data.width());
      EXPECT (image_data.height() == label_data.height());
      EXPECT (image_data.depth() == label_data.depth());

      const size_t n1 = data_to_weights_.size();
      process_image_and_labels(image_data, label_data);
      const size_t n2 = data_to_weights_.size();
      
      std::cerr << n2  << " records, " << double(n2 - n1) / image_data.size() << " miss ratio\n";
    }
  }
};

int main(int argc, char *argv[]) {
  try {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    TrainingSetBuilder trb;
    trb.build_training_set();
  } catch (std::exception const &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
