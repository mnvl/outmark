
#include <stdint.h>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <gflags/gflags.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <unordered_map>
#include <image/image.hpp>

DEFINE_string(image_dir, "/large/data/Abdomen/RawData/Training/img", "");
DEFINE_string(image_prefix, "img", "");
DEFINE_string(label_dir, "/large/data/Abdomen/RawData/Training/label", "");
DEFINE_string(label_prefix, "label", "");
DEFINE_string(model_file, "mrf.model", "");

#define EXPECT(x) if (!(x)) { throw std::runtime_error(std::string(#x " failed in ") + __PRETTY_FUNCTION__); }

class MRF {
private:
    uint64_t total_voxels_ = 0;
    std::unordered_map<int, std::unordered_map<int, uint64_t>> label_to_intensity_;
    std::unordered_map<int, std::unordered_map<int, uint64_t>> label_to_label_;

    void learn_image_and_labels(const image::basic_image<short, 3> &image_data,
                                const image::basic_image<short, 3> &label_data) {
        for (int x = 1; x < image_data.width() - 1; ++x) {
            for (int y = 1; y < image_data.height() - 1; ++y) {
                for (int z = 1; z < image_data.depth() - 1; ++z) {
                    short intensity = image_data.at(x, y, z);
                    short label = label_data.at(x, y ,z);

                    total_voxels_ += image_data.size();
                    ++label_to_intensity_[label][intensity];
                    ++label_to_label_[label_data.at(x - 1, y ,z)][label];
                    ++label_to_label_[label_data.at(x, y - 1, z)][label];
                    ++label_to_label_[label_data.at(x, y, z - 1)][label];
                    ++label_to_label_[label][label_data.at(x + 1, y ,z)];
                    ++label_to_label_[label][label_data.at(x, y + 1, z)];
                    ++label_to_label_[label][label_data.at(x, y, z + 1)];
                }
            }
        }

        store_model();
    }

    void store_model() {
        std::ofstream fs(FLAGS_model_file);
        boost::archive::text_oarchive oarch(fs);

        oarch << total_voxels_;
        oarch << label_to_label_;
        oarch << label_to_label_;
    }

public:
  void build_probabilities() {
    for (auto &image_file : boost::make_iterator_range(
             boost::filesystem::directory_iterator(boost::filesystem::path(FLAGS_image_dir)), {})) {
      boost::filesystem::path label_file =
          boost::filesystem::path(FLAGS_label_dir) /
          boost::replace_all_copy<std::string>(
              image_file.path().filename().string(), FLAGS_image_prefix, FLAGS_label_prefix);

      std::cerr << "\n*** loading " << image_file << " with labels "
                << label_file << "...\n";

      image::basic_image<short, 3> image_data;
      EXPECT (image_data.load_from_file<image::io::nifti>(
          image_file.path().string().c_str()));

      image::basic_image<short, 3> label_data;
      EXPECT (label_data.load_from_file<image::io::nifti>(
          label_file.string().c_str()));

      std::cerr << "image dimensions are " << image_data.width() << "x"
                << image_data.height() << "x" << image_data.depth() << " ("
                << label_data.width() << "x" << label_data.height() << "x"
                << label_data.depth() << " for label)\n";

      EXPECT (image_data.width() == label_data.width());
      EXPECT (image_data.height() == label_data.height());
      EXPECT (image_data.depth() == label_data.depth());

      learn_image_and_labels(image_data, label_data);
    }
  }
};

int main() {
    MRF mrf;
    mrf.build_probabilities();
}
