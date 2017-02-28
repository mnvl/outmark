
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
#include <GridCut/GridGraph_3D_6C_MT.h>
#include <AlphaExpansion/AlphaExpansion_3D_6C_MT.h>
#include <image/image.hpp>

DEFINE_string(image_dir, "/large/data/Abdomen/RawData/Training/img", "");
DEFINE_string(image_prefix, "img", "");
DEFINE_string(label_dir, "/large/data/Abdomen/RawData/Training/label", "");
DEFINE_string(label_prefix, "label", "");
DEFINE_string(model_file, "mrf.model", "");
DEFINE_bool(rebuild_model, false, "");

#define EXPECT(x) if (!(x)) { throw std::runtime_error(std::string(#x " failed in ") + __PRETTY_FUNCTION__); }

class MRF {
private:
  uint64_t total_voxels_ = 0;
  std::unordered_map<int, std::unordered_map<int, uint64_t>>
      intensity_to_label_;
  std::unordered_map<int, uint64_t> label_count_;
  std::unordered_map<int, std::unordered_map<int, uint64_t>> label_to_label_;

  void learn_image_and_labels(const image::basic_image<short, 3> &image_data,
                              const image::basic_image<short, 3> &label_data) {
    for (int x = 1; x < image_data.width() - 1; ++x) {
      for (int y = 1; y < image_data.height() - 1; ++y) {
        for (int z = 1; z < image_data.depth() - 1; ++z) {
          short intensity = image_data.at(x, y, z);
          short label = label_data.at(x, y, z);

          total_voxels_ += image_data.size();
          ++label_count_[label];
          ++intensity_to_label_[intensity][label];
          ++label_to_label_[label_data.at(x - 1, y, z)][label];
          ++label_to_label_[label_data.at(x, y - 1, z)][label];
          ++label_to_label_[label_data.at(x, y, z - 1)][label];
          ++label_to_label_[label][label_data.at(x + 1, y, z)];
          ++label_to_label_[label][label_data.at(x, y + 1, z)];
          ++label_to_label_[label][label_data.at(x, y, z + 1)];
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
  void store_model() const {
    std::ofstream fs(FLAGS_model_file);
    boost::archive::text_oarchive oarch(fs);

    oarch << total_voxels_;
    oarch << label_count_;
    oarch << intensity_to_label_;
    oarch << label_to_label_;
  }

  void load_model() {
    std::ifstream fs(FLAGS_model_file);
    boost::archive::text_iarchive iarch(fs);

    iarch >> total_voxels_;
    iarch >> label_count_;
    iarch >> intensity_to_label_;
    iarch >> label_to_label_;
  }

  void build_probabilities() {
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

      learn_image_and_labels(image_data, label_data);
    }
  }

  void classify_images() {
    for (const std::pair<std::string, std::string> &files :
             build_files_list()) {
      if (files.first.find("img0001.nii") == std::string::npos) continue;

      std::cerr << "\n*** segmenting " << files.first << "...\n";

      image::basic_image<short, 3> image_data;
      EXPECT (image_data.load_from_file<image::io::nifti>(files.first.c_str()));

      image::basic_image<short, 3> label_data;
      EXPECT (label_data.load_from_file<image::io::nifti>(files.second.c_str()));

      classify_image(image_data, label_data);
    }
  }

  void classify_image(image::basic_image<short, 3> const &image_data,
                      image::basic_image<short, 3> const &image_labels) const {
    const int width = image_data.width();
    const int height = image_data.height();
    const int depth = image_data.depth();
    const int voxels = width * height * depth;
    const int num_labels = label_count_.size();
    const int num_threads = 4;
    const int block_size = 20;

    clock_t t1 = clock();

    std::unique_ptr<float[]> cost(
        new float[width * height * depth * num_labels]);
    for (int x = 0; x < width; ++x) {
      for (int y = 0; y < height; ++y) {
        for (int z = 0; z < depth; ++z) {
          for (int l = 0; l < num_labels; ++l) {
            const int pos = (x + (y + z * height) * width) * num_labels + l;

            const auto label_count_it = label_count_.find(l);
            if (label_count_it == label_count_.end()) {
              cost[pos] = 1;
              continue;
            }

            const short intensity = image_data.at(x, y, z);
            const auto intensity_it = intensity_to_label_.find(intensity);
            if (intensity_it == intensity_to_label_.end()) {
              cost[pos] = 1;
              continue;
            }
            const auto intensity_label_it = intensity_it->second.find(l);
            if (intensity_label_it == intensity_it->second.end()) {
              cost[pos] = 1;
              continue;
            }

            double freq = double(intensity_label_it->second) / double(label_count_it->second);
            cost[pos] = float(1 - freq);
          }
        }
      }
    }

    std::vector<float> label_to_label(num_labels * num_labels);
    for (int i = 0; i < num_labels; ++i) {
      for (int j = 0; j < num_labels; ++j) {
        int pos = i * num_labels + j;

        const auto label_count_it = label_count_.find(i);
        if (label_count_it == label_count_.end()) {
          label_to_label[pos] = 1;
          continue;
        }

        const auto it = label_to_label_.find(i);
        if (it == label_to_label_.end()) {
          label_to_label[pos] = 1;
          continue;
        }

        const auto jt = it->second.find(j);
        if (jt == it->second.end()) {
          label_to_label[pos] = 1;
          continue;
        }

        label_to_label[pos] =
            float(1 - double(jt->second) / double(label_count_it->second));
      }
    }
    std::unique_ptr<float *[]> smooth(
        new float *[width * height * depth * 6]);
    for (int i = 0; i < width * height * depth * 6; ++i) {
      smooth[i] = &label_to_label[0];
    }

    clock_t t2 = clock();

    std::cerr << "building data structures took " << float(t2 - t1)/CLOCKS_PER_SEC << " secs\n";
    typedef AlphaExpansion_3D_6C_MT<int, float, float> AlphaExpansion;
    std::unique_ptr<AlphaExpansion> alpha_expansion(
        new AlphaExpansion(width, height, depth, num_labels, cost.release(), smooth.release(),
                           num_threads, block_size));

    const int ncycles = 1;
    alpha_expansion->perform(ncycles);

    clock_t t3 = clock();

    std::cerr << "\n\nalpha expansion took " << float(t3 - t2) / CLOCKS_PER_SEC
              << " secs for " << ncycles << " cycles\n";

    int correct_labels = 0;
    int incorrect_labels = 0;
    const int *labeling = alpha_expansion->get_labeling();
    for (int x = 0; x < width; ++x) {
      for (int y = 0; y < height; ++y) {
        for (int z = 0; z < depth; ++z) {
          const int pos = x + (y + z * height) * width;
          if (labeling[pos] == image_labels.at(x, y, z)) {
            ++correct_labels;
          } else {
            ++incorrect_labels;
          }
        }
      }

      std::cerr << "correct_labels = " << correct_labels << " "
                << (float(correct_labels) / voxels) << "%\n";
      std::cerr << "incorrect_labels = " << incorrect_labels << " "
                << (float(incorrect_labels) / voxels) << "%\n";
    }
  }
};

int main(int argc, char *argv[]) {
  try {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    MRF mrf;

    if (FLAGS_rebuild_model) {
      mrf.build_probabilities();
      mrf.store_model();
    } else {
      mrf.load_model();
    }

    mrf.classify_images();
  } catch (std::exception const &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}