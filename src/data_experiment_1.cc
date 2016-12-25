
#include <string>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/range/iterator_range.hpp>
#include <image/image.hpp>

static boost::filesystem::path const DATA_DIR = "/large/data/Abdomen/RawData/Training";
static boost::filesystem::path const IMAGE_DIR = DATA_DIR / "img";
static boost::filesystem::path const LABEL_DIR = DATA_DIR / "label";

void build_probabilities() {
    for(auto& image_file : boost::make_iterator_range(boost::filesystem::directory_iterator(IMAGE_DIR), {})) {
      boost::filesystem::path label_file =
          LABEL_DIR /
          boost::replace_all_copy<std::string>(
              image_file.path().filename().string(), "img", "label");

      std::cerr << "\n*** loading " << image_file << " with labels "
                << label_file << "...\n";

      image::basic_image<short, 3> image_data;
      assert(image_data.load_from_file<image::io::nifti>(
          image_file.path().string().c_str()));

      image::basic_image<short, 3> label_data;
      assert(label_data.load_from_file<image::io::nifti>(
          label_file.string().c_str()));

      std::cerr << "image dimensions are " << image_data.width() << "x"
                << image_data.height() << "x" << image_data.depth() << " ("
                << label_data.width() << "x" << label_data.height() << "x"
                << label_data.depth() << " for label)\n";

      assert(image_data.width() == label_data.width());
      assert(image_data.height() == label_data.height());
      assert(image_data.depth() == label_data.depth());
    }
}

int main() {
    build_probabilities();
}
