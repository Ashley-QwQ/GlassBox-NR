#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

// Release-flag publication for launches 61 and 63. The only elected
// writer is tid=(0,0,0). Every CTA x writes its z stage to the current
// release word; the last stage also zeros the successor release word.
// The recorded map shows one u32 slot per CTA x in each release region.
static py::bytes words(int grid_x, uint32_t value) {
    if (grid_x <= 0 || grid_x > 1'000'000) throw std::invalid_argument("grid x");
    std::string out(static_cast<size_t>(grid_x) * 4, '\0');
    for (int i = 0; i < grid_x; ++i) {
        for (int byte = 0; byte < 4; ++byte)
            out[static_cast<size_t>(i) * 4 + byte] = static_cast<char>((value >> (8 * byte)) & 0xff);
    }
    return py::bytes(out);
}

static py::dict execute_release(const std::string& family, int grid_x) {
    py::dict out;
    if (family == "qkv") {
        out["release61"] = words(grid_x, 1);
        out["release62"] = words(grid_x, 0);
    } else if (family == "projection") {
        out["release63"] = words(grid_x, 3);
        out["release64"] = words(grid_x, 0);
    } else throw std::invalid_argument("unknown family");
    return out;
}

PYBIND11_MODULE(dlssnr_fam_qkv_projection_release_resume1, m) {
    m.doc() = "release flag publication for qkv/projection R01 resume1";
    m.def("execute_release", &execute_release);
}
