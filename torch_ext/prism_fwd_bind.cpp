#include <torch/extension.h>

torch::Tensor prism_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &prism_forward,
          "OFT forward pass: C = A * R * B^T",
          py::arg("A"),
          py::arg("B"),
          py::arg("R"),
          py::arg("group_size") = 256,
          py::arg("reconn_sz") = 8);
}
