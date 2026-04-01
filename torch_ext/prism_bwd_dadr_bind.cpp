#include <torch/extension.h>

std::vector<torch::Tensor> prism_backward_dA_dR(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_dA_dR", &prism_backward_dA_dR,
          "Prism backward: compute dA, dR",
          py::arg("dC"),
          py::arg("A"),
          py::arg("B"),
          py::arg("R"),
          py::arg("group_size") = 256,
          py::arg("reconn_sz") = 8);
}
