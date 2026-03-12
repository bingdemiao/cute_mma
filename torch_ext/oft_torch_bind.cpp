#include <torch/extension.h>

torch::Tensor oft_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz);

std::vector<torch::Tensor> oft_backward_dA_dR(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    bool gated);

torch::Tensor oft_backward_dB(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    bool gated);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &oft_forward,
          "OFT forward pass: C = A * R * B^T",
          py::arg("A"),
          py::arg("B"),
          py::arg("R"),
          py::arg("group_size") = 256,
          py::arg("reconn_sz") = 8);
    m.def("backward_dA_dR", &oft_backward_dA_dR,
          "OFT backward: compute dA, dR",
          py::arg("dC"),
          py::arg("A"),
          py::arg("B"),
          py::arg("R"),
          py::arg("group_size") = 256,
          py::arg("reconn_sz") = 8,
          py::arg("gated") = false);
    m.def("backward_dB", &oft_backward_dB,
          "OFT backward: compute dB",
          py::arg("dC"),
          py::arg("A"),
          py::arg("R"),
          py::arg("group_size") = 256,
          py::arg("reconn_sz") = 8,
          py::arg("gated") = false);
}
