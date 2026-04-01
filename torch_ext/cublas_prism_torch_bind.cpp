#include <torch/extension.h>

torch::Tensor cublas_prism_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    bool rw_mode,
    bool gated);

std::vector<torch::Tensor> cublas_backward_dA_dR(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    bool gated,
    c10::optional<at::ScalarType> dR_dtype);

torch::Tensor cublas_backward_dB(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    bool gated,
    c10::optional<at::ScalarType> dB_dtype);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cublas_prism_forward,
          "OFT forward pass (cuBLAS): C = A * R * B^T",
          py::arg("A"),
          py::arg("B"),
          py::arg("R"),
          py::arg("group_size") = 256,
          py::arg("reconn_sz") = 8,
          py::arg("rw_mode") = false,
          py::arg("gated") = false);
    m.def("backward_dA_dR", &cublas_backward_dA_dR,
          "OFT backward (cuBLAS): compute dA, dR",
          py::arg("dC"),
          py::arg("A"),
          py::arg("B"),
          py::arg("R"),
          py::arg("group_size") = 256,
          py::arg("reconn_sz") = 8,
          py::arg("gated") = false,
          py::arg("dR_dtype") = py::none());
    m.def("backward_dB", &cublas_backward_dB,
          "OFT backward (cuBLAS): compute dB",
          py::arg("dC"),
          py::arg("A"),
          py::arg("R"),
          py::arg("group_size") = 256,
          py::arg("reconn_sz") = 8,
          py::arg("gated") = false,
          py::arg("dB_dtype") = py::none());
}
