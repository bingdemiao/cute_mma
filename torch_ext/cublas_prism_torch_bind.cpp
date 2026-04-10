#include <torch/extension.h>

std::vector<torch::Tensor> cublas_prism_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    bool rw_mode,
    bool gated,
    c10::optional<torch::Tensor> internal_bias,
    double dropout_p,
    bool training,
    c10::optional<torch::Tensor> seg_pairs);

std::vector<torch::Tensor> cublas_backward_dA_dR(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    bool gated,
    c10::optional<at::ScalarType> dR_dtype,
    c10::optional<torch::Tensor> internal_bias,
    c10::optional<torch::Tensor> dropout_seeds,
    double dropout_p,
    c10::optional<torch::Tensor> seg_pairs);

torch::Tensor cublas_backward_dB(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    bool gated,
    c10::optional<at::ScalarType> dB_dtype,
    c10::optional<torch::Tensor> internal_bias,
    c10::optional<torch::Tensor> dropout_seeds,
    double dropout_p,
    c10::optional<torch::Tensor> seg_pairs);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cublas_prism_forward,
          "Prism forward pass (cuBLAS): C = A * R * B^T",
          py::arg("A"),
          py::arg("B"),
          py::arg("R"),
          py::arg("group_size") = 256,
          py::arg("reconn_sz") = 8,
          py::arg("rw_mode") = false,
          py::arg("gated") = false,
          py::arg("internal_bias") = py::none(),
          py::arg("dropout_p") = 0.0,
          py::arg("training") = false,
          py::arg("seg_pairs") = py::none());
    m.def("backward_dA_dR", &cublas_backward_dA_dR,
          "Prism backward (cuBLAS): compute dA, dR, d_internal_bias",
          py::arg("dC"),
          py::arg("A"),
          py::arg("B"),
          py::arg("R"),
          py::arg("group_size") = 256,
          py::arg("reconn_sz") = 8,
          py::arg("gated") = false,
          py::arg("dR_dtype") = py::none(),
          py::arg("internal_bias") = py::none(),
          py::arg("dropout_seeds") = py::none(),
          py::arg("dropout_p") = 0.0,
          py::arg("seg_pairs") = py::none());
    m.def("backward_dB", &cublas_backward_dB,
          "Prism backward (cuBLAS): compute dB",
          py::arg("dC"),
          py::arg("A"),
          py::arg("R"),
          py::arg("group_size") = 256,
          py::arg("reconn_sz") = 8,
          py::arg("gated") = false,
          py::arg("dB_dtype") = py::none(),
          py::arg("internal_bias") = py::none(),
          py::arg("dropout_seeds") = py::none(),
          py::arg("dropout_p") = 0.0,
          py::arg("seg_pairs") = py::none());
}
