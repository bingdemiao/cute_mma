#include <torch/extension.h>

std::vector<torch::Tensor> cublas_prism_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    bool rw_mode,
    int64_t gate_kind,
    c10::optional<torch::Tensor> internal_bias,
    double dropout_p,
    bool training,
    c10::optional<torch::Tensor> shuffle_masks);

std::vector<torch::Tensor> cublas_backward_dA_dR(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    int64_t gate_kind,
    c10::optional<at::ScalarType> dR_dtype,
    c10::optional<torch::Tensor> internal_bias,
    c10::optional<torch::Tensor> dropout_seeds,
    double dropout_p,
    c10::optional<torch::Tensor> shuffle_masks);

torch::Tensor cublas_backward_dB(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    int64_t gate_kind,
    c10::optional<at::ScalarType> dB_dtype,
    c10::optional<torch::Tensor> internal_bias,
    c10::optional<torch::Tensor> dropout_seeds,
    double dropout_p,
    c10::optional<torch::Tensor> shuffle_masks);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // gate_kind: 0 = none (linear), 1 = silu_gate, 2 = sigmoid_gate (A * 2*sigmoid(S))
    m.def("forward", &cublas_prism_forward,
          "Prism forward pass (cuBLAS): C = A * R * B^T",
          py::arg("A"),
          py::arg("B"),
          py::arg("R"),
          py::arg("group_size") = 256,
          py::arg("reconn_sz") = 8,
          py::arg("rw_mode") = false,
          py::arg("gate_kind") = 0,
          py::arg("internal_bias") = py::none(),
          py::arg("dropout_p") = 0.0,
          py::arg("training") = false,
          py::arg("shuffle_masks") = py::none());
    m.def("backward_dA_dR", &cublas_backward_dA_dR,
          "Prism backward (cuBLAS): compute dA, dR, d_internal_bias",
          py::arg("dC"),
          py::arg("A"),
          py::arg("B"),
          py::arg("R"),
          py::arg("group_size") = 256,
          py::arg("reconn_sz") = 8,
          py::arg("gate_kind") = 0,
          py::arg("dR_dtype") = py::none(),
          py::arg("internal_bias") = py::none(),
          py::arg("dropout_seeds") = py::none(),
          py::arg("dropout_p") = 0.0,
          py::arg("shuffle_masks") = py::none());
    m.def("backward_dB", &cublas_backward_dB,
          "Prism backward (cuBLAS): compute dB",
          py::arg("dC"),
          py::arg("A"),
          py::arg("R"),
          py::arg("group_size") = 256,
          py::arg("reconn_sz") = 8,
          py::arg("gate_kind") = 0,
          py::arg("dB_dtype") = py::none(),
          py::arg("internal_bias") = py::none(),
          py::arg("dropout_seeds") = py::none(),
          py::arg("dropout_p") = 0.0,
          py::arg("shuffle_masks") = py::none());
}
