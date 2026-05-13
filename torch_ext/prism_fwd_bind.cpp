#include <torch/extension.h>

torch::Tensor prism_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    c10::optional<torch::Tensor> internal_bias,
    c10::optional<torch::Tensor> shuffle_masks,
    c10::optional<torch::Tensor> dropout_seeds,
    double dropout_p);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &prism_forward,
          "Prism forward pass: C = A * R * B^T (with optional gating, internal_bias, input_shuffle, dropout)",
          py::arg("A"),
          py::arg("B"),
          py::arg("R"),
          py::arg("group_size") = 256,
          py::arg("reconn_sz") = 8,
          py::arg("internal_bias") = py::none(),
          py::arg("shuffle_masks") = py::none(),
          py::arg("dropout_seeds") = py::none(),
          py::arg("dropout_p") = 0.0);
}
