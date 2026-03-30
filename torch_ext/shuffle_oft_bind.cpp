#include <torch/extension.h>
#include <tuple>

torch::Tensor shuffle_oft_forward(
    torch::Tensor A, torch::Tensor B, torch::Tensor R,
    torch::Tensor seg_pairs,
    int64_t group_size, int64_t reconn_sz, bool gated);

std::tuple<torch::Tensor, torch::Tensor> shuffle_oft_backward_dA_dR(
    torch::Tensor dC, torch::Tensor A, torch::Tensor B, torch::Tensor R,
    torch::Tensor seg_pairs,
    int64_t group_size, int64_t reconn_sz, bool gated);

torch::Tensor shuffle_oft_backward_dB(
    torch::Tensor dC, torch::Tensor A, torch::Tensor R,
    torch::Tensor seg_pairs,
    int64_t group_size, int64_t reconn_sz, bool gated);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shuffle_oft_forward,
          "Shuffle OFT forward");
    m.def("backward_dA_dR", &shuffle_oft_backward_dA_dR,
          "Shuffle OFT backward dA + dR");
    m.def("backward_dB", &shuffle_oft_backward_dB,
          "Shuffle OFT backward dB");
}
