#include <torch/extension.h>
#include <vector>
#include <unordered_map>
#include <algorithm>

torch::Tensor mode_except_negative_one(torch::Tensor input) {
    auto sizes = input.sizes();
    int N = sizes[0];
    int D = sizes[1];
    
    auto input_a = input.accessor<int32_t, 2>();
    torch::Tensor result = torch::empty({N}, input.options().dtype(torch::kInt32));
    auto result_a = result.accessor<int32_t, 1>();
    
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        std::unordered_map<int32_t, int> freq;
        for (int j = 0; j < D; j++) {
            freq[input_a[i][j]]++;
        }
        
        std::vector<std::pair<int32_t, int>> freq_vec(freq.begin(), freq.end());
        std::sort(freq_vec.begin(), freq_vec.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
        
        if (freq_vec[0].first == 0 && freq_vec.size() > 1) {
            result_a[i] = freq_vec[1].first;
        } else {
            result_a[i] = freq_vec[0].first;
        }
    }
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mode_except_negative_one", &mode_except_negative_one, "Mode except 0 (parallel over N)");
}