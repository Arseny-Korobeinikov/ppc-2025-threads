#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>
#include <iostream>

#include "core/util/include/util.hpp"

#include "../include/ops_tbb_korobeinikov.hpp"

bool korobeinikov_simpson_multidim::SimpsonTaskTbb::ValidationImpl() {
  const auto arity = task_data->inputs_count[0];

  const bool inputs_are_present = task_data->inputs.size() == 3 && arity > 0;
  const bool outputs_are_present = task_data->outputs.size() == 1 && task_data->outputs_count[0] == 1;
  if (!inputs_are_present || !outputs_are_present) {
    return false;
  }

  const auto* bounds = reinterpret_cast<Bound*>(task_data->inputs[0]);
  return std::all_of(bounds, bounds + arity, [](const auto& b) { return b.lo <= b.hi; });
}

bool korobeinikov_simpson_multidim::SimpsonTaskTbb::PreProcessingImpl() {
  arity_ = task_data->inputs_count[0];
  const auto* bsrc = reinterpret_cast<Bound*>(task_data->inputs[0]);
  bounds_.assign(bsrc, bsrc + arity_);

  func_ = reinterpret_cast<IntegrandFunction>(task_data->inputs[1]);
  approxs_ = *reinterpret_cast<std::size_t*>(task_data->inputs[2]);

  steps_.resize(arity_);
  std::ranges::transform(bounds_, steps_.begin(), [n = approxs_](const auto& b) { return (b.hi - b.lo) / n; });

  gridcap_ = static_cast<std::size_t>(std::pow(approxs_, arity_));
  scale_ = std::accumulate(steps_.begin(), steps_.end(), 1., [](double cur, double step) { return cur * step / 3.; });

  return true;
}

bool korobeinikov_simpson_multidim::SimpsonTaskTbb::RunImpl() {
  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  double isum = arena.execute([&] {
    return oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<std::size_t>(0, gridcap_), 0.,
        [&](const tbb::blocked_range<std::size_t>& r, double threadsum) {
          std::vector<std::size_t> gridpos(arity_);
          std::vector<double> coordbuf(arity_);
          for (std::size_t ip = r.begin(); ip < r.end(); ip++) {
            {
              auto p = ip;
              for (size_t i = 0; i < arity_; i++) {
                gridpos[i] = p % approxs_;
                p /= approxs_;
              }
            }

            for (size_t i = 0; i < arity_; i++) {
              coordbuf[i] = bounds_[i].lo + (static_cast<double>(gridpos[i]) * steps_[i]);
            }

            double coefficient = 1.;
            for (auto pos : gridpos) {
              if (pos == 0 || pos == (approxs_ - 1)) {
                coefficient *= 1.;
              } else if (pos % 2 != 0) {
                coefficient *= 4.;
              } else {
                coefficient *= 2.;
              }
            }

            threadsum += coefficient * func_(coordbuf);
          }

          return threadsum;
        },
        std::plus<>());
  });

  result_ = isum * scale_;
  
  return true;
}

bool korobeinikov_simpson_multidim::SimpsonTaskTbb::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
  return true;
}
