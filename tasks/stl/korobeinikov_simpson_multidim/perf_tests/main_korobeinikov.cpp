#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "../include/ops_stl_korobeinikov.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

double TestFunction_1(const korobeinikov_simpson_multidim::Coordinate &coord) {
  return coord[0] + coord[1] + coord[2] + coord[3] + coord[4];
}

TEST(stl_korobeinikov_perf_test, test_task_run) {
  using namespace korobeinikov_simpson_multidim;
  // Input data
  std::vector<Bound> bounds = {
      {.lo = 0., .hi = 1.}, {.lo = 0., .hi = 1.}, {.lo = 0., .hi = 1.},
      {.lo = 0., .hi = 1.}, {.lo = 0., .hi = 1.}, {.lo = 0., .hi = 1.},
  };
  IntegrandFunction func = &TestFunction_1;
  std::size_t approxs = 18;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t *>(bounds.data()), reinterpret_cast<uint8_t *>(func),
                       reinterpret_cast<uint8_t *>(&approxs)};
  task_data->inputs_count.emplace_back(bounds.size());

  task_data->outputs.resize(1);
  double result = 0.0;
  task_data->outputs[0] = reinterpret_cast<uint8_t *>(&result);
  task_data->outputs_count.push_back(1);

  auto task = std::make_shared<SimpsonTaskStl>(task_data);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  EXPECT_NEAR(result, 1.5, 0.3);
}
