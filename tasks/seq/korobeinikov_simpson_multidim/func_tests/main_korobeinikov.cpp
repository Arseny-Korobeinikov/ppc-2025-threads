#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "../include/ops_seq_korobeinikov.hpp"
#include "core/task/include/task.hpp"

double TestFunction_1(const korobeinikov_simpson_multidim::Coordinate& coord) { return std::sin(coord[0]); }

TEST(SimpsonTaskSeqTest, OneDimensionalIntegral) {
  using namespace korobeinikov_simpson_multidim;

  // Input data
  std::vector<Bound> bounds = {{0.0, 1.0}};
  IntegrandFunction func = &TestFunction_1;
  std::size_t approxs = 128;

  // Data for task
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.resize(3);
  task_data->inputs[0] = reinterpret_cast<uint8_t*>(bounds.data());
  task_data->inputs[1] = reinterpret_cast<uint8_t*>(func);
  task_data->inputs[2] = reinterpret_cast<uint8_t*>(&approxs);

  task_data->outputs.resize(1);
  double result = 0.0;
  task_data->outputs[0] = reinterpret_cast<uint8_t*>(&result);
  task_data->outputs_count.push_back(1);

  // Make task
  SimpsonTaskSeq task(task_data);

  ASSERT_TRUE(task.ValidationImpl()) << "Validation failed!";
  ASSERT_TRUE(task.PreProcessingImpl()) << "PreProcessing failed!";
  ASSERT_TRUE(task.RunImpl()) << "Run failed!";
  ASSERT_TRUE(task.PostProcessingImpl()) << "PostProcessing failed!";

  double expected = 1 - std::cos(1);

  EXPECT_NEAR(result, expected, 0.01) << "Integration result is incorrect!";
}

double TestFunction_2(const korobeinikov_simpson_multidim::Coordinate& coord) {
  return coord[0] * coord[0] + coord[1] * coord[1];
}

TEST(SimpsonTaskSeqTest, TwoDimensionalIntegral) {
  using namespace korobeinikov_simpson_multidim;

  // Input data
  std::vector<Bound> bounds = {{0.0, 1.0}, {0.0, 1.0}}; 
  IntegrandFunction func = &TestFunction_2;
  std::size_t approxs = 512; 

  // Data for task
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.resize(3);
  task_data->inputs[0] = reinterpret_cast<uint8_t*>(bounds.data());
  task_data->inputs[1] = reinterpret_cast<uint8_t*>(func);
  task_data->inputs[2] = reinterpret_cast<uint8_t*>(&approxs);

  task_data->outputs.resize(1);
  double result = 0.0;
  task_data->outputs[0] = reinterpret_cast<uint8_t*>(&result);
  task_data->outputs_count.push_back(1);

  // Make task
  SimpsonTaskSeq task(task_data);

  ASSERT_TRUE(task.ValidationImpl()) << "Validation failed!";
  ASSERT_TRUE(task.PreProcessingImpl()) << "PreProcessing failed!";
  ASSERT_TRUE(task.RunImpl()) << "Run failed!";
  ASSERT_TRUE(task.PostProcessingImpl()) << "PostProcessing failed!";

  double expected = 2.0 / 3.0;  // 0.6667

  EXPECT_NEAR(result, expected, 0.01) << "Integration result is incorrect!";
}

double TestFunction_3(const korobeinikov_simpson_multidim::Coordinate& coord) {
  return coord[0] * 2 - coord[1] + coord[2];
}

TEST(SimpsonTaskSeqTest, ThreeDimensionalIntegral) {
  using namespace korobeinikov_simpson_multidim;

  // Input data
  std::vector<Bound> bounds = {{-1.0, 1.0}, {0.0, 1.0}, {0.0, 2.0}};
  IntegrandFunction func = &TestFunction_3;
  std::size_t approxs = 256;

  // Data for task
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.resize(3);
  task_data->inputs[0] = reinterpret_cast<uint8_t*>(bounds.data());
  task_data->inputs[1] = reinterpret_cast<uint8_t*>(func);
  task_data->inputs[2] = reinterpret_cast<uint8_t*>(&approxs);

  task_data->outputs.resize(1);
  double result = 0.0;
  task_data->outputs[0] = reinterpret_cast<uint8_t*>(&result);
  task_data->outputs_count.push_back(1);

  // Make task
  SimpsonTaskSeq task(task_data);

  ASSERT_TRUE(task.ValidationImpl()) << "Validation failed!";
  ASSERT_TRUE(task.PreProcessingImpl()) << "PreProcessing failed!";
  ASSERT_TRUE(task.RunImpl()) << "Run failed!";
  ASSERT_TRUE(task.PostProcessingImpl()) << "PostProcessing failed!";

  double expected = 2;

  EXPECT_NEAR(result, expected, 0.1) << "Integration result is incorrect!";
}