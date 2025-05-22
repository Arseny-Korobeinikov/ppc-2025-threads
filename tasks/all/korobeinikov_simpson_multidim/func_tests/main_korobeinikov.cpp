#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "../include/ops_all_korobeinikov.hpp"
#include "core/task/include/task.hpp"

double TestFunction_1(const korobeinikov_simpson_multidim::Coordinate& coord) { return std::sin(coord[0]); }

TEST(SimpsonTaskAllTest, OneDimensionalIntegral) {
  using namespace korobeinikov_simpson_multidim;
  boost::mpi::communicator world;

  // Input data
  std::vector<Bound> bounds = {{0.0, 1.0}};
  IntegrandFunction func = &TestFunction_1;
  std::size_t approxs = 128;

  // Output data
  double result = 0.0;

  // Data for task
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs_count.push_back(bounds.size());
    task_data->inputs.resize(3);
    task_data->inputs[0] = reinterpret_cast<uint8_t*>(bounds.data());
    task_data->inputs[1] = reinterpret_cast<uint8_t*>(func);
    task_data->inputs[2] = reinterpret_cast<uint8_t*>(&approxs);

    task_data->outputs.resize(1);

    task_data->outputs[0] = reinterpret_cast<uint8_t*>(&result);
    task_data->outputs_count.push_back(1);
  } else {
    task_data->inputs = {reinterpret_cast<uint8_t*>(func)};
  }
  // Make task
  SimpsonTaskAll task(task_data);

  ASSERT_TRUE(task.ValidationImpl()) << "Validation failed!";
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  if (world.rank() == 0) {
    double expected = 1 - std::cos(1);
    EXPECT_NEAR(result, expected, 0.01);
  }
}

double TestFunction_2(const korobeinikov_simpson_multidim::Coordinate& coord) {
  return coord[0] * coord[0] + coord[1] * coord[1];
}

TEST(SimpsonTaskAllTest, TwoDimensionalIntegral) {
  using namespace korobeinikov_simpson_multidim;
  boost::mpi::communicator world;

  // Input data
  std::vector<Bound> bounds = {{0.0, 1.0}, {0.0, 1.0}};
  IntegrandFunction func = &TestFunction_2;
  std::size_t approxs = 512;

  // Output data
  double result = 0.0;

  // Data for task
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs_count.push_back(bounds.size());
    task_data->inputs.resize(3);
    task_data->inputs[0] = reinterpret_cast<uint8_t*>(bounds.data());
    task_data->inputs[1] = reinterpret_cast<uint8_t*>(func);
    task_data->inputs[2] = reinterpret_cast<uint8_t*>(&approxs);

    task_data->outputs.resize(1);

    task_data->outputs[0] = reinterpret_cast<uint8_t*>(&result);
    task_data->outputs_count.push_back(1);
  } else {
    task_data->inputs = {reinterpret_cast<uint8_t*>(func)};
  }
  // Make task
  SimpsonTaskAll task(task_data);

  ASSERT_TRUE(task.ValidationImpl()) << "Validation failed!";
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();


  if (world.rank() == 0) {
    double expected = 2.0 / 3.0;  // 0.6667
    EXPECT_NEAR(result, expected, 0.01);
  }
}

double TestFunction_3(const korobeinikov_simpson_multidim::Coordinate& coord) {
  return coord[0] * 2 - coord[1] + coord[2];
}

TEST(SimpsonTaskAllTest, ThreeDimensionalIntegral) {
  using namespace korobeinikov_simpson_multidim;
  boost::mpi::communicator world;

  // Input data
  std::vector<Bound> bounds = {{-1.0, 1.0}, {0.0, 1.0}, {0.0, 2.0}};
  IntegrandFunction func = &TestFunction_3;
  std::size_t approxs = 256;

  // Output data
  double result = 0.0;

  // Data for task
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs_count.push_back(bounds.size());
    task_data->inputs.resize(3);
    task_data->inputs[0] = reinterpret_cast<uint8_t*>(bounds.data());
    task_data->inputs[1] = reinterpret_cast<uint8_t*>(func);
    task_data->inputs[2] = reinterpret_cast<uint8_t*>(&approxs);

    task_data->outputs.resize(1);

    task_data->outputs[0] = reinterpret_cast<uint8_t*>(&result);
    task_data->outputs_count.push_back(1);
  } else {
    task_data->inputs = {reinterpret_cast<uint8_t*>(func)};
  }

  // Make task
  SimpsonTaskAll task(task_data);

  ASSERT_TRUE(task.ValidationImpl()) << "Validation failed!";
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  if (world.rank() == 0) {
    double expected = 2;
    EXPECT_NEAR(result, expected, 0.01);
  }
}
