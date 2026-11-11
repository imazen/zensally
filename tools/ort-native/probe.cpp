// Native deployment/benchmark probe, not a detector or crop API.
#include <onnxruntime_cxx_api.h>
#include "embedded_model.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "zensally-single-file");
    env.DisableTelemetryEvents();
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::Session session(env, kModel, sizeof(kModel), options);
    std::vector<float> input(3 * 640 * 640);
    uint32_t state = 911;
    for (auto& x : input) {
      state = state * 1664525u + 1013904223u;
      x = static_cast<float>(state >> 24);
    }
    int64_t shape[] = {1, 3, 640, 640};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape, 4);
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};
    std::vector<Ort::AllocatedStringPtr> owners;
    std::vector<const char*> names;
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
      owners.push_back(session.GetOutputNameAllocated(i, allocator));
      names.push_back(owners.back().get());
    }
    auto run = [&]() { return session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, names.data(), names.size()); };
    for (int i = 0; i < 3; ++i) run();
    // Three fixed batches; construction, warmup and evidence writing excluded.
    for (int round = 0; round < 3; ++round) {
      const auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < 10; ++i) run();
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()/10.;
      std::cout << "elapsed_ms=" << std::setprecision(12) << ms << '\n';
    }
    auto outputs = run();
    size_t scalars = 0;
    std::ofstream file;
    if (argc == 2) {
      std::filesystem::create_directories(argv[1]);
      file.open(std::filesystem::path(argv[1])/"outputs.f32", std::ios::binary);
      if (!file) throw std::runtime_error("cannot create output evidence");
    } else if (argc != 1) throw std::runtime_error("usage: ort-native-probe [evidence-directory]");
    double checksum = 0;
    for (auto& value : outputs) {
      auto info = value.GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw std::runtime_error("non-f32 output");
      const auto n = info.GetElementCount();
      const float* data = value.GetTensorData<float>();
      for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(data[i])) throw std::runtime_error("nonfinite output");
        checksum += data[i];
      }
      if (file.is_open()) file.write(reinterpret_cast<const char*>(data), n*sizeof(float));
      scalars += n;
    }
    if (!scalars || !std::isfinite(checksum)) throw std::runtime_error("invalid output summary");
    std::cout << "{\"ort\":\"" << OrtGetApiBase()->GetVersionString() << "\",\"outputs\":" << outputs.size()
              << ",\"scalars\":" << scalars << ",\"checksum\":" << std::setprecision(17) << checksum
              << ",\"embedded_model_bytes\":" << sizeof(kModel) << "}\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
