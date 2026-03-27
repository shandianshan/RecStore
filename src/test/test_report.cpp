#include <gtest/gtest.h>
#include "base/report/report_client.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <string>
#include <iostream>

using namespace std::chrono_literals;

namespace {

std::mutex& EnvTestMutex() {
  static std::mutex mutex;
  return mutex;
}

class ScopedEnvVar {
public:
  ScopedEnvVar(const char* key, const char* value) : key_(key) {
    const char* original = std::getenv(key_);
    if (original != nullptr) {
      had_original_ = true;
      original_value_ = original;
    }

    if (value != nullptr) {
      setenv(key_, value, 1);
    } else {
      unsetenv(key_);
    }
  }

  ~ScopedEnvVar() {
    if (had_original_) {
      setenv(key_, original_value_.c_str(), 1);
    } else {
      unsetenv(key_);
    }
  }

private:
  const char* key_;
  bool had_original_ = false;
  std::string original_value_;
};

class ScopedCurrentPath {
public:
  explicit ScopedCurrentPath(const std::filesystem::path& path)
      : original_(std::filesystem::current_path()) {
    std::filesystem::current_path(path);
  }

  ~ScopedCurrentPath() { std::filesystem::current_path(original_); }

private:
  std::filesystem::path original_;
};

}  // namespace

TEST(ReportTest, TimelineBasic) {
  {
    ReportTimeline tl("cpp_easy_test_timeline_map", "timeline_test_2");
    std::cout << "Running phase ... " << std::endl;
    std::this_thread::sleep_for(1s);
  }

  std::this_thread::sleep_for(500ms);
  SUCCEED();
}

TEST(ReportTest, ReportModeCanBeDisabledByEnvVar) {
  std::lock_guard<std::mutex> lock(EnvTestMutex());
  ScopedEnvVar report_mode("RECSTORE_REPORT_MODE", "local");

  EXPECT_FALSE(is_report_remote_enabled_for_test());
  EXPECT_TRUE(report("cpp_easy_test_report_map",
                     "env_local_test",
                     "duration_us",
                     1.0));
}

TEST(ReportTest, EmptyReportApiDisablesRemoteReporting) {
  std::lock_guard<std::mutex> lock(EnvTestMutex());
  ScopedEnvVar report_mode("RECSTORE_REPORT_MODE", nullptr);

  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() /
      "recstore_test_report_empty_api";
  std::filesystem::create_directories(temp_dir);

  const std::filesystem::path config_path = temp_dir / "recstore_config.json";
  std::ofstream ofs(config_path);
  ofs << "{\n"
         "  \"report_API\": \"\"\n"
         "}\n";
  ofs.close();

  ScopedCurrentPath cwd_guard(temp_dir);
  EXPECT_FALSE(is_report_remote_enabled_for_test());
  EXPECT_TRUE(report("cpp_easy_test_report_map",
                     "config_local_test",
                     "duration_us",
                     2.0));

  std::filesystem::remove(config_path);
  std::filesystem::remove(temp_dir);
}

TEST(ReportTest, FlameGraphData) {
  FlameGraphData root     = {"main", 0.0, 0, 100.0, 10.0};
  FlameGraphData child1   = {"do_work", 10.0, 1, 60.0, 20.0};
  FlameGraphData child2   = {"do_io", 70.0, 1, 30.0, 30.0};
  FlameGraphData child1_1 = {"compute", 20.0, 2, 40.0, 40.0};

  report_flame_graph("cpp_easy_test_flame_map", "flame_test_2", root);
  report_flame_graph("cpp_easy_test_flame_map", "flame_test_2", child1);
  report_flame_graph("cpp_easy_test_flame_map", "flame_test_2", child2);
  report_flame_graph("cpp_easy_test_flame_map", "flame_test_2", child1_1);

  std::this_thread::sleep_for(500ms);
  SUCCEED();
}
