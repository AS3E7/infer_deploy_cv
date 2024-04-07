/*************************************************************************
 * Copyright (C) [2020] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#ifndef GDDEPLOY_API_H_
#define GDDEPLOY_API_H_

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "buffer.h"
#include "opencv2/core/hal/interface.h"
#include "shape.h"
#include "util/any.h"
#include "util/base_object.h"
// #include "edk_config.h"

namespace gddeploy
{

  /**
   * @brief Enumeration to specify data type of model input and output
   */
  enum class DataType
  {
    INT8,
    UINT8,
    FLOAT32,
    FLOAT16,
    UINT16,
    INT16,
    INT32,
    INVALID
  };

  /**
   * @brief Enumeration to specify dim order of model input and output
   */
  enum class DimOrder
  {
    NCHW,
    NHWC,
    HWCN,
    TNC,
    NTC
  };

  /**
   * @brief Describe data layout on MLU or CPU
   */
  struct DataLayout
  {
    DataType dtype; ///< @see DataType
    DimOrder order; ///< @see DimOrder
  };

  /**
   * @brief Get size in bytes of type
   *
   * @param type Data type enumeration
   * @return size_t size of specified type
   */
  size_t GetTypeSize(DataType type) noexcept;

  /**
   * @brief An enum describes InferServer request return values.
   */
  enum class Status
  {
    SUCCESS = 0,         ///< The operation was successful
    ERROR_READWRITE = 1, ///< Read / Write file failed
    ERROR_MEMORY = 2,    ///< Memory error, such as out of memory, memcpy failed
    INVALID_PARAM = 3,   ///< Invalid parameters
    WRONG_TYPE = 4,      ///< Invalid data type in `any`
    ERROR_BACKEND = 5,   ///< Error occurred in processor
    NOT_IMPLEMENTED = 6, ///< Function not implemented
    TIMEOUT = 7,         ///< Time expired
    STATUS_COUNT = 8,    ///< Number of status
    INVALID_MODEL = 9,   ///< Invalid model
  };

  /**
   * @brief An enum describes batch strategy
   */
  enum class BatchStrategy
  {
    DYNAMIC = 0,        ///< Cross-request batch
    STATIC = 1,         ///< In-request batch
    SEQUENCE = 2,       ///< Sequence model, unsupported for now
    STRATEGY_COUNT = 3, ///< Number of strategy
  };

  /**
   * @brief Convert BatchStrategy to string
   *
   * @param strategy batch strategy
   * @return std::string Stringified batch strategy
   */
  std::string ToString(BatchStrategy strategy) noexcept;

  /**
   * @brief Put BatchStrategy into ostream
   *
   * @param os ostream
   * @param s BatchStrategy
   * @return std::ostream& ostream
   */
  inline std::ostream &operator<<(std::ostream &os, BatchStrategy s) { return os << ToString(s); }

  /**
   * @brief Set current deivce for this thread
   *
   * @param device_id device id
   *
   * @retval true success
   * @retval false set device failed
   */
  bool SetCurrentDevice(int device_id) noexcept;

  /**
   * @brief Check whether device is accessible
   *
   * @param device_id device id
   *
   * @retval true device is accessible
   * @retval false no such device
   */
  bool CheckDevice(int device_id) noexcept;

  /**
   * @brief Get total device count
   *
   * @retval device count
   */
  uint32_t TotalDeviceCount() noexcept;

  /**
   * @brief Model interface
   */
  class ModelInfo
  {
  public:
    virtual ~ModelInfo() = default;

    // ----------- Observers -----------

    /**
     * @brief Get input shape
     *
     * @param index index of input
     * @return const Shape& shape of specified input
     */
    virtual const Shape &InputShape(int index) const noexcept = 0;

    /**
     * @brief Get output shape
     *
     * @param index index of output
     * @return const Shape& shape of specified output
     */
    virtual const Shape &OutputShape(int index) const noexcept = 0;
    /**
     * @brief Check if output shapes are fixed
     *
     * @return Returns true if all output shapes are fixed, otherwise returns false.
     */
    virtual int FixedOutputShape() noexcept = 0;

    /**
     * @brief Get input layout on MLU
     *
     * @param index index of input
     * @return const DataLayout& data layout of specified input
     */
    virtual const DataLayout &InputLayout(int index) const noexcept = 0;

    /**
     * @brief Get output layout on MLU
     *
     * @param index index of output
     * @return const DataLayout& data layout of specified output
     */
    virtual const DataLayout &OutputLayout(int index) const noexcept = 0;

    /**
     * @brief Get number of input
     *
     * @return uint32_t number of input
     */
    virtual uint32_t InputNum() const noexcept = 0;

    /**
     * @brief Get number of output
     *
     * @return uint32_t number of output
     */
    virtual uint32_t OutputNum() const noexcept = 0;

    /**
     * @brief Get number of input
     *
     * @return uint32_t number of input
     */
    virtual std::vector<float> InputScale() const noexcept = 0;

    /**
     * @brief Get number of output
     *
     * @return uint32_t number of output
     */
    virtual std::vector<float> OutputScale() const noexcept = 0;

    /**
     * @brief Get Zeropoint of input
     *
     * @return uint32_t number of input
     */
    virtual std::vector<int> InputZp() const noexcept { return std::vector<int>{0, 0, 0}; }

    /**
     * @brief Get number of output
     *
     * @return uint32_t number of output
     */
    virtual std::vector<int> OutputZp() const noexcept { return std::vector<int>{0, 0, 0};}

    /**
     * @brief Get model batch size
     *
     * @return uint32_t batch size
     */
    virtual uint32_t BatchSize() const noexcept = 0;

    /**
     * @brief Get model key
     *
     * @return const std::string& model key
     */
    virtual std::string GetKey() const noexcept = 0;

    // ----------- Observers End -----------
  }; // class ModelInfo

  using ModelInfoPtr = std::shared_ptr<ModelInfo>;

  class RequestControl;

  struct FrameInfo
  {
    uint64_t frame_id;
    int stream_id;
    int width;
    int height;
  }; // struct FrameInfo

  /**
   * @brief Inference data unit
   */
  struct InferData
  {
    /**
     * @brief Set any data into inference data
     *
     * @tparam T data type
     * @param v data value
     */
    template <typename T>
    void Set(T &&v)
    {
      data = std::forward<T>(v);
    }

    /**
     * @brief Get data by value
     *
     * @tparam T data type
     * @return std::remove_reference<T>::type a copy of data
     */
    template <typename T>
    typename std::remove_reference<T>::type Get() const
    {
      return any_cast<typename std::remove_reference<T>::type>(data);
    }

    /**
     * @brief Get data by lvalue reference
     *
     * @tparam T data type
     * @return std::add_lvalue_reference<T>::type lvalue reference to data
     */
    template <typename T>
    typename std::add_lvalue_reference<T>::type GetLref() &
    {
      return any_cast<typename std::add_lvalue_reference<T>::type>(data);
    }

    /**
     * @brief Get data by const lvalue reference
     *
     * @tparam T data type
     * @return std::add_lvalue_reference<typename std::add_const<T>::type>::type const lvalue reference to data
     */
    template <typename T>
    typename std::add_lvalue_reference<typename std::add_const<T>::type>::type GetLref() const &
    {
      return any_cast<typename std::add_lvalue_reference<typename std::add_const<T>::type>::type>(data);
    }

    /**
     * @brief Check if InferData has value
     *
     * @retval true InferData has value
     * @retval false InferData does not have value
     */
    bool HasValue() noexcept
    {
      return data.has_value();
    }

    /**
     * @brief Set user data for postprocess
     *
     * @tparam T data type
     * @param v data value
     */
    template <typename T>
    void SetUserData(T &&v)
    {
      user_data = std::forward<T>(v);
    }

    /**
     * @brief Get user data by value
     *
     * @note if T is lvalue reference, data is returned by lvalue reference.
     *       if T is bare type, data is returned by value.
     * @tparam T data type
     * @return data
     */
    template <typename T>
    T GetUserData() const
    {
      return any_cast<T>(user_data);
    }

    /**
     * @brief Set user data for postprocess
     *
     * @tparam T data type
     * @param v data value
     */
    template <typename T>
    void SetFrameData(T &&v)
    {
      frame_data = std::forward<T>(v);
    }

    /**
     * @brief Get user data by value
     *
     * @note if T is lvalue reference, data is returned by lvalue reference.
     *       if T is bare type, data is returned by value.
     * @tparam T data type
     * @return data
     */
    template <typename T>
    T GetFrameData() const
    {
      return any_cast<T>(frame_data);
    }

    /**
     * @brief Set user data for postprocess
     *
     * @tparam T data type
     * @param v data value
     */
    template <typename T>
    void SetMetaData(T &&v)
    {
      meta_data = std::forward<T>(v);
    }

    /**
     * @brief Get user data by value
     *
     * @note if T is lvalue reference, data is returned by lvalue reference.
     *       if T is bare type, data is returned by value.
     * @tparam T data type
     * @return data
     */
    template <typename T>
    T GetMetaData() const
    {
      return any_cast<T>(meta_data);
    }

    bool HasMetaValue() noexcept
    {
      return meta_data.has_value();
    }

    /// stored data
    any data;
    /// user data passed to postprocessor
    any user_data;
    /// passed to postprocessor
    any frame_data;
    /// stored metadata, include all result of per infer
    any meta_data;
    /// private member
    RequestControl *ctrl{nullptr};
    /// private member
    uint32_t index{0};
  };

  using InferDataPtr = std::shared_ptr<InferData>;
  using BatchData = std::vector<InferDataPtr>;

  // TODO: 这里需要修改，保存推理的结果，作为第二模型使用
  /**
   * @brief Data package, used in request and response
   */
  struct Package
  {
    /// a batch of data, origin data
    BatchData data;

    /// private member, intermediate storage，可能会作为前处理和推理后数据临时存储
    InferDataPtr predict_io{nullptr};

    /// tag of this package (such as stream_id, client ip, etc.)
    std::string tag;

    /// perf statistics of one request
    std::map<std::string, float> perf;

    /// private member
    int64_t priority;

    static std::shared_ptr<Package> Create(uint32_t data_num, const std::string &tag = "") noexcept
    {
      auto ret = std::make_shared<Package>();
      ret->data.reserve(data_num);
      for (uint32_t idx = 0; idx < data_num; ++idx)
      {
        ret->data.emplace_back(std::shared_ptr<InferData>(new InferData));
      }
      ret->tag = tag;
      return ret;
    }
  };
  using PackagePtr = std::shared_ptr<Package>;

  /**
   * @brief Processor interface
   */
  class Processor : public BaseObject
  {
  public:
    Processor() = default;
    /**
     * @brief Construct a new Processor object
     *
     * @param type_name type name of derived processor
     */
    Processor(const std::string &type_name) noexcept : type_name_(type_name) {}

    /**
     * @brief Get type name of processor
     *
     * @return const std::string& type name
     */
    const std::string &TypeName() const noexcept { return type_name_; }

    /**
     * @brief Destroy the Processor object
     */
    virtual ~Processor() = default;

    /**
     * @brief Initialize processor
     *
     * @retval Status::SUCCESS Init succeeded
     * @retval other Init failed
     */
    virtual Status Init() noexcept = 0;

    /**
     * @brief Process data in package
     *
     * @param data Processed data
     * @retval Status::SUCCESS Process succeeded
     * @retval other Process failed
     */
    virtual Status Process(PackagePtr data) noexcept = 0;

    /**
     * @brief Fork an initialized processor which have the same params as this
     *
     * @return std::shared_ptr<Processor> A new processor
     */
    virtual std::shared_ptr<Processor> Fork() = 0;

  private:
    friend class TaskNode;
    std::unique_lock<std::mutex> Lock() noexcept { return std::unique_lock<std::mutex>(process_lock_); }
    std::string type_name_;
    std::mutex process_lock_;
  }; // class Processor
  using ProcessorPtr = std::shared_ptr<Processor>;

  /**
   * @brief A convenient CRTP template provided `Fork` and `Create` function
   *
   * @tparam T Type of derived class
   */
  template <typename T>
  class ProcessorForkable : public Processor
  {
  public:
    ProcessorForkable() = default;

    /**
     * @brief Construct a new Processor Forkable object
     *
     * @param type_name type name of derived processor
     */
    explicit ProcessorForkable(const std::string &type_name) noexcept : Processor(type_name) {}

    /**
     * @brief Destroy the Processor Forkable object
     */
    virtual ~ProcessorForkable() = default;

    /**
     * @brief Fork an initialized processor which have the same params as this
     *
     * @return std::shared_ptr<Processor> A new processor
     */
    virtual std::shared_ptr<Processor> Fork()
    {
      auto p = std::make_shared<T>();
      p->CopyParamsFrom(*this);
      if (p->Init() != Status::SUCCESS)
        return nullptr;
      return p;
    }

    /**
     * @brief Create a processor
     *
     * @return std::shared_ptr<T> A new processor
     */
    static std::shared_ptr<T> Create() noexcept(std::is_nothrow_default_constructible<T>::value)
    {
      return std::make_shared<T>();
    }
  };

  /**
   * @brief Base class of response observer, only used for async Session
   */
  class Observer
  {
  public:
    /**
     * @brief Notify the observer one response
     *
     * @param status Request status code
     * @param data Response data
     * @param user_data User data
     */
    virtual void Response(Status status, PackagePtr data, any user_data) noexcept = 0;

    /**
     * @brief Destroy the Observer object
     */
    virtual ~Observer() = default;
  };

  /**
   * @brief A struct to describe execution graph
   */
  struct SessionDesc
  {
    /// session name, distinct session in log
    std::string name{};
    /// model pointer
    ModelInfoPtr model{nullptr};
    /// batch strategy
    BatchStrategy strategy{BatchStrategy::DYNAMIC};
    /**
     * @brief host input data layout, work when input data is on cpu
     *
     * @note built-in processor will transform data from host input layout into MLU input layout
     *       ( @see ModelInfo::InputLayout(int index) ) automatically before infer
     */
    DataLayout host_input_layout{DataType::UINT8, DimOrder::NHWC};
    /**
     * @brief host output data layout
     *
     * @note built-in processor will transform from MLU output layout ( @see ModelInfo::OutputLayout(int index) )
     *       into host output layout automatically after infer
     */
    DataLayout host_output_layout{DataType::FLOAT32, DimOrder::NHWC};
    /// preprocessor
    std::shared_ptr<Processor> preproc{nullptr};
    /// postprocessor
    std::shared_ptr<Processor> postproc{nullptr};
    /// timeout in milliseconds, only work for BatchStrategy::DYNAMIC
    uint32_t batch_timeout{100};
    /// Session request priority
    int priority{0};
    /**
     * @brief engine number
     *
     * @note multi engine can boost process, but will take more MLU resources
     */
    uint32_t engine_num{1};
    /// whether print performance
    bool show_perf{true};
  };

  /**
   * @brief Latency statistics
   */
  struct LatencyStatistic
  {
    /// Total processed unit count
    uint32_t unit_cnt{0};
    /// Total recorded value
    double total{0};
    /// Maximum value of one unit
    float max{0};
    /// Minimum value of one unit
    float min{std::numeric_limits<float>::max()};
  };

  /**
   * @brief Throughout statistics
   */
  struct ThroughoutStatistic
  {
    /// total request count
    uint32_t request_cnt{0};
    /// total unit cnt
    uint32_t unit_cnt{0};
    /// request per second
    float rps{0};
    /// unit per second
    float ups{0};
    /// real time rps
    float rps_rt{0};
    /// real time ups
    float ups_rt{0};
  };

  /// A structure describes linked session of server
  class Session;
  /// pointer to Session
  using Session_t = Session *;


  static int GetMD5String(const unsigned char *data, unsigned int len, char *md5_str);

  class InferServerPrivate;
  /**
   * @brief Inference server api class
   */
  class InferServer
  {
  public:
    /**
     * @brief Construct a new Infer Server object
     *
     * @param device_id Specified MLU device ID
     */
    explicit InferServer(int device_id) noexcept;

    /* ------------------------- Request API -------------------------- */

    /**
     * @brief Create a Session
     *
     * @param desc Session description
     * @param observer Response observer
     * @return Session_t a Session
     */
    Session_t CreateSession(SessionDesc desc, std::shared_ptr<Observer> observer) noexcept;

    /**
     * @brief Create a synchronous Session
     *
     * @param desc Session description
     * @return Session_t a Session
     */
    Session_t CreateSyncSession(SessionDesc desc) noexcept { return CreateSession(desc, nullptr); }

    /**
     * @brief Destroy session
     *
     * @param session a Session
     * @retval true Destroy succeeded
     * @retval false session does not belong to this server
     */
    bool DestroySession(Session_t session) noexcept;

    /**
     * @brief send a inference request
     *
     * @warning async api, can be invoked with async Session only.
     *
     * @param session link handle
     * @param input input package
     * @param user_data user data
     * @param timeout timeout threshold (milliseconds), -1 for endless
     */
    bool Request(Session_t session, PackagePtr input, any user_data, int timeout = -1) noexcept;

    /**
     * @brief send a inference request and wait for response
     *
     * @warning synchronous api, can be invoked with synchronous Session only.
     *
     * @param session session
     * @param input input package
     * @param status execute status
     * @param response output result
     * @param timeout timeout threshold (milliseconds), -1 for endless
     */
    bool RequestSync(Session_t session, PackagePtr input, Status *status, PackagePtr response, int timeout = -1) noexcept;

    /**
     * @brief Wait task with specified tag done, @see Package::tag
     *
     * @note Usually used at EOS
     *
     * @param session a Session
     * @param tag specified tag
     */
    void WaitTaskDone(Session_t session, const std::string &tag) noexcept;

    /**
     * @brief Discard task with specified tag done, @see Package::tag
     *
     * @note Usually used when you need to stop the process as soon as possible
     * @param session a Session
     * @param tag specified tag
     */
    void DiscardTask(Session_t session, const std::string &tag) noexcept;

    /**
     * @brief Get model from session
     *
     * @param session a Session
     * @return ModelInfoPtr A model
     */
    ModelInfoPtr GetModel(Session_t session) noexcept;

    /* --------------------- Model API ----------------------------- */

    /**
     * @brief Set directory to save downloaded model file
     *
     * @param model_dir model directory
     * @retval true Succeeded
     * @retval false Model not exist
     */
    static bool SetModelDir(const std::string &model_dir) noexcept;

    /**
     * @brief Load model from uri, model won't be loaded again if it is already in cache
     *
     * @note support download model from remote by HTTP, HTTPS, FTP, while compiled with flag `WITH_CURL`,
     *       use uri such as `../../model_file`, or "https://someweb/model_file"
     * @param pattern1 offline model uri
     * @param pattern2 extracted function name, work only if backend is cnrt
     * @return ModelInfoPtr A model
     */
    static ModelInfoPtr LoadModel(const std::string &pattern1, const std::string &pattern2, std::string param) noexcept;

#ifdef CNIS_USE_MAGICMIND
    /**
     * @brief Load model from memory, model won't be loaded again if it is already in cache
     *
     * @param ptr serialized model data in memory
     * @param size size of model data in memory
     * @return ModelInfoPtr A model
     */
    static ModelInfoPtr LoadModel(void *ptr, size_t size) noexcept;
#else
    /**
     * @brief Load model from memory, model won't be loaded again if it is already in cache
     *
     * @param ptr serialized model data in memory
     * @param func_name name of function to be extracted
     * @return ModelInfoPtr A model
     */
    // static ModelInfoPtr LoadModel(void* ptr, size_t mem_size, std::string param) noexcept;
#endif

    /**
     * @brief Remove model from cache, model won't be destroyed if still in use
     *
     * @param model a model
     * @return true Succeed
     * @return false Model is not in cache
     */
    static bool UnloadModel(ModelInfoPtr model) noexcept;

    /**
     * @brief Clear all the models in cache, model won't be destroyed if still in use
     */
    static void ClearModelCache() noexcept;

    /* ----------------------- Perf API ---------------------------- */
    /**
     * @brief Get the latency statistics
     *
     * @param session a session
     * @return std::map<std::string, PerfStatistic> latency statistics
     */
    std::map<std::string, LatencyStatistic> GetLatency(Session_t session) const noexcept;

    /**
     * @brief Get the performance statistics
     *
     * @param session a session
     * @return ThroughoutStatistic throughout statistic
     */
    ThroughoutStatistic GetThroughout(Session_t session) const noexcept;

    /**
     * @brief Get the throughout statistics of specified tag
     *
     * @param session a session
     * @param tag tag
     * @return ThroughoutStatistic throughout statistic
     */
    ThroughoutStatistic GetThroughout(Session_t session, const std::string &tag) const noexcept;

  private:
    InferServer() = delete;
    InferServerPrivate *priv_;
  }; // class InferServer

  template <class T>
  class Creator
  {
    virtual const char *GetName() const { return "Net"; }
    virtual std::shared_ptr<T> Create(any value) = 0;
  }; // class Creator

} // namespace gddeploy

#endif // GDDEPLOY_API_H_
