
// 设计内存池，双向循环队列请求释放


#include "core/buffer.h"
#include <cstddef>
#include <cassert>
#include <memory.h>

using namespace gddeploy;

namespace gddeploy {

}

static inline void* DataOffset(void* data, size_t offset) noexcept {
  return reinterpret_cast<void*>(reinterpret_cast<int64_t>(data) + offset);
}

Buffer::Buffer(int memory_size)
{
  data_ = std::make_shared<Memory>(nullptr, -1, [](void* memory, int /*unused*/) {
  });
  memory_size_ = memory_size;
}

/**
 * @brief Construct a new Buffer object contained MLU memory
 *
 * @param memory_size Memory size in bytes
 * @param device_id memory on which device
 */
Buffer::Buffer(int memory_size, int device_id)
{
  data_ = std::make_shared<Memory>(nullptr, device_id, [](void* memory, int device_id) {
  });
  memory_size_ = memory_size;
}

Buffer::Buffer(void* cpu_memory, size_t memory_size, MemoryDeallocator d) : memory_size_(memory_size) {
//   if (!cpu_memory || !memory_size) {
//     THROW_EXCEPTION(Exception::INVALID_ARG, "[EasyDK InferServer] [Buffer] Memory cannot be empty");
//   }
  data_ = std::make_shared<Memory>(cpu_memory, -1, std::move(d));
  type_ = MemoryType::E_MEMORY_TYPE_CPU;
  memory_size_ = memory_size;
}

/**
 * @brief Construct a new Buffer object with raw MLU memory
 *
 * @param mlu_memory raw pointer
 * @param memory_size Memory size in bytes
 * @param d A function to handle memory when destruct
 * @param device_id memory on which device
 */
//   Buffer(void *mlu_memory, int memory_size, MemoryDeallocator d, int device_id)
//   {}

/**
 * @brief Construct a new Buffer object with raw CPU memory
 *
 * @param cpu_memory raw pointer
 * @param memory_size Memory size in bytes
 * @param d A function to handle memory when destruct
 */
//   Buffer(void *cpu_memory, int memory_size, MemoryDeallocator d)
//   {

//   }

/**
 * @brief Get a shallow copy of buffer by offset
 *
 * @param offset offset
 * @return copied buffer
 */
Buffer Buffer::operator()(int offset) const
{
	if (offset + this->offset_ >= memory_size_) {
		printf("[EasyDK InferServer] [Buffer] Offset out of range");
	}
	Buffer buf;
	buf.data_ = this->data_;
	buf.type_ = this->type_;
	buf.memory_size_ = this->memory_size_;
	buf.offset_ = this->offset_ + offset;
	return buf;
}

void Buffer::LazyMalloc()
{
	assert(memory_size_);
  	assert(data_);
	if (!data_->data) {
		data_->data = malloc(memory_size_);
	}
}

/**
 * @brief Get mutable raw pointer
 *
 * @return raw pointer
 */
void *Buffer::MutableData()
{
	LazyMalloc();
  	return DataOffset(data_->data, offset_);
}

/**
 * @brief Get const raw pointer
 *
 * @return raw pointer
 */
const void *Buffer::Data() const
{
	return DataOffset(data_->data, offset_);
}

/**
 * @brief Get device id
 *
 * @return device id
 */
int Buffer::DeviceId() const noexcept
{
	return 0;
}


/**
 * @brief query whether Buffer own memory
 *
 * @retval true own memory
 * @retval false not own memory
 */
bool Buffer::OwnMemory() const noexcept
{
	return false;
}

/**
 * @brief Copy data from raw CPU memory
 *
 * @param cpu_src Copy source, data on CPU
 * @param copy_size Memory size in bytes
 */
void Buffer::CopyFrom(void *cpu_src, int copy_size)
{
	LazyMalloc();

	if (type_ == MemoryType::E_MEMORY_TYPE_CPU) {
		memcpy(DataOffset(data_->data, offset_), cpu_src, copy_size);
	}
}

/**
 * @brief Copy data from another buffer
 *
 * @param src Copy source
 * @param copy_size Memory size in bytes
 */
void Buffer::CopyFrom(const Buffer &src, int copy_size)
{
	
}

/**
 * @brief Copy data to raw CPU memory
 *
 * @param cpu_dst Copy destination, memory on CPU
 * @param copy_size Memory size in bytes
 */
void Buffer::CopyTo(void *cpu_dst, int copy_size) const
{
	if (type_ == MemoryType::E_MEMORY_TYPE_CPU) {
		memcpy(cpu_dst, DataOffset(data_->data, offset_), copy_size);
	}
}

/**
 * @brief Copy data to another buffer
 *
 * @param dst Copy source
 * @param copy_size Memory size in bytes
 */
void Buffer::CopyTo(Buffer *dst, int copy_size) const
{

}


/* -------- CPUMemoryPool -----------*/
CPUMemoryPool::CPUMemoryPool(size_t memory_size, size_t max_buffer_num, int device_id)
    : memory_size_(memory_size), max_buffer_num_(max_buffer_num), buffer_num_(0), device_id_(device_id) {
  running_.store(true);
}

CPUMemoryPool::~CPUMemoryPool() {
	running_.store(false);
	size_t remain_memory = buffer_num_;

	std::unique_lock<std::mutex> lk(q_mutex_);
	while (remain_memory) {
		if (cache_.empty()) {
			empty_cond_.wait(lk, [this]() { return !cache_.empty(); });
		}

		free(cache_.front());
		--remain_memory;
	}
}

Buffer CPUMemoryPool::Request(int timeout_ms) {
	std::unique_lock<std::mutex> lk(q_mutex_);
	if (cache_.empty()) {
		if (buffer_num_ < max_buffer_num_) {
			void* data{nullptr};
			data = (void *)malloc(memory_size_);
			cache_.push(data);
			++buffer_num_;
		} else {
			auto not_empty = [this]() { return !cache_.empty(); };
			if (timeout_ms >= 0) {
				empty_cond_.wait_for(lk, std::chrono::milliseconds(timeout_ms), not_empty);
			} else {
				empty_cond_.wait(lk, not_empty);
			}
		}
	}

	void* m = cache_.front();
	cache_.pop();
	return Buffer(m, memory_size_,
				[this](void* m, int /*unused*/) {
					std::unique_lock<std::mutex> lk(q_mutex_);
					cache_.push(m);
					empty_cond_.notify_one();
				});
}
/* -------- CPUMemoryPool END -----------*/