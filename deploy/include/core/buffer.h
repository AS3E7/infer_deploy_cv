/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
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

/**
 * @file buffer.h
 *
 * This file contains a declaration of the Buffer class.
 */

#ifndef GDDEPLOY_BUFFER_H_
#define GDDEPLOY_BUFFER_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>

namespace gddeploy {

/**
 * @brief Enumerator of memory type
 */
enum class MemoryType {
	E_MEMORY_TYPE_CPU = 0,  ///< memory on CPU
	E_MEMORY_TYPE_MLU = 1,  ///< memory on MLU
	E_MEMORY_TYPE_BMNN = 2,  ///< memory on BMNN
};

class Memory;

class Buffer {
public:
	/// callback function to deallocate memory
	using MemoryDeallocator = std::function<void(void *memory, int device_id)>;

	/**
	 * @brief Construct a new Buffer object contained CPU memory
	 *
	 * @param memory_size Memory size in bytes
	 */
	explicit Buffer(int memory_size);

	/**
	 * @brief Construct a new Buffer object contained MLU memory
	 *
	 * @param memory_size Memory size in bytes
	 * @param device_id memory on which device
	 */
	explicit Buffer(int memory_size, int device_id);

	/**
	 * @brief Construct a new Buffer object with raw CPU memory
	 *
	 * @param cpu_memory raw pointer
	 * @param memory_size Memory size in bytes
	 * @param d A function to handle memory when destruct
	 */
	Buffer(void *cpu_memory, size_t memory_size, MemoryDeallocator d);

	/**
	 * @brief Construct a new Buffer object with raw MLU memory
	 *
	 * @param mlu_memory raw pointer
	 * @param memory_size Memory size in bytes
	 * @param d A function to handle memory when destruct
	 * @param device_id memory on which device
	 */
	Buffer(void *mlu_memory, int memory_size, MemoryDeallocator d, int device_id){}

	/**
	 * @brief Construct a new Buffer object with raw CPU memory
	 *
	 * @param cpu_memory raw pointer
	 * @param memory_size Memory size in bytes
	 * @param d A function to handle memory when destruct
	 */
	// Buffer(void *cpu_memory, int memory_size, MemoryDeallocator d);

	/**
	 * @brief default constructor
	 *
	 * @warning generated Buffer cannot be used until assigned
	 */
	Buffer() = default;

	/**
	 * @brief default copy constructor (shallow)
	 */
	Buffer(const Buffer &another) = default;

	/**
	 * @brief default copy assign (shallow)
	 */
	Buffer &operator=(const Buffer &another) = default;

	/**
	 * @brief default move construct
	 */
	Buffer(Buffer &&another) = default;

	/**
	 * @brief default move assign
	 */
	Buffer &operator=(Buffer &&another) = default;

	/**
	 * @brief Get a shallow copy of buffer by offset
	 *
	 * @param offset offset
	 * @return copied buffer
	 */
	Buffer operator()(int offset) const;

	/**
	 * @brief Get mutable raw pointer
	 *
	 * @return raw pointer
	 */
	virtual void *MutableData();

	/**
	 * @brief Get const raw pointer
	 *
	 * @return raw pointer
	 */
	virtual const void *Data() const;

	/**
	 * @brief Get size of MLU memory
	 *
	 * @return memory size in bytes
	 */
	virtual int MemorySize() const noexcept { return memory_size_ - offset_; }

	/**
	 * @brief Get device id
	 *
	 * @return device id
	 */
	virtual int DeviceId() const noexcept;

	/**
	 * @brief Get memory type
	 *
	 * @return memory type
	 */
	virtual MemoryType Type() const noexcept { return type_; }

	/**
	 * @brief Query whether memory is on MLU
	 *
	 * @retval true memory on MLU
	 * @retval false memory on CPU
	 */
	virtual bool OnDevice() const noexcept { return type_ == MemoryType::E_MEMORY_TYPE_MLU; }

	/**
	 * @brief query whether Buffer own memory
	 *
	 * @retval true own memory
	 * @retval false not own memory
	 */
	virtual bool OwnMemory() const noexcept;

	/**
	 * @brief Copy data from raw CPU memory
	 *
	 * @param cpu_src Copy source, data on CPU
	 * @param copy_size Memory size in bytes
	 */
	virtual void CopyFrom(void *cpu_src, int copy_size);

	/**
	 * @brief Copy data from another buffer
	 *
	 * @param src Copy source
	 * @param copy_size Memory size in bytes
	 */
	virtual void CopyFrom(const Buffer &src, int copy_size);

	/**
	 * @brief Copy data to raw CPU memory
	 *
	 * @param cpu_dst Copy destination, memory on CPU
	 * @param copy_size Memory size in bytes
	 */
	virtual void CopyTo(void *cpu_dst, int copy_size) const;

	/**
	 * @brief Copy data to another buffer
	 *
	 * @param dst Copy source
	 * @param copy_size Memory size in bytes
	 */
	virtual void CopyTo(Buffer *dst, int copy_size) const;

private:
	void LazyMalloc();
	std::shared_ptr<Memory> data_{nullptr};

	int memory_size_{0};
	int offset_{0};

	MemoryType type_{MemoryType::E_MEMORY_TYPE_CPU};
};


/**
 * @brief CPUMemoryPool is a MLU memory helper class.
 *
 * @note It provides a easy way to manage memory on MLU.
 */
class CPUMemoryPool {
public:
	/**
	 * @brief Construct a new Mlu Memory Pool object
	 *
	 * @param memory_size Memory size in bytes
	 * @param max_buffer_num max number of memory cached in pool
	 * @param device_id memory on which device
	 */
	CPUMemoryPool(size_t memory_size, size_t max_buffer_num, int device_id = 0);

	/**
	 * @brief A destructor
	 * @note wait until all MluMemory requested is released
	 */
	~CPUMemoryPool();

	/**
	 * @brief Request Buffer from pool, wait for timeout_ms if pool is empty
	 *
	 * @param timeout_ms wait timeout in milliseconds
	 * @return a Buffer
	 */
	virtual Buffer Request(int timeout_ms = -1);

	/**
	 * @brief Get size of MLU memory
	 *
	 * @return memory size in bytes
	 */
	size_t MemorySize() const noexcept { return memory_size_; }

	/**
	 * @brief Get how many pieces of MLU memory cached
	 *
	 * @return number of memory cached
	 */
	size_t BufferNum() const noexcept { return buffer_num_; }

	/**
	 * @brief Get device id
	 *
	 * @return device id
	 */
	int DeviceId() const noexcept { return device_id_; }

private:
	std::queue<void *> cache_;
	std::mutex q_mutex_;
	std::condition_variable empty_cond_;
	size_t memory_size_;
	size_t max_buffer_num_;
	size_t buffer_num_;
	int device_id_;
	std::atomic<bool> running_{false};
};

class Memory {
public:
	Memory(void* _data, int _device_id, Buffer::MemoryDeallocator&& _deallocator)
		: data(_data), deallocator(std::forward<Buffer::MemoryDeallocator>(_deallocator)), device_id(_device_id) {}
	~Memory() {
	if (data && deallocator) {
		deallocator(data, device_id);
	}
	}
	Memory(const Memory&) = delete;
	Memory& operator=(const Memory&) = delete;
	void* data{nullptr};
	Buffer::MemoryDeallocator deallocator{nullptr};
	int device_id{-1};
};

}  // namespace gddeploy

#endif  // GDDEPLOY_BUFFER_H_
