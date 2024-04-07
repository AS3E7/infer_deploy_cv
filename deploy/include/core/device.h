#pragma once

#include <string>
#include <future>

#include "core/model.h"
#include "core/infer_server.h"

namespace gddeploy{

typedef struct {
    int ip_num;
    std::string ip_type;    // 比如pre/infer/post
    std::string ip_name;    // 比如vpp/tpu/cpu/npu
} DeviceIp;

// using SingerTask = std::pair<PackagePtr, std::packaged_task<int(PackagePtr pack)>>;
// class SingerTask {
// public:
//     SingerTask() {}
//     int Run(){
//         // 跑task_func任务
//         (*task_func)();
//         return 0;
//     }

//     PackagePtr data;
//     std::shared_ptr<std::packaged_task<int()>> task_func;
// };

// 定义PriorityCmp
// struct PriorityCmp {
//     bool operator()(const SingerTask &lhs, const SingerTask &rhs) const {
//         return lhs.data->priority < rhs.data->priority;
//     }
// };
// typedef std::pair<PackagePtr, std::packaged_task<int(PackagePtr pack)>> TaskPair;

struct SingerTask {
    // std::function<int()> func;
    std::shared_ptr<std::packaged_task<int()>> task_func;
    int priority;
};

class TaskQueue {
public:
    TaskQueue() : stop_(false), queue_([](const SingerTask& lhs, const SingerTask& rhs) { 
        return lhs.priority > rhs.priority; 
    }) {
         // 创建线程
        std::thread worker_thread(&TaskQueue::WorkerLoop, this);
        // 线程分离
        worker_thread.detach();
    }

    void push(const SingerTask& task) {
        // printf("#######!TaskQueue push1\n");
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(task);
        cv_.notify_one();
        // printf("#######!TaskQueue push2\n");
    }

    bool try_pop(SingerTask& task) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        task = std::move(queue_.top());
        queue_.pop();
        return true;
    }

    void stop() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
    }

    bool empty() {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() {
        // std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void WorkerLoop() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !queue_.empty() || stop_; });
            if (stop_ && queue_.empty()) {
                return;
            }
            SingerTask task = std::move(queue_.top());
            
            // lock.unlock();
            (*task.task_func)();
            queue_.pop();
        }
    }
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::priority_queue<SingerTask, std::vector<SingerTask>, 
                        std::function<bool(const SingerTask&, const SingerTask&)>> queue_;
    bool stop_;
};

// using TaskQueue = std::priority_queue<SingerTask, std::vector<SingerTask>, PriorityCmp>;

// 定义一个worker类，每个worker包含loop一个线程，每个线程都会从优先级队列中获取任务，然后执行任务

class DeviceWorker{
public:
    // 构造函数接受队列对象
    DeviceWorker(std::string name, std::string ip_name, TaskQueue *task_queue) : 
            name_(name), ip_name_(ip_name), task_queue_(task_queue){}

    // 定义workerLoop函数，每个worker线程都会执行这个函数
    void WorkerLoop() {
        // 循环读取队列中的包，在processor中执行process函数
        while (true) {
            printf("!!!!!!!!!!!!!!!hahahah\n");
            if (stop_flag_) {
                break;
            }

            // 如果队列为空，就等待
            // if (task_queue_->empty()) {
            //     continue;
            // }
            // printf("#######!hahahah:%d\n", task_queue_->size());
            // // 从队列中取出一个包
            // SingerTask task = task_queue_->top();
            // task_queue_->pop();

            // // 如果队列不为空，就执行process函数
            // task.Run();
        }
    }

    // start函数开启线程不断读取task和执行Process函数
    void Start(){
        // 创建线程
        std::thread worker_thread(&DeviceWorker::WorkerLoop, this);
        // 线程分离
        worker_thread.detach();
    }

    // stop函数停止线程
    void Stop(){
        stop_flag_ = true;
    }

    std::string GetName(){ return name_; }

private:
    TaskQueue *task_queue_;
    std::string ip_name_;

    std::string name_;

    // 结束标志
    bool stop_flag_ = false;
};

class Device{
public:

    // static Device* Instance() noexcept{
    //     if (pInstance_ == nullptr){
    //         pInstance_ = new Pipeline();
    //     }
    //     return pInstance_;
    // }
    Device(){}
    Device(std::string name) : device_name_(name){
    }
    Device(std::string name, std::vector<DeviceIp> device_ips) {
        device_name_ = name;
        device_ips_ = device_ips;
    }
    ~Device(){
        // DestroyWorker("pre", "vpp");
    }

    virtual std::string GetDeviceSN(){ return ""; }
    static std::string GetDeviceUUID();

    std::string GetDeviceName() { return device_name_; }

    // virtual void init_handle();

    // virtual uint32_t get_device_count() ;

    // virtual std::string get_device_compute_capability(unsigned int device_id);

    // virtual std::pair<uint32_t, uint32_t> get_device_power_status(uint32_t device_id) ;

    // virtual bool get_device_memory_info(unsigned int device_id, MemoryInfo &info) ;

    // virtual std::string get_device_name(unsigned int device_id) ;

    // virtual std::string get_driver_version(unsigned int device_id);

    // virtual std::string get_device_serial_number(unsigned int device_id) ;

    // virtual std::string get_device_uuid(unsigned int device_id) ;

    // virtual bool get_devide_utilization_rate(unsigned int device_id,
    //                                          UtilizationRate &dev_utilization);

    // ------------------------------------硬件资源创建对应的worker和线程和队列相关 start------------------------------------
    // 根据自己的前处理、推理、后处理是否有加速IP，如果有的话需要注册
    // 需要表明自己的IP的名称，以及IP的数量
    std::vector<DeviceIp> GetDeviceIps() { return std::vector<DeviceIp>(); }
    int SetDeviceIps(std::vector<DeviceIp> &device_ips) {         
        device_ips_.assign(device_ips.begin(), device_ips.end());

        return 0;
    }

    // 创建workers，虚函数，根据config配置，如果不配置，则按照device_ips_创建
    void CreateWorkers(std::string config = "") {
        // 解析config，暂不解析
        // 创建所有的队列
        CreatePriorityQueues();

        // 根据device_ips_创建workers
        // for (auto & iter : device_ips_){
        //     std::string type = iter.ip_type;
        //     std::string name = iter.ip_name;
        //     int num = iter.ip_num;

        //     for (int i = 0; i < num; i++){
        //         std::string work_name = type + "_" + name + "_" + std::to_string(i);
        //         // CreateWorker(work_name, type, name);
        //     }
        // }
    }

    // 创建worker线程，全局只有一个
    void CreateWorker(std::string work_name, std::string ip_type, std::string ip_name){
        // 获取对于的优先级队列
        TaskQueue *task_queue = GetTaskQueue(work_name);
        // 创建worker
        DeviceWorker worker(work_name, ip_name, task_queue);
        worker_v_.push_back(worker);

        // 开启线程
        worker.Start();
    }

    // 销毁worker
    void DestroyWorker(std::string work_name){
        // 销毁队列，并且删除task_queue_v_中对应元素
        for (auto iter = task_queue_v_.begin(); iter != task_queue_v_.end(); iter++){
            std::string type = iter->first;
            delete iter->second;

            if (type == work_name){
                task_queue_v_.erase(iter);
                break;
            }
        }

        // 销毁worker
        for (auto & iter : worker_v_){
            if (iter.GetName() == work_name){
                iter.Stop();
            }
        }
    }

    int CreatePriorityQueues(){
        // 读取task_queue_v_，获取work_name，填充
        for (auto & iter : device_ips_){
            for (int i = 0; i < iter.ip_num; i++){
                auto work_name = iter.ip_type + "_" + iter.ip_name + "_" + std::to_string(i);
                std::pair<std::string, TaskQueue*> queue = {work_name, new TaskQueue()};
                task_queue_v_.emplace_back(queue);
            }
        }
        return 0;
    }

    // 根据ip type和name获取对于的worker所对应的优先级队列
    TaskQueue *GetTaskQueue(std::string work_name){
        // 遍历所有的优先级队列，找到对于的优先级队列
        for (auto & iter : task_queue_v_){
            std::string type = iter.first;

            if (type == work_name){
                return iter.second;
            }
        }
    }

    // 往某个worker的优先级队列中添加任务，这个任务是一个PackagePtr
    int AddTask(std::string ip_type, SingerTask &task){
        // 根据ip_type获取对于的work_name
        std::string work_name = "";
        std::vector<std::string> work_names;
        // TODO 改为轮训多个设备，查看对应的队列长度，选择队列最少的那个获取相对于队列后push task
        // int index =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        // for (auto & iter : device_ips_){
        //     for (int i = 0; i < iter.ip_num; i++){
        //         if (iter.ip_type == ip_type){
        //             // work_name = iter.ip_type + "_" + iter.ip_name + "_" + std::to_string(i);
        //             work_names.emplace_back(iter.ip_type + "_" + iter.ip_name + "_" + std::to_string(i));
        //             // iter.ip_num++;
        //             // break;
        //         }
        //     }
        // }
        // // if (work_name == "") {
        // if (work_names.size() == 0) {
        //     printf("work_name is null\n");
        //     return -1;
        // }
        // 获取对于的优先级队列
        // TaskQueue *task_queue = GetTaskQueue(work_name);
        TaskQueue *task_queue = nullptr;

        // // 循环task_queue_v_，找到对于的优先级队列，并且根据队列长度选择最短的那个
        // for (auto & wn : work_names){
        //     for (auto & iter : task_queue_v_){
        //         std::string type = iter.first;

        //         if (type == wn){
        //             if (task_queue == nullptr){
        //                 printf("work_name:%s, %d, size:%d\n", wn.c_str(), iter.second->size(), iter.second->size());
        //             } else {
        //                 printf("work_name:%s, %d, size:%d\n", wn.c_str(), task_queue->size(), iter.second->size());
        //             }

        //             if (task_queue == nullptr){
        //                 task_queue = iter.second;
        //                 work_name = wn;
        //             } else {
        //                 if (task_queue->size() > iter.second->size()){
        //                     task_queue = iter.second;
        //                     work_name = wn;
        //                 }
        //             }
        //         }
        //     }
        // }

        for (auto & iter : device_ips_){        // 根据已有注册轮训查找相同IP类型
            if (iter.ip_type == ip_type){   // 找到相同IP类型，比如都是preproc/infer/postproc
                for (int i = 0; i < iter.ip_num; i++){
                    auto wn = iter.ip_type + "_" + iter.ip_name + "_" + std::to_string(i);
                    // work_names.emplace_back(iter.ip_type + "_" + iter.ip_name + "_" + std::to_string(i));

                    for (auto & iter : task_queue_v_){  // 根据已有注册任务队列，轮训查找相同key
                        std::string type = iter.first;

                        if (type == wn){
                            // if (task_queue == nullptr){
                            //     printf("work_name:%s, %d, size:%d\n", wn.c_str(), iter.second->size(), iter.second->size());
                            // } else {
                            //     printf("work_name:%s, %d, size:%d\n", wn.c_str(), task_queue->size(), iter.second->size());
                            // }

                            if (task_queue == nullptr){ // 找到的第一个任务队列
                                task_queue = iter.second;
                                work_name = wn;
                            } else {
                                if (task_queue->size() > iter.second->size()){  // 找到任务队列长度最短的
                                    task_queue = iter.second;
                                    work_name = wn;
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }

        if (work_name == "") {
            printf("work_name is null\n");
            return -1;
        }

        // if (ip_type == "infer")
        //     printf("work_name:%s, size:%d\n", work_name.c_str(), task_queue->size());

        // 将任务加入到优先级队列中
        task_queue->push(task);
        
        return 0;
    }

    void PrintQueueSize(){
        for (auto & iter : task_queue_v_){
            printf("work_name:%s, size:%d\n", iter.first.c_str(), iter.second->size());
        }
    }

    // ------------------------------------硬件资源创建对应的worker和线程和队列相关 end------------------------------------

public:
    // 定义多个优先级队列, key 是 work_name, value 是优先级队列
    // work_name 定义推荐：pre_vpp_1, pre_vpp_2, infer_tpu_1, infer_tpu_2, post_cpu_1, post_cpu_2
    std::vector<std::pair<std::string, TaskQueue*>> task_queue_v_;

    // 定义多个worker
    std::vector<DeviceWorker> worker_v_;

    // 定义多级索引，分别是ip_type, ip_name, ip_num;
    std::vector<DeviceIp> device_ips_;

    // static Device *pInstance_;
    std::string device_name_;
};

class DeviceManager{
public: 
    DeviceManager(){}
    ~DeviceManager(){
        if (pInstance_ != nullptr){
            delete pInstance_;
            pInstance_ = nullptr;
        }
    }

    static DeviceManager* Instance() noexcept{
        if (pInstance_ == nullptr){
            pInstance_ = new DeviceManager();
        }
        return pInstance_;
    }
    Device* GetDevice() noexcept{
        if (device_creator_.size() == 0)
            return nullptr;
        
        auto tmp = device_creator_[0].second;
        return tmp.second;
    }

    Device* GetDevice(std::string manufacturer, std::string chip) noexcept{
        for (auto & iter_manu : device_creator_){
            std::string manu = iter_manu.first;
            if (manu != manufacturer)
                continue;

            std::pair<std::string, Device*> chip_creator = iter_manu.second;

            if (chip_creator.first != chip)
                continue;

            return chip_creator.second;
        }

        return nullptr;
    }

    int RegisterDevice(std::string manufacturer, std::string chip, Device *device_creator) noexcept{
        //TODO: 判断是否已经注册
        auto chip_device_creator = std::pair<std::string, Device*>(chip, device_creator);
        device_creator_.emplace_back(std::pair<std::string, std::pair<std::string, Device*>>(manufacturer, chip_device_creator));

        device_creator->CreateWorkers();

        return 0;
    }

private:
    std::vector<std::pair<std::string, std::pair<std::string, Device*>>> device_creator_;

    static DeviceManager *pInstance_;
};

int register_device_module();


}