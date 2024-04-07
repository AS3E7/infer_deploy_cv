#include <algorithm>
#include <cstddef>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>

#include "core/model.h"
#include "core/device.h"
#include "../util/env.h"
#include "common/logger.h"

#include "modcrypt.h"

#include "openssl/md5.h"
#include <fstream>

namespace gddeploy
{

std::unordered_map<std::string, std::shared_ptr<Model>> ModelManager::model_cache_;
std::mutex ModelManager::model_cache_mutex_;
std::unordered_map<std::string, std::unordered_map<std::string, ModelCreator *>> ModelManager::model_creator_;

inline std::string GetModelKey(const std::string &model_path, const std::string &func_name) noexcept
{
    // 读取文件，获取md5值作为函数返回
    std::ifstream model_file(model_path, std::ios::binary);
    if (!model_file.is_open())
    {
        return model_path + "_" + func_name;
    }
    model_file.seekg(0, std::ios::end);
    uint64_t file_len = model_file.tellg();
    model_file.seekg(0, std::ios::beg);
    auto model_data = std::vector<uint8_t>(file_len);
    model_file.read(reinterpret_cast<char *>(model_data.data()), file_len);
    model_file.close();

    unsigned char md5[16] = {0};
    MD5(model_data.data(), file_len, md5);
    char md5_str[32 + 1] = {0};
    for (int i = 0; i < 16; i++)
    {
        sprintf(md5_str + i * 2, "%02x", md5[i]);
    }

    return std::string(md5_str) + "_" + func_name;
}

int GetMD5String(const unsigned char *data, unsigned int len, char *md5_str)
{
    if (data == nullptr || md5_str == nullptr)
    {
        return -1;
    }

    unsigned char md5[16] = {0};
    MD5(data, len, md5);
    for (int i = 0; i < 16; i++)
    {
        sprintf(md5_str + i * 2, "%02x", md5[i]);
    }

    return 0;
}

static inline std::string GetModelKey(const void *mem_ptr, const std::string &func_name = "") noexcept
{
    std::ostringstream ss;
    ss << mem_ptr << "_" << func_name;
    return ss.str();
}

void ModelManager::CheckAndCleanCache() noexcept
{
    if (model_cache_.size() >= GetUlongFromEnv("CNIS_MODEL_CACHE_LIMIT", 10))
    {
        for (auto &p : model_cache_)
        {
            if (p.second.use_count() == 1)
            {
                model_cache_.erase(p.first);
                break;
            }
        }
    }
}

#include <dirent.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/types.h>
#include <fcntl.h>
#include <fstream>
#include <unistd.h>
#include <openssl/md5.h>

bool Model::InitExpiredTimeFile(std::string license_file_path, long int expired_time)
{
    unsigned char md5[16] = {0};
    char cache_hash_name[32*2+1] = {0};
    std::ifstream license_fstream(license_file_path, std::ios::binary);
    if (!license_fstream.is_open()) {
            std::cout << "Init() can't open license file: " << license_file_path << std::endl;
            return false;
    }
        
    license_fstream.seekg(0, std::ios::end);
    uint64_t file_len = license_fstream.tellg();
    license_fstream.seekg(0, std::ios::beg);
    auto encrypt_license = std::vector<uint8_t>(file_len);
    license_fstream.read((char *)encrypt_license.data(), encrypt_license.size());
    license_fstream.close();
    MD5((const unsigned char*)encrypt_license.data(),encrypt_license.size(),md5);
    for(int i = 0; i < 16; i++)
    {
        sprintf(cache_hash_name + (i * 2), "%02x", md5[i]);
    }

    std::string::size_type position = license_file_path.find_last_of('//');
    license_root_ = license_file_path.substr(0, position);
    license_root_ = license_root_+"/."+cache_hash_name;
    if(access(license_root_.c_str(), F_OK ) == -1)
    {
        int exp_lock_file_ = open(license_root_.c_str(),O_CREAT|O_RDWR|O_TRUNC, 0666);
        if(exp_lock_file_ < 0)
        {
            printf("could not create pxe file.\n");
            return false;
        }
        write(exp_lock_file_,&expired_time,sizeof(expired_time));
        close(exp_lock_file_);
    }
    else
    {
        long int expired_time;
        int exp_lock_file_ = open(license_root_.c_str(), O_RDWR, 0666);
        if(exp_lock_file_ < 0)
        {
            printf("could not create pxe file.\n");
            return false;
        }
        read(exp_lock_file_,&expired_time,sizeof(expired_time));
        if(expired_time < 10000)
        {
            printf("license time out \n");
            return false;
        }
        close(exp_lock_file_);
    }

    license_time_recode_thread_ = std::thread(&Model::licenseThread, this);
    return true;
}


#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/time.h>

void Model::licenseThread()
{
    unsigned long recode_expired_time = 0;
    
    long int expired_time;

    struct timeval time_v = {0};
    gettimeofday(&time_v, NULL);
    long tic_ = (long)time_v.tv_sec*1000 + time_v.tv_usec/1000;

    int exp_lock_file_ = open(license_root_.c_str(), O_RDWR, 0666);
    if(exp_lock_file_ < 0)
    {
        printf("could not create pxe file.\n");
        return ;
    }

    struct flock fl;
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = 100;
    fl.l_len = 10;
 
    if (fcntl(exp_lock_file_, F_SETLK, &fl) == -1) {
        if (errno == EACCES || errno == EAGAIN) {
            fl.l_type = F_RDLCK;
            printf("Already locked by another process\n"); 
            if (fcntl(exp_lock_file_, F_SETLK, &fl) == -1) {
                while(1){
                    if (exit_flag_){
                        break;
                    }

                    lseek(exp_lock_file_,0,SEEK_SET);
                    read(exp_lock_file_,&expired_time, sizeof(expired_time));

                    if(expired_time == 0){
                        GDDEPLOY_ERROR("Authorization expired !!! ");
                        this->authorization_expired_flag_ = true;
                        break;
                    }

                    usleep(1000*200);
                }
            }
        } else {
            /* Handle unexpected error */;
        }
    } else { /* Lock was granted... */
        lseek(exp_lock_file_,0,SEEK_SET);
        read(exp_lock_file_,&expired_time, sizeof(expired_time));

        while(1){
            if (this->exit_flag_){
                lseek(exp_lock_file_,0,SEEK_SET);
                write(exp_lock_file_,&expired_time,sizeof(expired_time));

                break;
            }

            struct timeval time_v = {0};
            gettimeofday(&time_v, NULL);
            long toc_ = (long)time_v.tv_sec*1000 + time_v.tv_usec/1000;

            if(toc_ - tic_ > 2000){
                expired_time -= (toc_ - tic_);
                tic_ = toc_;
                
                if(expired_time > 10000){
                    lseek(exp_lock_file_,0,SEEK_SET);
                    write(exp_lock_file_,&expired_time, sizeof(expired_time));

                }else{  // 超过授权时间
                    expired_time = 0;
                    lseek(exp_lock_file_,0,SEEK_SET);
                    write(exp_lock_file_,&expired_time,sizeof(expired_time));
                    
                    GDDEPLOY_ERROR("Authorization expired !!! ");
                    this->authorization_expired_flag_ = true;
                    break;
                }
            }

            usleep(10);
        }
    }
    /* Unlock the locked bytes */
    fl.l_type = F_UNLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = 100;
    fl.l_len = 10;
    if (fcntl(exp_lock_file_, F_SETLK, &fl) == -1){
        printf("Already locked by another process\n"); 
    }

    close(exp_lock_file_);
}

std::vector<std::string> Model::GetLabels()
{
    return model_info_priv_->GetLabels();
}

ModelPtr ModelManager::Load(const std::string &model_path, const std::string &properties_path, const std::string &license_path, std::string param) noexcept
{
    // TODO: 这里要插入解密部分代码

    std::vector<uint8_t> module;
    std::string config;
    // ret = modelDecryptOnline(modelName, secretKey, config, module, uuid);
    std::string sn = gddeploy::DeviceManager::Instance()->GetDevice()->GetDeviceSN();

#ifdef WITH_NVIDIA
    //兼容之前sdk的序列号
    std::string uuid = gddi::get_device_uuid(sn, "9mflg7sl2pqved84");
#else
    std::string uuid = gddi::get_device_uuid(sn, "inference-engine");
#endif
    gddi::ModCrypto mod_crypto(uuid);

    bool ret = false;
    long int model_expired_time_l = 0;
    if (license_path.empty())
    {
        ret = mod_crypto.decrypt_model(model_path, config, module);
    } else {
        ret = mod_crypto.decrypt_model_v2(license_path, model_path, config, module, model_expired_time_l);
    }
    

    // int ret = gddi::model_decrypt_with_salt(model_path, "inference-engine", config, module);
    if (ret == false)
    {
        GDDEPLOY_ERROR("Model Decrypt fail: the uuid: {}", uuid);
        return nullptr;
    }

    ModelPropertiesPtr model_info_priv = std::make_shared<ModelProperties>(config);

    std::string model_key = GetModelKey(model_path, properties_path);

    std::unique_lock<std::mutex> lk(model_cache_mutex_);
    if (model_cache_.find(model_key) == model_cache_.cend())
    {
        auto model = Load(module.data(), module.size(), license_path, model_expired_time_l, model_info_priv, model_key);
        if (model == nullptr)
        {
            GDDEPLOY_ERROR("[ModelManager] Model Init fail, please check param");
            return nullptr;
        }

        model_cache_[model_key] = model;

        return model;
    }
    else
    {
        // cache hit
        GDDEPLOY_INFO("[ModelManager] Get model from cache");
        return model_cache_.at(model_key);
    }

    return nullptr;
}

ModelPtr ModelManager::Load(void *mem_ptr, size_t mem_size, const std::string &license_path, long int model_expired_time_l, ModelPropertiesPtr model_info, std::string param) noexcept
{
    GDDEPLOY_INFO("[ModelManager] Load model from memory");

    std::string manu = model_info->GetProductType();
    std::string chip = model_info->GetChipType();

    // auto creator = model_creator_[manu][chip];
    // auto creator = model_creator_["bmnn"]["bm1684"];
    auto creator = GetModelCreator(manu, chip);
    if (creator == nullptr)
    {
        GDDEPLOY_ERROR("[ModelManager] ModelManager can't find manu:{}, chip:{} creator", manu, chip);
        return nullptr;
    }

    auto model = creator->Create(model_info);

    if (false == license_path.empty())
    {
        if(false == model->InitExpiredTimeFile(license_path, model_expired_time_l * 1000))
        {
            std::cout << "Model InitExpiredTimeFile fail: " << std::endl;
            return nullptr;
        }
    }

    model->SetModelInfoPriv(model_info);
    if (model->Init(mem_ptr, mem_size, param))
    {
        return nullptr;
    }

    CheckAndCleanCache();

    return model;
}

std::shared_ptr<Model> ModelManager::GetModel(const std::string &name) noexcept
{
    std::unique_lock<std::mutex> lk(model_cache_mutex_);
    if (model_cache_.find(name) == model_cache_.cend())
    {
        return nullptr;
    }
    return model_cache_.at(name);
}

int ModelManager::CacheSize() noexcept { return model_cache_.size(); }

bool ModelManager::Unload(ModelInfoPtr model) noexcept
{
    if (!model)
    {
        GDDEPLOY_ERROR("[ModelManager] Model is nullptr!");
    }
    const std::string &model_key = model->GetKey();
    std::lock_guard<std::mutex> lk(model_cache_mutex_);
    if (model_cache_.find(model_key) == model_cache_.cend())
    {
        GDDEPLOY_WARN("[ModelManager] Model is not in cache");
        return false;
    }
    else
    {
        model_cache_.erase(model_key);
        return true;
    }
}

void ModelManager::ClearCache() noexcept
{
    std::lock_guard<std::mutex> lk(model_cache_mutex_);
    model_cache_.clear();
}


ModelCreator* ModelManager::GetModelCreator(std::string manu, std::string chip)
{
    ModelCreator* creator = nullptr;
    if (manu == "SOPHGO"){
    // if (false){
        creator = model_creator_["SOPHGO"]["SE5"];
    } else if (manu == "Tsingmicro"){
        creator = model_creator_["Tsingmicro"]["TX5368A"];
    } else if (manu == "Nvidia") {
        creator = model_creator_["NVIDIA"]["any"];
    } else if (manu == "Cambricon") {
        // creator = model_creator_["Cambricon"]["MLU220"];
        creator = model_creator_["Cambricon"][chip];
    } else if (manu == "Intel") {
        // creator = model_creator_["Cambricon"]["MLU220"];
        creator = model_creator_["Intel"]["any"];
    } else if (manu == "Rockchip") {
        // creator = model_creator_["Cambricon"]["MLU220"];
        creator = model_creator_["Rockchip"]["3588"];
    } else{
        creator = model_creator_["ort"]["cpu"];
    }
    
    return creator;
}
}