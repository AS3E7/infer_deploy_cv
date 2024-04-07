#ifndef GDDEPLOY_BMNN_MODEL_H
#define GDDEPLOY_BMNN_MODEL_H

#include <memory>
#include "core/model.h"


namespace gddeploy
{

class BmnnModelPrivate;
class BmnnModel : public Model {
public:
    BmnnModel(const any& value):Model(){}
    ~BmnnModel() = default;
    
    int Init(const std::string& model_path, const std::string& param) override;
    int Init(void* mem_ptr,  size_t mem_size, const std::string& param) override;

    /**
     * @brief Get input shape
     *
     * @param index index of input
     * @return const Shape& shape of specified input
     */
    const Shape& InputShape(int index) const noexcept override;

    /**
     * @brief Get output shape
     *
     * @param index index of output
     * @return const Shape& shape of specified output
     */
    const Shape& OutputShape(int index) const noexcept override;
    /**
     * @brief Check if output shapes are fixed
     *
     * @return Returns true if all output shapes are fixed, otherwise returns false.
     */
    int FixedOutputShape() noexcept override;

    /**
     * @brief Get input layout on MLU
     *
     * @param index index of input
     * @return const DataLayout& data layout of specified input
     */
    const DataLayout& InputLayout(int index) const noexcept override;

    /**
     * @brief Get output layout on MLU
     *
     * @param index index of output
     * @return const DataLayout& data layout of specified output
     */
    const DataLayout& OutputLayout(int index) const noexcept override;

    /**
     * @brief Get number of input
     *
     * @return uint32_t number of input
     */
    uint32_t InputNum() const noexcept override;

    /**
     * @brief Get number of output
     *
     * @return uint32_t number of output
     */
    uint32_t OutputNum() const noexcept override;

    /**
     * @brief Get number of input
     *
     * @return uint32_t number of input
     */
    virtual std::vector<float> InputScale() const noexcept override;

    /**
     * @brief Get number of output
     *
     * @return uint32_t number of output
     */
    virtual std::vector<float> OutputScale() const noexcept override;

    /**
     * @brief Get model batch size
     *
     * @return uint32_t batch size
     */
    uint32_t BatchSize() const noexcept override;

    /**
     * @brief Get model key
     *
     * @return const std::string& model key
     */
    std::string GetKey() const noexcept override;

    any GetModel() override;

private:

    std::shared_ptr<BmnnModelPrivate> bmnn_model_priv_; 

    std::vector<Shape> inputs_shape_;
    std::vector<DataLayout> intput_data_layout_;

    std::vector<float> input_scale_;
    std::vector<float> output_scale_;

    std::vector<Shape> outputs_shape_;
    std::vector<DataLayout> output_data_layout_;
}; // class BmnnModel


class BmnnModelCreator : public ModelCreator{
public:
    BmnnModelCreator() : ModelCreator("BmnnModelCreator"){
    }

    std::string GetName() const override { return model_creator_name_; }

    std::shared_ptr<Model> Create(const any& value) override {
        return std::shared_ptr<BmnnModel>(new BmnnModel(value));
    }

private:
    std::string model_creator_name_ = "BmnnModelCreator";

};  // class BmnnModelCreator
}

#endif
