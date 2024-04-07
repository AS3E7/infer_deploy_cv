#include "core/model_properties.h"
#include <algorithm>
using namespace gddeploy;

static void split(const std::string &s, std::vector<std::string> &tokens, const std::string &delimiters = " ")
{
    std::string::size_type lastPos = s.find_first_not_of(delimiters, 0);
    std::string::size_type pos = s.find_first_of(delimiters, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos)
    {
        tokens.push_back(s.substr(lastPos, pos - lastPos));
        lastPos = s.find_first_not_of(delimiters, pos);
        pos = s.find_first_of(delimiters, lastPos);
    }
}

ModelProperties::ModelProperties(std::string raw_info)
{
    std::vector<std::string> config_v;
    split(raw_info, config_v, "\n");

    for (auto &c : config_v)
    {
        std::vector<std::string> config_item;
        split(c, config_item, "=");
        if (config_item[0] == "label")
        {
            if (config_item.size() > 2)
            {
                auto tmp = config_item[1] + "=" + config_item[2];
                config_item[1] = tmp;
            }
            auto start = 0;
            auto len = config_item[1].size();
            while ((start = config_item[1].find_first_of('\'', start)) != config_item[1].npos)
            {

                if (start + 1 >= len)
                    break;

                auto end = config_item[1].find_first_of('\'', start + 1);
                if (config_item[1][end + 1] == '\'')
                {
                    end = end + 1;
                }

                if (end >= len)
                    break;

                labels_.push_back(config_item[1].substr(start + 1, end - start - 1));
                start = end + 1;
            }
        }
        else if (config_item[0] == "conf_thres")
        {
            conf_thresh_ = atof(config_item[1].c_str());
        }
        else if (config_item[0] == "model_type")
        {
            model_type_ = config_item[1];
        }
        else if (config_item[0] == "product_type")
        {
            product_type_ = config_item[1];
        }
        else if (config_item[0] == "chip_type")
        {
            chip_type_ = config_item[1];
        }
        else if (config_item[0] == "net_type")
        {
            net_type_ = config_item[1];
        }else if (config_item[0] == "input_size"){
            input_size_ = std::atoi(config_item[1].c_str());
        }
        else if (config_item[0] == "forward_type")
        {
            chip_ip_ = config_item[1];
        }
        else if (config_item[0] == "input_dest")
        {
            input_dest_ = config_item[1];
        }
        else if (config_item[0] == "anchors")
        {
            auto start = 1, end = 0;
            std::vector<std::string> anchors;

            config_item[1] = config_item[1].substr(1, config_item[1].length() - 2);

            while ((end = config_item[1].find_first_of(']', end + 1)) != config_item[1].npos)
            {
                anchors.push_back(config_item[1].substr(start, end - start));
                start = config_item[1].find_first_of('[', start + 1) + 1;
            }
            for (auto a : anchors)
            {
                std::vector<std::string> values;
                split(a, values, ",");

                for (auto value : values)
                {
                    anchors_.push_back(atof(value.c_str()));
                }
            }
        }
        else if (config_item[0] == "qat")
        {
            std::transform(config_item[1].begin(), config_item[1].end(), config_item[1].begin(), ::toupper);
            if (config_item[1] == "TRUE")
            {
                // printf("use qat int8 model.\n");
                qat_ = true;
            }
        }
        else if (config_item[0] == "slice_w")
        {
            slice_w_ = atoi(config_item[1].c_str()); 
        }
        else if (config_item[0] == "slice_h")
        {
            slice_h_ = atoi(config_item[1].c_str()); 
        }
        else if (config_item[0] == "need_clip")
        {
            // need_clip = atoi(config_item[1].c_str());
            std::transform(config_item[1].begin(), config_item[1].end(), config_item[1].begin(), ::toupper);
            if (config_item[1] == "TRUE")
            {
                need_clip = true;
            }
        }

        config_item.clear();
    }
}