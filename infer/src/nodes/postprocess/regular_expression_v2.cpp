#include "regular_expression_v2.h"

namespace gddi {
namespace nodes {

std::wstring string2wstring(const std::string &strInput) {
    if (strInput.empty()) { return L""; }
    std::string strLocale = setlocale(LC_ALL, "C");
    const char *pSrc = strInput.c_str();
    unsigned int iDestSize = mbstowcs(NULL, pSrc, 0) + 1;
    wchar_t *szDest = new wchar_t[iDestSize];
    wmemset(szDest, 0, iDestSize);
    mbstowcs(szDest, pSrc, iDestSize);
    std::wstring wstrResult = szDest;
    delete[] szDest;
    setlocale(LC_ALL, strLocale.c_str());
    return wstrResult;
}

void RegularExpression_v2::on_setup() {
    // 正则表达式
    pattern_ = std::wregex(converter_.from_bytes(str_expression_));
}

void RegularExpression_v2::on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame) {

    auto clone_frame = std::make_shared<msgs::cv_frame>(frame);

    auto &back_ext_info = clone_frame->frame_info->ext_info.back();
    auto iter = back_ext_info.map_ocr_info.begin();
    while (iter != back_ext_info.map_ocr_info.end()) {
        if (std::regex_match(converter_.from_bytes(iter->second.labels.front().str), pattern_)) {
            ++iter;
        } else {
            iter = back_ext_info.map_ocr_info.erase(iter);
        }
    }

    output_image_(clone_frame);
}

}// namespace nodes
}// namespace gddi