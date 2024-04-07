#include <regex>
#include <spdlog/spdlog.h>
#include <string>
#include <codecvt>

int main() {
    std::string str = "^([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[a-zA-Z](([DF]((?![IO])[a-zA-Z0-9](?![IO]))[0-9]{4})|([0-9]{5}[DF]))|[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}[A-Z0-9]{4}[A-Z0-9挂学警港澳]{1})$";

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wregex reg(converter.from_bytes(str));

    std::string plate_str = "皖ADB2676";

    // 使用正则表达式模式进行匹配
    if (std::regex_match(converter.from_bytes(plate_str), reg)) {
        spdlog::info("车牌号码有效: {}", plate_str);
    } else {
        spdlog::info("车牌号码无效: {}", plate_str);
    }

    return 0;
}