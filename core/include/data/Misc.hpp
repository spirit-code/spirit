#pragma once

#include <string_view>
#include <vector>

namespace Data
{

template<typename T>
using labeled = std::pair<std::string_view, T>;

template<typename T>
using vectorlabeled = std::vector<labeled<T>>;

} // namespace Data
