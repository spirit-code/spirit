#pragma once
#ifndef SPIRIT_CORE_UTILITY_TYPE_TRAITS_HPP
#define SPIRIT_CORE_UTILITY_TYPE_TRAITS_HPP
#include <string>
#include <type_traits>

namespace Utility
{

template<typename T>
struct is_string : std::false_type
{
};

template<class T, class Traits, class Alloc>
struct is_string<std::basic_string<T, Traits, Alloc>> : std::true_type
{
};

template<typename T>
static constexpr bool is_string_v = is_string<T>::value;

} // namespace Utility
#endif
