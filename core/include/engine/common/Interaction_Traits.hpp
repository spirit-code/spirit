#pragma once

#include <type_traits>

namespace Engine
{

namespace Common
{

namespace Interaction
{

namespace detail
{

template<typename Interaction, typename = void>
struct is_local_impl : std::true_type
{
};

template<typename Interaction>
struct is_local_impl<Interaction, std::void_t<decltype( Interaction::local )>> : std::bool_constant<Interaction::local>
{
};

template<typename T, typename = void>
struct has_geometry_member : std::false_type
{
};

template<typename T>
struct has_geometry_member<T, std::void_t<decltype( std::declval<T>().geometry )>> : std::true_type
{
};

template<typename T, typename = void>
struct has_bc_member : std::false_type
{
};

template<typename T>
struct has_bc_member<T, std::void_t<decltype( std::declval<T>().boundary_conditions )>> : std::true_type
{
};

} // namespace detail

template<typename Interaction>
using is_local = detail::is_local_impl<Interaction>;

template<typename T>
using has_geometry_member = detail::has_geometry_member<T>;

template<typename T>
using has_bc_member = detail::has_bc_member<T>;

} // namespace Interaction

} // namespace Spin

} // namespace Engine
