#pragma once
#ifndef SPIRIT_CORE_DATA_PARAMETER_DEFINES_HPP
#define SPIRIT_CORE_DATA_PARAMETER_DEFINES_HPP

namespace Data
{
namespace Definitions
{

// Which propagotor to use for the suzuki-trotter solver
enum class ST_Propagator
{
    SA  = 0, // Spin aligned
    IMP = 1  // Implicit
};

} // namespace Definitions
} // namespace Data
#endif