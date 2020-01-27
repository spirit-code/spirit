#pragma once
#ifndef UTILITY_CUSTOM_FIELD_H
#define UTILITY_CUSTOM_FIELD_H

#include <engine/Vectormath_Defines.hpp>

namespace Utility
{
    namespace Custom_Field
    {
            void CustomField(size_t n_cells_total, const Vector3 * positions, const Vector3 center, scalar picoseconds_passed, Vector3 * external_field);
    }//end namespace Custom_Field
}//end namespace Utility

#endif
