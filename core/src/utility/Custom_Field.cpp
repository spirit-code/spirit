#ifndef SPIRIT_USE_CUDA
#include <utility/Custom_Field.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <cmath>

namespace Utility
{
    namespace Custom_Field
    {
		Vector3 Custom_Field(Vector3 x, scalar t)
		{
			Vector3 out={0,0,0};
			if (x[0]*x[0]+x[1]*x[1]<10000) {
				out = {0,0,0.0};
			}
			else{
				out = {0,0,-0.0};
			}
			return out;
		}

		void CustomField(size_t n_cells_total, const Vector3 * positions, const Vector3 center, const scalar picoseconds_passed, Vector3 * external_field)
		{
			#pragma omp parallel for
			for( int icell = 0; icell < n_cells_total; ++icell )
			{
				external_field[icell]=Custom_Field(positions[icell]-center, picoseconds_passed);
			}

		}
    }//end namespace Custom_Field
}//end namespace Utility

#endif
