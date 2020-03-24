#ifdef SPIRIT_USE_CUDA
#include <utility/Custom_Field.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <cmath>

namespace Utility
{
    namespace Custom_Field
    {
        __device__ Vector3 CU_Custom_Field(Vector3 x, scalar t)
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

        __global__ void CU_Gradient_Zeeman_Set_External_Field(size_t n_cells_total, const Vector3 * positions, const Vector3 center, scalar picoseconds_passed, Vector3 * external_field){
            for (auto icell = blockIdx.x * blockDim.x + threadIdx.x;
                        icell < n_cells_total;
                        icell += blockDim.x * gridDim.x)
            {
                external_field[icell]=CU_Custom_Field(positions[icell]-center, picoseconds_passed);
            }
        }
        void CustomField(size_t n_cells_total, const Vector3 * positions, const Vector3 center, scalar picoseconds_passed, Vector3 * external_field)
        {
            CU_Gradient_Zeeman_Set_External_Field << <(n_cells_total + 1023) / 1024, 1024 >> > (n_cells_total, positions,center, picoseconds_passed, external_field);
            CU_CHECK_AND_SYNC();
        }
    }//end namespace Custom_Field
}//end namespace Utility

#endif
