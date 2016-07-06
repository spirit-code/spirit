#include <Geometry.h>

namespace Data
{
	Geometry::Geometry(std::vector<std::vector<double>> basis_i, const std::vector<std::vector<double>> translation_vectors_i,
		const std::vector<int> n_translations_i, const int n_spins_basic_domain_i, const std::vector<std::vector<double>> spin_pos_i) :
		basis(basis_i), translation_vectors(translation_vectors_i),
		n_translations(n_translations_i), n_spins_basic_domain(n_spins_basic_domain_i),
		spin_pos(spin_pos_i), nos( n_spins_basic_domain_i * (n_translations_i[0] + 1) * (n_translations_i[1] + 1) *(n_translations_i[2] + 1))
	{
		// Calculate Bounds of the System
		this->bounds_min = std::vector<double>(3);
		this->bounds_max = std::vector<double>(3);
		this->center = std::vector<double>(3);

		for (int dim = 0; dim < 3; ++dim)
		{
			for (int iatom = 0; iatom < nos; ++iatom)
			{
				if (this->spin_pos[dim][iatom] < this->bounds_min[dim]) this->bounds_min[dim] = spin_pos[dim][iatom];
				if (this->spin_pos[dim][iatom] > this->bounds_max[dim]) this->bounds_max[dim] = spin_pos[dim][iatom];
			}
		}

		// Calculate Center of the System
		for (int dim = 0; dim < 3; ++dim)
		{
			this->center[dim] = (this->bounds_min[dim] + this->bounds_max[dim]) / 2.0;
		}
	}
}