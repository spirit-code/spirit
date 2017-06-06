#include <data/Parameters_Method.hpp>

namespace Data
{
	Pinning::Pinning(std::shared_ptr<Geometry> geometry,
			int na_left, int na_right,
			int nb_left, int nb_right,
			int nc_left, int nc_right,
			vectorfield pinned_cell) :
		geometry(geometry)
    {
		//this->mask_pinned = intfield(geometry->nos);
		this->mask_unpinned = intfield(geometry->nos);
		this->mask_pinned_cells = vectorfield(geometry->nos);
		int N  = geometry->n_spins_basic_domain;
		int Na = geometry->n_cells[0];
		int Nb = geometry->n_cells[1];
		int Nc = geometry->n_cells[2];
		int ispin;

		for (int iatom = 0; iatom < N; ++iatom)
		{
			for (int na = 0; na < Na; ++na)
			{
				for (int nb = 0; nb < Nb; ++nb)
				{
					for (int nc = 0; nc < Nc; ++nc)
					{
						ispin = N*na + N*Na*nb + N*Na*Nb*nc + iatom;
						if ((na < na_left || na > Na - na_right) ||
							(nb < nb_left || nb > Nb - nb_right) ||
							(nc < nc_left || nc > Nc - nc_right))
						{
							// this->mask_pinned[ispin] = 1;
							this->mask_unpinned[ispin] = 0;
							this->mask_pinned_cells[ispin] = pinned_cell[iatom];
						}
						else
						{
							// this->mask_pinned[ispin] = 0;
							this->mask_unpinned[ispin] = 1;
							this->mask_pinned_cells[ispin] = { 0,0,0 };
						}
					}
				}
			}
		}
    }

	Pinning::Pinning(std::shared_ptr<Geometry> geometry,
			intfield mask_unpinned,
			vectorfield mask_pinned_cells):
		geometry(geometry),
		mask_unpinned(mask_unpinned),
		mask_pinned_cells(mask_pinned_cells)
	{
	}


	void Pinning::Apply(vectorfield & vf)
	{
		int N = geometry->n_spins_basic_domain;
		int Na = geometry->n_cells[0];
		int Nb = geometry->n_cells[1];
		int Nc = geometry->n_cells[2];
		int ispin;

		for (int iatom = 0; iatom < N; ++iatom)
		{
			for (int na = 0; na < Na; ++na)
			{
				for (int nb = 0; nb < Nb; ++nb)
				{
					for (int nc = 0; nc < Nc; ++nc)
					{
						ispin = N*na + N*Na*nb + N*Na*Nb*nc + iatom;
						if (!mask_unpinned[ispin]) vf[ispin] = mask_pinned_cells[ispin];
					}
				}
			}
		}
	}
}