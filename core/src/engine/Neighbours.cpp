#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <Eigen/Dense>

#include <numeric>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>

using namespace Utility;

namespace Engine
{
	namespace Neighbours
	{
		std::vector<scalar> Get_Shell_Radius(const Data::Geometry & geometry, const int n_shells)
		{
			auto shell_radius = std::vector<scalar>(n_shells);
			
			Vector3 a = geometry.bravais_vectors[0];
			Vector3 b = geometry.bravais_vectors[1];
			Vector3 c = geometry.bravais_vectors[2];

			scalar current_radius=0, dx, min_distance=0;
			int i=0, j=0, k=0;
			int ii, jj, kk;

			// The 15 is a value that is big enough by experience to 
			// produce enough needed shells, but is small enough to run sufficiently fast
			int imax = 15, jmax = 15, kmax = 15;
			Vector3 x0={0,0,0}, x1={0,0,0};

			// Abort condidions for all 3 vectors
			if (a.norm() == 0.0) imax = 0;
			if (b.norm() == 0.0) jmax = 0;
			if (c.norm() == 0.0) kmax = 0;

			for (int n = 0; n < n_shells; ++n)
			{
				current_radius = min_distance;
				min_distance = 1e10;
				for (int iatom = 0; iatom < geometry.n_cell_atoms; ++iatom)
				{
				    x0 = geometry.cell_atoms[iatom][0] * a + geometry.cell_atoms[iatom][1] * b + geometry.cell_atoms[iatom][2] * c;
					for (ii = imax; ii >= -imax; --ii)
					{
						for (jj = jmax; jj >= -jmax; --jj)
						{
							for (kk = kmax; kk >= -kmax; --kk)
							{
								for (int jatom = 0; jatom < geometry.n_cell_atoms; ++jatom)
								{
									if ( !( iatom==jatom && ii==0 && jj==0 && kk==0 ) )
									{
									    x1 = geometry.cell_atoms[jatom][0] * a + geometry.cell_atoms[jatom][1] * b + geometry.cell_atoms[jatom][2] * c + ii*a + jj*b + kk*c;
										dx = (x0-x1).norm();

										if (dx - current_radius > 1e-6 && dx < min_distance)
										{
											min_distance = dx;
											shell_radius[n] = dx;
										}
									}
								}//endfor jatom
							}//endfor kk
						}//endfor jj
					}//endfor ii
				}//endfor iatom
			}
			
			return shell_radius;
		}
		
		pairfield Get_Pairs_in_Shells(const Data::Geometry & geometry, int nShells)
		{
			auto pairs = pairfield(0);

			auto shell_radius = Get_Shell_Radius(geometry, nShells);
			
			Vector3 a = geometry.bravais_vectors[0];
			Vector3 b = geometry.bravais_vectors[1];
			Vector3 c = geometry.bravais_vectors[2];

			// The nShells + 10 is a value that is big enough by experience to 
			// produce enough needed shells, but is small enough to run sufficiently fast
			int tMax = nShells + 10;
			int imax = tMax, jmax = tMax, kmax = tMax;
			int i,j,k;
			scalar dx, delta, radius;
			Vector3 x0={0,0,0}, x1={0,0,0};

			// Abort condidions for all 3 vectors
			if (a.norm() == 0.0) imax = 0;
			if (b.norm() == 0.0) jmax = 0;
			if (c.norm() == 0.0) kmax = 0;

			for (int iatom = 0; iatom < geometry.n_cell_atoms; ++iatom)
			{
				x0 = geometry.cell_atoms[iatom];
				for (int ishell = 0; ishell < nShells; ++ishell)
				{
					radius = shell_radius[ishell];
					for (i = imax; i >= -imax; --i)
					{
						for (j = jmax; j >= -jmax; --j)
						{
							for (k = kmax; k >= -kmax; --k)
							{
								for (int jatom = 0; jatom < geometry.n_cell_atoms; ++jatom)
								{
									x1 = geometry.cell_atoms[jatom] + i*a + j*b + k*c;
									dx = (x0-x1).norm();
									delta = std::abs(dx - radius);
									if (delta < 1e-6)
									{
										pairs.push_back( {iatom, jatom, {i, j, k} } );
									}
								}//endfor jatom
							}//endfor k
						}//endfor j
					}//endfor i
				}//endfor ishell
			}//endfor iatom

			return pairs;
		}

		neighbourfield Get_Neighbours_in_Shells(const Data::Geometry & geometry, int nShells)
		{
			auto neighbours = neighbourfield(0);

			auto shell_radius = Get_Shell_Radius(geometry, nShells);
			
			Vector3 a = geometry.bravais_vectors[0];
			Vector3 b = geometry.bravais_vectors[1];
			Vector3 c = geometry.bravais_vectors[2];

			// The nShells + 10 is a value that is big enough by experience to 
			// produce enough needed shells, but is small enough to run sufficiently fast
			int tMax = nShells + 10;
			int imax = std::min(tMax, geometry.n_cells[0]-1), jmax = std::min(tMax, geometry.n_cells[1]-1), kmax = std::min(tMax, geometry.n_cells[2]-1);
			int i,j,k;
			scalar dx, delta, radius;
			Vector3 x0={0,0,0}, x1={0,0,0};

			// Abort condidions for all 3 vectors
			if (a.norm() == 0.0) imax = 0;
			if (b.norm() == 0.0) jmax = 0;
			if (c.norm() == 0.0) kmax = 0;

			for (int iatom = 0; iatom < geometry.n_cell_atoms; ++iatom)
			{
				x0 = geometry.cell_atoms[iatom][0] * a + geometry.cell_atoms[iatom][1] * b + geometry.cell_atoms[iatom][2] * c;
				for (int ishell = 0; ishell < nShells; ++ishell)
				{
					radius = shell_radius[ishell];
					for (i = imax; i >= -imax; --i)
					{
						for (j = jmax; j >= -jmax; --j)
						{
							for (k = kmax; k >= -kmax; --k)
							{
								for (int jatom = 0; jatom < geometry.n_cell_atoms; ++jatom)
								{
									x1 = geometry.cell_atoms[jatom][0] * a + geometry.cell_atoms[jatom][1] * b + geometry.cell_atoms[jatom][2] * c + i*a + j*b + k*c;
									dx = (x0-x1).norm();
									delta = std::abs(dx - radius);
									if (delta < 1e-6)
									{
										Neighbour neigh;
										neigh.i = iatom;
										neigh.j = jatom;
										neigh.translations[0] = i;
										neigh.translations[1] = j;
										neigh.translations[2] = k;
										neigh.idx_shell = ishell;
										neighbours.push_back( neigh );
									}
								}//endfor jatom
							}//endfor k
						}//endfor j
					}//endfor i
				}//endfor ishell
			}//endfor iatom

			return neighbours;
		}


		pairfield Get_Pairs_in_Radius(const Data::Geometry & geometry, scalar radius)
		{
			auto pairs = pairfield(0);

			if (radius > 1e-6)
			{
				Vector3 a = geometry.bravais_vectors[0];
				Vector3 b = geometry.bravais_vectors[1];
				Vector3 c = geometry.bravais_vectors[2];

				Vector3 bounds_diff = geometry.bounds_max - geometry.bounds_min;
				Vector3 ratio = {
					bounds_diff[0]/std::max(1, geometry.n_cells[0]),
					bounds_diff[1]/std::max(1, geometry.n_cells[1]),
					bounds_diff[2]/std::max(1, geometry.n_cells[2]) };

				// This should give enough translations to contain all DDI pairs
				int imax = 0, jmax = 0, kmax = 0;
				if ( bounds_diff[0] > 0 )
					imax = std::min(geometry.n_cells[0], (int)(1.1 * radius * geometry.n_cells[0] / bounds_diff[0]));
				if ( bounds_diff[1] > 0 )
					jmax = std::min(geometry.n_cells[1], (int)(1.1 * radius * geometry.n_cells[1] / bounds_diff[1]));
				if ( bounds_diff[2] > 0 )
					kmax = std::min(geometry.n_cells[2], (int)(1.1 * radius * geometry.n_cells[2] / bounds_diff[2]));

				int i,j,k;
				scalar dx;
				Vector3 x0={0,0,0}, x1={0,0,0};

				// Abort condidions for all 3 vectors
				if (a.norm() == 0.0) imax = 0;
				if (b.norm() == 0.0) jmax = 0;
				if (c.norm() == 0.0) kmax = 0;

				for (int iatom = 0; iatom < geometry.n_cell_atoms; ++iatom)
				{
					x0 = geometry.cell_atoms[iatom];
					for (i = imax; i >= -imax; --i)
					{
						for (j = jmax; j >= -jmax; --j)
						{
							for (k = kmax; k >= -kmax; --k)
							{
								for (int jatom = 0; jatom < geometry.n_cell_atoms; ++jatom)
								{
									x1 = geometry.cell_atoms[jatom] + i*a + j*b + k*c;
									dx = (x0-x1).norm();
									if (dx < radius)
									{
										pairs.push_back( {iatom, jatom, {i, j, k} } );
									}
								}//endfor jatom
							}//endfor k
						}//endfor j
					}//endfor i
				}//endfor iatom
			}

			return pairs;
		}


		neighbourfield Get_Neighbours_in_Radius(const Data::Geometry & geometry, scalar radius)
		{
			auto neighbours = neighbourfield(0);

			if (radius > 1e-6)
			{
				Vector3 a = geometry.bravais_vectors[0];
				Vector3 b = geometry.bravais_vectors[1];
				Vector3 c = geometry.bravais_vectors[2];

				Vector3 bounds_diff = geometry.bounds_max - geometry.bounds_min;
				Vector3 ratio = {
					bounds_diff[0]/std::max(1, geometry.n_cells[0]),
					bounds_diff[1]/std::max(1, geometry.n_cells[1]),
					bounds_diff[2]/std::max(1, geometry.n_cells[2]) };
					
				// This should give enough translations to contain all DDI pairs
				int imax = 0, jmax = 0, kmax = 0;
				if ( bounds_diff[0] > 0 )
					imax = std::min(geometry.n_cells[0], (int)(1.1 * radius * geometry.n_cells[0] / bounds_diff[0]));
				if ( bounds_diff[1] > 0 )
					jmax = std::min(geometry.n_cells[1], (int)(1.1 * radius * geometry.n_cells[1] / bounds_diff[1]));
				if ( bounds_diff[2] > 0 )
					kmax = std::min(geometry.n_cells[2], (int)(1.1 * radius * geometry.n_cells[2] / bounds_diff[2]));
					
				int i, j, k;
				scalar dx;
				Vector3 x0 = { 0,0,0 }, x1 = { 0,0,0 };

				// Abort condidions for all 3 vectors
				if (a.norm() == 0.0) imax = 0;
				if (b.norm() == 0.0) jmax = 0;
				if (c.norm() == 0.0) kmax = 0;

				for (int iatom = 0; iatom < geometry.n_cell_atoms; ++iatom)
				{
					x0 = geometry.cell_atoms[iatom];
					for (i = imax; i >= -imax; --i)
					{
						for (j = jmax; j >= -jmax; --j)
						{
							for (k = kmax; k >= -kmax; --k)
							{
								for (int jatom = 0; jatom < geometry.n_cell_atoms; ++jatom)
								{
									x1 = geometry.cell_atoms[jatom] + i*a + j*b + k*c;
									dx = (x0 - x1).norm();
									if (dx < radius)
									{
										Neighbour neigh;
										neigh.i = iatom;
										neigh.j = jatom;
										neigh.translations[0] = i;
										neigh.translations[1] = j;
										neigh.translations[2] = k;
										neigh.idx_shell = 0;
										neighbours.push_back( neigh );
									}
								}//endfor jatom
							}//endfor k
						}//endfor j
					}//endfor i
				}//endfor iatom
			}

			return neighbours;
		}


		Vector3 DMI_Normal_from_Pair(const Data::Geometry & geometry, const Pair & pair, int chirality)
		{
			Vector3 ta = geometry.bravais_vectors[0];
			Vector3 tb = geometry.bravais_vectors[1];
			Vector3 tc = geometry.bravais_vectors[2];

			int da = pair.translations[0];
			int db = pair.translations[1];
			int dc = pair.translations[2];

			Vector3 ipos = geometry.cell_atoms[pair.i];
			Vector3 jpos = geometry.cell_atoms[pair.j] + da*ta + db*tb + dc*tc;

			if (chirality == 1)
			{
				// Bloch chirality
				return (jpos - ipos).normalized();
			}
			else if (chirality == -1)
			{
				// Inverse Bloch chirality
				return (ipos - jpos).normalized();
			}
			else if (chirality == 2)
			{
				// Neel chirality (surface)
				return (jpos - ipos).normalized().cross(Vector3{0,0,1});
			}
			else if (chirality == -2)
			{
				// Inverse Neel chirality (surface)
				return Vector3{0,0,1}.cross((jpos - ipos).normalized());
			}
			else
			{
				return Vector3{ 0,0,0 };
			}
		}

		void DDI_from_Pair(const Data::Geometry & geometry, const Pair & pair, scalar & magnitude, Vector3 & normal)
		{
			Vector3 ta = geometry.bravais_vectors[0];
			Vector3 tb = geometry.bravais_vectors[1];
			Vector3 tc = geometry.bravais_vectors[2];

			int da = pair.translations[0];
			int db = pair.translations[1];
			int dc = pair.translations[2];

			Vector3 ipos = geometry.cell_atoms[pair.i];
			Vector3 jpos = geometry.cell_atoms[pair.j] + da*ta + db*tb + dc*tc;

			// Calculate positions and difference vector
			Vector3 vector_ij = jpos - ipos;

			// Length of difference vector
			magnitude = vector_ij.norm();
			normal = vector_ij.normalized();
		}

	}// end Namespace Neighbours
}// end Namespace Engine
