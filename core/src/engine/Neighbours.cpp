#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/IO.hpp>
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
	std::vector<scalar> Get_Shell_Radius(const Data::Geometry & geometry, const int n_shells)
	{
		auto shell_radius = std::vector<scalar>(n_shells);
		
		Vector3 a = geometry.basis[0];
		Vector3 b = geometry.basis[1];
		Vector3 c = geometry.basis[2];

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
			for (int iatom = 0; iatom < geometry.n_spins_basic_domain; ++iatom)
			{
				x0 = geometry.basis_atoms[iatom];
				for (ii = imax; ii >= -imax; --ii)
				{
					for (jj = jmax; jj >= -jmax; --jj)
					{
						for (kk = kmax; kk >= -kmax; --kk)
						{
							for (int jatom = 0; jatom < geometry.n_spins_basic_domain; ++jatom)
							{
								if ( !( iatom==jatom && ii==0 && jj==0 && kk==0 ) )
								{
									x1 = geometry.basis_atoms[jatom] + ii*a + jj*b + kk*c;
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

	void Pairs_from_Neighbour_Shells(const Data::Geometry & geometry, int nShells, std::vector<int> & shellIndex, pairfield & pairs)
	{
		shellIndex = std::vector<int>(0);
		pairs = pairfield(0);

		auto shell_radius = Get_Shell_Radius(geometry, nShells);
		
		Vector3 a = geometry.translation_vectors[0];
		Vector3 b = geometry.translation_vectors[1];
		Vector3 c = geometry.translation_vectors[2];

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

		for (int iatom = 0; iatom < geometry.n_spins_basic_domain; ++iatom)
		{
			x0 = geometry.basis_atoms[iatom];
			for (int ishell = 0; ishell < nShells; ++ishell)
			{
				radius = shell_radius[ishell];
				for (i = imax; i >= -imax; --i)
				{
					for (j = jmax; j >= -jmax; --j)
					{
						for (k = kmax; k >= -kmax; --k)
						{
							for (int jatom = 0; jatom < geometry.n_spins_basic_domain; ++jatom)
							{
								x1 = geometry.basis_atoms[jatom] + i*a + j*b + k*c;
								dx = (x0-x1).norm();
								delta = std::abs(dx - radius);
								if (delta < 1e-6)
								{
									shellIndex.push_back(ishell);
									pairs.push_back( {iatom, jatom, {i, j, k} } );
								}
							}//endfor jatom
						}//endfor k
					}//endfor j
				}//endfor i
			}//endfor ishell
		}//endfor iatom
	}

	Vector3 DMI_Normal_from_Pair(const Data::Geometry & geometry, Pair pair, int chirality)
	{
		Vector3 ta = geometry.translation_vectors[0];
		Vector3 tb = geometry.translation_vectors[1];
		Vector3 tc = geometry.translation_vectors[2];

		int da = pair.translations[0];
		int db = pair.translations[1];
		int dc = pair.translations[2];

		auto ipos = geometry.basis_atoms[pair.i];
		auto jpos = geometry.basis_atoms[pair.j] + da*ta + db*tb + dc*tc;

		if (chirality == 1)
		{
			return (jpos - ipos).normalized();
		}
		else if (chirality == -1)
		{
			return (ipos - jpos).normalized();
		}
		else
		{
			return Vector3{0,0,0};
		}
	}

	void DDI_Pairs_from_Neighbours(const Data::Geometry & geometry, scalar radius, pairfield & pairs)
	{
		auto diagonal = geometry.bounds_max - geometry.bounds_min;
		scalar maxradius = std::min(diagonal[0], std::min(diagonal[1], diagonal[2]));
		// scalar ratio = radius/diagonal.norm();

		// Check for too large DDI radius
		if (radius > maxradius)
		{
			radius = maxradius;
			Log(Log_Level::Warning, Log_Sender::All, "DDI radius is larger than your system! Setting to minimum of system bounds: " + std::to_string(radius));
		}

		if (radius > 1e-6)
		{
			Vector3 a = geometry.translation_vectors[0];
			Vector3 b = geometry.translation_vectors[1];
			Vector3 c = geometry.translation_vectors[2];

			Vector3 ratio{radius/diagonal[0], radius/diagonal[1], radius/diagonal[2]};

			// This should give enough translations to contain all DDI pairs
			int imax = std::min(geometry.n_cells[0], (int)(1.5 * ratio[0] * geometry.n_cells[0]) + 1);
			int jmax = std::min(geometry.n_cells[1], (int)(1.5 * ratio[1] * geometry.n_cells[1]) + 1);
			int kmax = std::min(geometry.n_cells[1], (int)(1.5 * ratio[2] * geometry.n_cells[2]) + 1);

			int i,j,k;
			scalar dx, delta, radius;
			Vector3 x0={0,0,0}, x1={0,0,0};

			// Abort condidions for all 3 vectors
			if (a.norm() == 0.0) imax = 0;
			if (b.norm() == 0.0) jmax = 0;
			if (c.norm() == 0.0) kmax = 0;

			for (int iatom = 0; iatom < geometry.n_spins_basic_domain; ++iatom)
			{
				x0 = geometry.basis_atoms[iatom];
				for (i = imax; i >= -imax; --i)
				{
					for (j = jmax; j >= -jmax; --j)
					{
						for (k = kmax; k >= -kmax; --k)
						{
							for (int jatom = 0; jatom < geometry.n_spins_basic_domain; ++jatom)
							{
								x1 = geometry.basis_atoms[jatom] + i*a + j*b + k*c;
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
	}

	void Neighbours::Create_Dipole_Pairs(const Data::Geometry & geometry, scalar dd_radius,
		std::vector<indexPairs> & DD_indices, std::vector<scalarfield> & DD_magnitude, std::vector<vectorfield> & DD_normal)
	{
		// ------ Find the pairs for the first cell ------
		Vector3 vector_ij,  build_array, ipos, jpos;
		scalar magnitude;
		
		int iatom, jatom;
		int da, db, dc;
		int sign_a, sign_b, sign_c;
		int pair_da, pair_db, pair_dc;
		int na, nb, nc;

		int idx_i = 0, idx_j = 0;
		int Na = geometry.n_cells[0];
		int Nb = geometry.n_cells[1];
		int Nc = geometry.n_cells[2];
		int N = geometry.n_spins_basic_domain;
		int nos = geometry.nos;
		
		int periods_a, periods_b, periods_c, pair_periodicity;

		// Loop over all basis atoms
		for (iatom = 0; iatom < N; ++iatom)
		{
			for (jatom = 0; jatom < N; ++jatom)
			{
				// Because the terms with the largest distance are the smallest, we start with the largest indices
				for (da = Na-1; da >= 0; --da)
				{
					for (db = Nb-1; db >= 0; --db)
					{
						for (dc = Nc-1; dc >= 0; --dc)
						{
							for (sign_a = -1; sign_a <= 1; sign_a+=2)
							{
							for (sign_b = -1; sign_b <= 1; sign_b+=2)
							{
							for (sign_c = -1; sign_c <= 1; sign_c+=2)
							{
								pair_da = sign_a * da;
								pair_db = sign_b * db;
								pair_dc = sign_c * dc;
								// Calculate positions and difference vector
								ipos 		= geometry.spin_pos[iatom];
								jpos 		= geometry.spin_pos[jatom]
													+ geometry.translation_vectors[0]*pair_da
													+ geometry.translation_vectors[1]*pair_db
													+ geometry.translation_vectors[2]*pair_dc;
								vector_ij  = jpos - ipos;
								
								// Length of difference vector
								magnitude = vector_ij.norm();
								if ( magnitude==0.0 || (da==0 && sign_a==-1) || (db==0 && sign_b==-1) || (dc==0 && sign_c==-1) )
								{
									magnitude = dd_radius + 1.0;
								}
								// Check if inside DD radius
								if ( magnitude - dd_radius < 1.0E-5 )
								{
									// std::cerr << "found " << iatom << " " << jatom << std::endl;
									// std::cerr << "      " << pair_da << " " << pair_db << " " << pair_dc << std::endl;
									// Normal
									vector_ij.normalize();

									// ------ Translate for the whole lattice ------
									// Create all Pairs of this Kind through translation
									for (na = 0; na < Na; ++na)
									{
										for (nb = 0; nb < Nb; ++nb)
										{
											for (nc = 0; nc < Nc; ++nc)
											{
												idx_i = iatom + N*na + N*Na*nb + N*Na*Nb*nc;
												// na + pair_da is absolute position of cell in x direction
												// if (na + pair_da) > Na (number of atoms in x)
												// go to the other side with % Na
												// if (na + pair_da) negative (or multiply (of Na) negative)
												// add Na and modulo again afterwards
												// analogous for y and z direction with nb, nc
												idx_j = jatom	+ N*( (((na + pair_da) % Na) + Na) % Na )
																+ N*Na*( (((nb + pair_db) % Nb) + Nb) % Nb )
																+ N*Na*Nb*( (((nc + pair_dc) % Nc) + Nc) % Nc );
												// Determine the periodicity
												periods_a = (na + pair_da) / Na;
												periods_b = (nb + pair_db) / Nb;
												periods_c = (nc + pair_dc) / N;
												//		none
												if (periods_a == 0 && periods_b == 0 && periods_c == 0)
												{
													pair_periodicity = 0;
												}
												//		a
												else if (periods_a != 0 && periods_b == 0 && periods_c == 0)
												{
													pair_periodicity = 1;
												}
												//		b
												else if (periods_a == 0 && periods_b != 0 && periods_c == 0)
												{
													pair_periodicity = 2;
												}
												//		c
												else if (periods_a == 0 && periods_b == 0 && periods_c != 0)
												{
													pair_periodicity = 3;
												}
												//		ab
												else if (periods_a != 0 && periods_b != 0 && periods_c == 0)
												{
													pair_periodicity = 4;
												}
												//		ac
												else if (periods_a != 0 && periods_b == 0 && periods_c != 0)
												{
													pair_periodicity = 5;
												}
												//		bc
												else if (periods_a == 0 && periods_b != 0 && periods_c != 0)
												{
													pair_periodicity = 6;
												}
												//		abc
												else if (periods_a != 0 && periods_b != 0 && periods_c != 0)
												{
													pair_periodicity = 7;
												}

												// Add the indices and parameters to the corresponding lists
												if (idx_i < idx_j)
												{
													// std::cerr << "   made pair " << idx_i << " " << idx_j << std::endl;
													DD_indices[pair_periodicity].push_back(indexPair{ idx_i, idx_j });
													DD_magnitude[pair_periodicity].push_back(magnitude);
													DD_normal[pair_periodicity].push_back(vector_ij);
												}
											}// end for nc
										}// end for nb
									}// end for na
								}// end if in radius
							}
							}
							}
							// else 
							// 	std::cerr << "outed " << iatom << " " << jatom << " with d=" << magnitude << std::endl;
						}// end for pair_dc
					}// end for pair_db
				}// end for pair_da
			}// end for jatom
		}// end for iatom
	}

}// end Namespace Engine