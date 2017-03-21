#include <utility/IO.hpp>
#include <utility/IO_Filter_File_Handle.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>
#include <engine/Vectormath.hpp>

#include <iostream>
#include <fstream>
#include <thread>
#include <string>
#include <sstream>
#include <algorithm>

namespace Utility
{
	namespace IO
	{
		std::string int_to_formatted_string(int in, int n)
		{
			// The format string
			std::string format = "%";
			format += std::to_string(0) + std::to_string(n) + "i";
			// The buffer
			const int buffer_length = 80;
			std::string out = "";
			char buffer_string_conversion[buffer_length + 2];
			//std::cout << format << std::endl;
			// Write formatted into buffer
			snprintf(buffer_string_conversion, buffer_length, format.c_str(), in);
			// Write buffer into out string
			out.append(buffer_string_conversion);
			// Return
			return out;
		}

		// TODO: this function does not make much sense... need to do this stuff coherently throughout the parser...
		std::vector<scalar> split_string_to_scalar(const std::string& source, const std::string& delimiter)
		{
			std::vector<scalar> result;

			scalar temp;
			std::stringstream ss(source);
			while (ss >> temp)
			{
				result.push_back(temp);

				if (ss.peek() == ',' || ss.peek() == ' ')
					ss.ignore();
			}

			return result;
		}

		/*
		Reads a configuration file into an existing Spin_System
		*/
		void Read_Spin_Configuration(std::shared_ptr<Data::Spin_System> s, const std::string file, VectorFileFormat format)
		{
			std::ifstream myfile(file);
			if (myfile.is_open())
			{
				Log(Log_Level::Info, Log_Sender::IO, std::string("Reading Spins File ").append(file));
				std::string line = "";
				std::istringstream iss(line);
				std::size_t found;
				int i = 0;
				if (format == VectorFileFormat::CSV_POS_SPIN)
				{
					auto& spins = *s->spins;
					while (getline(myfile, line))
					{
						if (i >= s->nos) { Log(Log_Level::Warning, Log_Sender::IO, "NOS mismatch in Read Spin Configuration - Aborting"); myfile.close(); return; }
						found = line.find("#");
						// Read the line if # is not found (# marks a comment)
						if (found == std::string::npos)
						{
							auto x = split_string_to_scalar(line, ",");
							spins[i][0] = x[3];
							spins[i][1] = x[4];
							spins[i][2] = x[5];
							++i;
						}// endif (# not found)
						 // discard line if # is found
					}// endif new line (while)
				}
				else
				{
					auto& spins = *s->spins;
					while (getline(myfile, line))
					{
						if (i >= s->nos) { Log(Log_Level::Warning, Log_Sender::IO, "NOS mismatch in Read Spin Configuration - Aborting"); myfile.close(); return; }
						found = line.find("#");
						// Read the line if # is not found (# marks a comment)
						if (found == std::string::npos)
						{
							//scalar x, y, z;
							iss.clear();
							iss.str(line);
							//iss >> x >> y >> z;
							iss >> spins[i][0] >> spins[i][1] >> spins[i][2];
							++i;
						}// endif (# not found)
						 // discard line if # is found
					}// endif new line (while)
				}
				if (i < s->nos) { Log(Log_Level::Warning, Log_Sender::IO, "NOS mismatch in Read Spin Configuration"); }
				myfile.close();
				Log(Log_Level::Info, Log_Sender::IO, "Done");
			}
		}


		void Read_SpinChain_Configuration(std::shared_ptr<Data::Spin_System_Chain> c, const std::string file)
		{
			std::ifstream myfile(file);
			if (myfile.is_open())
			{
				Log(Log_Level::Info, Log_Sender::IO, std::string("Reading SpinChain File ").append(file));
				std::string line = "";
				std::istringstream iss(line);
				std::size_t found;
				int i = 0, iimage = -1, nos = c->images[0]->nos, noi = c->noi;
				while (getline(myfile, line))
				{
					found = line.find("#");
					if (found == std::string::npos)		// Read the line if # is not found (# marks a comment)
					{
						found = line.find("Image No");
						if (found != std::string::npos)	// Set counters if 'Image No' was found
						{
							if (i < nos && iimage>0)	// Check if less than NOS spins were read for the image before
							{
								Log(Log_Level::Warning, Log_Sender::IO, std::string("NOS(image) > NOS(file) in image ").append(std::to_string(iimage)));
							}
							++iimage;
							i = 0;
							if (iimage >= noi)
							{
								Log(Log_Level::Warning, Log_Sender::IO, "NOI(file) > NOI(chain)");
							}
							else
							{
								nos = c->images[iimage]->nos; // Note: different NOS in different images is currently not supported
							}
						}//endif "Image No"
						else	// The line should contain a spin
						{
							if (iimage >= noi)
							{
								Log(Log_Level::Warning, Log_Sender::IO, "NOI(file) > NOI(chain). Appending image " + std::to_string(iimage));
								auto new_system = std::make_shared<Data::Spin_System>(Data::Spin_System(*c->images[iimage-1]));
								c->images.push_back(new_system);
							}
							nos = c->images[iimage]->nos; // Note: different NOS in different images is currently not supported

							if (i >= nos)
							{
								Log(Log_Level::Warning, Log_Sender::IO, std::string("NOS missmatch in image ").append(std::to_string(iimage)));
								Log(Log_Level::Warning, Log_Sender::IO, std::string("NOS(file) > NOS(image)"));
								//Log(Log_Level::Warning, Log_Sender::IO, std::string("Aborting Loading of SpinChain Configuration ").append(file));
								//myfile.close();
								//return;
							}
							else
							{
								iss.clear();
								iss.str(line);
								auto& spins = *c->images[iimage]->spins;
								//iss >> x >> y >> z;
								iss >> spins[i][0] >> spins[i][1] >> spins[i][2];
							}
							++i;
						}//end else
					}// endif (# not found)
					 // discard line if # is found
				}// endif new line (while)
				if (i < nos) Log(Log_Level::Warning, Log_Sender::IO, std::string("NOS(image) > NOS(file) in image ").append(std::to_string(iimage - 1)));
				if (iimage < noi-1) Log(Log_Level::Warning, Log_Sender::IO, "NOI(chain) > NOI(file)");
				myfile.close();
				Log(Log_Level::Info, Log_Sender::IO, std::string("Done Reading SpinChain File ").append(file));
			}
		}

		/*std::vector<std::vector<scalar>> External_Field_from_File(int nos, const std::string externalFieldFile)
		{

		}*/


		/*
		Read from Anisotropy file
		*/
		void Anisotropy_from_File(const std::string anisotropyFile, Data::Geometry geometry, int & n_indices,
			intfield & anisotropy_index, scalarfield & anisotropy_magnitude,
			vectorfield & anisotropy_normal)
		{
			Log(Log_Level::Info, Log_Sender::IO, "Reading anisotropy from file " + anisotropyFile);
			try {
				n_indices = 0;
				std::vector<std::string> columns(5);	// at least: 1 (index) + 3 (K)
				// column indices of pair indices and interactions
				int col_i = -1, col_K = -1, col_Kx = -1, col_Ky = -1, col_Kz = -1, col_Ka = -1, col_Kb = -1, col_Kc = -1;
				bool K_magnitude = false, K_xyz = false, K_abc = false;
				Vector3 K_temp = { 0, 0, 0 };
				// Get column indices
				IO::Filter_File_Handle file(anisotropyFile);
				file.GetLine(); // first line contains the columns
				for (unsigned int i = 0; i < columns.size(); ++i)
				{
					file.iss >> columns[i];
					if (!columns[i].compare(0, 1, "i"))	col_i = i;
					else if (!columns[i].compare(0, 2, "K")) { col_K = i;	K_magnitude = true; }
					else if (!columns[i].compare(0, 2, "Kx"))	col_Kx = i;
					else if (!columns[i].compare(0, 2, "Ky"))	col_Ky = i;
					else if (!columns[i].compare(0, 2, "Kz"))	col_Kz = i;
					else if (!columns[i].compare(0, 2, "Ka"))	col_Ka = i;
					else if (!columns[i].compare(0, 2, "Kb"))	col_Kb = i;
					else if (!columns[i].compare(0, 2, "Kc"))	col_Kc = i;

					if (col_Kx >= 0 && col_Ky >= 0 && col_Kz >= 0) K_xyz = true;
					if (col_Ka >= 0 && col_Kb >= 0 && col_Kc >= 0) K_abc = true;
				}

				if (!K_xyz && !K_abc) Log(Log_Level::Warning, Log_Sender::IO, "No anisotropy data could be found in header of file " + anisotropyFile);

				// Catch horizontal separation Line
				// file.GetLine();
				// Get number of lines
				while (file.GetLine()) { ++n_indices; }

				// Indices
				int spin_i = 0;
				scalar spin_K = 0, spin_K1 = 0, spin_K2 = 0, spin_K3 = 0;
				// Arrays
				anisotropy_index = intfield(0);
				anisotropy_magnitude = scalarfield(0);
				anisotropy_normal = vectorfield(0);

				// Get actual Data
				file.ResetStream();
				int i_pair = 0;
				std::string sdump;
				file.GetLine();	// skip first line
								//dataHandle.GetLine();	// skip second line
				while (file.GetLine())
				{
					// Read a line from the File
					for (unsigned int i = 0; i < columns.size(); ++i)
					{
						if (i == col_i)
							file.iss >> spin_i;
						else if (i == col_K)
							file.iss >> spin_K;
						else if (i == col_Kx && K_xyz)
							file.iss >> spin_K1;
						else if (i == col_Ky && K_xyz)
							file.iss >> spin_K2;
						else if (i == col_Kz && K_xyz)
							file.iss >> spin_K3;
						else if (i == col_Ka && K_abc)
							file.iss >> spin_K1;
						else if (i == col_Kb && K_abc)
							file.iss >> spin_K2;
						else if (i == col_Kc && K_abc)
							file.iss >> spin_K3;
						else
							file.iss >> sdump;
					}
					K_temp = { spin_K1, spin_K2, spin_K3 };
					K_temp.normalize();
					spin_K1 = K_temp[0]; spin_K2 = K_temp[1]; spin_K3 = K_temp[2];
					// Anisotropy vector orientation
					if (K_abc)
					{
						spin_K1 = K_temp.dot(geometry.basis[0]);
						spin_K2 = K_temp.dot(geometry.basis[1]);
						spin_K3 = K_temp.dot(geometry.basis[2]);
					}
					// Anisotropy vector normalisation
					if (K_magnitude)
					{
						scalar dnorm = std::sqrt(std::pow(spin_K1, 2) + std::pow(spin_K2, 2) + std::pow(spin_K3, 2));
						if (dnorm != 0)
						{
							spin_K1 = spin_K1 / dnorm;
							spin_K2 = spin_K2 / dnorm;
							spin_K3 = spin_K3 / dnorm;
						}
					}
					else
					{
						spin_K = std::sqrt(std::pow(spin_K1, 2) + std::pow(spin_K2, 2) + std::pow(spin_K3, 2));
						if (spin_K != 0)
						{
							spin_K1 = spin_K1 / spin_K;
							spin_K2 = spin_K2 / spin_K;
							spin_K3 = spin_K3 / spin_K;
						}
					}

					// TODO: propagation of basis anisotropy across lattice

					if (spin_K != 0)
					{
						anisotropy_index.push_back(spin_i);
						anisotropy_magnitude.push_back(spin_K);
						anisotropy_normal.push_back(Vector3{spin_K1, spin_K2, spin_K3});
					}

				}// end while getline
			}// end try
			catch (Exception ex)
			{
				throw ex;
			}
		}

		/*
		Read from Pairs file by Markus & Bernd
		*/
		void Pairs_from_File(const std::string pairsFile, Data::Geometry geometry, int & nop,
			std::vector<indexPairs> & Exchange_indices, std::vector<scalarfield> & Exchange_magnitude,
			std::vector<indexPairs> & DMI_indices, std::vector<scalarfield> & DMI_magnitude, std::vector<vectorfield> & DMI_normal)
		{
			Log(Log_Level::Info, Log_Sender::IO, "Reading spin pairs from file " + pairsFile);
			try {
				nop = 0;
				std::vector<std::string> columns(20);	// at least: 2 (indices) + 3 (J) + 3 (DMI)
				// column indices of pair indices and interactions
				int col_i = -1, col_j = -1, col_da = -1, col_db = -1, col_dc = -1,
					col_J = -1, col_DMIx = -1, col_DMIy = -1, col_DMIz = -1,
					col_Dij = -1, col_DMIa = -1, col_DMIb = -1, col_DMIc = -1;
				bool J = false, DMI_xyz = false, DMI_abc = false, Dij = false;
				int pair_periodicity = 0;
				Vector3 pair_D_temp = { 0, 0, 0 };
				// Get column indices
				IO::Filter_File_Handle file(pairsFile);
				file.GetLine(); // first line contains the columns
				for (unsigned int i = 0; i < columns.size(); ++i)
				{
					file.iss >> columns[i];
					if      (!columns[i].compare(0, 1, "i"))	col_i = i;
					else if (!columns[i].compare(0, 1, "j"))	col_j = i;
					else if (!columns[i].compare(0, 2, "da"))	col_da = i;
					else if (!columns[i].compare(0, 2, "db"))	col_db = i;
					else if (!columns[i].compare(0, 2, "dc"))	col_dc = i;
					else if (!columns[i].compare(0, 2, "J"))	{ col_J = i;	J = true; }
					else if (!columns[i].compare(0, 2, "Dij"))	{ col_Dij = i;	Dij = true; }
					else if (!columns[i].compare(0, 2, "Dx"))	col_DMIx = i;
					else if (!columns[i].compare(0, 2, "Dy"))	col_DMIy = i;
					else if (!columns[i].compare(0, 2, "Dz"))	col_DMIz = i;
					else if (!columns[i].compare(0, 2, "Da"))	col_DMIx = i;
					else if (!columns[i].compare(0, 2, "Db"))	col_DMIy = i;
					else if (!columns[i].compare(0, 2, "Dc"))	col_DMIz = i;

					if (col_DMIx >= 0 && col_DMIy >= 0 && col_DMIz >= 0) DMI_xyz = true;
					if (col_DMIa >= 0 && col_DMIb >= 0 && col_DMIc >= 0) DMI_abc = true;
				}

				// Check if interactions have been found in header
				if (!J && !DMI_xyz && !DMI_abc) Log(Log_Level::Warning, Log_Sender::IO, "No interactions could be found in header of pairs file " + pairsFile);

				// Catch horizontal separation Line
				file.GetLine();
				// Get number of pairs
				while (file.GetLine()) { ++nop; }

				// Create Pairs Vector
				//pairs = std::vector<Data::Spin_Pair>(nop);

				// Pair Indices
				int pair_i = 0, pair_j = 0, pair_da = 0, pair_db = 0, pair_dc = 0;
				scalar pair_Jij = 0, pair_Dij = 0, pair_D1 = 0, pair_D2 = 0, pair_D3 = 0;
				// Get actual Pairs Data
				file.ResetStream();
				int i_pair = 0;
				std::string sdump;
				file.GetLine();	// skip first line
				//dataHandle.GetLine();	// skip second line
				while (file.GetLine())
				{
					// Read a Pair from the File
					for (unsigned int i = 0; i < columns.size(); ++i)
					{
						if (i == col_i)
							file.iss >> pair_i;
						else if (i == col_j)
							file.iss >> pair_j;
						else if (i == col_da)
							file.iss >> pair_da;
						else if (i == col_db)
							file.iss >> pair_db;
						else if (i == col_dc)
							file.iss >> pair_dc;
						else if (i == col_J && J)
							file.iss >> pair_Jij;
						else if (i == col_Dij && Dij)
							file.iss >> pair_Dij;
						else if (i == col_DMIa && DMI_abc)
							file.iss >> pair_D1;
						else if (i == col_DMIb && DMI_abc)
							file.iss >> pair_D2;
						else if (i == col_DMIc && DMI_abc)
							file.iss >> pair_D3;
						else if (i == col_DMIx && DMI_xyz)
							file.iss >> pair_D1;
						else if (i == col_DMIy && DMI_xyz)
							file.iss >> pair_D2;
						else if (i == col_DMIz && DMI_xyz)
							file.iss >> pair_D3;
						else
							file.iss >> sdump;
					}// end for columns

					// DMI vector orientation
					if (DMI_abc)
					{
						pair_D_temp = { pair_D1, pair_D2, pair_D3 };
						pair_D1 = pair_D_temp.dot(geometry.basis[0]);
						pair_D2 = pair_D_temp.dot(geometry.basis[1]);
						pair_D3 = pair_D_temp.dot(geometry.basis[2]);
					}
					// DMI vector normalisation
					if (Dij)
					{
						scalar dnorm = std::sqrt(std::pow(pair_D1, 2) + std::pow(pair_D2, 2) + std::pow(pair_D3, 2));
						if (dnorm != 0)
						{
							pair_D1 = pair_D1 / dnorm;
							pair_D2 = pair_D2 / dnorm;
							pair_D3 = pair_D3 / dnorm;
						}
					}
					else
					{
						pair_Dij = std::sqrt(std::pow(pair_D1, 2) + std::pow(pair_D2, 2) + std::pow(pair_D3, 2));
						if (pair_Dij != 0)
						{
							pair_D1 = pair_D1 / pair_Dij;
							pair_D2 = pair_D2 / pair_Dij;
							pair_D3 = pair_D3 / pair_Dij;
						}
					}
					

					// Create all Pairs of this Kind through translation
					int idx_i = 0, idx_j = 0;
					int Na = geometry.n_cells[0];
					int Nb = geometry.n_cells[1];
					int Nc = geometry.n_cells[2];
					int N = geometry.n_spins_basic_domain;
					int periods_a = 0, periods_b = 0, periods_c = 0;
					for (int na = 0; na < Na; ++na)
					{
						for (int nb = 0; nb < Nb; ++nb)
						{
							for (int nc = 0; nc < Nc; ++nc)
							{
								idx_i = pair_i + N*na + N*Na*nb + N*Na*Nb*nc;
								// na + pair_da is absolute position of cell in x direction
								// if (na + pair_da) > Na (number of atoms in x)
								// go to the other side with % Na
								// if (na + pair_da) negative (or multiply (of Na) negative)
								// add Na and modulo again afterwards
								// analogous for y and z direction with nb, nc
								periods_a = (na + pair_da) / Na;
								periods_b = (nb + pair_db) / Nb;
								periods_c = (nc + pair_dc) / Nc;

								// Catch cases of negative periodicity
								if (na + pair_da < 0) periods_a = -1;
								if (nb + pair_db < 0) periods_b = -1;
								if (nc + pair_dc < 0) periods_c = -1;

								idx_j = pair_j	+ N*( (((na + pair_da) % Na) + Na) % Na )
												+ N*Na*( (((nb + pair_db) % Nb) + Nb) % Nb )
												+ N*Na*Nb*( (((nc + pair_dc) % Nc) + Nc) % Nc );
								// Determine the periodicity
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
								if (pair_Jij != 0)
								{
									Exchange_indices[pair_periodicity].push_back(indexPair{ idx_i, idx_j });
									Exchange_magnitude[pair_periodicity].push_back(pair_Jij);
								}
								if (pair_Dij != 0)
								{
									DMI_indices[pair_periodicity].push_back(indexPair{ idx_i, idx_j });
									DMI_magnitude[pair_periodicity].push_back(pair_Dij);
									DMI_normal[pair_periodicity].push_back(Vector3{pair_D1, pair_D2, pair_D3});
								}
							}
						}
					}// end for translations

					++i_pair;
				}// end while GetLine
				Log(Log_Level::Info, Log_Sender::IO, "Done reading " + std::to_string(i_pair) + " spin pairs from file " + pairsFile);
			}// end try
			catch (Exception ex)
			{
				if (ex == Exception::File_not_Found)
					Log(Log_Level::Error, Log_Sender::IO, "Could not read pairs file " + pairsFile);
				else
					throw ex;
			}
		}



		/*
		Read from Quadruplet file
		*/
		void Quadruplets_from_File(const std::string quadrupletsFile, Data::Geometry geometry, int & noq,
			std::vector<indexQuadruplets> & quadruplet_indices, std::vector<scalarfield> & quadruplet_magnitude)
		{
			Log(Log_Level::Info, Log_Sender::IO, "Reading spin quadruplets from file " + quadrupletsFile);
			try {
				noq = 0;
				std::vector<std::string> columns(20);	// at least: 4 (indices) + 3*3 (positions) + 1 (magnitude)
				// column indices of pair indices and interactions
				int col_i = -1;
				int col_j = -1, col_da_j = -1, col_db_j = -1, col_dc_j = -1, periodicity_j = 0;
				int col_k = -1, col_da_k = -1, col_db_k = -1, col_dc_k = -1, periodicity_k = 0;
				int col_l = -1, col_da_l = -1, col_db_l = -1, col_dc_l = -1, periodicity_l = 0;
				int	col_Q = -1;
				bool Q = false;
				int max_periods_a = 0, max_periods_b = 0, max_periods_c = 0;
				int quadruplet_periodicity = 0;
				// Get column indices
				IO::Filter_File_Handle file(quadrupletsFile);
				file.GetLine(); // first line contains the columns
				for (unsigned int i = 0; i < columns.size(); ++i)
				{
					file.iss >> columns[i];
					if      (!columns[i].compare(0, 1, "i"))	col_i = i;
					else if (!columns[i].compare(0, 1, "j"))	col_j = i;
					else if (!columns[i].compare(0, 4, "da_j"))	col_da_j = i;
					else if (!columns[i].compare(0, 4, "db_j"))	col_db_j = i;
					else if (!columns[i].compare(0, 4, "dc_j"))	col_dc_j = i;
					else if (!columns[i].compare(0, 1, "k"))	col_k = i;
					else if (!columns[i].compare(0, 4, "da_k"))	col_da_k = i;
					else if (!columns[i].compare(0, 4, "db_k"))	col_db_k = i;
					else if (!columns[i].compare(0, 4, "dc_k"))	col_dc_k = i;
					else if (!columns[i].compare(0, 1, "l"))	col_l = i;
					else if (!columns[i].compare(0, 4, "da_l"))	col_da_l = i;
					else if (!columns[i].compare(0, 4, "db_l"))	col_db_l = i;
					else if (!columns[i].compare(0, 4, "dc_l"))	col_dc_l = i;
					else if (!columns[i].compare(0, 1, "Q"))	{ col_Q = i;	Q = true; }
				}

				// Check if interactions have been found in header
				if (!Q) Log(Log_Level::Warning, Log_Sender::IO, "No interactions could be found in header of quadruplets file " + quadrupletsFile);

				// Catch horizontal separation Line
				file.GetLine();
				// Get number of pairs
				while (file.GetLine()) { ++noq; }

				// Quadruplet Indices
				int q_i = 0;
				int q_j = 0, q_da_j = 0, q_db_j = 0, q_dc_j = 0;
				int q_k = 0, q_da_k = 0, q_db_k = 0, q_dc_k = 0;
				int q_l = 0, q_da_l = 0, q_db_l = 0, q_dc_l = 0;
				scalar q_Q;
				// Get actual Pairs Data
				file.ResetStream();
				int i_quadruplet = 0;
				std::string sdump;
				file.GetLine();	// skip first line
				//dataHandle.GetLine();	// skip second line
				while (file.GetLine())
				{
					// Read a Pair from the File
					for (unsigned int i = 0; i < columns.size(); ++i)
					{
						// i
						if (i == col_i)
							file.iss >> q_i;
						// j
						else if (i == col_j)
							file.iss >> q_j;
						else if (i == col_da_j)
							file.iss >> q_da_j;
						else if (i == col_db_j)
							file.iss >> q_db_j;
						else if (i == col_dc_j)
							file.iss >> q_dc_j;
						// k
						else if (i == col_k)
							file.iss >> q_k;
						else if (i == col_da_k)
							file.iss >> q_da_k;
						else if (i == col_db_k)
							file.iss >> q_db_k;
						else if (i == col_dc_k)
							file.iss >> q_dc_k;
						// l
						else if (i == col_l)
							file.iss >> q_l;
						else if (i == col_da_l)
							file.iss >> q_da_l;
						else if (i == col_db_l)
							file.iss >> q_db_l;
						else if (i == col_dc_l)
							file.iss >> q_dc_l;
						// Quadruplet magnitude
						else if (i == col_Q && Q)
							file.iss >> q_Q;
						// Otherwise dump the line
						else
							file.iss >> sdump;
					}// end for columns
					

					auto periodicity = [] (int periods_a, int periods_b, int periods_c) -> int
					{
						// Determine the periodicity
						//		none
						if (periods_a == 0 && periods_b == 0 && periods_c == 0)
						{
							return 0;
						}
						//		a
						else if (periods_a != 0 && periods_b == 0 && periods_c == 0)
						{
							return 1;
						}
						//		b
						else if (periods_a == 0 && periods_b != 0 && periods_c == 0)
						{
							return 2;
						}
						//		c
						else if (periods_a == 0 && periods_b == 0 && periods_c != 0)
						{
							return 3;
						}
						//		ab
						else if (periods_a != 0 && periods_b != 0 && periods_c == 0)
						{
							return 4;
						}
						//		ac
						else if (periods_a != 0 && periods_b == 0 && periods_c != 0)
						{
							return 5;
						}
						//		bc
						else if (periods_a == 0 && periods_b != 0 && periods_c != 0)
						{
							return 6;
						}
						//		abc
						else if (periods_a != 0 && periods_b != 0 && periods_c != 0)
						{
							return 7;
						}
						else return 0;
					};

					// Create all Pairs of this Kind through translation
					int idx_i = 0, idx_j = 0, idx_k = 0, idx_l = 0;
					int Na = geometry.n_cells[0];
					int Nb = geometry.n_cells[1];
					int Nc = geometry.n_cells[2];
					int N = geometry.n_spins_basic_domain;
					int periods_a_j = 0, periods_b_j = 0, periods_c_j = 0;
					int periods_a_k = 0, periods_b_k = 0, periods_c_k = 0;
					int periods_a_l = 0, periods_b_l = 0, periods_c_l = 0;
					for (int na = 0; na < Na; ++na)
					{
						for (int nb = 0; nb < Nb; ++nb)
						{
							for (int nc = 0; nc < Nc; ++nc)
							{
								idx_i = q_i + N*na + N*Na*nb + N*Na*Nb*nc;
								// na + pair_da is absolute position of cell in x direction
								// if (na + pair_da) > Na (number of atoms in x)
								// go to the other side with % Na
								// if (na + pair_da) negative (or multiply (of Na) negative)
								// add Na and modulo again afterwards
								// analogous for y and z direction with nb, nc

								// j
								periods_a_j = (na + q_da_j) / Na;
								periods_b_j = (nb + q_db_j) / Nb;
								periods_c_j = (nc + q_dc_j) / Nc;
								idx_j = q_j	+ N*( (((na + q_da_j) % Na) + Na) % Na )
											+ N*Na*( (((nb + q_db_j) % Nb) + Nb) % Nb )
											+ N*Na*Nb*( (((nc + q_dc_j) % Nc) + Nc) % Nc );

								// k
								periods_a_k = (na + q_da_k) / Na;
								periods_b_k = (nb + q_db_k) / Nb;
								periods_c_k = (nc + q_dc_k) / Nc;
								idx_k = q_k	+ N*( (((na + q_da_k) % Na) + Na) % Na )
											+ N*Na*( (((nb + q_db_k) % Nb) + Nb) % Nb )
											+ N*Na*Nb*( (((nc + q_dc_k) % Nc) + Nc) % Nc );
								
								// l
								periods_a_l = (na + q_da_l) / Na;
								periods_b_l = (nb + q_db_l) / Nb;
								periods_c_l = (nc + q_dc_l) / Nc;
								idx_l = q_l	+ N*( (((na + q_da_l) % Na) + Na) % Na )
											+ N*Na*( (((nb + q_db_l) % Nb) + Nb) % Nb )
											+ N*Na*Nb*( (((nc + q_dc_l) % Nc) + Nc) % Nc );

								// Periodicity
								// periodicity_j = periodicity(periods_a_j, periods_b_j, periods_c_j)
								// periodicity_k = periodicity(periods_a_k, periods_b_k, periods_c_k)
								// periodicity_l = periodicity(periods_a_l, periods_b_l, periods_c_l)
								max_periods_a = std::max({periods_a_j, periods_a_k, periods_a_l});
								max_periods_b = std::max({periods_b_j, periods_b_k, periods_b_l});
								max_periods_c = std::max({periods_c_j, periods_c_k, periods_c_l});
								quadruplet_periodicity = periodicity(max_periods_a, max_periods_b, max_periods_c);
								

								// Add the indices and parameter to the corresponding list
								if (q_Q != 0)
								{
									quadruplet_indices[quadruplet_periodicity].push_back(indexQuadruplet{ idx_i, idx_j, idx_k, idx_l });
									quadruplet_magnitude[quadruplet_periodicity].push_back(q_Q);
								}
							}
						}
					}// end for translations

					++i_quadruplet;
				}// end while GetLine
				Log(Log_Level::Info, Log_Sender::IO, "Done reading " + std::to_string(i_quadruplet) + " spin quadruplets from file " + quadrupletsFile);
			}// end try
			catch (Exception ex)
			{
				throw ex;
			}
		}
	}
}