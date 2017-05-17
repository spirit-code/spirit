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
								Log(Log_Level::Warning, Log_Sender::IO, "NOI(file) > NOI(chain). Appending image " + std::to_string(iimage+1));
								// Copy Image
								auto new_system = std::make_shared<Data::Spin_System>(Data::Spin_System(*c->images[iimage-1]));
								// Add to chain
								c->noi++;
								c->images.push_back(new_system);
								c->image_type.push_back(Data::GNEB_Image_Type::Normal);
								noi = c->noi;
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


		void External_Field_from_File(const std::string externalFieldFile, const Data::Geometry & geometry, int & n_indices,
			intfield & external_field_index, scalarfield & external_field_magnitude, vectorfield & external_field_normal)
		{
			Log(Log_Level::Info, Log_Sender::IO, "Reading external field from file " + externalFieldFile);
			try
			{
				n_indices = 0;
				std::vector<std::string> columns(5);	// at least: 1 (index) + 3 (K)
				// column indices of pair indices and interactions
				int col_i = -1, col_B = -1, col_Bx = -1, col_By = -1, col_Bz = -1, col_Ba = -1, col_Bb = -1, col_Bc = -1;
				bool B_magnitude = false, B_xyz = false, B_abc = false;
				Vector3 B_temp = { 0, 0, 0 };
				// Get column indices
				IO::Filter_File_Handle file(externalFieldFile);
				file.GetLine(); // first line contains the columns
				for (unsigned int i = 0; i < columns.size(); ++i)
				{
					file.iss >> columns[i];
					if (!columns[i].compare(0, 1, "i"))	col_i = i;
					else if (!columns[i].compare(0, 2, "B")) { col_B = i;	B_magnitude = true; }
					else if (!columns[i].compare(0, 2, "Bx"))	col_Bx = i;
					else if (!columns[i].compare(0, 2, "By"))	col_By = i;
					else if (!columns[i].compare(0, 2, "Bz"))	col_Bz = i;
					else if (!columns[i].compare(0, 2, "Ba"))	col_Ba = i;
					else if (!columns[i].compare(0, 2, "Bb"))	col_Bb = i;
					else if (!columns[i].compare(0, 2, "Bc"))	col_Bc = i;

					if (col_Bx >= 0 && col_By >= 0 && col_Bz >= 0) B_xyz = true;
					if (col_Ba >= 0 && col_Bb >= 0 && col_Bc >= 0) B_abc = true;
				}

				if (!B_xyz && !B_abc) Log(Log_Level::Warning, Log_Sender::IO, "No external field data could be found in header of file " + externalFieldFile);

				// Catch horizontal separation Line
				// file.GetLine();
				// Get number of lines
				while (file.GetLine()) { ++n_indices; }

				// Indices
				int spin_i = 0;
				scalar spin_B = 0, spin_B1 = 0, spin_B2 = 0, spin_B3 = 0;
				// Arrays
				external_field_index = intfield(0);
				external_field_magnitude = scalarfield(0);
				external_field_normal = vectorfield(0);

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
						else if (i == col_B)
							file.iss >> spin_B;
						else if (i == col_Bx && B_xyz)
							file.iss >> spin_B1;
						else if (i == col_By && B_xyz)
							file.iss >> spin_B2;
						else if (i == col_Bz && B_xyz)
							file.iss >> spin_B3;
						else if (i == col_Ba && B_abc)
							file.iss >> spin_B1;
						else if (i == col_Bb && B_abc)
							file.iss >> spin_B2;
						else if (i == col_Bc && B_abc)
							file.iss >> spin_B3;
						else
							file.iss >> sdump;
					}
					B_temp = { spin_B1, spin_B2, spin_B3 };
					// B_temp.normalize();
					// spin_B1 = B_temp[0]; spin_B2 = B_temp[1]; spin_B3 = B_temp[2];
					// Anisotropy vector orientation
					if (B_abc)
					{
						spin_B1 = B_temp.dot(geometry.basis[0]);
						spin_B2 = B_temp.dot(geometry.basis[1]);
						spin_B3 = B_temp.dot(geometry.basis[2]);
						B_temp = { spin_B1, spin_B2, spin_B3 };
					}
					// Anisotropy vector normalisation
					if (B_magnitude)
					{
						scalar dnorm = B_temp.norm();
						if (dnorm != 0)
							B_temp.normalize();
					}
					else
					{
						spin_B = B_temp.norm();
						if (spin_B != 0)
							B_temp.normalize();
					}

					if (spin_B != 0)
					{
						external_field_index.push_back(spin_i);
						external_field_magnitude.push_back(spin_B);
						external_field_normal.push_back(B_temp);
					}

				}// end while getline
			}// end try
			catch (Exception ex)
			{
				throw ex;
			}

		}


		/*
		Read from Anisotropy file
		*/
		void Anisotropy_from_File(const std::string anisotropyFile, const Data::Geometry & geometry, int & n_indices,
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
					// K_temp.normalize();
					// spin_K1 = K_temp[0]; spin_K2 = K_temp[1]; spin_K3 = K_temp[2];
					// Anisotropy vector orientation
					if (K_abc)
					{
						spin_K1 = K_temp.dot(geometry.basis[0]);
						spin_K2 = K_temp.dot(geometry.basis[1]);
						spin_K3 = K_temp.dot(geometry.basis[2]);
						K_temp = { spin_K1, spin_K2, spin_K3 };
					}
					// Anisotropy vector normalisation
					if (K_magnitude)
					{
						scalar dnorm = K_temp.norm();
						if (dnorm != 0)
							K_temp.normalize();
					}
					else
					{
						spin_K = K_temp.norm();
						if (spin_K != 0)
							K_temp.normalize();
					}

					// TODO: propagation of basis anisotropy across lattice

					if (spin_K != 0)
					{
						anisotropy_index.push_back(spin_i);
						anisotropy_magnitude.push_back(spin_K);
						anisotropy_normal.push_back(K_temp);
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
			pairfield & exchange_pairs, scalarfield & exchange_magnitudes,
			pairfield & dmi_pairs, scalarfield & dmi_magnitudes, vectorfield & dmi_normals)
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
					

					// Add the indices and parameters to the corresponding lists
					if (pair_Jij != 0)
					{
						exchange_pairs.push_back({ pair_i, pair_i, { pair_da, pair_db, pair_dc } });
						exchange_magnitudes.push_back(pair_Jij);
					}
					if (pair_Dij != 0)
					{
						dmi_pairs.push_back({ pair_i, pair_i, { pair_da, pair_db, pair_dc } });
						dmi_magnitudes.push_back(pair_Dij);
						dmi_normals.push_back(Vector3{pair_D1, pair_D2, pair_D3});
					}

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
			quadrupletfield & quadruplets, scalarfield & quadruplet_magnitudes)
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
					

					// Add the indices and parameter to the corresponding list
					if (q_Q != 0)
					{
						quadruplets.push_back({ q_i, q_j, q_k, q_l, { q_da_j, q_da_k, q_da_l } });
						quadruplet_magnitudes.push_back(q_Q);
					}

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