#include "IO.h"
#include "Vectormath.h"
#include "IO_Filter_File_Handle.h"

#include <iostream>
#include <fstream>
#include <thread>
#include <string>
#include <sstream>
#include "Logging.h"
#include "Exception.h"

//extern Utility::LoggingHandler Log;

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

		/*
		Reads a configuration file into an existing Spin_System
		*/
		void Read_Spin_Configuration(std::shared_ptr<Data::Spin_System> s, const std::string file)
		{
			std::ifstream myfile(file);
			if (myfile.is_open())
			{
				Log.Send(Utility::Log_Level::INFO, Utility::Log_Sender::IO, std::string("Reading Spins File ").append(file));
				std::string line = "";
				std::istringstream iss(line);
				std::size_t found;
				int i = 0;
				while (getline(myfile, line))
				{
					if (i >= s->nos) { Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::IO, "NOS mismatch in Read Spin Configuration - Aborting"); myfile.close(); return; }
					found = line.find("#");
					// Read the line if # is not found (# marks a comment)
					if (found == std::string::npos)
					{
						//double x, y, z;
						iss.clear();
						iss.str(line);
						//iss >> x >> y >> z;
						iss >> s->spins[i] >> s->spins[1 * s->nos + i] >> s->spins[2 * s->nos + i];
						++i;
					}// endif (# not found)
					 // discard line if # is found
				}// endif new line (while)
				if (i < s->nos) { Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::IO, "NOS mismatch in Read Spin Configuration"); }
				myfile.close();
				Log.Send(Utility::Log_Level::INFO, Utility::Log_Sender::IO, "Done");
			}
		}
		void Read_SpinChain_Configuration(std::shared_ptr<Data::Spin_System_Chain> c, const std::string file)
		{
			std::ifstream myfile(file);
			if (myfile.is_open())
			{
				Log.Send(Utility::Log_Level::INFO, Utility::Log_Sender::IO, std::string("Reading SpinChain File ").append(file));
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
								Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::IO, std::string("NOS(image) > NOS(file) in image ").append(std::to_string(iimage)));
							}
							++iimage;
							i = 0;
							if (iimage >= noi)
							{
								Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::IO, "NOI(file) > NOI(chain)");
							}
							else
							{
								nos = c->images[iimage]->nos; // Note: different NOS in different images is currently not supported
							}
						}//endif "Image No"
						else if (iimage < noi)	// The line should contain a spin
						{

							if (i >= nos)
							{
								Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::IO, std::string("NOS missmatch in image ").append(std::to_string(iimage)));
								Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::IO, std::string("NOS(file) > NOS(image)"));
								//Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::IO, std::string("Aborting Loading of SpinChain Configuration ").append(file));
								//myfile.close();
								//return;
							}
							else
							{
								iss.clear();
								iss.str(line);
								//iss >> x >> y >> z;
								iss >> c->images[iimage]->spins[i] >> c->images[iimage]->spins[1 * nos + i] >> c->images[iimage]->spins[2 * nos + i];
							}
							++i;
						}//end else
					}// endif (# not found)
					 // discard line if # is found
				}// endif new line (while)
				if (i < nos) Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::IO, std::string("NOS(image) > NOS(file) in image ").append(std::to_string(iimage - 1)));
				if (iimage < noi-1) Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::IO, "NOI(chain) > NOI(file)");
				myfile.close();
				Log.Send(Utility::Log_Level::INFO, Utility::Log_Sender::IO, std::string("Done Reading SpinChain File ").append(file));
			}
		}
		/*std::vector<std::vector<double>> External_Field_from_File(int nos, const std::string externalFieldFile)
		{

		}*/


		/*
		Read from Pairs file by Markus & Bernd
		*/
		void Pairs_from_File(const std::string pairsFile, Data::Geometry geometry, int & nop,
			std::vector<std::vector<std::vector<int>>> & Exchange_indices, std::vector<std::vector<double>> & Exchange_magnitude,
			std::vector<std::vector<std::vector<int>>> & DMI_indices, std::vector<std::vector<double>> & DMI_magnitude, std::vector<std::vector<std::vector<double>>> & DMI_normal,
			std::vector<std::vector<std::vector<int>>> & BQC_indices, std::vector<std::vector<double>> & BQC_magnitude)
		{
			Log.Send(Utility::Log_Level::INFO, Utility::Log_Sender::IO, "Reading spin pairs from file " + pairsFile);
			try {
				nop = 0;
				std::vector<std::string> columns(20);	// at least: 2 (indices) + 3 (J) + 3 (DMI) + 1 (BQC)
				// column indices of pair indices and interactions
				int col_i = -1, col_j = -1, col_da = -1, col_db = -1, col_dc = -1,
					col_J = -1, col_DMIx = -1, col_DMIy = -1, col_DMIz = -1, col_BQC = -1,
					col_Dij = -1, col_DMIa = -1, col_DMIb = -1, col_DMIc = -1;
				bool J = false, DMI_xyz = false, DMI_abc = false, Dij = false, BQC = false;
				int pair_periodicity = 0;
				std::vector<double> pair_D_temp = { 0, 0, 0 };
				// Get column indices
				Utility::IO::Filter_File_Handle file(pairsFile);
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
					else if (!columns[i].compare(0, 3, "BQC"))	{ col_BQC = i; BQC = true; }

					if (col_DMIx >= 0 && col_DMIy >= 0 && col_DMIz >= 0) DMI_xyz = true;
					if (col_DMIa >= 0 && col_DMIb >= 0 && col_DMIc >= 0) DMI_abc = true;
				}

				// Check if interactions have been found in header
				if (!J && !BQC && !DMI_xyz && !DMI_abc) Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::IO, "No interactions could be found in header of pairs file " + pairsFile);

				// Catch horizontal separation Line
				file.GetLine();
				// Get number of pairs
				while (file.GetLine()) { ++nop; }

				// Create Pairs Vector
				//pairs = std::vector<Data::Spin_Pair>(nop);

				// Pair Indices
				int pair_i = 0, pair_j = 0, pair_da = 0, pair_db = 0, pair_dc = 0;
				double pair_Jij = 0, pair_Dij = 0, pair_D1 = 0, pair_D2 = 0, pair_D3 = 0, pair_Bij = 0;
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
						else if (i == col_BQC && BQC)
							file.iss >> pair_Bij;
						else
							file.iss >> sdump;
					}// end for columns

					// DMI vector orientation
					if (DMI_abc)
					{
						pair_D_temp = { pair_D1, pair_D2, pair_D3 };
						pair_D1 = pair_D_temp[0] * geometry.basis[0][0] + pair_D_temp[1] * geometry.basis[0][1] + pair_D_temp[2] * geometry.basis[0][2];
						pair_D2 = pair_D_temp[0] * geometry.basis[1][0] + pair_D_temp[1] * geometry.basis[1][1] + pair_D_temp[2] * geometry.basis[1][2];
						pair_D3 = pair_D_temp[0] * geometry.basis[2][0] + pair_D_temp[1] * geometry.basis[2][1] + pair_D_temp[2] * geometry.basis[2][2];
					}
					// DMI vector normalisation
					if (Dij)
					{
						double dnorm = std::sqrt(std::pow(pair_D1, 2) + std::pow(pair_D2, 2) + std::pow(pair_D3, 2));
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
					int Na = geometry.n_translations[0] + 1;
					int nTa = geometry.n_translations[0];
					int Nb = geometry.n_translations[1] + 1;
					int nTb = geometry.n_translations[1];
					int Nc = geometry.n_translations[2] + 1;
					int nTc = geometry.n_translations[2];
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
								periods_c = (nc + pair_dc) / N;
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
									Exchange_indices[pair_periodicity].push_back(std::vector<int>{ idx_i, idx_j });
									Exchange_magnitude[pair_periodicity].push_back(pair_Jij);
								}
								if (pair_Dij != 0)
								{
									DMI_indices[pair_periodicity].push_back(std::vector<int>{ idx_i, idx_j });
									DMI_magnitude[pair_periodicity].push_back(pair_Dij);
									DMI_normal[pair_periodicity].push_back(std::vector<double>{pair_D1, pair_D2, pair_D3});
								}
								if (pair_Bij != 0)
								{
									BQC_indices[pair_periodicity].push_back(std::vector<int>{ idx_i, idx_j });
									BQC_magnitude[pair_periodicity].push_back(pair_Bij);
								}
								//pairs.push_back(Data::Spin_Pair(idx_i, idx_j, pair_Jij, pair_Dij, std::vector<double>{pair_Dx, pair_Dy, pair_Dz}, pair_Bij));
							}
						}
					}// end for translations

					++i_pair;
				}// end while GetLine
				Log.Send(Utility::Log_Level::INFO, Utility::Log_Sender::IO, "Done reading " + std::to_string(i_pair) + " spin pairs from file " + pairsFile);
			}// end try
			catch (Exception ex)
			{
				throw ex;
			}
		}
	}
}