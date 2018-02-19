#include <io/IO.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Dataparser.hpp>
#include <io/OVF_File.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <iostream>
#include <fstream>
#include <thread>
#include <string>
#include <sstream>
#include <algorithm>

using namespace Utility;
using namespace Engine;

namespace IO
{
    // A helper function for splitting a string with a delimiter
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
    
    // A helper function 
    void check_defects( std::shared_ptr<Data::Spin_System> s )
    {
        auto& spins = *s->spins;
        int nos = s->geometry->nos;
        
        // Detecet the defects 
        for (int i=0; i<nos; i++)
        {
            if (spins[i].norm() < 1e-5)
            {
                spins[i] = {0, 0, 1};
                
                // in case of spin vector close to zero we have a vacancy
            #ifdef SPIRIT_ENABLE_DEFECTS
                s->geometry->atom_types[i] = -1;
            #endif
            }            
        }
    }
    
    /*
    Reads a configuration file into an existing Spin_System
    */
    void Read_Spin_Configuration( std::shared_ptr<Data::Spin_System> s, const std::string file, 
                                  VF_FileFormat format )
    {
        std::ifstream myfile(file);
        if (myfile.is_open())
        {
            Log(Log_Level::Info, Log_Sender::IO, std::string("Reading Spins File ").append(file));
            std::string line = "";
            std::istringstream iss(line);
            std::size_t found;
            int i = 0;
            if (format == VF_FileFormat::SPIRIT_CSV_POS_SPIN)
            {
                auto& spins = *s->spins;
                while (getline(myfile, line))
                {
                    if (i >= s->nos) 
                    { 
                        Log( Log_Level::Warning, Log_Sender::IO, "NOS mismatch in Read Spin "
                             "Configuration - Aborting" ); 
                        myfile.close(); 
                        return; 
                    }
                    
                    found = line.find("#");
                    
                    // Read the line if # is not found (# marks a comment)
                    if (found == std::string::npos)
                    {
                        auto x = split_string_to_scalar(line, ",");

                        if (x[3]*x[3] + x[4]*x[4] + x[5]*x[5] < 1e-5)
                        {
                            spins[i][0] = 0;
                            spins[i][1] = 0;
                            spins[i][2] = 1;
                            #ifdef SPIRIT_ENABLE_DEFECTS
                            s->geometry->atom_types[i] = -1;
                            #endif
                        }
                        else
                        {
                            spins[i][0] = x[3];
                            spins[i][1] = x[4];
                            spins[i][2] = x[5];
                        }
                        ++i;
                    }// endif (# not found)
                    
                    // discard line if # is found
                }// endif new line (while)
            
                if (i < s->nos) { Log(Log_Level::Warning, Log_Sender::IO, "NOS mismatch in Read Spin Configuration"); }
            }
            else if ( format == VF_FileFormat::OVF_BIN8 || 
                      format == VF_FileFormat::OVF_BIN4 || 
                      format == VF_FileFormat::OVF_TEXT )
            {
                auto& spins = *s->spins;
                auto& geometry = *s->geometry;
                
                OVF_File ovf_file( file, format );
                ovf_file.read_image( spins, geometry ); 
            }
            else
            {
                auto& spins = *s->spins;
                Vector3 spin;
                while (getline(myfile, line))
                {
                    if (i >= s->nos) 
                    { 
                        Log( Log_Level::Warning, Log_Sender::IO, "NOS mismatch in Read Spin "
                             "Configuration - Aborting"); 
                        myfile.close(); 
                        return; 
                    }
                    found = line.find("#");
                    // Read the line if # is not found (# marks a comment)
                    if (found == std::string::npos)
                    {
                        //scalar x, y, z;
                        iss.clear();
                        iss.str(line);
                        //iss >> x >> y >> z;
                        iss >> spin[0] >> spin[1] >> spin[2];
                        if (spin.norm() < 1e-5)
                        {
                            spin = {0, 0, 1};
                            // in case of spin vector close to zero we have a vacancy
                            #ifdef SPIRIT_ENABLE_DEFECTS
                            s->geometry->atom_types[i] = -1;
                            #endif
                        }
                        spins[i] = spin;
                        ++i;
                    }// endif (# not found)
                        // discard line if # is found
                }// endif new line (while)
                if (i < s->nos) { Log(Log_Level::Warning, Log_Sender::IO, "NOS mismatch in Read Spin Configuration"); }
            }
        
        #ifdef SPIRIT_ENABLE_DEFECTS
            // assure that defects are treated right
            check_defects(s);
        #endif
            
            // normalize read in spins
            Vectormath::normalize_vectors(*s->spins);
            
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
            std::size_t image_no;
            int ispin = 0, iimage = -1, nos = c->images[0]->nos, noi = c->noi;
            Vector3 spin;
        
            while (getline(myfile, line))
            {
                // First we check if the line declares a new image
                
                image_no = line.find("Image No"); 
                
                // if there is a "Image No" in that line increment the indeces appropriately
                if ( image_no != std::string::npos )
                {
                    if (ispin < nos && iimage>0)    // Check if less than NOS spins were read for the image before
                    {
                        Log(Log_Level::Warning, Log_Sender::IO, fmt::format("NOS(image) = {} > NOS(file) = {} in image {}", nos, ispin+1, iimage+1));
                    }
                    // set new image index
                    ++iimage;
                    // re-set spin counter
                    ispin = 0;
                    // jump to next line
                    getline(myfile, line);
                    if (iimage >= noi)
                    {
                        Log(Log_Level::Warning, Log_Sender::IO, fmt::format("NOI(file) = {} > NOI(chain) = {}", iimage+1, noi));
                    }
                    else
                    {
                        nos = c->images[iimage]->nos; // Note: different NOS in different images is currently not supported
                    }
                }//endif "Image No"
                
                // Then check if the line contains "#" charachter which means that is a comment.
                // This will not affect the "Image No" since is already been done.
                found = line.find("#");
                
                // Read the line if # is not found (# marks a comment)
                if (found == std::string::npos)
                {
                    if (iimage < 0) iimage = 0;

                    if (iimage >= noi)
                    {
                        Log(Log_Level::Warning, Log_Sender::IO, fmt::format("NOI(file) = {} > NOI(chain) = {}. Appending image {}", iimage+1, noi, iimage+1));
                        // Copy Image
                        auto new_system = std::make_shared<Data::Spin_System>(Data::Spin_System(*c->images[iimage-1]));
                        new_system->Lock();
                        // Add to chain
                        c->noi++;
                        c->images.push_back(new_system);
                        c->image_type.push_back(Data::GNEB_Image_Type::Normal);
                        noi = c->noi;
                    }
                    nos = c->images[iimage]->nos; // Note: different NOS in different images is currently not supported

                    if (ispin >= nos)
                    {
                        Log(Log_Level::Warning, Log_Sender::IO, fmt::format("NOS missmatch in image {}", iimage+1));
                        Log(Log_Level::Warning, Log_Sender::IO, fmt::format("NOS(file) = {} > NOS(image) = {}", ispin+1, nos));
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
                        iss >> spin[0] >> spin[1] >> spin[2];
                        if (spin.norm() < 1e-5)
                        {
                            spin = {0, 0, 1};
                            #ifdef SPIRIT_ENABLE_DEFECTS
                            c->images[iimage]->geometry->atom_types[ispin] = -1;
                            #endif
                        }
                        spins[ispin] = spin;
                    }
                    ++ispin;
                }// endif (# not found)
                
                // Discard line if # is found. This will work also in the case of finding "Image No"
                // keyword since a line like that would containi "#"
                
            }// endif new line (while)
            
            // for every image of the chain
            for (int i=0; i<iimage; i++)
            {
            #ifdef SPIRIT_ENABLE_DEFECTS
                // assure that defects are treated right
                check_defects(c->images[iimage]);
            #endif
                
                // normalize read in spins
                Vectormath::normalize_vectors(*c->images[iimage]->spins);
            }
            
            if (ispin < nos) Log(Log_Level::Warning, Log_Sender::IO, fmt::format("NOS(image) = {} > NOS(file) = {} in image {}", nos, ispin+1, iimage+1));
            if (iimage < noi-1) Log(Log_Level::Warning, Log_Sender::IO, fmt::format("NOI(chain) = {} > NOI(file) = {}", noi, iimage+1));
            
            myfile.close();
            Log(Log_Level::Info, Log_Sender::IO, std::string("Done Reading SpinChain File ").append(file));
        }
    }

    void Read_Eigenmodes( std::shared_ptr<Data::Spin_System> image, const std::string filename, 
                          VF_FileFormat format )
    {
        if (format == VF_FileFormat::OVF_BIN8 || 
            format == VF_FileFormat::OVF_BIN4 || 
            format == VF_FileFormat::OVF_TEXT )
        {
            auto& spins = *image->spins;
            auto& geometry = *image->geometry;
           
            //OVF_File ovf_file( filename, format );
            //ovf_file.read_eigenmodes( spins, geometry );           
        }
        else
        {
            Log( Log_Level::Warning, Log_Sender::IO, "Read Eigenmodes supports only OVF files"); 
        }
    }

    /*
    Read from Anisotropy file
    */
    void Anisotropy_from_File(const std::string anisotropyFile, const std::shared_ptr<Data::Geometry> geometry, int & n_indices,
        intfield & anisotropy_index, scalarfield & anisotropy_magnitude,
        vectorfield & anisotropy_normal)
    {
        Log(Log_Level::Info, Log_Sender::IO, "Reading anisotropy from file " + anisotropyFile);
        try
        {
            std::vector<std::string> columns(5);    // at least: 1 (index) + 3 (K)
            // column indices of pair indices and interactions
            int col_i = -1, col_K = -1, col_Kx = -1, col_Ky = -1, col_Kz = -1, col_Ka = -1, col_Kb = -1, col_Kc = -1;
            bool K_magnitude = false, K_xyz = false, K_abc = false;
            Vector3 K_temp = { 0, 0, 0 };
            int n_anisotropy;

            Filter_File_Handle file(anisotropyFile);

            if (file.Find("n_anisotropy"))
            {
                // Read n interaction pairs
                file.iss >> n_anisotropy;
                Log(Log_Level::Info, Log_Sender::IO, fmt::format("Anisotropy file {} should have {} vectors", anisotropyFile, n_anisotropy));
            }
            else
            {
                // Read the whole file
                n_anisotropy = (int)1e8;
                // First line should contain the columns
                file.ResetStream();
                Log(Log_Level::Info, Log_Sender::IO, "Trying to parse anisotropy columns from top of file " + anisotropyFile);
            }

            // Get column indices
            file.GetLine(); // first line contains the columns
            for (unsigned int i = 0; i < columns.size(); ++i)
            {
                file.iss >> columns[i];
                if (!columns[i].compare(0, 1, "i"))    col_i = i;
                else if (!columns[i].compare(0, 2, "k")) { col_K = i;    K_magnitude = true; }
                else if (!columns[i].compare(0, 2, "kx"))    col_Kx = i;
                else if (!columns[i].compare(0, 2, "ky"))    col_Ky = i;
                else if (!columns[i].compare(0, 2, "kz"))    col_Kz = i;
                else if (!columns[i].compare(0, 2, "ka"))    col_Ka = i;
                else if (!columns[i].compare(0, 2, "kb"))    col_Kb = i;
                else if (!columns[i].compare(0, 2, "kc"))    col_Kc = i;

                if (col_Kx >= 0 && col_Ky >= 0 && col_Kz >= 0) K_xyz = true;
                if (col_Ka >= 0 && col_Kb >= 0 && col_Kc >= 0) K_abc = true;
            }

            if (!K_xyz && !K_abc) Log(Log_Level::Warning, Log_Sender::IO, "No anisotropy data could be found in header of file " + anisotropyFile);

            // Indices
            int spin_i = 0;
            scalar spin_K = 0, spin_K1 = 0, spin_K2 = 0, spin_K3 = 0;
            // Arrays
            anisotropy_index = intfield(0);
            anisotropy_magnitude = scalarfield(0);
            anisotropy_normal = vectorfield(0);

            // Get actual Data
            int i_anisotropy = 0;
            std::string sdump;
            while (file.GetLine() && i_anisotropy < n_anisotropy)
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

                // Anisotropy vector orientation
                if (K_abc)
                {
                    spin_K1 = K_temp.dot(geometry->bravais_vectors[0]);
                    spin_K2 = K_temp.dot(geometry->bravais_vectors[1]);
                    spin_K3 = K_temp.dot(geometry->bravais_vectors[2]);
                    K_temp = { spin_K1, spin_K2, spin_K3 };
                }

                // Anisotropy vector normalisation
                if (K_magnitude)
                {
                    K_temp.normalize();
                    if (K_temp.norm() == 0)
                        K_temp = Vector3{0, 0, 1};
                }
                else
                {
                    spin_K = K_temp.norm();
                    if (spin_K != 0)
                        K_temp.normalize();
                }

                // Add the index and parameters to the corresponding lists
                if (spin_K != 0)
                {
                    anisotropy_index.push_back(spin_i);
                    anisotropy_magnitude.push_back(spin_K);
                    anisotropy_normal.push_back(K_temp);
                }

                ++i_anisotropy;
            }// end while getline
            n_indices = i_anisotropy;
        }// end try
        catch( ... )
        {
            spirit_rethrow(    fmt::format("Could not read anisotropies from file  \"{}\"", anisotropyFile) );
        }
    }

    /*
    Read from Pairs file by Markus & Bernd
    */
    void Pairs_from_File(const std::string pairsFile, const std::shared_ptr<Data::Geometry> geometry, int & nop,
        pairfield & exchange_pairs, scalarfield & exchange_magnitudes,
        pairfield & dmi_pairs, scalarfield & dmi_magnitudes, vectorfield & dmi_normals)
    {
        Log(Log_Level::Info, Log_Sender::IO, fmt::format("Reading spin pairs from file \"{}\"", pairsFile));
        try
        {
            std::vector<std::string> columns(20);    // at least: 2 (indices) + 3 (J) + 3 (DMI)
            // column indices of pair indices and interactions
            int n_pairs = 0;
            int col_i = -1, col_j = -1, col_da = -1, col_db = -1, col_dc = -1,
                col_J = -1, col_DMIx = -1, col_DMIy = -1, col_DMIz = -1,
                col_Dij = -1, col_DMIa = -1, col_DMIb = -1, col_DMIc = -1;
            bool J = false, DMI_xyz = false, DMI_abc = false, Dij = false;
            int pair_periodicity = 0;
            Vector3 pair_D_temp = { 0, 0, 0 };
            // Get column indices
            Filter_File_Handle file(pairsFile);

            if (file.Find("n_interaction_pairs"))
            {
                // Read n interaction pairs
                file.iss >> n_pairs;
                Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("File {} should have {} pairs", pairsFile, n_pairs));
            }
            else
            {
                // Read the whole file
                n_pairs = (int)1e8;
                // First line should contain the columns
                file.ResetStream();
                Log(Log_Level::Info, Log_Sender::IO, "Trying to parse spin pairs columns from top of file " + pairsFile);
            }

            file.GetLine(); 
            for (unsigned int i = 0; i < columns.size(); ++i)
            {
                file.iss >> columns[i];
                if      (columns[i] == "i")    col_i = i;
                else if (columns[i] == "j")    col_j = i;
                else if (columns[i] == "da")   col_da = i;
                else if (columns[i] == "db")   col_db = i;
                else if (columns[i] == "dc")   col_dc = i;
                else if (columns[i] == "jij")  { col_J = i;    J = true; }
                else if (columns[i] == "dij")  { col_Dij = i;    Dij = true; }
                else if (columns[i] == "dijx") col_DMIx = i;
                else if (columns[i] == "dijy") col_DMIy = i;
                else if (columns[i] == "dijz") col_DMIz = i;
                else if (columns[i] == "dija") col_DMIx = i;
                else if (columns[i] == "dijb") col_DMIy = i;
                else if (columns[i] == "dijc") col_DMIz = i;

                if (col_DMIx >= 0 && col_DMIy >= 0 && col_DMIz >= 0) DMI_xyz = true;
                if (col_DMIa >= 0 && col_DMIb >= 0 && col_DMIc >= 0) DMI_abc = true;
            }

            // Check if interactions have been found in header
            if (!J && !DMI_xyz && !DMI_abc) Log(Log_Level::Warning, Log_Sender::IO, "No interactions could be found in pairs file " + pairsFile);

            // Pair Indices
            int pair_i = 0, pair_j = 0, pair_da = 0, pair_db = 0, pair_dc = 0;
            scalar pair_Jij = 0, pair_Dij = 0, pair_D1 = 0, pair_D2 = 0, pair_D3 = 0;

            // Get actual Pairs Data
            int i_pair = 0;
            std::string sdump;
            while (file.GetLine() && i_pair < n_pairs)
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
                    pair_D1 = pair_D_temp.dot(geometry->bravais_vectors[0]);
                    pair_D2 = pair_D_temp.dot(geometry->bravais_vectors[1]);
                    pair_D3 = pair_D_temp.dot(geometry->bravais_vectors[2]);
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
                    exchange_pairs.push_back({ pair_i, pair_j, { pair_da, pair_db, pair_dc } });
                    exchange_magnitudes.push_back(pair_Jij);
                }
                if (pair_Dij != 0)
                {
                    dmi_pairs.push_back({ pair_i, pair_j, { pair_da, pair_db, pair_dc } });
                    dmi_magnitudes.push_back(pair_Dij);
                    dmi_normals.push_back(Vector3{pair_D1, pair_D2, pair_D3});
                }

                ++i_pair;
            }// end while GetLine
            Log(Log_Level::Info, Log_Sender::IO, fmt::format("Done reading {} spin pairs from file \"{}\"", i_pair, pairsFile));
            nop = i_pair;
        }// end try
        catch( ... )
        {
            spirit_rethrow(fmt::format("Could not read pairs file \"{}\"", pairsFile));
        }
    }



    /*
    Read from Quadruplet file
    */
    void Quadruplets_from_File(const std::string quadrupletsFile, const std::shared_ptr<Data::Geometry>, int & noq,
        quadrupletfield & quadruplets, scalarfield & quadruplet_magnitudes)
    {
        Log(Log_Level::Info, Log_Sender::IO, "Reading spin quadruplets from file " + quadrupletsFile);
        try
        {
            std::vector<std::string> columns(20);    // at least: 4 (indices) + 3*3 (positions) + 1 (magnitude)
            // column indices of pair indices and interactions
            int col_i = -1;
            int col_j = -1, col_da_j = -1, col_db_j = -1, col_dc_j = -1, periodicity_j = 0;
            int col_k = -1, col_da_k = -1, col_db_k = -1, col_dc_k = -1, periodicity_k = 0;
            int col_l = -1, col_da_l = -1, col_db_l = -1, col_dc_l = -1, periodicity_l = 0;
            int    col_Q = -1;
            bool Q = false;
            int max_periods_a = 0, max_periods_b = 0, max_periods_c = 0;
            int quadruplet_periodicity = 0;
            int n_quadruplets = 0;

            // Get column indices
            Filter_File_Handle file(quadrupletsFile);

            if (file.Find("n_interaction_quadruplets"))
            {
                // Read n interaction quadruplets
                file.iss >> n_quadruplets;
                Log(Log_Level::Debug, Log_Sender::IO, fmt::format("File {} should have {} quadruplets", quadrupletsFile, n_quadruplets));
            }
            else
            {
                // Read the whole file
                n_quadruplets = (int)1e8;
                // First line should contain the columns
                file.ResetStream();
                Log(Log_Level::Info, Log_Sender::IO, "Trying to parse quadruplet columns from top of file " + quadrupletsFile);
            }

            file.GetLine();
            for (unsigned int i = 0; i < columns.size(); ++i)
            {
                file.iss >> columns[i];
                if      (columns[i] == "i")    col_i = i;
                else if (columns[i] == "j")    col_j = i;
                else if (columns[i] == "da_j")    col_da_j = i;
                else if (columns[i] == "db_j")    col_db_j = i;
                else if (columns[i] == "dc_j")    col_dc_j = i;
                else if (columns[i] == "k")    col_k = i;
                else if (columns[i] == "da_k")    col_da_k = i;
                else if (columns[i] == "db_k")    col_db_k = i;
                else if (columns[i] == "dc_k")    col_dc_k = i;
                else if (columns[i] == "l")    col_l = i;
                else if (columns[i] == "da_l")    col_da_l = i;
                else if (columns[i] == "db_l")    col_db_l = i;
                else if (columns[i] == "dc_l")    col_dc_l = i;
                else if (columns[i] == "q")    { col_Q = i;    Q = true; }
            }

            // Check if interactions have been found in header
            if (!Q) Log(Log_Level::Warning, Log_Sender::IO, "No interactions could be found in header of quadruplets file " + quadrupletsFile);

            // Quadruplet Indices
            int q_i = 0;
            int q_j = 0, q_da_j = 0, q_db_j = 0, q_dc_j = 0;
            int q_k = 0, q_da_k = 0, q_db_k = 0, q_dc_k = 0;
            int q_l = 0, q_da_l = 0, q_db_l = 0, q_dc_l = 0;
            scalar q_Q;

            // Get actual Quadruplets Data
            int i_quadruplet = 0;
            std::string sdump;
            while (file.GetLine() && i_quadruplet < n_quadruplets)
            {
                // Read a Quadruplet from the File
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
                    quadruplets.push_back({ q_i, q_j, q_k, q_l,
                        { q_da_j, q_db_j, q_db_j },
                        { q_da_k, q_db_k, q_db_k },
                        { q_da_l, q_db_l, q_db_l } });
                    quadruplet_magnitudes.push_back(q_Q);
                }

                ++i_quadruplet;
            }// end while GetLine
            Log(Log_Level::Info, Log_Sender::IO, fmt::format("Done reading {} spin quadruplets from file {}", i_quadruplet, quadrupletsFile));
            noq = i_quadruplet;
        }// end try
        catch( ... )
        {
            spirit_rethrow( fmt::format("Could not read quadruplets from file  \"{}\"", quadrupletsFile) );
        }
    } // End Quadruplets_from_File


    void Defects_from_File(const std::string defectsFile, int & n_defects,
        intfield & defect_indices, intfield & defect_types)
    {
        n_defects = 0;

        int nod = 0;
        intfield indices(0), types(0);
        try
        {
            Log(Log_Level::Info, Log_Sender::IO, "Reading Defects");
            Filter_File_Handle myfile(defectsFile);

            if (myfile.Find("n_defects"))
            {
                // Read n interaction pairs
                myfile.iss >> nod;
                Log(Log_Level::Debug, Log_Sender::IO, fmt::format("File {} should have {} defects", defectsFile, nod));
            }
            else
            {
                // Read the whole file
                nod = (int)1e8;
                // First line should contain the columns
                myfile.ResetStream();
                Log(Log_Level::Debug, Log_Sender::IO, "Trying to parse defects from top of file " + defectsFile);
            }

            int i_defect = 0;
            while (myfile.GetLine() && i_defect < nod)
            {
                int index, type;
                myfile.iss >> index >> type;
                indices.push_back(index);
                types.push_back(type);
                ++i_defect;
            }

            defect_indices = indices;
            defect_types = types;
            n_defects = i_defect;

            Log(Log_Level::Info, Log_Sender::IO, "Done Reading Defects");
        }
        catch( ... )
        {
            spirit_rethrow(    fmt::format("Could not read defects file  \"{}\"", defectsFile) );
        }
    } // End Defects_from_File

    void Pinned_from_File(const std::string pinnedFile, int & n_pinned,
        intfield & pinned_indices, vectorfield & pinned_spins)
    {
        n_pinned = 0;

        int nop = 0;
        intfield indices(0);
        vectorfield spins(0);
        try
        {
            Log(Log_Level::Info, Log_Sender::IO, "Reading pinned sites");
            Filter_File_Handle myfile(pinnedFile);

            if (myfile.Find("n_pinned"))
            {
                // Read n interaction pairs
                myfile.iss >> nop;
                Log(Log_Level::Debug, Log_Sender::IO, fmt::format("File {} should have {} pinned sites", pinnedFile, nop));
            }
            else
            {
                // Read the whole file
                nop = (int)1e8;
                // First line should contain the columns
                myfile.ResetStream();
                Log(Log_Level::Debug, Log_Sender::IO, "Trying to parse pinned sites from top of file " + pinnedFile);
            }

            int i_pinned = 0;
            while (myfile.GetLine() && i_pinned < nop)
            {
                int index;
                scalar sx, sy, sz;
                myfile.iss >> index >> sx >> sy >> sz;
                indices.push_back(index);
                spins.push_back({sx, sy, sz});
                ++i_pinned;
            }

            pinned_indices = indices;
            pinned_spins = spins;
            n_pinned = i_pinned;

            Log(Log_Level::Info, Log_Sender::IO, "Done reading pinned sites");
        }
        catch( ... )
        {
            spirit_rethrow(    fmt::format("Could not read pinned sites file  \"{}\"", pinnedFile) );
        }
    } // End Pinned_from_File


    int ReadHeaderLine(FILE * fp, char * line)
    {
        char c;
        int pos=0;
        
        do
        {
            c = (char)fgetc(fp); // Get current char and move pointer to the next position
            if (c != EOF && c != '\n') line[pos++] = c; // If it's not the end of the file
        }
        while(c != EOF && c != '\n'); // If it's not the end of the file or end of the line
        
        line[pos] = 0; // Complete the read line
        if ((pos==0 || line[0]!='#') && c != EOF)
            return ReadHeaderLine(fp, line);// Recursive call for ReadHeaderLine if the current line is empty
        
        // The last symbol is the line end symbol
        return pos-1;
    }
    
    void ReadDataLine(FILE * fp, char * line)
    {
        char c;
        int pos=0;

        do
        {
            c = (char)fgetc(fp);
            if (c != EOF && c != '\n') line[pos++] = c;
        }
        while(c != EOF && c != '\n');

        line[pos] = 0;
    }
    
}// end namespace IO
