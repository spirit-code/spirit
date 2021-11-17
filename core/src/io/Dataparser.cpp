#include <engine/Vectormath.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/IO.hpp>
#include <io/OVF_File.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

using Utility::Log_Level;
using Utility::Log_Sender;

namespace IO
{

// Reads a non-OVF spins file with plain text and discarding any headers starting with '#'
void Read_NonOVF_Spin_Configuration(
    vectorfield & spins, Data::Geometry & geometry, const int nos, const int idx_image_infile,
    const std::string & file )
{
    IO::Filter_File_Handle file_handle( file, "#" );

    // Jump to the specified image in the file
    for( int i = 0; i < ( nos * idx_image_infile ); i++ )
        file_handle.GetLine();

    for( int i = 0; i < nos && file_handle.GetLine( "," ); i++ )
    {
        file_handle.iss >> spins[i][0];
        file_handle.iss >> spins[i][1];
        file_handle.iss >> spins[i][2];

        if( spins[i].norm() < 1e-5 )
        {
            spins[i] = { 0, 0, 1 };
            // In case of spin vector close to zero we have a vacancy
#ifdef SPIRIT_ENABLE_DEFECTS
            geometry.atom_types[i] = -1;
#endif
        }
    }

    // normalize read in spins
    Engine::Vectormath::normalize_vectors( spins );
}

void Check_NonOVF_Chain_Configuration(
    std::shared_ptr<Data::Spin_System_Chain> chain, const std::string & file, int start_image_infile,
    int end_image_infile, const int insert_idx, int & noi_to_add, int & noi_to_read, const int idx_chain )
{
    IO::Filter_File_Handle file_handle( file, "#" );

    int nol = file_handle.Get_N_Non_Comment_Lines();
    int noi = chain->noi;
    int nos = chain->images[0]->nos;

    int noi_infile = nol / nos;
    int remainder  = nol % nos;

    if( remainder != 0 )
    {
        Log( Utility::Log_Level::Warning, Utility::Log_Sender::IO,
             fmt::format( "Calculated number of images in the nonOVF file is not integer" ), insert_idx, idx_chain );
    }

    // Check if the ending image is valid otherwise set it to the last image infile
    if( end_image_infile < start_image_infile || end_image_infile >= noi_infile )
    {
        end_image_infile = noi_infile - 1;
        Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
             fmt::format( "Invalid end_image_infile. Value was set to the last image "
                          "of the file" ),
             insert_idx, idx_chain );
    }

    // If the idx of the starting image is valid
    if( start_image_infile < noi_infile )
    {
        noi_to_read = end_image_infile - start_image_infile + 1;

        noi_to_add = noi_to_read - ( noi - insert_idx );
    }
    else
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::IO,
             fmt::format( "Invalid starting_idx. File {} has {} noi", file, noi_infile ), insert_idx, idx_chain );
    }
}

// Read from Anisotropy file
void Anisotropy_from_File(
    const std::string & anisotropyFile, const std::shared_ptr<Data::Geometry> geometry, int & n_indices,
    intfield & anisotropy_index, scalarfield & anisotropy_magnitude, vectorfield & anisotropy_normal ) noexcept
try
{
    Log( Log_Level::Debug, Log_Sender::IO, "Reading anisotropy from file " + anisotropyFile );

    std::vector<std::string> columns( 5 ); // at least: 1 (index) + 3 (K)
    // column indices of pair indices and interactions
    int col_i = -1, col_K = -1, col_Kx = -1, col_Ky = -1, col_Kz = -1, col_Ka = -1, col_Kb = -1, col_Kc = -1;
    bool K_magnitude = false, K_xyz = false, K_abc = false;
    Vector3 K_temp = { 0, 0, 0 };
    int n_anisotropy;

    Filter_File_Handle file( anisotropyFile );

    if( file.Find( "n_anisotropy" ) )
    {
        // Read n interaction pairs
        file.iss >> n_anisotropy;
        Log( Log_Level::Debug, Log_Sender::IO,
             fmt::format( "Anisotropy file {} should have {} vectors", anisotropyFile, n_anisotropy ) );
    }
    else
    {
        // Read the whole file
        n_anisotropy = (int)1e8;
        // First line should contain the columns
        file.ResetStream();
        Log( Log_Level::Debug, Log_Sender::IO,
             "Trying to parse anisotropy columns from top of file " + anisotropyFile );
    }

    // Get column indices
    file.GetLine(); // first line contains the columns
    for( unsigned int i = 0; i < columns.size(); ++i )
    {
        file.iss >> columns[i];
        std::transform( columns[i].begin(), columns[i].end(), columns[i].begin(), ::tolower );
        if( columns[i] == "i" )
            col_i = i;
        else if( columns[i] == "k" )
        {
            col_K       = i;
            K_magnitude = true;
        }
        else if( columns[i] == "kx" )
            col_Kx = i;
        else if( columns[i] == "ky" )
            col_Ky = i;
        else if( columns[i] == "kz" )
            col_Kz = i;
        else if( columns[i] == "ka" )
            col_Ka = i;
        else if( columns[i] == "kb" )
            col_Kb = i;
        else if( columns[i] == "kc" )
            col_Kc = i;

        if( col_Kx >= 0 && col_Ky >= 0 && col_Kz >= 0 )
            K_xyz = true;
        if( col_Ka >= 0 && col_Kb >= 0 && col_Kc >= 0 )
            K_abc = true;
    }

    if( !K_xyz && !K_abc )
        Log( Log_Level::Warning, Log_Sender::IO,
             fmt::format( "No anisotropy data could be found in header of file \"{}\"", anisotropyFile ) );

    // Indices
    int spin_i    = 0;
    scalar spin_K = 0, spin_K1 = 0, spin_K2 = 0, spin_K3 = 0;
    // Arrays
    anisotropy_index     = intfield( 0 );
    anisotropy_magnitude = scalarfield( 0 );
    anisotropy_normal    = vectorfield( 0 );

    // Get actual Data
    int i_anisotropy = 0;
    std::string sdump;
    while( file.GetLine() && i_anisotropy < n_anisotropy )
    {
        // Read a line from the File
        for( unsigned int i = 0; i < columns.size(); ++i )
        {
            if( i == col_i )
                file.iss >> spin_i;
            else if( i == col_K )
                file.iss >> spin_K;
            else if( i == col_Kx && K_xyz )
                file.iss >> spin_K1;
            else if( i == col_Ky && K_xyz )
                file.iss >> spin_K2;
            else if( i == col_Kz && K_xyz )
                file.iss >> spin_K3;
            else if( i == col_Ka && K_abc )
                file.iss >> spin_K1;
            else if( i == col_Kb && K_abc )
                file.iss >> spin_K2;
            else if( i == col_Kc && K_abc )
                file.iss >> spin_K3;
            else
                file.iss >> sdump;
        }
        K_temp = { spin_K1, spin_K2, spin_K3 };
        // Anisotropy vector orientation
        if( K_abc )
        {
            spin_K1 = K_temp.dot( geometry->lattice_constant * geometry->bravais_vectors[0] );
            spin_K2 = K_temp.dot( geometry->lattice_constant * geometry->bravais_vectors[1] );
            spin_K3 = K_temp.dot( geometry->lattice_constant * geometry->bravais_vectors[2] );
            K_temp  = { spin_K1, spin_K2, spin_K3 };
        }

        // Anisotropy vector normalisation
        if( K_magnitude )
        {
            K_temp.normalize();
            if( K_temp.norm() == 0 )
                K_temp = Vector3{ 0, 0, 1 };
        }
        else
        {
            spin_K = K_temp.norm();
            if( spin_K != 0 )
                K_temp.normalize();
        }

        // Add the index and parameters to the corresponding lists
        if( spin_K != 0 )
        {
            anisotropy_index.push_back( spin_i );
            anisotropy_magnitude.push_back( spin_K );
            anisotropy_normal.push_back( K_temp );
        }
        ++i_anisotropy;
    } // end while getline
    n_indices = i_anisotropy;
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read anisotropies from file \"{}\"", anisotropyFile ) );
}

// Read from Pairs file by Markus & Bernd
void Pairs_from_File(
    const std::string & pairs_file, const std::shared_ptr<Data::Geometry> geometry, int & nop,
    pairfield & exchange_pairs, scalarfield & exchange_magnitudes, pairfield & dmi_pairs, scalarfield & dmi_magnitudes,
    vectorfield & dmi_normals ) noexcept
try
{
    Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "Reading spin pairs from file \"{}\"", pairs_file ) );

    std::vector<std::string> columns( 20 ); // at least: 2 (indices) + 3 (J) + 3 (DMI)
    // column indices of pair indices and interactions
    int n_pairs = 0;
    int col_i = -1, col_j = -1, col_da = -1, col_db = -1, col_dc = -1, col_J = -1, col_DMIx = -1, col_DMIy = -1,
        col_DMIz = -1, col_Dij = -1, col_DMIa = -1, col_DMIb = -1, col_DMIc = -1;
    bool J = false, DMI_xyz = false, DMI_abc = false, Dij = false;
    int pair_periodicity = 0;
    Vector3 pair_D_temp  = { 0, 0, 0 };
    // Get column indices
    Filter_File_Handle file( pairs_file );

    if( file.Find( "n_interaction_pairs" ) )
    {
        // Read n interaction pairs
        file.iss >> n_pairs;
        Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "File {} should have {} pairs", pairs_file, n_pairs ) );
    }
    else
    {
        // Read the whole file
        n_pairs = (int)1e8;
        // First line should contain the columns
        file.ResetStream();
        Log( Log_Level::Debug, Log_Sender::IO, "Trying to parse spin pairs columns from top of file " + pairs_file );
    }

    file.GetLine();
    for( unsigned int i = 0; i < columns.size(); ++i )
    {
        file.iss >> columns[i];
        std::transform( columns[i].begin(), columns[i].end(), columns[i].begin(), ::tolower );
        if( columns[i] == "i" )
            col_i = i;
        else if( columns[i] == "j" )
            col_j = i;
        else if( columns[i] == "da" )
            col_da = i;
        else if( columns[i] == "db" )
            col_db = i;
        else if( columns[i] == "dc" )
            col_dc = i;
        else if( columns[i] == "jij" )
        {
            col_J = i;
            J     = true;
        }
        else if( columns[i] == "dij" )
        {
            col_Dij = i;
            Dij     = true;
        }
        else if( columns[i] == "dijx" )
            col_DMIx = i;
        else if( columns[i] == "dijy" )
            col_DMIy = i;
        else if( columns[i] == "dijz" )
            col_DMIz = i;
        else if( columns[i] == "dija" )
            col_DMIx = i;
        else if( columns[i] == "dijb" )
            col_DMIy = i;
        else if( columns[i] == "dijc" )
            col_DMIz = i;

        if( col_DMIx >= 0 && col_DMIy >= 0 && col_DMIz >= 0 )
            DMI_xyz = true;
        if( col_DMIa >= 0 && col_DMIb >= 0 && col_DMIc >= 0 )
            DMI_abc = true;
    }

    // Check if interactions have been found in header
    if( !J && !DMI_xyz && !DMI_abc )
        Log( Log_Level::Warning, Log_Sender::IO,
             fmt::format( "No interactions could be found in pairs file \"{}\"", pairs_file ) );

    // Get actual Pairs Data
    int i_pair = 0;
    std::string sdump;
    while( file.GetLine() && i_pair < n_pairs )
    {
        // Pair Indices
        int pair_i = 0, pair_j = 0, pair_da = 0, pair_db = 0, pair_dc = 0;
        scalar pair_Jij = 0, pair_Dij = 0, pair_D1 = 0, pair_D2 = 0, pair_D3 = 0;
        // Read a Pair from the File
        for( unsigned int i = 0; i < columns.size(); ++i )
        {
            if( i == col_i )
                file.iss >> pair_i;
            else if( i == col_j )
                file.iss >> pair_j;
            else if( i == col_da )
                file.iss >> pair_da;
            else if( i == col_db )
                file.iss >> pair_db;
            else if( i == col_dc )
                file.iss >> pair_dc;
            else if( i == col_J && J )
                file.iss >> pair_Jij;
            else if( i == col_Dij && Dij )
                file.iss >> pair_Dij;
            else if( i == col_DMIa && DMI_abc )
                file.iss >> pair_D1;
            else if( i == col_DMIb && DMI_abc )
                file.iss >> pair_D2;
            else if( i == col_DMIc && DMI_abc )
                file.iss >> pair_D3;
            else if( i == col_DMIx && DMI_xyz )
                file.iss >> pair_D1;
            else if( i == col_DMIy && DMI_xyz )
                file.iss >> pair_D2;
            else if( i == col_DMIz && DMI_xyz )
                file.iss >> pair_D3;
            else
                file.iss >> sdump;
        } // end for columns

        // DMI vector orientation
        if( DMI_abc )
        {
            pair_D_temp = pair_D1 * geometry->lattice_constant * geometry->bravais_vectors[0]
                          + pair_D2 * geometry->lattice_constant * geometry->bravais_vectors[1]
                          + pair_D3 * geometry->lattice_constant * geometry->bravais_vectors[2];
            pair_D1 = pair_D_temp[0];
            pair_D2 = pair_D_temp[1];
            pair_D3 = pair_D_temp[2];
        }
        // DMI vector normalisation
        scalar dnorm = std::sqrt( std::pow( pair_D1, 2 ) + std::pow( pair_D2, 2 ) + std::pow( pair_D3, 2 ) );
        if( dnorm != 0 )
        {
            pair_D1 = pair_D1 / dnorm;
            pair_D2 = pair_D2 / dnorm;
            pair_D3 = pair_D3 / dnorm;
        }
        if( !Dij )
        {
            pair_Dij = dnorm;
        }

        // Add the indices and parameters to the corresponding lists
        if( pair_Jij != 0 )
        {
            bool already_in{ false };
            int atposition = -1;
            for( unsigned int icheck = 0; icheck < exchange_pairs.size(); ++icheck )
            {
                auto & p                = exchange_pairs[icheck];
                auto & t                = p.translations;
                std::array<int, 3> tnew = { pair_da, pair_db, pair_dc };
                if( ( pair_i == p.i && pair_j == p.j && tnew == std::array<int, 3>{ t[0], t[1], t[2] } )
                    || ( pair_i == p.j && pair_j == p.i && tnew == std::array<int, 3>{ -t[0], -t[1], -t[2] } ) )
                {
                    already_in = true;
                    atposition = icheck;
                    break;
                }
            }
            if( already_in )
            {
                exchange_magnitudes[atposition] += pair_Jij;
            }
            else
            {
                exchange_pairs.push_back( { pair_i, pair_j, { pair_da, pair_db, pair_dc } } );
                exchange_magnitudes.push_back( pair_Jij );
            }
        }
        if( pair_Dij != 0 )
        {
            bool already_in{ false };
            int dfact      = 1;
            int atposition = -1;
            for( unsigned int icheck = 0; icheck < dmi_pairs.size(); ++icheck )
            {
                auto & p                = dmi_pairs[icheck];
                auto & t                = p.translations;
                std::array<int, 3> tnew = { pair_da, pair_db, pair_dc };
                if( pair_i == p.i && pair_j == p.j && tnew == std::array<int, 3>{ t[0], t[1], t[2] } )
                {
                    already_in = true;
                    atposition = icheck;
                    break;
                }
                else if( pair_i == p.j && pair_j == p.i && tnew == std::array<int, 3>{ -t[0], -t[1], -t[2] } )
                {
                    // If the inverted pair is present, the DMI vector has to be mirrored due to its pseudo-vector behaviour
                    dfact      = -1;
                    already_in = true;
                    atposition = icheck;
                    break;
                }
            }
            if( already_in )
            {
                // Calculate new D vector by adding the two redundant ones and normalize again
                Vector3 newD = dmi_magnitudes[atposition] * dmi_normals[atposition]
                               + dfact * pair_Dij * Vector3{ pair_D1, pair_D2, pair_D3 };
                scalar newdnorm = std::sqrt( std::pow( newD[0], 2 ) + std::pow( newD[1], 2 ) + std::pow( newD[2], 2 ) );
                dmi_magnitudes[atposition] = newdnorm;
                dmi_normals[atposition]    = newD / newdnorm;
            }
            else
            {
                dmi_pairs.push_back( { pair_i, pair_j, { pair_da, pair_db, pair_dc } } );
                dmi_magnitudes.push_back( pair_Dij );
                dmi_normals.push_back( Vector3{ pair_D1, pair_D2, pair_D3 } );
            }
        }

        ++i_pair;
    } // end while GetLine
    Log( Log_Level::Parameter, Log_Sender::IO,
         fmt::format(
             "Done reading {} spin pairs from file \"{}\", giving {} exchange and {} DM (symmetry-reduced) pairs.",
             i_pair, pairs_file, exchange_pairs.size(), dmi_pairs.size() ) );
    nop = i_pair;
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read pairs file \"{}\"", pairs_file ) );
}

// Read from Quadruplet file
void Quadruplets_from_File(
    const std::string & quadruplets_file, const std::shared_ptr<Data::Geometry>, int & noq,
    quadrupletfield & quadruplets, scalarfield & quadruplet_magnitudes ) noexcept
try
{
    Log( Log_Level::Debug, Log_Sender::IO,
         fmt::format( "Reading spin quadruplets from file \"{}\"", quadruplets_file ) );

    std::vector<std::string> columns( 20 ); // at least: 4 (indices) + 3*3 (positions) + 1 (magnitude)
    // column indices of pair indices and interactions
    int col_i = -1;
    int col_j = -1, col_da_j = -1, col_db_j = -1, col_dc_j = -1, periodicity_j = 0;
    int col_k = -1, col_da_k = -1, col_db_k = -1, col_dc_k = -1, periodicity_k = 0;
    int col_l = -1, col_da_l = -1, col_db_l = -1, col_dc_l = -1, periodicity_l = 0;
    int col_Q         = -1;
    bool Q            = false;
    int max_periods_a = 0, max_periods_b = 0, max_periods_c = 0;
    int quadruplet_periodicity = 0;
    int n_quadruplets          = 0;

    // Get column indices
    Filter_File_Handle file( quadruplets_file );

    if( file.Find( "n_interaction_quadruplets" ) )
    {
        // Read n interaction quadruplets
        file.iss >> n_quadruplets;
        Log( Log_Level::Debug, Log_Sender::IO,
             fmt::format( "File {} should have {} quadruplets", quadruplets_file, n_quadruplets ) );
    }
    else
    {
        // Read the whole file
        n_quadruplets = (int)1e8;
        // First line should contain the columns
        file.ResetStream();
        Log( Log_Level::Debug, Log_Sender::IO,
             "Trying to parse quadruplet columns from top of file " + quadruplets_file );
    }

    file.GetLine();
    for( unsigned int i = 0; i < columns.size(); ++i )
    {
        file.iss >> columns[i];
        std::transform( columns[i].begin(), columns[i].end(), columns[i].begin(), ::tolower );
        if( columns[i] == "i" )
            col_i = i;
        else if( columns[i] == "j" )
            col_j = i;
        else if( columns[i] == "da_j" )
            col_da_j = i;
        else if( columns[i] == "db_j" )
            col_db_j = i;
        else if( columns[i] == "dc_j" )
            col_dc_j = i;
        else if( columns[i] == "k" )
            col_k = i;
        else if( columns[i] == "da_k" )
            col_da_k = i;
        else if( columns[i] == "db_k" )
            col_db_k = i;
        else if( columns[i] == "dc_k" )
            col_dc_k = i;
        else if( columns[i] == "l" )
            col_l = i;
        else if( columns[i] == "da_l" )
            col_da_l = i;
        else if( columns[i] == "db_l" )
            col_db_l = i;
        else if( columns[i] == "dc_l" )
            col_dc_l = i;
        else if( columns[i] == "q" )
        {
            col_Q = i;
            Q     = true;
        }
    }

    // Check if interactions have been found in header
    if( !Q )
        Log( Log_Level::Warning, Log_Sender::IO,
             fmt::format( "No interactions could be found in header of quadruplets file ", quadruplets_file ) );

    // Quadruplet Indices
    int q_i = 0;
    int q_j = 0, q_da_j = 0, q_db_j = 0, q_dc_j = 0;
    int q_k = 0, q_da_k = 0, q_db_k = 0, q_dc_k = 0;
    int q_l = 0, q_da_l = 0, q_db_l = 0, q_dc_l = 0;
    scalar q_Q;

    // Get actual Quadruplets Data
    int i_quadruplet = 0;
    std::string sdump;
    while( file.GetLine() && i_quadruplet < n_quadruplets )
    {
        // Read a Quadruplet from the File
        for( unsigned int i = 0; i < columns.size(); ++i )
        {
            // i
            if( i == col_i )
                file.iss >> q_i;
            // j
            else if( i == col_j )
                file.iss >> q_j;
            else if( i == col_da_j )
                file.iss >> q_da_j;
            else if( i == col_db_j )
                file.iss >> q_db_j;
            else if( i == col_dc_j )
                file.iss >> q_dc_j;
            // k
            else if( i == col_k )
                file.iss >> q_k;
            else if( i == col_da_k )
                file.iss >> q_da_k;
            else if( i == col_db_k )
                file.iss >> q_db_k;
            else if( i == col_dc_k )
                file.iss >> q_dc_k;
            // l
            else if( i == col_l )
                file.iss >> q_l;
            else if( i == col_da_l )
                file.iss >> q_da_l;
            else if( i == col_db_l )
                file.iss >> q_db_l;
            else if( i == col_dc_l )
                file.iss >> q_dc_l;
            // Quadruplet magnitude
            else if( i == col_Q && Q )
                file.iss >> q_Q;
            // Otherwise dump the line
            else
                file.iss >> sdump;
        } // end for columns

        // Add the indices and parameter to the corresponding list
        if( q_Q != 0 )
        {
            quadruplets.push_back( { q_i,
                                     q_j,
                                     q_k,
                                     q_l,
                                     { q_da_j, q_db_j, q_dc_j },
                                     { q_da_k, q_db_k, q_dc_k },
                                     { q_da_l, q_db_l, q_dc_l } } );
            quadruplet_magnitudes.push_back( q_Q );
        }

        ++i_quadruplet;
    } // end while GetLine
    Log( Log_Level::Parameter, Log_Sender::IO,
         fmt::format( "Done reading {} spin quadruplets from file \"{}\"", i_quadruplet, quadruplets_file ) );
    noq = i_quadruplet;
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read quadruplets from file  \"{}\"", quadruplets_file ) );
}

void Defects_from_File(
    const std::string & defects_file, int & n_defects, field<Site> & defect_sites, intfield & defect_types ) noexcept
try
{
    n_defects    = 0;
    defect_sites = field<Site>( 0 );
    defect_types = intfield( 0 );

    Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "Reading defects from file \"{}\"", defects_file ) );
    Filter_File_Handle myfile( defects_file );
    int nod = 0;

    if( myfile.Find( "n_defects" ) )
    {
        // Read n interaction pairs
        myfile.iss >> nod;
        Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "File \"{}\" should have {} defects", defects_file, nod ) );
    }
    else
    {
        // Read the whole file
        nod = (int)1e8;
        // First line should contain the columns
        myfile.ResetStream();
        Log( Log_Level::Debug, Log_Sender::IO,
             fmt::format( "Trying to parse defects from top of file \"{}\"", defects_file ) );
    }

    while( myfile.GetLine() && n_defects < nod )
    {
        int _i, _da, _db, _dc, type;
        myfile.iss >> _i >> _da >> _db >> _dc >> type;
        defect_sites.push_back( { _i, { _da, _db, _dc } } );
        defect_types.push_back( type );
        ++n_defects;
    }

    Log( Log_Level::Parameter, Log_Sender::IO,
         fmt::format( "Done reading {} defects from file \"{}\"", n_defects, defects_file ) );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read defects file \"{}\"", defects_file ) );
}

void Pinned_from_File(
    const std::string & pinned_file, int & n_pinned, field<Site> & pinned_sites, vectorfield & pinned_spins ) noexcept
try
{
    int nop      = 0;
    n_pinned     = 0;
    pinned_sites = field<Site>( 0 );
    pinned_spins = vectorfield( 0 );

    Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "Reading pinned sites from file \"{}\"", pinned_file ) );
    Filter_File_Handle myfile( pinned_file );

    if( myfile.Find( "n_pinned" ) )
    {
        // Read n interaction pairs
        myfile.iss >> nop;
        Log( Log_Level::Debug, Log_Sender::IO,
             fmt::format( "File \"{}\" should have {} pinned sites", pinned_file, nop ) );
    }
    else
    {
        // Read the whole file
        nop = (int)1e8;
        // First line should contain the columns
        myfile.ResetStream();
        Log( Log_Level::Debug, Log_Sender::IO,
             fmt::format( "Trying to parse pinned sites from top of file \"{}\"", pinned_file ) );
    }

    while( myfile.GetLine() && n_pinned < nop )
    {
        int _i, _da, _db, _dc;
        scalar sx, sy, sz;
        myfile.iss >> _i >> _da >> _db >> _dc >> sx >> sy >> sz;
        pinned_sites.push_back( { _i, { _da, _db, _dc } } );
        pinned_spins.push_back( { sx, sy, sz } );
        ++n_pinned;
    }

    Log( Log_Level::Parameter, Log_Sender::IO,
         fmt::format( "Done reading {} pinned sites from file \"{}\"", n_pinned, pinned_file ) );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read pinned sites file  \"{}\"", pinned_file ) );
}

int ReadHeaderLine( FILE * fp, char * line )
{
    char c;
    int pos = 0;

    do
    {
        c = (char)fgetc( fp ); // Get current char and move pointer to the next position
        if( c != EOF && c != '\n' )
            line[pos++] = c;          // If it's not the end of the file
    } while( c != EOF && c != '\n' ); // If it's not the end of the file or end of the line

    line[pos] = 0; // Complete the read line
    if( ( pos == 0 || line[0] != '#' ) && c != EOF )
        return ReadHeaderLine( fp, line ); // Recursive call for ReadHeaderLine if the current line is empty

    // The last symbol is the line end symbol
    return pos - 1;
}

void ReadDataLine( FILE * fp, char * line )
{
    char c;
    int pos = 0;

    do
    {
        c = (char)fgetc( fp );
        if( c != EOF && c != '\n' )
            line[pos++] = c;
    } while( c != EOF && c != '\n' );

    line[pos] = 0;
}

} // namespace IO