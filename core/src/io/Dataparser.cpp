#include <engine/Vectormath.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/IO.hpp>
#include <io/OVF_File.hpp>
#include <io/Tableparser.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <Eigen/Core>
#include <Eigen/Dense>

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
        file_handle >> spins[i][0];
        file_handle >> spins[i][1];
        file_handle >> spins[i][2];

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
    std::shared_ptr<State::chain_t> chain, const std::string & file, int start_image_infile,
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
    const std::string & anisotropy_file, const Data::Geometry& geometry, int & n_indices,
    intfield & anisotropy_index, scalarfield & anisotropy_magnitude, vectorfield & anisotropy_normal,
    intfield & cubic_anisotropy_index, scalarfield & cubic_anisotropy_magnitude ) noexcept
try
{
    Log( Log_Level::Debug, Log_Sender::IO, "Reading anisotropy from file " + anisotropy_file );

    // parser initialization
    using AnisotropyTableParser = TableParser<int, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar>;
    const AnisotropyTableParser parser( { "i", "k", "kx", "ky", "kz", "ka", "kb", "kc", "k4" } );

    // factory function for creating a lambda that transforms the row that is read
    auto transform_factory = [&anisotropy_file, &geometry]( const std::map<std::string_view, int> & idx )
    {
        bool K_xyz = false, K_abc = false, K_magnitude = false;

        if( idx.at( "kx" ) >= 0 && idx.at( "ky" ) >= 0 && idx.at( "kz" ) >= 0 )
            K_xyz = true;
        if( idx.at( "ka" ) >= 0 && idx.at( "kb" ) >= 0 && idx.at( "kc" ) >= 0 )
            K_abc = true;
        if( idx.at( "k" ) >= 0 )
            K_magnitude = true;

        if( !K_xyz && !K_abc )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "No anisotropy data could be found in header of file \"{}\"", anisotropy_file ) );

        return [K_xyz, K_abc, K_magnitude,
                &geometry]( const AnisotropyTableParser::read_row_t & row ) -> std::tuple<int, scalar, Vector3, scalar>
        {
            auto [i, k, kx, ky, kz, ka, kb, kc, k4] = row;

            Vector3 K_temp;
            if( K_xyz )
                K_temp = { kx, ky, kz };
            // Anisotropy vector orientation
            if( K_abc )
            {
                K_temp = { ka, kb, kc };
                K_temp = { K_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[0] ),
                           K_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[1] ),
                           K_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[2] ) };
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
                k = K_temp.norm();
                if( k != 0 )
                    K_temp.normalize();
            }

            return std::make_tuple( i, k, K_temp, k4 );
        };
    };

    const std::string anisotropy_size_id = "n_anisotropy";
    const auto data                      = parser.parse( anisotropy_file, anisotropy_size_id, 6ul, transform_factory );
    n_indices                            = data.size();

    // Arrays
    anisotropy_index           = intfield( 0 );
    anisotropy_magnitude       = scalarfield( 0 );
    anisotropy_normal          = vectorfield( 0 );
    cubic_anisotropy_index     = intfield( 0 );
    cubic_anisotropy_magnitude = scalarfield( 0 );

    for( const auto & [i, k, k_vec, k4] : data )
    {
        if( k != 0 )
        {
            anisotropy_index.push_back( i );
            anisotropy_magnitude.push_back( k );
            anisotropy_normal.push_back( k_vec );
        }
        if( k4 != 0 )
        {
            Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "appending spin K4\"{}\"", k4 ) );
            cubic_anisotropy_index.push_back( i );
            cubic_anisotropy_magnitude.push_back( k4 );
        }
    }
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read anisotropies from file \"{}\"", anisotropy_file ) );
}

void Biaxial_Anisotropy_Axes_from_File(
    const std::string & anisotropy_axes_file, const Data::Geometry& geometry, int & n_axes,
    std::map<int, std::pair<Vector3, Vector3>> & anisotropy_axes ) noexcept
try
{
    // parser initialization
    using AnisotropyTableParser = TableParser<
        int, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar>;
    const AnisotropyTableParser parser(
        { "i", "k1x", "k1y", "k1z", "k1a", "k1b", "k1c", "k2x", "k2y", "k2z", "k2a", "k2b", "k2c" } );

    // factory function for creating a lambda that transforms the row that is read
    auto transform_factory = [&anisotropy_axes_file, &geometry]( const std::map<std::string_view, int> & idx )
    {
        bool K1_xyz = ( idx.at( "k1x" ) >= 0 && idx.at( "k1y" ) >= 0 && idx.at( "k1z" ) >= 0 );
        bool K1_abc = ( idx.at( "k1a" ) >= 0 && idx.at( "k1b" ) >= 0 && idx.at( "k1c" ) >= 0 );
        bool K2_xyz = ( idx.at( "k2x" ) >= 0 && idx.at( "k2y" ) >= 0 && idx.at( "k2z" ) >= 0 );
        bool K2_abc = ( idx.at( "k2a" ) >= 0 && idx.at( "k2b" ) >= 0 && idx.at( "k2c" ) >= 0 );

        if( !( ( K1_xyz || K1_abc ) && ( K2_xyz || K2_abc ) ) )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "No anisotropy data could be found in header of file \"{}\"", anisotropy_axes_file ) );

        return [K1_xyz, K1_abc, K2_xyz, K2_abc, &geometry](
                   const AnisotropyTableParser::read_row_t & row ) -> std::pair<int, std::pair<Vector3, Vector3>>
        {
            auto [i, k1x, k1y, k1z, k1a, k1b, k1c, k2x, k2y, k2z, k2a, k2b, k2c] = row;

            Vector3 K1_temp, K2_temp;
            if( K1_xyz )
                K1_temp = { k1x, k1y, k1z };
            // Anisotropy vector orientation
            if( K1_abc )
            {
                K1_temp = { k1a, k1b, k1c };
                K1_temp = { K1_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[0] ),
                            K1_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[1] ),
                            K1_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[2] ) };
            }
            K1_temp.normalize();

            if( K2_xyz )
                K2_temp = { k2x, k2y, k2z };
            // Anisotropy vector orientation
            if( K2_abc )
            {
                K2_temp = { k2a, k2b, k2c };
                K2_temp = { K2_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[0] ),
                            K2_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[1] ),
                            K2_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[2] ) };
            }

            // orthogonalize and normalize
            K2_temp = K2_temp - K1_temp.dot( K2_temp ) * K1_temp;
            K2_temp.normalize();

            return std::pair( i, std::pair{ K1_temp, K2_temp } );
        };
    };

    const auto data = parser.parse( anisotropy_axes_file, "n_biaxial_anisotropy_axes", 7ul, transform_factory );
    n_axes          = data.size();

    anisotropy_axes = std::map( begin( data ), end( data ) );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read anisotropy axes from file \"{}\"", anisotropy_axes_file ) );
}

void Biaxial_Anisotropy_Terms_from_File(
    const std::string & anisotropy_terms_file, const Data::Geometry &, int & n_terms,
    std::map<int, field<PolynomialTerm>> & anisotropy_terms ) noexcept
try
{
    // parser initialization
    using AnisotropyTableParser = TableParser<int, unsigned int, unsigned int, unsigned int, scalar>;
    const AnisotropyTableParser parser( { "i", "n1", "n2", "n3", "k" } );

    // factory function for creating a lambda that transforms the row that is read
    auto transform_factory = [&anisotropy_terms_file]( const std::map<std::string_view, int> & idx )
    {
        if( idx.at( "i" ) < 0 || idx.at( "k" ) < 0
            || ( idx.at( "n1" ) < 0 && idx.at( "n2" ) < 0 && idx.at( "n3" ) < 0 ) )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "No anisotropy data could be found in header of file \"{}\"", anisotropy_terms_file ) );

        return []( AnisotropyTableParser::read_row_t row ) -> std::pair<int, PolynomialTerm>
        {
            auto [i, n1, n2, n3, k] = row;
            return { i, PolynomialTerm{ k, n1, n2, n3 } };
        };
    };

    const auto data = parser.parse( anisotropy_terms_file, "n_biaxial_anisotropy_terms", 6ul, transform_factory );
    n_terms         = data.size();

    anisotropy_terms.clear();
    for( const auto & [i, term] : data )
        anisotropy_terms[i].push_back( term );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read anisotropy terms from file \"{}\"", anisotropy_terms_file ) );
}

void Biaxial_Anisotropy_from_File(
    const std::string & anisotropy_axes_file, const std::string & anisotropy_terms_file,
    const Data::Geometry & geometry, int & n_indices, intfield & anisotropy_indices,
    field<PolynomialBasis> & anisotropy_polynomial_bases, field<unsigned int> & anisotropy_polynomial_site_p,
    field<PolynomialTerm> & anisotropy_polynomial_terms ) noexcept
try
{
    int n_axes = 0, n_terms = 0;
    auto anisotropy_axes  = std::map<int, std::pair<Vector3, Vector3>>();
    auto anisotropy_terms = std::map<int, field<PolynomialTerm>>();

    Log( Log_Level::Debug, Log_Sender::IO, "Reading anisotropy axes from file " + anisotropy_axes_file );
    Biaxial_Anisotropy_Axes_from_File( anisotropy_axes_file, geometry, n_axes, anisotropy_axes );

    Log( Log_Level::Debug, Log_Sender::IO, "Reading anisotropy terms from file " + anisotropy_terms_file );
    Biaxial_Anisotropy_Terms_from_File( anisotropy_terms_file, geometry, n_terms, anisotropy_terms );

    n_indices = n_axes + n_terms;

    // Arrays
    anisotropy_indices           = intfield{};
    anisotropy_polynomial_bases  = field<PolynomialBasis>{};
    anisotropy_polynomial_site_p = field<unsigned int>{};
    anisotropy_polynomial_terms  = field<PolynomialTerm>{};

    if( n_terms > 0 )
    {
        anisotropy_polynomial_site_p.push_back( 0 );
        anisotropy_polynomial_terms.reserve( n_terms );
    }

    const scalar thresh = 1e-5;
    for( const auto & [i, axes] : anisotropy_axes )
    {
        if( axes.first.norm() > thresh && axes.second.norm() > thresh )
        {
            if( const auto & terms = anisotropy_terms[i]; !terms.empty() )
            {
                anisotropy_indices.push_back( i );
                anisotropy_polynomial_bases.push_back(
                    PolynomialBasis{ axes.first, axes.second, axes.first.cross( axes.second ).normalized() } );
                anisotropy_polynomial_site_p.push_back( anisotropy_polynomial_site_p.back() + terms.size() );
                std::copy( begin( terms ), end( terms ), std::back_inserter( anisotropy_polynomial_terms ) );
            }
            else
            {
                Log( Log_Level::Warning, Log_Sender::IO,
                     fmt::format( "Anisotropy axes specified at site i={} but no polynomial terms were found.", i ) );
            }
        }
        else
        {
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format(
                     "Discarding anisotropy axes at site i={} because they are smaller than threshold ({})", i,
                     thresh ) );
        }
    }

    if( int diff = anisotropy_terms.size() - anisotropy_axes.size(); diff > 0 )
    {
        Log( Log_Level::Warning, Log_Sender::IO,
             fmt::format( "There were polynomials specified without any matching axes at {} sites.", diff ) );
    }
}
catch( ... )
{
    spirit_rethrow( fmt::format(
        "Could not read anisotropies from files \"{}\" & \"{}\" ", anisotropy_axes_file, anisotropy_terms_file ) );
}

// Read Basis from file
void Basis_from_File(
    const std::string & basis_file, Data::Basis_Cell_Composition & cell_composition, std::vector<Vector3> & cell_atoms,
    std::size_t & n_cell_atoms ) noexcept
{

    Log( Log_Level::Info, Log_Sender::IO, "Reading basis from file " + basis_file );

    Filter_File_Handle basis_file_handle( basis_file );

    // Read basis cell
    if( basis_file_handle.Find( "basis" ) )
    {
        // Read number of atoms in the basis cell
        basis_file_handle.GetLine();
        basis_file_handle >> n_cell_atoms;
        cell_atoms = std::vector<Vector3>( n_cell_atoms );
        cell_composition.iatom.resize( n_cell_atoms );
        cell_composition.atom_type = std::vector<int>( n_cell_atoms, 0 );
        cell_composition.mu_s      = std::vector<scalar>( n_cell_atoms, 1 );

        // Read atom positions
        for( std::size_t iatom = 0; iatom < n_cell_atoms; ++iatom )
        {
            basis_file_handle.GetLine();
            basis_file_handle >> cell_atoms[iatom][0] >> cell_atoms[iatom][1] >> cell_atoms[iatom][2];
            cell_composition.iatom[iatom] = static_cast<int>( iatom );
        }
    }
}

// Read from Pairs file by Markus & Bernd
void Pairs_from_File(
    const std::string & pairs_file, const Data::Geometry & geometry, int & nop,
    pairfield & exchange_pairs, scalarfield & exchange_magnitudes, pairfield & dmi_pairs, scalarfield & dmi_magnitudes,
    vectorfield & dmi_normals ) noexcept
try
{
    Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "Reading spin pairs from file \"{}\"", pairs_file ) );

    using PairTableParser
        = TableParser<int, int, int, int, int, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar>;
    const PairTableParser parser(
        { "i", "j", "da", "db", "dc", "dij", "dijx", "dijy", "dijz", "dija", "dijb", "dijc", "jij" } );

    auto transform_factory = [&pairs_file, &geometry]( const std::map<std::string_view, int> & idx )
    {
        bool DMI_xyz = false, DMI_abc = false, DMI_magnitude = false;

        if( idx.at( "dijx" ) >= 0 && idx.at( "dijy" ) >= 0 && idx.at( "dijz" ) >= 0 )
            DMI_xyz = true;
        if( idx.at( "dija" ) >= 0 && idx.at( "dijb" ) >= 0 && idx.at( "dijc" ) >= 0 )
            DMI_abc = true;
        if( idx.at( "dij" ) >= 0 )
            DMI_magnitude = true;

        if( idx.at( "j" ) < 0 && !DMI_xyz && !DMI_abc )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "No interactions could be found in pairs file \"{}\"", pairs_file ) );

        return [DMI_xyz, DMI_abc, DMI_magnitude,
                &geometry]( const PairTableParser::read_row_t & row ) -> std::tuple<Pair, scalar, Vector3, scalar>
        {
            auto [i, j, da, db, dc, Dij, Dijx, Dijy, Dijz, Dija, Dijb, Dijc, Jij] = row;
            Vector3 D_temp;
            if( DMI_xyz )
                D_temp = { Dijx, Dijy, Dijz };
            // Anisotropy vector orientation
            if( DMI_abc )
            {
                D_temp = { Dija, Dijb, Dijc };
                D_temp = { D_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[0] ),
                           D_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[1] ),
                           D_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[2] ) };
            }

            if( !DMI_magnitude )
                Dij = D_temp.norm();

            D_temp.normalize();

            return std::make_tuple( Pair{ i, j, { da, db, dc } }, Jij, D_temp, Dij );
        };
    };

    const auto data = parser.parse( pairs_file, "n_interaction_pairs", 20, transform_factory );
    nop             = data.size();

    {
        auto predicate = []( const auto & first, const auto & second ) -> int
        {
            const auto t1 = std::array{ first.translations[0], first.translations[1], first.translations[2] };
            const auto t2 = std::array{ second.translations[0], second.translations[1], second.translations[2] };

            if( first.i == second.i && first.j == second.j && t1 == std::array{ t2[0], t2[1], t2[2] } )
                return 1;
            else if( first.i == second.j && first.j == second.i && t1 == std::array{ -t2[0], -t2[1], -t2[2] } )
                return -1;
            else
                return 0;
        };

        // Add the indices and parameters to the corresponding lists and deduplicate entries
        for( const auto & [pair, Jij, D_vec, Dij] : data )
        {
            if( Jij != 0 )
            {
                bool already_in{ false };
                int atposition = -1;
                for( std::size_t icheck = 0; icheck < exchange_pairs.size(); ++icheck )
                {
                    if( predicate( pair, exchange_pairs[icheck] ) == 0 )
                        continue;

                    already_in = true;
                    atposition = icheck;
                    break;
                }
                if( already_in )
                {
                    exchange_magnitudes[atposition] += Jij;
                }
                else
                {
                    exchange_pairs.push_back( pair );
                    exchange_magnitudes.push_back( Jij );
                }
            }
            if( Dij != 0 )
            {
                bool already_in{ false };
                int dfact      = 1;
                int atposition = -1;
                for( std::size_t icheck = 0; icheck < dmi_pairs.size(); ++icheck )
                {
                    const auto pred = predicate( pair, dmi_pairs[icheck] );
                    if( pred == 0 )
                        continue;

                    already_in = true;
                    dfact      = pred;
                    break;
                }
                if( already_in )
                {
                    // Calculate new D vector by adding the two redundant ones and normalize again
                    Vector3 newD    = dmi_magnitudes[atposition] * dmi_normals[atposition] + dfact * Dij * D_vec;
                    scalar newdnorm = newD.norm();
                    newD.normalize();
                    dmi_magnitudes[atposition] = newdnorm;
                    dmi_normals[atposition]    = newD;
                }
                else
                {
                    dmi_pairs.push_back( pair );
                    dmi_magnitudes.push_back( Dij );
                    dmi_normals.push_back( D_vec );
                }
            }
        }
    }

    Log( Log_Level::Parameter, Log_Sender::IO,
         fmt::format(
             "Done reading {} spin pairs from file \"{}\", giving {} exchange and {} DM (symmetry-reduced) pairs.", nop,
             pairs_file, exchange_pairs.size(), dmi_pairs.size() ) );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read pairs file \"{}\"", pairs_file ) );
}

// Read from Quadruplet file
void Quadruplets_from_File(
    const std::string & quadruplets_file, const Data::Geometry &, int & noq,
    quadrupletfield & quadruplets, scalarfield & quadruplet_magnitudes ) noexcept
try
{
    Log( Log_Level::Debug, Log_Sender::IO,
         fmt::format( "Reading spin quadruplets from file \"{}\"", quadruplets_file ) );

    // parser initialization
    using QuadrupletTableParser = TableParser<int, int, int, int, int, int, int, int, int, int, int, int, int, scalar>;
    const QuadrupletTableParser parser(
        { "i", "j", "da_j", "db_j", "dc_j", "k", "da_k", "db_k", "dc_k", "l", "da_l", "db_l", "dc_l", "q" } );

    // factory function for creating a lambda that transforms the row that is read
    auto transform_factory = [&quadruplets_file]( const std::map<std::string_view, int> & idx )
    {
        if( idx.at( "q" ) < 0 )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "No interactions could be found in header of quadruplets file ", quadruplets_file ) );

        return []( const QuadrupletTableParser::read_row_t & row ) -> std::tuple<Quadruplet, scalar>
        {
            const auto & [i, j, da_j, db_j, dc_j, k, da_k, db_k, dc_k, l, da_l, db_l, dc_l, Q] = row;
            return std::make_tuple(
                Quadruplet{ i, j, k, l, { da_j, db_j, dc_j }, { da_k, db_k, dc_k }, { da_l, db_l, dc_l } }, Q );
        };
    };
    const auto data = parser.parse( quadruplets_file, "n_interaction_quadruplets", 20, transform_factory );
    noq             = data.size();

    for( const auto & [quadruplet, magnitude] : data )
    {
        if( magnitude != 0 )
        {
            quadruplets.push_back( quadruplet );
            quadruplet_magnitudes.push_back( magnitude );
        }
    }
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
        myfile >> nod;
        Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "File \"{}\" should have {} defects", defects_file, nod ) );
    }
    else
    {
        // Read the whole file
        nod = (int)1e8;
        // First line should contain the columns
        myfile.To_Start();
        Log( Log_Level::Debug, Log_Sender::IO,
             fmt::format( "Trying to parse defects from top of file \"{}\"", defects_file ) );
    }

    while( myfile.GetLine() && n_defects < nod )
    {
        Site site{};
        int type{ 0 };
        myfile >> site.i >> site.translations[0] >> site.translations[1] >> site.translations[2] >> type;
        defect_sites.push_back( site );
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
        myfile >> nop;
        Log( Log_Level::Debug, Log_Sender::IO,
             fmt::format( "File \"{}\" should have {} pinned sites", pinned_file, nop ) );
    }
    else
    {
        // Read the whole file
        nop = (int)1e8;
        // First line should contain the columns
        myfile.To_Start();
        Log( Log_Level::Debug, Log_Sender::IO,
             fmt::format( "Trying to parse pinned sites from top of file \"{}\"", pinned_file ) );
    }

    while( myfile.GetLine() && n_pinned < nop )
    {
        Site site{};
        Vector3 orientation{};
        myfile >> site.i >> site.translations[0] >> site.translations[1] >> site.translations[2] >> orientation.x()
            >> orientation.y() >> orientation.z();
        pinned_sites.push_back( site );
        pinned_spins.push_back( orientation );
        ++n_pinned;
    }

    Log( Log_Level::Parameter, Log_Sender::IO,
         fmt::format( "Done reading {} pinned sites from file \"{}\"", n_pinned, pinned_file ) );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read pinned sites file  \"{}\"", pinned_file ) );
}

} // namespace IO
