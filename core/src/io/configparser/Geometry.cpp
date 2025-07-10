#include <io/Configparser.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>
#include <utility/Logging.hpp>
#include <utility/Timing.hpp>

#include <fmt/format.h>

#include <string>

using Utility::Log_Level;
using Utility::Log_Sender;

namespace IO
{

namespace detail
{

namespace
{

struct BravaisConfig
{
    std::vector<Vector3> vectors          = std::vector{ Vector3{ 1, 0, 0 }, Vector3{ 0, 1, 0 }, Vector3{ 0, 0, 1 } };
    Data::BravaisLatticeType lattice_type = Data::BravaisLatticeType::SC;
    std::string lattice_type_str          = "sc";
};

auto Bravais_Vectors_from_TOML( const toml::table & tbl ) -> BravaisConfig
{
    using Data::Geometry;
    using Type = Data::BravaisLatticeType;

    std::optional<BravaisConfig> bravais;
    if( auto bravais_lattice = tbl["bravais_lattice"].value<std::string>() )
    {
        if( bravais_lattice == "sc" )
            bravais.emplace( BravaisConfig{ Geometry::BravaisVectorsSC(), Type::SC, "simple cubic" } );
        else if( bravais_lattice == "fcc" )
            bravais.emplace( BravaisConfig{ Geometry::BravaisVectorsFCC(), Type::FCC, "face-centered cubic" } );
        else if( bravais_lattice == "bcc" )
            bravais.emplace( BravaisConfig{ Geometry::BravaisVectorsBCC(), Type::BCC, "body-centered cubic" } );
        else if( bravais_lattice == "hex2d" )
            bravais.emplace( BravaisConfig{ Geometry::BravaisVectorsHex2D60(), Type::Hex2D,
                                            "hexagonal 2D (default: 60deg angle)" } );
        else if( bravais_lattice == "hex2d60" )
            bravais.emplace(
                BravaisConfig{ Geometry::BravaisVectorsHex2D60(), Type::Hex2D, "hexagonal 2D 60deg angle" } );
        else if( bravais_lattice == "hex2d120" )
            bravais.emplace(
                BravaisConfig{ Geometry::BravaisVectorsHex2D120(), Type::Hex2D, "hexagonal 2D 120deg angle" } );
        else
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( *bravais_lattice, "Bravais lattice \"{}\" unknown." ) );
    }

    if( auto vectors = tbl["bravais_vectors"] )
    {
        if( !bravais )
        {
            try
            {
                using matrix_t = std::array<std::array<scalar, 3>, 3>;
                if( const auto bravais_vectors = toml_transform<matrix_t>( *vectors.node() ) )
                {
                    bravais.emplace(
                        BravaisConfig{ std::vector<Vector3>( 3, Vector3::Zero() ), Type::Irregular, "irregular" } );
                    for( int i = 0; i < 3; ++i )
                        for( int j = 0; j < 3; ++j )
                            bravais->vectors[i][j] = ( *bravais_vectors )[i][j];
                    Log( Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: irregular" );
                }
            }
            catch( ... )
            {
                spirit_handle_exception_core( "Error parsing bravais vectors!" );
            }
        }
        else
            Log( Log_Level::Warning, Log_Sender::IO, "Ignoring specified field 'bravais_vectors'!" );
    }

    if( auto matrix = tbl["bravais_matrix"] )
    {
        if( !bravais )
        {
            try
            {
                using matrix_t = std::array<std::array<scalar, 3>, 3>;
                if( const auto bravais_matrix = toml_transform<matrix_t>( *matrix.node() ) )
                {
                    bravais.emplace(
                        BravaisConfig{ std::vector<Vector3>( 3, Vector3::Zero() ), Type::Irregular, "irregular" } );
                    for( int i = 0; i < 3; ++i )
                        for( int j = 0; j < 3; ++j )
                            bravais->vectors[i][j] = ( *bravais_matrix )[j][i];
                    Log( Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: irregular" );
                }
            }
            catch( ... )
            {
                spirit_handle_exception_core( "Error parsing bravais matrix!" );
            };
        }
        else
            Log( Log_Level::Warning, Log_Sender::IO, "Ignoring specified field 'bravais_matrix'!" );
    }

    if( !bravais )
    {
        Log( Log_Level::Warning, Log_Sender::IO, "No valid bravais lattice specified. Using simple cubic..." );
        bravais.emplace( BravaisConfig{ Geometry::BravaisVectorsSC(), Type::SC, "sc" } );
    }
    return *bravais;
}

auto Pinning_from_TOML( const toml::table & tbl, std::size_t n_cell_atoms ) -> Data::Pinning
{
#ifndef SPIRIT_ENABLE_PINNING
    Log( Log_Level::Parameter, Log_Sender::IO, "Pinning is disabled" );
    if( tbl["pinning"] )
        Log( Log_Level::Warning, Log_Sender::IO, "You specified a 'pinning' section even though pinning is disabled!" );
    return Data::Pinning{ 0, 0, 0, 0, 0, 0, vectorfield( 0 ), field<Site>( 0 ), vectorfield( 0 ) };
#else
    if( !tbl["pinning"] )
    {
        Log( Log_Level::Warning, Log_Sender::IO, "Missing section 'pinning'. Using defaults..." );
        return Data::Pinning{ 0, 0, 0, 0, 0, 0, vectorfield( 0 ), field<Site>( 0 ), vectorfield( 0 ) };
    }

    vectorfield pinned_cell( n_cell_atoms, Vector3{ 0, 0, 1 } );
    //-------------- Insert default values here -----------------------------
    int na_left = 0, na_right = 0;
    int nb_left = 0, nb_right = 0;
    int nc_left = 0, nc_right = 0;
    // Additional pinned sites
    field<Site> pinned_sites( 0 );
    vectorfield pinned_spins( 0 );
    int n_pinned = 0;

    //------------------------------- Parser --------------------------------
    Log( Log_Level::Debug, Log_Sender::IO, "going to read pinning" );
    if( auto pinning_boundary = tbl.at_path( "pinning.boundary" ).as_array() )
    {
        auto read_boundary = []( const auto & node, auto & left, auto & right )
        {
            if( auto value = node.template value<int>() )
            {
                left  = *value;
                right = *value;
            }
            else if( auto pair = node.as_array(); pair && pair->size() == 2 )
            {
                left  = ( *pair )[0].value_or( 0 );
                right = ( *pair )[1].value_or( 0 );
            }
        };
        if( pinning_boundary->size() == 3 )
        {
            read_boundary( ( *pinning_boundary )[0], na_left, na_right );
            read_boundary( ( *pinning_boundary )[1], nb_left, nb_right );
            read_boundary( ( *pinning_boundary )[2], nc_left, nc_right );
        }
        else
            Log( Log_Level::Error, Log_Sender::IO, "Wrong number of pinning boundary regions specified!" );
    }

    if( na_left > 0 || na_right > 0 || nb_left > 0 || nb_right > 0 || nc_left > 0 || nc_right > 0 )
    {
        if( auto cell = tbl.at_path( "pinning.cell" ); cell && cell.is_array() )
        {
            if( auto parsed_cell = toml_transform<vectorfield>( *cell.node() ) )
            {
                pinned_cell = std::move( *parsed_cell );
                if( pinned_cell.size() > n_cell_atoms )
                {
                    Log( Log_Level::Warning, Log_Sender::IO, "Too many pinned cells specified. Truncating..." );
                    pinned_cell.resize( n_cell_atoms );
                }
                else if( pinned_cell.size() < n_cell_atoms )
                    spirit_throw(
                        Utility::Exception_Classifier::Input_parse_failed, Log_Level::Error,
                        "Too few pinned cells specified." );
            }
        }
        else
        {
            na_left  = 0;
            na_right = 0;
            nb_left  = 0;
            nb_right = 0;
            nc_left  = 0;
            nc_right = 0;
            Log( Log_Level::Warning, Log_Sender::IO,
                 "Pinning specified, but keyword 'pinning.cell' not found. Won't pin any spins!" );
        }
    }

    if( auto pinned = tbl.at_path( "pinning.pinned" ) )
    {
        if( auto str = pinned.as_string() )
        {
            auto file_handle = Filter_File_Handle::from_string( str->get() );
            Pinned_from_File( file_handle, n_pinned, pinned_sites, pinned_spins );
        }
        else
            Log( Log_Level::Error, Log_Sender::IO, "Unable to read pinning cell from config!" );
    }

    // Create Pinning
    auto pinning = Data::Pinning{
        na_left, na_right, nb_left, nb_right, nc_left, nc_right, pinned_cell, pinned_sites, pinned_spins,
    };

    // Return Pinning
    Log( Log_Level::Debug, Log_Sender::IO, "pinning has been read" );
    return pinning;
#endif // SPIRIT_ENABLE_PINNING
}

auto Basis_Cell_Composition_from_TOML( const toml::table & tbl, std::size_t n_cell_atoms )
    -> Data::Basis_Cell_Composition
{
#ifdef SPIRIT_ENABLE_DEFECTS
    static constexpr bool enable_defects = true;
#else
    static constexpr bool enable_defects = false;
#endif

    if( auto atom_types = tbl["atom_types"].value<std::string>(); enable_defects && atom_types )
    {
        using AtomTypesParser = TableParser<int, int, scalar, scalar, int>;
        const AtomTypesParser parser( { "i", "type", "c", "mu_s", "spin_qn" } );

        // Disorder
        auto file_handle = Filter_File_Handle::from_string( *atom_types );

        const auto factory = []( const auto & headings )
        {
            const auto set_default
                = [&headings]( const char * label, auto default_value ) -> std::optional<decltype( default_value )>
            {
                if( headings.at( label ) < 0 )
                {
                    Log( Log_Level::Warning, Log_Sender::IO,
                         fmt::format(
                             "Table 'atom_types' has missing column '{}', using default value {}", label,
                             default_value ) );
                    return std::optional{ default_value };
                }
                else
                    return std::nullopt;
            };

            auto default_mu_s    = set_default( "mu_s", scalar( 1.0 ) );
            auto default_spin_qn = set_default( "spin_qn", 1 );

            if( headings.at( "i" ) < 0 || headings.at( "type" ) < 0 || headings.at( "c" ) < 0 )
            {
                spirit_throw(
                    Utility::Exception_Classifier::Input_parse_failed, Utility::Log_Level::Error,
                    "The columns 'i', 'type' and 'c' are required. Cannot parse atom types!" );
            }

            return [default_spin_qn, default_mu_s]( const AtomTypesParser::read_row_t & row )
            {
                struct Row
                {
                    int iatom, type;
                    scalar concentration;
                    scalar mu_s;
                    int spin_qn;
                };

                auto result = IO::make_from_tuple<Row>( row );

                if( default_mu_s )
                    result.mu_s = *default_mu_s;

                if( default_spin_qn )
                    result.spin_qn = *default_spin_qn;

                return result;
            };
        };
        const auto data = parser.parse( file_handle, "n_atom_types", 5, factory );

        auto cell_composition = Data::Basis_Cell_Composition{ true, {}, {}, {}, {}, {} };
        cell_composition.iatom.reserve( data.size() );
        cell_composition.atom_type.reserve( data.size() );
        cell_composition.concentration.reserve( data.size() );
        cell_composition.mu_s.reserve( data.size() );
        cell_composition.spin_qn.reserve( data.size() );

        for( const auto & row : data )
        {
            cell_composition.iatom.push_back( row.iatom );
            cell_composition.atom_type.push_back( row.type );
            cell_composition.concentration.push_back( row.concentration );
            cell_composition.mu_s.push_back( row.mu_s );
            cell_composition.spin_qn.push_back( row.spin_qn );
        }

        if( !data.empty() )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format(
                     "{} atom types, iatom={} atom type={} concentration={}", data.size(), cell_composition.iatom[0],
                     cell_composition.atom_type[0], cell_composition.concentration[0] ) );

        return cell_composition;
    }
    else
    {
        auto cell_composition = Data::Basis_Cell_Composition::make_default( n_cell_atoms, /*disordered=*/false );
        auto parse_quantity   = [&tbl]( const std::string_view label, auto & container )
        {
            const auto node = tbl[label];
            if( !node )
            {
                Log( Log_Level::Warning, Log_Sender::IO,
                     fmt::format(
                         "Keyword '{}' not found. Using Default: {}", label, container.empty() ? 0 : container[0] ) );
                return;
            }

            using ValueType = typename std::decay_t<decltype( container )>::value_type;
            auto msg        = [label]( auto count, auto idx, auto value )
            {
                return fmt::format(
                    "Not enough values specified after '{}'. Expected {}. Using "
                    "{}[{}]={}[0]={}",
                    label, count, label, idx, label, value );
            };
            if( const auto value = node.template value<ValueType>() )
            {
                std::fill( container.begin(), container.end(), *value );
                for( unsigned int iatom = 1; iatom < container.size(); ++iatom )
                    Log( Log_Level::Warning, Log_Sender::IO, msg( container.size(), iatom, *value ) );
            }
            else if( const auto array = node.as_array() )
            {
                for( unsigned int iatom = 0; iatom < container.size(); ++iatom )
                    if( iatom < array->size() )
                    {
                        if( auto arr_value = ( *array )[iatom].template value<ValueType>() )
                            container[iatom] = *arr_value;
                        else
                            Log( Log_Level::Error, Log_Sender::IO,
                                 fmt::format(
                                     "Invalid entry while parsing array for '{}', keeping default value {}", label,
                                     container[iatom] ) );
                    }
                    else
                    {
                        Log( Log_Level::Warning, Log_Sender::IO, msg( container.size(), iatom, container[0] ) );
                        container[iatom] = container[0];
                    }
            }
        };

        parse_quantity( "mu_s", cell_composition.mu_s );
        parse_quantity( "spin_qn", cell_composition.spin_qn );

        return cell_composition;
    }
}

} // namespace

} // namespace detail

auto Geometry_from_TOML( const toml::table & root ) -> Data::Geometry
{
    Log( Log_Level::Debug, Log_Sender::IO, "Geometry: building" );
    const toml::table default_table{};
    const auto & tbl = [node_view = root.at_path( "geometry" ).as_table(), &default_table]() -> const toml::table &
    {
        if( node_view )
            return *node_view;
        else
        {
            Log( Log_Level::Warning, Log_Sender::IO, "Missing config section: 'geometry'. Using defaults..." );
            return default_table;
        }
    }();

    const auto lattice_constant = [&tbl]
    {
        scalar lattice_constant = 1.0;
        read_value( tbl, "lattice_constant", lattice_constant );
        return lattice_constant;
    }();
    const auto bravais = detail::Bravais_Vectors_from_TOML( tbl );
    const auto n_cells = [&tbl]
    {
        if( auto n_cells = tbl["n_basis_cells"].as_array() )
        {
            auto result = intfield{ 100, 1, 1 };
            for( unsigned int i = 0; i < 3 && i < n_cells->size(); ++i )
            {
                if( auto value = ( *n_cells )[i].value<int>() )
                    result[i] = *value;
                else
                    result[i] = 1; // TODO: This codepath should error instead!
            }
            return result;
        }
        else
            return intfield{ 100, 100, 1 };
    }();

    const auto cell_atoms = [&tbl]
    {
        if( auto value = tbl["basis"].value<std::string>() )
        {
            auto file_handle = Filter_File_Handle::from_string( *value );
            auto basis       = Basis_from_File( file_handle );
            if( basis.size() == 0 )
                basis = std::vector<Vector3>{ { 0, 0, 0 } };
            return basis;
        }
        else
            return std::vector<Vector3>{ { 0, 0, 0 } };
    }();
    const auto n_cell_atoms = cell_atoms.size();

    const auto pinning          = detail::Pinning_from_TOML( tbl, n_cell_atoms );
    const auto cell_composition = detail::Basis_Cell_Composition_from_TOML( tbl, n_cell_atoms );
    const auto defects          = [&tbl]
    {
        if( auto defects_node = tbl["defects"].value<std::string>() )
        {
            auto file_handle = Filter_File_Handle::from_string( *defects_node );
            return Defects_from_File( file_handle );
        }
        else
            return Data::Defects{};
    }();

    // Log the parameters
    std::vector<std::string> parameter_log;
    parameter_log.emplace_back( "Geometry:" );
    parameter_log.emplace_back( fmt::format( "    lattice constant = {} Angstrom", lattice_constant ) );
    parameter_log.emplace_back( fmt::format( "    Bravais lattice type: {}", bravais.lattice_type_str ) );
    Log( Log_Level::Debug, Log_Sender::IO,
         {
             "    Bravais vectors in units of lattice constant",
             fmt::format( "        a = {}", bravais.vectors[0].transpose() / lattice_constant ),
             fmt::format( "        b = {}", bravais.vectors[1].transpose() / lattice_constant ),
             fmt::format( "        c = {}", bravais.vectors[2].transpose() / lattice_constant ),
         } );
    parameter_log.emplace_back( "    Bravais vectors" );
    parameter_log.emplace_back( fmt::format( "        a = {}", bravais.vectors[0].transpose() ) );
    parameter_log.emplace_back( fmt::format( "        b = {}", bravais.vectors[1].transpose() ) );
    parameter_log.emplace_back( fmt::format( "        c = {}", bravais.vectors[2].transpose() ) );
    parameter_log.emplace_back( fmt::format( "    basis cell: {} atom(s)", n_cell_atoms ) );
    parameter_log.emplace_back( "    relative positions (first 10):" );
    for( std::size_t iatom = 0; iatom < n_cell_atoms && iatom < 10; ++iatom )
        parameter_log.emplace_back( fmt::format(
            "        atom {} at ({}), mu_s={}, spin_qn={}", iatom, cell_atoms[iatom].transpose(),
            cell_composition.mu_s[iatom], cell_composition.spin_qn[iatom] ) );

    parameter_log.emplace_back( "    absolute atom positions (first 10):" );
    for( std::size_t iatom = 0; iatom < n_cell_atoms && iatom < 10; ++iatom )
    {
        Vector3 cell_atom = lattice_constant
                            * ( bravais.vectors[0] * cell_atoms[iatom][0] + bravais.vectors[1] * cell_atoms[iatom][1]
                                + bravais.vectors[2] * cell_atoms[iatom][2] );
        parameter_log.emplace_back( fmt::format( "        atom {} at ({})", iatom, cell_atom.transpose() ) );
    }

    if( cell_composition.disordered )
        parameter_log.emplace_back( "    note: the lattice has some disorder!" );

#ifdef SPIRIT_ENABLE_PINNING
    // Log pinning
    auto n_pinned_cell_sites = pinning.sites.size();
    if( n_pinned_cell_sites == 0 && pinning.na_left == 0 && pinning.na_right == 0 && pinning.nb_left == 0
        && pinning.nb_right == 0 && pinning.nc_left == 0 && pinning.nc_right == 0 )
    {
        parameter_log.emplace_back( "    no pinned spins" );
    }
    else
    {
        parameter_log.emplace_back( "    pinning of boundary cells:" );
        parameter_log.emplace_back(
            fmt::format( "        n_a: left={}, right={}", pinning.na_left, pinning.na_right ) );
        parameter_log.emplace_back(
            fmt::format( "        n_b: left={}, right={}", pinning.nb_left, pinning.nb_right ) );
        parameter_log.emplace_back(
            fmt::format( "        n_c: left={}, right={}", pinning.nc_left, pinning.nc_right ) );

        parameter_log.emplace_back( "        pinned to (showing first 10 sites):" );
        for( std::size_t i = 0; i < std::min( pinning.pinned_cell.size(), static_cast<std::size_t>( 10 ) ); ++i )
        {
            parameter_log.emplace_back(
                fmt::format( "          cell atom[{}] = ({})", i, pinning.pinned_cell[i].transpose() ) );
        }
        if( n_pinned_cell_sites == 0 )
            parameter_log.emplace_back( "    no individually pinned sites" );
        else
        {
            parameter_log.emplace_back(
                fmt::format( "    {} individually pinned sites. Showing the first 10:", n_pinned_cell_sites ) );
            for( std::size_t i = 0; i < std::min( pinning.sites.size(), static_cast<std::size_t>( 10 ) ); ++i )
            {
                parameter_log.emplace_back( fmt::format(
                    "        pinned site[{}]: {} at ({} {} {}) = ({})", i, pinning.sites[i].i,
                    pinning.sites[i].translations[0], pinning.sites[i].translations[1],
                    pinning.sites[i].translations[2], pinning.spins[i].transpose() ) );
            }
        }
    }
#endif

    // Defects
#ifdef SPIRIT_ENABLE_DEFECTS
    if( defect_sites.empty() )
        parameter_log.emplace_back( "    no defects" );
    else
    {
        parameter_log.emplace_back( fmt::format( "    {} defects (showing first 10 sites):", defect_sites.size() ) );
        for( std::size_t i = 0; i < std::min( defect_sites.size(), static_cast<std::size_t>( 10 ) ); ++i )
        {
            parameter_log.emplace_back( fmt::format(
                "        defect[{}]: translations=({} {} {}), type=", i, defect_sites[i].translations[0],
                defect_sites[i].translations[1], defect_sites[i].translations[2], defect_types[i] ) );
        }
    }
#endif

    // Log parameters
    parameter_log.emplace_back( "    lattice: n_basis_cells" );
    parameter_log.emplace_back( fmt::format( "        na = {}", n_cells[0] ) );
    parameter_log.emplace_back( fmt::format( "        nb = {}", n_cells[1] ) );
    parameter_log.emplace_back( fmt::format( "        nc = {}", n_cells[2] ) );

    // Return geometry
    const auto geometry
        = Data::Geometry( bravais.vectors, n_cells, cell_atoms, cell_composition, lattice_constant, pinning, defects );

    parameter_log.emplace_back( fmt::format( "    {} spins", geometry.nos ) );
    parameter_log.emplace_back( fmt::format( "    the geometry is {}-dimensional", geometry.dimensionality ) );

    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Geometry: built" );

    return geometry;
};

} // namespace IO
