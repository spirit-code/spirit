#include <io/Configparser.hpp>
#include <io/Configwriter.hpp>
#include <utility/Formatters_Eigen.hpp>

#include <fmt/format.h>

namespace IO
{

auto Geometry_to_TOML( const Data::Geometry & geometry ) -> toml::table
{
    toml::table tbl;

    tbl.insert( "boundary_conditions", toml_array_from_container( geometry.boundary_conditions ) );
    tbl.insert( "lattice_constant", geometry.lattice_constant );
    tbl.insert( "n_basis_cells", toml_array_from_container( geometry.n_cells ) );
    if( geometry.classifier == Data::BravaisLatticeType::SC )
        tbl.insert( "bravais_lattice", "sc" );
    else if( geometry.classifier == Data::BravaisLatticeType::FCC )
        tbl.insert( "bravais_lattice", "fcc" );
    else if( geometry.classifier == Data::BravaisLatticeType::BCC )
        tbl.insert( "bravais_lattice", "bcc" );
    else if( geometry.classifier == Data::BravaisLatticeType::Hex2D )
        tbl.insert( "bravais_lattice", "hex2d120" );
    else
        tbl.insert(
            "bravais_vectors", fmt::format(
                                   "{0}\n{1}\n{2}\n", geometry.bravais_vectors[0].transpose(),
                                   geometry.bravais_vectors[1].transpose(), geometry.bravais_vectors[2].transpose() ) );

    if( geometry.n_cell_atoms > 1 )
    {
        std::ostringstream oss;
        oss << fmt::format( "\n{}\n", geometry.cell_atoms.size() );
        for( const auto & vector : geometry.cell_atoms )
            oss << fmt::format( "{}\n", vector.transpose() );
        tbl.insert( "basis", oss.str() );
    }

    if( !geometry.cell_composition.disordered )
    {
        auto partial_array = []( const auto & container, const std::size_t size )
        {
            toml::array array{};
            array.reserve( size );
            array.insert(
                array.end(), container.begin(), size >= container.size() ? container.end() : container.begin() + size );
            return array;
        };

        tbl.insert( "mu_s", partial_array( geometry.mu_s, geometry.n_cell_atoms ) );
        tbl.insert( "spin_qn", partial_array( geometry.spin_qn, geometry.n_cell_atoms ) );
    }
    else
    {
        const auto & iatom         = geometry.cell_composition.iatom;
        const auto & atom_type     = geometry.cell_composition.atom_type;
        const auto & concentration = geometry.cell_composition.concentration;
        const auto & mu_s          = geometry.cell_composition.mu_s;
        const auto & spin_qn       = geometry.cell_composition.spin_qn;

        std::ostringstream oss;
        oss << '\n' << "i   type   c   mu_s   spin_qn\n";
        for( std::size_t i = 0; i < iatom.size(); ++i )
            oss << fmt::format(
                "{}   {}   {}   {}   {}\n", iatom[i], atom_type[i], concentration[i], mu_s[i], spin_qn[i] );
        tbl.insert( "atom_types", oss.str() );
    }

    if( !geometry.defects.sites.empty() )
    {
        const auto & sites = geometry.defects.sites;
        const auto & types = geometry.defects.types;

        std::ostringstream oss;
        oss << '\n' << "i   da  db  dc   type";
        for( std::size_t i = 0; i < sites.size(); ++i )
        {
            const auto & t = sites[i].translations;
            oss << fmt::format( "{}   {}  {}  {}   {}\n", sites[i].i, t[0], t[1], t[2], types[i] );
        }
        tbl.insert( "defects", oss.str() );
    }

#ifdef SPIRIT_ENABLE_PINNING
    {
        const auto & pinning = geometry.pinning;
        if( pinning.na_left > 0 || pinning.na_right > 0 || pinning.nb_left > 0 || pinning.nb_right > 0
            || pinning.nc_left > 0 || pinning.nc_right > 0 )
        {
            tbl.insert(
                "pinning_boundary", toml::array{
                                        toml::array{ pinning.na_left, pinning.na_right },
                                        toml::array{ pinning.nb_left, pinning.nb_right },
                                        toml::array{ pinning.nc_left, pinning.nc_right },
                                    } );
            tbl.insert( "pinning_cell", toml_array_from_container( pinning.pinned_cell ) );
        }

        const auto & sites = pinning.sites;
        const auto & spins = pinning.spins;

        std::ostringstream oss;
        if( !pinning.sites.empty() )
        {
            oss << '\n' << "i  da db dc  x y z";
            for( unsigned int i = 0; i < pinning.sites.size(); ++i )
            {
                const auto & t = sites[i].translations;
                oss << fmt::format(
                    "{}  {} {} {}  {} {} {}", sites[i].i, t[0], t[1], t[2], spins[i][0], spins[i][1], spins[i][2] );
            }
            tbl.insert( "pinned", oss.str() );
        }
    }
#endif
    return tbl;
}

} // namespace IO
