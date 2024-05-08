#pragma once

#include <io/Filter_File_Handle.hpp>
#include <utility/Logging.hpp>

#include <map>
#include <string_view>
#include <tuple>

namespace IO
{

// default factory for forwarding the parsed data
template<typename In>
static constexpr auto forwarding_factory
    = []( const std::map<std::string_view, int> & ) { return []( In row ) -> In { return row; }; };

/*
 * Universal implementation of a table parser.
 * Initialized using the data types and header labels for each column. The `parse` method accepts a factory function
 * that takes a map of column labels to their respective index in the read buffer (implemented as a tuple) and returns a
 * function that transforms the data from the read buffer into a more readily usable representation of each row.
 */
template<typename... Args>
class TableParser
{
public:
    static constexpr std::size_t n_columns = sizeof...( Args );

    using read_row_t = std::tuple<Args...>;
    using labels_t   = std::array<std::string_view, n_columns>;

    constexpr TableParser( labels_t && labels ) : labels( std::forward<labels_t>( labels ) ){};
    constexpr TableParser( const labels_t & labels ) : labels( labels ){};

    template<typename... InitArgs, typename = std::enable_if_t<std::is_constructible_v<labels_t, InitArgs &&...>>>
    constexpr explicit TableParser( InitArgs &&... labels ) : labels( std::forward<InitArgs>( labels )... )
    {
        static_assert(
            sizeof...( InitArgs ) == n_columns, "The number of labels has to match the number of specified types" );
    };

private:
    // mapping: index in data tuple -> column name
    labels_t labels;

    // constexpr way to iterate over all indices and read the value into the right position using the right type
    template<std::size_t Index = 0>
    static void readTupleElements( IO::Filter_File_Handle & file_handle, read_row_t & container, std::size_t index )
    {
        if( index >= n_columns )
        {
            std::string sdump{};
            file_handle >> sdump;
        }
        else if constexpr( Index < n_columns )
        {
            if( Index == index )
            {
                // templated `std::get` in order to have access to the type at that position
                file_handle >> std::get<Index>( container );
            }
            // Continue with the next element
            readTupleElements<Index + 1>( file_handle, container, index );
        }
    };

public:
    // parse function, transform_factory expects a second oder function whose result can transform the read in row into
    // a format that should be stored
    template<typename F>
    [[nodiscard]] decltype( auto ) parse(
        const std::string & table_file, const std::string & table_size_id, const std::size_t n_columns_read,
        F transform_factory = forwarding_factory<read_row_t> ) const
    {
        using Utility::Log_Level, Utility::Log_Sender;

        std::vector<std::string> columns( n_columns_read, "" );   // data storage for parsed row
        std::vector<std::size_t> tuple_idx( n_columns_read, -1 ); // mapping: index in file -> index in data tuple
        std::array<int, n_columns> column_idx{};                  // mapping: index in data tuple -> index in file
        std::fill( begin( column_idx ), end( column_idx ), -1 );  // initialize with sentinal value
        int table_size{ 0 };                                      // table size for single config file setup

        IO::Filter_File_Handle file_handle( table_file );

        if( file_handle.Find( table_size_id ) )
        {
            // Read n interaction pairs
            file_handle >> table_size;
            Log( Log_Level::Debug, Log_Sender::IO,
                 fmt::format( "Table file {} should have {} rows", table_file, table_size ) );
            // if we know the expected size, we can reserve the necessary space
        }
        else
        {
            // Read the whole file
            table_size = (int)1e8;
            // First line should contain the columns
            file_handle.To_Start();
            Log( Log_Level::Debug, Log_Sender::IO, "Trying to parse columns from top of file " + table_file );
        }

        file_handle.GetLine();
        for( std::size_t i = 0; i < columns.size(); ++i )
        {
            // read in lower case
            file_handle >> columns[i];
            std::transform( begin( columns[i] ), end( columns[i] ), begin( columns[i] ), ::tolower );

            // find tuple position and update mappings
            const auto it = find( begin( labels ), end( labels ), columns[i] );
            if( it != end( labels ) )
            {
                const std::size_t pos = std::distance( begin( labels ), it );
                column_idx[pos]       = i;
                tuple_idx[i]          = pos;
            }
            else if( !columns[i].empty() )
            {
                Log( Log_Level::Warning, Log_Sender::IO,
                     fmt::format(
                         "Unknown column \"{}\" in header of table file \"{}\" below \"{}\"", columns[i], table_file,
                         table_size_id ) );
            }
        }

        // make the transform function based on the columns that are present.
        std::map<std::string_view, int> lookup;
        for( std::size_t i = 0; i < labels.size(); ++i )
            lookup.emplace( std::make_pair( labels[i], column_idx[i] ) );
        auto transform = transform_factory( lookup );

        // read data row by row and store a transformed version of it
        auto row = std::make_tuple( Args{}... );                           // read buffer
        std::vector<std::decay_t<decltype( transform( row ) )>> data( 0 ); // return value
        for( int row_idx = 0; row_idx < table_size; ++row_idx )
        {
            // read a new line and stop reading if EOF
            if( !file_handle.GetLine() )
                break;

            // Read a line from the File
            for( std::size_t i = 0; i < columns.size(); ++i )
            {

                readTupleElements( file_handle, row, tuple_idx[i] );
            }

            data.emplace_back( transform( row ) );
        }

        return data;
    };
};

} // namespace IO
