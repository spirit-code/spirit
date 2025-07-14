#include <io/Fileformat.hpp>
#include <io/VTK_Geometry.hpp>
#include <io/XML_File.hpp>
#include <utility/Base64.hpp>
#include <utility/Endian.hpp>
#include <utility/Enum.hpp>

#include <tinyxml2/tinyxml2.h>

namespace Enum = Utility::Enum;

namespace IO
{

namespace VTK
{

static const char * BYTE_ORDER_STR = Utility::ENDIAN == Utility::endianness::little ? "LittleEndian" : "BigEndian";

template<typename T>
struct type_str;

// clang-format off
#if (__cplusplus >= 202302L)
template<> struct type_str<std::float32_t> { static constexpr const char * value = "Float32"; };
template<> struct type_str<std::float64_t> { static constexpr const char * value = "Float64"; };
#else
template<> struct type_str<float> { static constexpr const char * value = "Float32"; };
template<> struct type_str<double> { static constexpr const char * value = "Float64"; };
#endif
template<> struct type_str<std::int8_t> { static constexpr const char * value = "Int8"; };
template<> struct type_str<std::int16_t> { static constexpr const char * value = "Int16"; };
template<> struct type_str<std::int32_t> { static constexpr const char * value = "Int32"; };
template<> struct type_str<std::int64_t> { static constexpr const char * value = "Int64"; };
template<> struct type_str<std::uint8_t> { static constexpr const char * value = "UInt8"; };
template<> struct type_str<std::uint16_t> { static constexpr const char * value = "UInt16"; };
template<> struct type_str<std::uint32_t> { static constexpr const char * value = "UInt32"; };
template<> struct type_str<std::uint64_t> { static constexpr const char * value = "UInt64"; };
// clang-format on

template<typename T>
static constexpr const char * type_str_v = type_str<T>::value;

} // namespace VTK

namespace XML
{

namespace
{

struct AsciiEncoder
{
    static constexpr const char * format = "ascii";

    template<typename T>
    std::string operator()( const T * src, const std::size_t size ) const
    {
        static constexpr const auto ascii_fmt = []
        {
            struct FMT
            {
                const char * head;
                const char * tail;
            };

            if constexpr( std::is_same_v<std::decay_t<T>, float> )
                return FMT{ "{:.6e}", " {:.6e}" };
            if constexpr( std::is_same_v<std::decay_t<T>, double> )
                return FMT{ "{:.12e}", " {:.12e}" };
            else if constexpr( std::is_integral_v<std::decay_t<T>> )
                return FMT{ "{:d}", " {:d}" };
            else
                return FMT{ "{}", " {}" };
        }();

        std::string output{};
        if( size == 0 )
            return output;

        // first without leading space
        fmt::format_to( std::back_inserter( output ), ascii_fmt.head, src[0] );

        // following numbers with leading space
        std::for_each_n(
            src + 1, size - 1, [output = std::back_inserter( output )]( const T & value )
            { fmt::format_to( output, ascii_fmt.tail, value ); } );

        return output;
    }
};

struct BinaryEncoder
{
    static constexpr const char * format = "binary";

    template<typename T>
    std::string operator()( const T * src, const std::size_t size ) const
    {
        std::string output{};
        output.reserve( ( ( sizeof( T ) * size + 7 ) / 3 ) * 4 );

        const std::uint32_t size_header = sizeof( T ) * size;
        Utility::b64::encode( &size_header, 1, std::back_inserter( output ) );

        Utility::b64::encode( src, size, std::back_inserter( output ) );

        return output;
    }
};

class DataElement
{
public:
    explicit DataElement( tinyxml2::XMLNode & parent ) noexcept : m_parent( &parent ) {}

    template<typename T, typename Encoder = AsciiEncoder>
    void append(
        const T * data, std::size_t size, const std::size_t components = 1, const char * name = nullptr,
        Encoder & encoder = Encoder{} )
    {
        auto * node = document()->NewElement( "DataArray" );
        node->SetAttribute( "type", VTK::type_str_v<T> );

        if( name != nullptr )
            node->SetAttribute( "Name", name );

        if( components > 1 )
            node->SetAttribute( "NumberOfComponents", fmt::format( "{:d}", components ).c_str() );

        node->SetAttribute( "format", std::decay_t<Encoder>::format );
        node->SetText( std::invoke( encoder, data, size ).c_str() );
        parent()->InsertEndChild( node );
    }

    template<typename T, typename Encoder = AsciiEncoder>
    void append_vectorfield( const field<T> & vf, const char * name = nullptr, Encoder & encoder = Encoder{} )
    {
        append( vf.front().data(), vf.front().size() * vf.size(), vf.front().size(), name, encoder );
    }

    tinyxml2::XMLNode * get()
    {
        return m_parent;
    }

private:
    tinyxml2::XMLNode * parent()
    {
        return m_parent;
    }

    tinyxml2::XMLDocument * document()
    {
        return m_parent->GetDocument();
    }

    tinyxml2::XMLNode * m_parent;
};

template<typename Encoder = AsciiEncoder>
void write_fields_impl(
    const std::string & outfile, const VTK::UnstructuredGrid & geometry,
    const std::vector<VTK::FieldDescriptor> & fields, Encoder encoder )
{
    static constexpr const char * grid_type = "UnstructuredGrid";
    static constexpr const char * version   = "1.0";

    tinyxml2::XMLDocument doc{};

    auto * root = doc.NewElement( "VTKFile" );
    root->SetAttribute( "type", grid_type );
    root->SetAttribute( "version", version );
    root->SetAttribute( "byte_order", VTK::BYTE_ORDER_STR );
    doc.InsertEndChild( root );

    auto * grid = doc.NewElement( grid_type );
    grid->SetAttribute( "NumberOfPoints", fmt::format( "{}", geometry.positions().size() ).c_str() );
    grid->SetAttribute( "NumberOfCells", fmt::format( "{}", geometry.cell_size() ).c_str() );
    root->InsertEndChild( grid );

    auto points = DataElement( *doc.NewElement( "Points" ) );
    points.append_vectorfield( geometry.positions(), nullptr, encoder );
    grid->InsertEndChild( points.get() );

    auto cells = DataElement( *doc.NewElement( "Cells" ) );
    cells.append( geometry.connectivity().data(), geometry.connectivity().size(), 1, "connectivity", encoder );
    cells.append( geometry.offsets().data() + 1, geometry.offsets().size() - 1, 1, "offsets", encoder );
    cells.append( geometry.types().data(), geometry.types().size(), 1, "types", encoder );
    grid->InsertEndChild( cells.get() );

    if( !fields.empty() )
    {
        auto node = DataElement( *doc.NewElement( "PointData" ) );
        for( const auto & [name, data] : fields )
        {
            if( data == nullptr )
                continue;

            if( data->size() != geometry.positions().size() )
                continue;

            node.append_vectorfield( *data, name.data(), encoder );
        }
        grid->InsertEndChild( node.get() );
    }

    doc.SaveFile( outfile.c_str(), /*compact=*/false );
}

} // namespace

void write_fields(
    const std::string & outfile, const VTK::UnstructuredGrid & geometry, const VF_FileFormat format,
    const std::vector<VTK::FieldDescriptor> & fields )
{
    switch( format )
    {
        case VF_FileFormat::VTK_XML_BIN: return write_fields_impl( outfile, geometry, fields, BinaryEncoder{} );
        case VF_FileFormat::VTK_XML_TEXT: return write_fields_impl( outfile, geometry, fields, AsciiEncoder{} );
        default:
            spirit_throw(
                Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Error,
                fmt::format( "Invalid format {} passed to XML::write_fields()", Enum::name( format ) ) );
    }
};

} // namespace XML

} // namespace IO
