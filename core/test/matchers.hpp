#include <Spirit/Spirit_Defines.h>
#include <fmt/format.h>
#include <map>

#include "catch.hpp"

namespace Catch
{

template<typename KeyType>
struct MapApproxMatcher : Catch::MatcherBase<std::map<KeyType, scalar>>
{
    constexpr MapApproxMatcher( const std::map<KeyType, scalar> & expected, scalar threshold )
            : expected( expected ), threshold( threshold )
    {
    }

    bool match( const std::map<KeyType, scalar> & actual ) const override
    {
        if( actual.size() != expected.size() )
        {
            return false;
        }

        auto predicate = [this]( const auto & pair )
        {
            const auto it = expected.find( pair.first );
            return !( it == expected.end() || std::abs( pair.second - it->second ) > threshold );
        };

        return std::all_of( cbegin( actual ), cend( actual ), predicate );
    }

    std::string describe() const override
    {
        return fmt::format( "is approximately equal to the expected map within {}\nExpected: {}", threshold, expected );
    }

private:
    const std::map<KeyType, scalar> & expected;
    scalar threshold;
};

namespace CustomMatchers
{
// Factory function for creating the matcher
template<typename KeyType>
MapApproxMatcher<KeyType> MapApprox( const std::map<KeyType, scalar> & expected, scalar threshold )
{
    return MapApproxMatcher<KeyType>( expected, threshold );
};

template<typename T>
auto within_digits( T value, int decimals_required_equal )
{
    double using_decimals = decimals_required_equal - int( std::ceil( std::log10( std::abs( value ) ) ) );
    INFO(
        "Requested " << decimals_required_equal << " decimals, meaning " << using_decimals
                     << " decimals after the floating point" );
    return Catch::Matchers::WithinAbs( value, std::pow( 10, -using_decimals ) );
}

} // namespace CustomMatchers

} // namespace Catch
