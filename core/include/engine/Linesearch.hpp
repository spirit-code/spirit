#include <fmt/format.h>
#include <Eigen/Geometry>
#include <engine/Backend_par.hpp>
#include <engine/Hamiltonian.hpp>
#include <utility/Exception.hpp>
#include <vector>

#include <iomanip>

namespace Engine
{

class Linesearch
{
    // Abstract Interface
public:
    virtual void
    run( std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<vectorfield> & directions )
        = 0;

    virtual void propagate( vectorfield & v, const vectorfield & directions, scalar alpha = 1.0 ) = 0;
};

class RotationLinesearch : public Linesearch
{
protected:
    vectorfield axes_temp;

    // Propagation via rotations
public:
    RotationLinesearch( int nos ) : axes_temp( vectorfield( nos ) ) {}

    void rotate( vectorfield & v, const vectorfield & axes, scalar alpha = 1.0 )
    {
        Backend::par::apply( v.size(), [v = v.data(), a = axes.data(), alpha] SPIRIT_LAMBDA( int idx ) {
            const Vector3 axis = a[idx];
            const scalar angle = alpha * axis.norm();

            if( axis.norm() > std::numeric_limits<scalar>::epsilon() )
            {
                const scalar angle = alpha * axis.norm();
                const Matrix3 rotation_matrix
                    = Eigen::AngleAxis<scalar>( angle, axis.normalized() ).toRotationMatrix().eval();
                const Vector3 vt = rotation_matrix * v[idx];
                v[idx]           = vt;
            }
        } );
    }

    void compute_axes( const vectorfield & v, const vectorfield & directions, vectorfield & axes )
    {
        if( v.size() != directions.size() )
            spirit_throw(
                Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Severe,
                "Error in Linsearch: configurations and directions have unequal lengths" );

        if( axes.size() != v.size() )
            axes.resize( v.size() );

        Backend::par::apply( v.size(), [a = axes.data(), v = v.data(), d = directions.data()] SPIRIT_LAMBDA( int idx ) {
            if( std::abs( v[idx].dot( d[idx].normalized() ) )
                >= v[idx].norm()
                       * ( 1 - std::numeric_limits<scalar>::epsilon() ) ) // Angle can become nan for collinear spins
                a[idx] = Vector3::Zero();
            else
                a[idx] = v[idx].normalized().cross( d[idx] );
        } );
    }

    virtual void propagate( vectorfield & v, const vectorfield & directions, scalar alpha = 1.0 )
    {
        // Compute temporary axes
        compute_axes( v, directions, axes_temp );
        // Rotate
        rotate( v, axes_temp, alpha );
    }
};

class Trivial_Linesearch : public RotationLinesearch
{
    // Propagates with step length 1
public:
    Trivial_Linesearch( int nos ) : RotationLinesearch( nos ) {}

    virtual void
    run( std::vector<std::shared_ptr<vectorfield>> & configurations,
         const std::vector<vectorfield> & directions ) override
    {
        for( int img = 0; img < configurations.size(); img++ )
        {
            propagate( *configurations[img], directions[img], 1.0 );
        }
    }
};

class Quadratic_Backtracking_Linesearch : public RotationLinesearch
{
private:
    vectorfield gradient_throwaway;

    scalarfield energy_start;
    scalarfield energy_end;
    scalar tolerance = 0.01;

    scalarfield a;
    scalarfield b;

    int noi;

    std::shared_ptr<Hamiltonian> hamiltonian;

    void determine_parabola_coefficients(
        std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<vectorfield> & directions )
    {
        // We expect the energy to fit a parabola:
        // E(x) = a * x^2 + b * x + energy_start
        // We determine the parabola from E(0), E'(0) and E'(1) where the search interval is [0,1]

        // Determine a, b and energy_start
        for( int img = 0; img < noi; img++ )
        {
            // Reset energy and gradient to 0
            energy_start[img] = 0;
            Vectormath::fill( gradient_throwaway, Vector3::Zero() );

            hamiltonian->Gradient_and_Energy( *configurations[img], gradient_throwaway, energy_start[img] );
            Manifoldmath::project_tangential( gradient_throwaway, *configurations[img] );
            b[img] = Vectormath::dot(
                directions[img], gradient_throwaway ); // b is the projection of the gradient along the tangent

            // Now do the computation at the end of the interval
            // Reset energy and gradient
            energy_end[img] = 0;
            Vectormath::fill( gradient_throwaway, Vector3::Zero() );

            // Compute axes of rotation
            compute_axes( *configurations[img], directions[img], axes_temp );
            // Use axes to propagate the configuration
            rotate( *configurations[img], axes_temp, 1.0 );
            hamiltonian->Gradient_and_Energy( *configurations[img], gradient_throwaway, energy_end[img] );
            Manifoldmath::project_tangential( gradient_throwaway, *configurations[img] );

            // To make comparisons between gradient and direction meaningful, we rotate the gradient back into the
            // tangent frame of the start and then project to its tangent plane
            rotate( gradient_throwaway, axes_temp, -1.0 );
            a[img] = 0.5 * ( Vectormath::dot( directions[img], gradient_throwaway ) - b[img] );
        }
    }

    // Check the energy predictions assuming the current energies are stored in energy_end and we are at step length alpha
    bool check_energy( scalar alpha, int img )
    {
        // Compute expected energy difference from parabola coefficients
        const scalar delta_e_expected = b[img] * alpha + a[img] * alpha * alpha;

        // Compute error of expected energy
        const scalar error_delta_e_expected
            = std::numeric_limits<scalar>::epsilon()
              * std::sqrt( std::pow( b[img] * alpha, 2 ) + std::pow( a[img] * alpha * alpha, 2 ) );

        // Compute the real energy difference
        const scalar delta_e = energy_end[img] - energy_start[img];
        const scalar error_delta_e
            = std::numeric_limits<scalar>::epsilon()
              * std::sqrt( std::pow( 1.0 + energy_end[img], 2 ) + std::pow( 1.0 + energy_start[img], 2 ) );

        // Error on the ratio between delta_e and delta_e_expected
        const scalar error_ratio
            = std::abs( delta_e / delta_e_expected + std::numeric_limits<scalar>::epsilon() )
              * std::sqrt(
                  std::pow( error_delta_e / delta_e, 2 ) + std::pow( error_delta_e_expected / delta_e_expected, 2 ) );

        const bool linesearch_applicable = error_ratio < tolerance * 1e-1;
        const scalar criterion           = std::abs( std::abs( delta_e_expected / delta_e ) - 1 );

        // fmt::print( "======\n" );
        // // fmt::print( "iter                        {}\n", iter );
        // // fmt::print( "ratio                       {}\n", ratio );
        // fmt::print( "alpha ls                    {:.15f}\n", alpha );
        // fmt::print( "energy_end                  {:.15f}\n", energy_end[img] );
        // fmt::print( "energy_start                {:.15f}\n", energy_start[img] );
        // fmt::print( "b                           {:.15f}\n", b[img] );
        // fmt::print( "a                           {:.15f}\n", a[img] );
        // fmt::print( "delta_e ls                  {:.15f} +- {:.15f}\n", delta_e, error_delta_e );
        // fmt::print(
        //     "delta_e_expected ls         {:.15f} +- {:.15f}\n", delta_e_expected, error_delta_e_expected );
        // fmt::print( "ratio {:.15f} +- {:.15f}\n", delta_e_expected / delta_e, error_ratio );
        // fmt::print( "criterion {}\n", std::abs( std::abs( delta_e_expected / delta_e ) - 1 ) );
        // fmt::print( "applicable {}\n", linesearch_applicable );

        if( !linesearch_applicable )
            return true;

        return criterion < tolerance;
    }

public:
    int max_iter = 50;

    Quadratic_Backtracking_Linesearch( int noi, int nos, std::shared_ptr<Hamiltonian> & hamiltonian )
            : RotationLinesearch( nos ),
              hamiltonian( hamiltonian ),
              noi( noi ),
              a( scalarfield( noi ) ),
              b( scalarfield( noi ) ),
              energy_start( scalarfield( noi ) ),
              energy_end( scalarfield( noi ) ),
              gradient_throwaway( vectorfield( nos ) )
    {
    }

    virtual void
    run( std::vector<std::shared_ptr<vectorfield>> & configurations,
         const std::vector<vectorfield> & directions ) override
    {
        determine_parabola_coefficients( configurations, directions );

        // Note: Configurations are at end of search interval now

        // For every image check if the prediction of the energy and the actual energy is within tolerance
        for( int img = 0; img < noi; img++ )
        {
            scalar alpha        = 1;
            scalar tau          = 0.5;
            int iter            = 0;
            bool stop_criterion = false;

            while( !stop_criterion )
            {
                iter++;
                const bool check = check_energy( alpha, img );

                if( check || iter >= max_iter )
                {
                    stop_criterion = true;
                }
                else
                {
                    alpha /= 2.0;
                    // propagate( *configurations[img], directions[img], -alpha ); //Note: negative sign is for backtracking
                    rotate( *configurations[img], axes_temp, -alpha );

                    Vectormath::fill( gradient_throwaway, Vector3::Zero() );
                    energy_end[img] = 0;
                    hamiltonian->Gradient_and_Energy( *configurations[img], gradient_throwaway, energy_end[img] );
                    Manifoldmath::project_tangential( gradient_throwaway, *configurations[img] );
                }
            }
        }
    }
};

} // namespace Engine