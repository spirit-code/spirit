#include <engine/Manifoldmath.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Dense>

#include <GenEigsRealShiftSolver.h>
#include <GenEigsSolver.h> // Also includes <MatOp/DenseGenMatProd.h>

#include <array>

namespace C = Utility::Constants;

#ifndef SPIRIT_USE_CUDA

namespace Engine
{
namespace Manifoldmath
{
void project_parallel( vectorfield & vf1, const vectorfield & vf2 )
{
    vectorfield vf3 = vf1;
    project_orthogonal( vf3, vf2 );
// TODO: replace the loop with Vectormath Kernel
#pragma omp parallel for
    for( unsigned int i = 0; i < vf1.size(); ++i )
        vf1[i] -= vf3[i];
}

void project_orthogonal( vectorfield & vf1, const vectorfield & vf2 )
{
    scalar x = Vectormath::dot( vf1, vf2 );
// TODO: replace the loop with Vectormath Kernel
#pragma omp parallel for
    for( unsigned int i = 0; i < vf1.size(); ++i )
        vf1[i] -= x * vf2[i];
}

void invert_parallel( vectorfield & vf1, const vectorfield & vf2 )
{
    scalar x = Vectormath::dot( vf1, vf2 );
// TODO: replace the loop with Vectormath Kernel
#pragma omp parallel for
    for( unsigned int i = 0; i < vf1.size(); ++i )
        vf1[i] -= 2 * x * vf2[i];
}

void invert_orthogonal( vectorfield & vf1, const vectorfield & vf2 )
{
    vectorfield vf3 = vf1;
    project_orthogonal( vf3, vf2 );
// TODO: replace the loop with Vectormath Kernel
#pragma omp parallel for
    for( unsigned int i = 0; i < vf1.size(); ++i )
        vf1[i] -= 2 * vf3[i];
}

void project_tangential( vectorfield & vf1, const vectorfield & vf2 )
{
#pragma omp parallel for
    for( unsigned int i = 0; i < vf1.size(); ++i )
        vf1[i] -= vf1[i].dot( vf2[i] ) * vf2[i];
}

scalar dist_geodesic( const vectorfield & v1, const vectorfield & v2 )
{
    scalar dist = 0;
#pragma omp parallel for reduction( + : dist )
    for( unsigned int i = 0; i < v1.size(); ++i )
        dist += pow( Vectormath::angle( v1[i], v2[i] ), 2 );
    return sqrt( dist );
}

/*
Calculates the 'tangent' vectors, i.e.in crudest approximation the difference between an image and the neighbouring
*/
void Tangents(
    std::vector<std::shared_ptr<vectorfield>> configurations, const std::vector<scalar> & energies,
    std::vector<vectorfield> & tangents )
{
    int noi = configurations.size();
    int nos = ( *configurations[0] ).size();

    for( int idx_img = 0; idx_img < noi; ++idx_img )
    {
        auto & image = *configurations[idx_img];

        // First Image
        if( idx_img == 0 )
        {
            auto & image_plus = *configurations[idx_img + 1];
            Vectormath::set_c_a( 1, image_plus, tangents[idx_img] );
            Vectormath::add_c_a( -1, image, tangents[idx_img] );
        }
        // Last Image
        else if( idx_img == noi - 1 )
        {
            auto & image_minus = *configurations[idx_img - 1];
            Vectormath::set_c_a( 1, image, tangents[idx_img] );
            Vectormath::add_c_a( -1, image_minus, tangents[idx_img] );
        }
        // Images Inbetween
        else
        {
            auto & image_plus  = *configurations[idx_img + 1];
            auto & image_minus = *configurations[idx_img - 1];

            // Energies
            scalar E_mid = 0, E_plus = 0, E_minus = 0;
            E_mid   = energies[idx_img];
            E_plus  = energies[idx_img + 1];
            E_minus = energies[idx_img - 1];

            // Vectors to neighbouring images
            vectorfield t_plus( nos ), t_minus( nos );

            Vectormath::set_c_a( 1, image_plus, t_plus );
            Vectormath::add_c_a( -1, image, t_plus );

            Vectormath::set_c_a( 1, image, t_minus );
            Vectormath::add_c_a( -1, image_minus, t_minus );

            // Near maximum or minimum
            if( ( E_plus < E_mid && E_mid > E_minus ) || ( E_plus > E_mid && E_mid < E_minus ) )
            {
                // Get a smooth transition between forward and backward tangent
                scalar E_max = std::max( std::abs( E_plus - E_mid ), std::abs( E_minus - E_mid ) );
                scalar E_min = std::min( std::abs( E_plus - E_mid ), std::abs( E_minus - E_mid ) );

                if( E_plus > E_minus )
                {
                    Vectormath::set_c_a( E_max, t_plus, tangents[idx_img] );
                    Vectormath::add_c_a( E_min, t_minus, tangents[idx_img] );
                }
                else
                {
                    Vectormath::set_c_a( E_min, t_plus, tangents[idx_img] );
                    Vectormath::add_c_a( E_max, t_minus, tangents[idx_img] );
                }
            }
            // Rising slope
            else if( E_plus > E_mid && E_mid > E_minus )
            {
                Vectormath::set_c_a( 1, t_plus, tangents[idx_img] );
            }
            // Falling slope
            else if( E_plus < E_mid && E_mid < E_minus )
            {
                Vectormath::set_c_a( 1, t_minus, tangents[idx_img] );
                // tangents = t_minus;
                for( int i = 0; i < nos; ++i )
                {
                    tangents[idx_img][i] = t_minus[i];
                }
            }
            // No slope(constant energy)
            else
            {
                Vectormath::set_c_a( 1, t_plus, tangents[idx_img] );
                Vectormath::add_c_a( 1, t_minus, tangents[idx_img] );
            }
        }

        // Project tangents into tangent planes of spin vectors to make them actual tangents
        project_tangential( tangents[idx_img], image );

        // Normalise in 3N - dimensional space
        Manifoldmath::normalize( tangents[idx_img] );

    } // end for idx_img
} // end Tangents
} // namespace Manifoldmath
} // namespace Engine

#endif

namespace Engine
{
namespace Manifoldmath
{
scalar norm( const vectorfield & vf )
{
    scalar x = Vectormath::dot( vf, vf );
    return std::sqrt( x );
}

void normalize( vectorfield & vf )
{
    scalar sc = 1.0 / norm( vf );
    Vectormath::scale( vf, sc );
}

MatrixX tangential_projector( const vectorfield & image )
{
    int nos  = image.size();
    int size = 3 * nos;

    // Get projection matrix M=1-S, blockwise S=x*x^T
    MatrixX proj = MatrixX::Identity( size, size );
    for( int i = 0; i < nos; ++i )
    {
        proj.block<3, 3>( 3 * i, 3 * i ) -= image[i] * image[i].transpose();
    }

    return proj;
}

// This gives an orthogonal matrix of shape (3N, 2N), meaning M^T=M^-1 or M^T*M=1.
// This assumes that the vectors of vf are normalized and that basis is 3N x 2N
// It can be used to transform a vector into or back from the tangent space of a
//      sphere w.r.t. euclidean 3N space.
// It is generated by column-wise normalization of the Jacobi matrix for the
//      transformation from (unit-)spherical coordinates to euclidean.
// It therefore consists of the local basis vectors of the spherical coordinates
//      of a unit sphere, represented in 3N, as the two columns of the matrix.
void tangent_basis_spherical( const vectorfield & vf, MatrixX & basis )
{
    Vector3 tmp, etheta, ephi;
    basis.setZero();
    for( unsigned int i = 0; i < vf.size(); ++i )
    {
        if( vf[i][2] > 1 - 1e-8 )
        {
            tmp                                   = Vector3{ 1, 0, 0 };
            basis.block<3, 1>( 3 * i, 2 * i )     = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
            tmp                                   = Vector3{ 0, 1, 0 };
            basis.block<3, 1>( 3 * i, 2 * i + 1 ) = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
        }
        else if( vf[i][2] < -1 + 1e-8 )
        {
            tmp                                   = Vector3{ 1, 0, 0 };
            basis.block<3, 1>( 3 * i, 2 * i )     = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
            tmp                                   = Vector3{ 0, -1, 0 };
            basis.block<3, 1>( 3 * i, 2 * i + 1 ) = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
        }
        else
        {
            scalar rxy   = std::sqrt( 1 - vf[i][2] * vf[i][2] );
            scalar z_rxy = vf[i][2] / rxy;

            // Note: these are not unit vectors, but derivatives!
            etheta = Vector3{ vf[i][0] * z_rxy, vf[i][1] * z_rxy, -rxy };
            ephi   = Vector3{ -vf[i][1] / rxy, vf[i][0] / rxy, 0 };

            basis.block<3, 1>( 3 * i, 2 * i )     = ( etheta - etheta.dot( vf[i] ) * vf[i] ).normalized();
            basis.block<3, 1>( 3 * i, 2 * i + 1 ) = ( ephi - ephi.dot( vf[i] ) * vf[i] ).normalized();
        }
    }
}

void sparse_tangent_basis_spherical( const vectorfield & vf, SpMatrixX & basis )
{
    typedef Eigen::Triplet<scalar> T;
    std::vector<T> triplet_list;
    triplet_list.reserve( vf.size() * 3 );

    Vector3 tmp, etheta, ephi, res;
    for( unsigned int i = 0; i < vf.size(); ++i )
    {
        if( vf[i][2] > 1 - 1e-8 )
        {
            tmp = Vector3{ 1, 0, 0 };
            res = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();

            triplet_list.push_back( T( 3 * i, 2 * i, res[0] ) );
            triplet_list.push_back( T( 3 * i + 1, 2 * i, res[1] ) );
            triplet_list.push_back( T( 3 * i + 2, 2 * i, res[2] ) );

            tmp = Vector3{ 0, 1, 0 };
            res = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
            triplet_list.push_back( T( 3 * i, 2 * i + 1, res[0] ) );
            triplet_list.push_back( T( 3 * i + 1, 2 * i + 1, res[1] ) );
            triplet_list.push_back( T( 3 * i + 2, 2 * i + 1, res[2] ) );
        }
        else if( vf[i][2] < -1 + 1e-8 )
        {
            tmp = Vector3{ 1, 0, 0 };
            res = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
            triplet_list.push_back( T( 3 * i, 2 * i, res[0] ) );
            triplet_list.push_back( T( 3 * i + 1, 2 * i, res[1] ) );
            triplet_list.push_back( T( 3 * i + 2, 2 * i, res[2] ) );

            tmp = Vector3{ 0, -1, 0 };
            res = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
            triplet_list.push_back( T( 3 * i, 2 * i + 1, res[0] ) );
            triplet_list.push_back( T( 3 * i + 1, 2 * i + 1, res[1] ) );
            triplet_list.push_back( T( 3 * i + 2, 2 * i + 1, res[2] ) );
        }
        else
        {
            scalar rxy   = std::sqrt( 1 - vf[i][2] * vf[i][2] );
            scalar z_rxy = vf[i][2] / rxy;

            // Note: these are not unit vectors, but derivatives!
            etheta = Vector3{ vf[i][0] * z_rxy, vf[i][1] * z_rxy, -rxy };
            ephi   = Vector3{ -vf[i][1] / rxy, vf[i][0] / rxy, 0 };

            res = ( etheta - etheta.dot( vf[i] ) * vf[i] ).normalized();
            triplet_list.push_back( T( 3 * i, 2 * i, res[0] ) );
            triplet_list.push_back( T( 3 * i + 1, 2 * i, res[1] ) );
            triplet_list.push_back( T( 3 * i + 2, 2 * i, res[2] ) );
            res = ( ephi - ephi.dot( vf[i] ) * vf[i] ).normalized();
            triplet_list.push_back( T( 3 * i, 2 * i + 1, res[0] ) );
            triplet_list.push_back( T( 3 * i + 1, 2 * i + 1, res[1] ) );
            triplet_list.push_back( T( 3 * i + 2, 2 * i + 1, res[2] ) );
        }
    }
    basis.setFromTriplets( triplet_list.begin(), triplet_list.end() );
}

// This calculates the basis via calculation of cross products
// This assumes that the vectors of vf are normalized and that basis is 3N x 2N
void tangent_basis_cross( const vectorfield & vf, MatrixX & basis )
{
    basis.setZero();
    for( int i = 0; i < vf.size(); ++i )
    {
        if( std::abs( vf[i].z() ) > 1 - 1e-8 )
        {
            basis.block<3, 1>( 3 * i, 2 * i )     = Vector3{ 0, 1, 0 }.cross( vf[i] ).normalized();
            basis.block<3, 1>( 3 * i, 2 * i + 1 ) = vf[i].cross( basis.block<3, 1>( 3 * i, 2 * i ) );
        }
        else
        {
            basis.block<3, 1>( 3 * i, 2 * i )     = Vector3{ 0, 0, 1 }.cross( vf[i] ).normalized();
            basis.block<3, 1>( 3 * i, 2 * i + 1 ) = vf[i].cross( basis.block<3, 1>( 3 * i, 2 * i ) );
        }
    }
}

// This calculates the basis via orthonormalization to a random vector
// This assumes that the vectors of vf are normalized and that basis is 3N x 2N
void tangent_basis_righthanded( const vectorfield & vf, MatrixX & basis )
{
    int size = vf.size();
    basis.setZero();

    // vf should be 3N
    // basis should be 3N x 2N

    // e1 and e2 will form a righthanded vectorset with the axis (though not orthonormal!)
    Vector3 e1, e2, v1;
    Vector3 ex{ 1, 0, 0 }, ey{ 0, 1, 0 }, ez{ 0, 0, 1 };

    for( int i = 0; i < size; ++i )
    {
        auto & axis = vf[i];

        // Choose orthogonalisation basis for Grahm-Schmidt
        //      We will need two vectors with which the axis always forms the
        //      same orientation (händigkeit des vektor-dreibeins)
        // If axis_z=0 its in the xy-plane
        //      the vectors should be: axis, ez, (axis x ez)
        if( axis[2] == 0 )
        {
            e1 = ez;
            e2 = axis.cross( ez );
        }
        // Else its either above or below the xy-plane.
        //      if its above the xy-plane, it points in z-direction
        //      the vectors should be: axis, ex, -ey
        else if( axis[2] > 0 )
        {
            e1 = ex;
            e2 = -ey;
        }
        //      if its below the xy-plane, it points in -z-direction
        //      the vectors should be: axis, ex, ey
        else if( axis[2] < 0 )
        {
            e1 = ex;
            e2 = ey;
        }

        // First vector: orthogonalize e1 w.r.t. axis
        v1                                = ( e1 - e1.dot( axis ) * axis ).normalized();
        basis.block<3, 1>( 3 * i, 2 * i ) = v1;

        // Second vector: orthogonalize e2 w.r.t. axis and v1
        basis.block<3, 1>( 3 * i, 2 * i + 1 ) = ( e2 - e2.dot( axis ) * axis - e2.dot( v1 ) * v1 ).normalized();
    }
}

// This gives the Jacobian matrix for the transformation from (unit-)spherical
// to euclidean coordinates. It consists of the derivative vectors d/d_theta
// and d/d_phi as the two columns of the matrix.
void spherical_to_cartesian_jacobian( const vectorfield & vf, MatrixX & jacobian )
{
    Vector3 tmp, etheta, ephi;
    jacobian.setZero();
    for( unsigned int i = 0; i < vf.size(); ++i )
    {
        if( vf[i][2] > 1 - 1e-8 )
        {
            tmp                                      = Vector3{ 1, 0, 0 };
            jacobian.block<3, 1>( 3 * i, 2 * i )     = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
            tmp                                      = Vector3{ 0, 1, 0 };
            jacobian.block<3, 1>( 3 * i, 2 * i + 1 ) = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
        }
        else if( vf[i][2] < -1 + 1e-8 )
        {
            tmp                                      = Vector3{ 1, 0, 0 };
            jacobian.block<3, 1>( 3 * i, 2 * i )     = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
            tmp                                      = Vector3{ 0, -1, 0 };
            jacobian.block<3, 1>( 3 * i, 2 * i + 1 ) = ( tmp - tmp.dot( vf[i] ) * vf[i] ).normalized();
        }
        else
        {
            scalar rxy   = std::sqrt( 1 - vf[i][2] * vf[i][2] );
            scalar z_rxy = vf[i][2] / rxy;

            // Note: these are not unit vectors, but derivatives!
            etheta = Vector3{ vf[i][0] * z_rxy, vf[i][1] * z_rxy, -rxy };
            ephi   = Vector3{ -vf[i][1], vf[i][0], 0 };

            jacobian.block<3, 1>( 3 * i, 2 * i )     = etheta - etheta.dot( vf[i] ) * vf[i];
            jacobian.block<3, 1>( 3 * i, 2 * i + 1 ) = ephi - ephi.dot( vf[i] ) * vf[i];
        }
    }
}

// The Hessian matrix of the transformation from spherical to euclidean coordinates
void spherical_to_cartesian_hessian( const vectorfield & vf, MatrixX & gamma_x, MatrixX & gamma_y, MatrixX & gamma_z )
{
    int nos = vf.size();
    gamma_x.setZero();
    gamma_y.setZero();
    gamma_z.setZero();

    for( unsigned int i = 0; i < nos; ++i )
    {
        scalar z_rxy = vf[i][2] / std::sqrt( 1 + 1e-6 - vf[i][2] * vf[i][2] );

        gamma_x.block<2, 2>( 2 * i, 2 * i ) << -vf[i][0], -vf[i][1] * z_rxy, -vf[i][1] * z_rxy, -vf[i][0];

        gamma_y.block<2, 2>( 2 * i, 2 * i ) << -vf[i][1], vf[i][0] * z_rxy, vf[i][0] * z_rxy, -vf[i][1];

        gamma_z.block<2, 2>( 2 * i, 2 * i ) << -vf[i][2], 0, 0, 0;
    }
}

// The (2Nx2N) Christoffel symbols of the transformation from (unit-)spherical coordinates to euclidean
void spherical_to_cartesian_christoffel_symbols( const vectorfield & vf, MatrixX & gamma_theta, MatrixX & gamma_phi )
{
    using std::acos;
    using std::atan2;
    using std::cos;
    using std::sin;
    using std::tan;

    int nos     = vf.size();
    gamma_theta = MatrixX::Zero( 2 * nos, 2 * nos );
    gamma_phi   = MatrixX::Zero( 2 * nos, 2 * nos );

    for( unsigned int i = 0; i < nos; ++i )
    {
        scalar theta = acos( vf[i][2] );
        scalar phi   = atan2( vf[i][1], vf[i][0] );
        scalar cot   = 0;
        if( std::abs( theta ) > 1e-4 )
            cot = -tan( C::Pi_2 + theta );

        gamma_theta( 2 * i + 1, 2 * i + 1 ) = -sin( theta ) * cos( theta );

        gamma_phi( 2 * i + 1, 2 * i ) = cot;
        gamma_phi( 2 * i, 2 * i + 1 ) = cot;
    }
}

void hessian_bordered(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & tangent_basis,
    MatrixX & hessian_out )
{
    // Calculates a 3Nx3N matrix in the bordered Hessian approach and transforms it into the tangent basis,
    // making the result a 2Nx2N matrix. The bordered Hessian's Lagrange multipliers assume a local extremum.

    int nos        = image.size();
    MatrixX tmp_3N = hessian;

    VectorX lambda( nos );
    for( int i = 0; i < nos; ++i )
        lambda[i] = image[i].dot( gradient[i] );

    for( int i = 0; i < nos; ++i )
    {
        for( int j = 0; j < 3; ++j )
        {
            tmp_3N( 3 * i + j, 3 * i + j ) -= lambda( i );
        }
    }

    // Calculate the basis transformation matrix
    tangent_basis = MatrixX::Zero( 3 * nos, 2 * nos );
    tangent_basis_spherical( image, tangent_basis );

    // Result is a 2Nx2N matrix
    hessian_out = tangent_basis.transpose() * tmp_3N * tangent_basis;
}

void hessian_projected(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & tangent_basis,
    MatrixX & hessian_out )
{
    // Calculates a 3Nx3N matrix in the projector approach and transforms it into the tangent basis,
    // making the result a 2Nx2N matrix

    int nos = image.size();
    hessian_out.setZero();

    // Calculate projector matrix
    auto P = tangential_projector( image );

    // Calculate tangential projection of Hessian
    hessian_out = P * hessian * P;

    // Calculate correction terms
    for( unsigned int i = 0; i < nos; ++i )
    {
        hessian_out.block<3, 3>( 3 * i, 3 * i )
            -= P.block<3, 3>( 3 * i, 3 * i ) * ( image[i].dot( gradient[i] ) )
               + ( P.block<3, 3>( 3 * i, 3 * i ) * gradient[i] ) * image[i].transpose();
    }

    // Calculate the basis transformation matrix
    tangent_basis = MatrixX::Zero( 3 * nos, 2 * nos );
    tangent_basis_spherical( image, tangent_basis );

    // Result is a 2Nx2N matrix
    hessian_out = tangent_basis.transpose() * hessian_out * tangent_basis;
}

void hessian_weingarten(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & tangent_basis,
    MatrixX & hessian_out )
{
    // Calculates a 3Nx3N matrix in the Weingarten map approach and transforms it into the tangent basis,
    // making the result a 2Nx2N matrix

    int nos = image.size();
    hessian_out.setZero();

    // Calculate projector matrix
    auto P = tangential_projector( image );

    // Calculate tangential projection of Hessian
    hessian_out = P * hessian;

    // Add the Weingarten map
    for( unsigned int i = 0; i < nos; ++i )
    {
        MatrixX proj = MatrixX::Identity( 3, 3 );
        hessian_out.block<3, 3>( 3 * i, 3 * i ) -= MatrixX::Identity( 3, 3 ) * image[i].dot( gradient[i] );
    }

    // Calculate the basis transformation matrix
    tangent_basis = MatrixX::Zero( 3 * nos, 2 * nos );
    tangent_basis_spherical( image, tangent_basis );

    // Result is a 2Nx2N matrix
    hessian_out = tangent_basis.transpose() * hessian_out * tangent_basis;
}

void hessian_spherical(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & hessian_out )
{
    // Calculates a 2Nx2N hessian matrix containing second order spherical derivatives

    int nos = image.size();

    MatrixX jacobian   = MatrixX::Zero( 3 * nos, 2 * nos );
    MatrixX sph_hess_x = MatrixX::Zero( 2 * nos, 2 * nos );
    MatrixX sph_hess_y = MatrixX::Zero( 2 * nos, 2 * nos );
    MatrixX sph_hess_z = MatrixX::Zero( 2 * nos, 2 * nos );

    // Calculate coordinate transformation jacobian
    Engine::Manifoldmath::spherical_to_cartesian_jacobian( image, jacobian );

    // Calculate coordinate transformation Hessian
    Engine::Manifoldmath::spherical_to_cartesian_hessian( image, sph_hess_x, sph_hess_y, sph_hess_z );

    // Calculate transformed Hessian
    hessian_out = jacobian.transpose() * hessian * jacobian;
    for( int i = 0; i < nos; ++i )
    {
        hessian_out.block<2, 2>( 2 * i, 2 * i ) += gradient[i][0] * sph_hess_x.block<2, 2>( 2 * i, 2 * i )
                                                   + gradient[i][1] * sph_hess_y.block<2, 2>( 2 * i, 2 * i )
                                                   + gradient[i][2] * sph_hess_z.block<2, 2>( 2 * i, 2 * i );
    }
}

void hessian_covariant(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & hessian_out )
{
    // Calculates a 2Nx2N covariant hessian matrix containing second order spherical derivatives
    // and correction terms (containing Christoffel symbols)

    int nos = image.size();

    // Calculate coordinate transformation jacobian
    MatrixX jacobian( 3 * nos, 2 * nos );
    Engine::Manifoldmath::spherical_to_cartesian_jacobian( image, jacobian );

    // Calculate the gradient in spherical coordinates
    Eigen::Ref<const VectorX> grad = Eigen::Map<const VectorX>( gradient[0].data(), 3 * nos );
    VectorX gradient_spherical     = jacobian.transpose() * grad;

    // Calculate the Hessian in spherical coordinates
    hessian_spherical( image, gradient, hessian, hessian_out );

    // Calculate the Christoffel symbols for spherical coordinates
    MatrixX christoffel_theta = MatrixX::Zero( 2 * nos, 2 * nos );
    MatrixX christoffel_phi   = MatrixX::Zero( 2 * nos, 2 * nos );
    Engine::Manifoldmath::spherical_to_cartesian_christoffel_symbols( image, christoffel_theta, christoffel_phi );

    // Calculate the covariant Hessian
    for( int i = 0; i < nos; ++i )
    {
        hessian_out.block<2, 2>( 2 * i, 2 * i )
            -= gradient_spherical[2 * i] * christoffel_theta.block<2, 2>( 2 * i, 2 * i )
               + gradient_spherical[2 * i + 1] * christoffel_phi.block<2, 2>( 2 * i, 2 * i );
    }
}

} // namespace Manifoldmath
} // namespace Engine