#pragma once
#ifndef SPIRIT_CORE_ENGINE_MANIFOLDMATH_HPP
#define SPIRIT_CORE_ENGINE_MANIFOLDMATH_HPP

#include <memory>
#include <vector>

#include <Eigen/Core>

#include <engine/Vectormath_Defines.hpp>

namespace Engine
{

// Basically all math inside this namespace assumes that the given
//      vectorfields contain unit vectors. They are therefore interpreted
//      as living in a manifold which is the direct product space of N
//      spheres.
// The only exception are functions which interpret the vectorfield
//      as a single 3N-dimensional vector.
// TODO: cleanly separate these two cases!
namespace Manifoldmath
{
// Get the norm of a vectorfield (interpreted as a 3N-vector)
scalar norm( const vectorfield & vf );
// Normalize a vectorfield (interpreted as a 3N-vector)
void normalize( vectorfield & vf );

// Project v1 to be parallel to v2
//      Note: this assumes normalized vectorfields
void project_parallel( vectorfield & vf1, const vectorfield & vf2 );
// Project v1 to be orthogonal to v2
//      Note: this assumes normalized vectorfields
void project_orthogonal( vectorfield & vf1, const vectorfield & vf2 );
// Invert v1's component parallel to v2
//      Note: this assumes normalized vectorfields
void invert_parallel( vectorfield & vf1, const vectorfield & vf2 );
// Invert v1's component orthogonal to v2
//      Note: this assumes normalized vectorfields
void invert_orthogonal( vectorfield & vf1, const vectorfield & vf2 );
// Project vf1's vectors into the tangent plane of vf2
//      Note: vf2 must have normalized vectors
void project_tangential( vectorfield & vf1, const vectorfield & vf2 );

// The tangential projector is a matrix which projects any vector into the tangent
//      space of a vectorfield, considered to live on the direct product of N unit
//      spheres. It is a 3N x 3N matrix.
MatrixX tangential_projector( const vectorfield & image );

// Calculate a matrix of orthonormal basis vectors that span the tangent space to
//      a vectorfield, considered to live on the direct product of N unit spheres.
//      The basis vectors will be the spherical unit vectors, except at the poles.
//      The basis will be a 3Nx2N matrix.
void tangent_basis_spherical( const vectorfield & vf, MatrixX & basis );

void sparse_tangent_basis_spherical( const vectorfield & vf, SpMatrixX & basis );

// Calculate a matrix of orthonormal basis vectors that span the tangent space to
//      a vectorfield, considered to live on the direct product of N unit spheres.
//      The basis vectors will be generated from cross products with euclidean basis
//      vectors. The basis will be a 3Nx2N matrix.
void tangent_basis_cross( const vectorfield & vf, MatrixX & basis );

// Calculate a matrix of orthonormal basis vectors that span the tangent space to
//      a vectorfield, considered to live on the direct product of N unit spheres.
//      The basis vectors will form righthanded sets with the vectors of vf.
//      The basis will be a 3Nx2N matrix.
void tangent_basis_righthanded( const vectorfield & vf, MatrixX & basis );

// Calculate a matrix of orthonormal basis vectors that span the tangent space to
//      a vectorfield, considered to live on the direct product of N unit spheres.
//      The basis vectors will be calculated from orthonormalizations with respect
//      to a random vector. The basis will be a 3Nx2N matrix.
// void tangent_basis_random(const vectorfield & vf, MatrixX & basis);

// This is the derivatives of the coordinate transformation from (unit-)spherical to cartesian
// i.e. d_cartesian/d_spherical
void spherical_to_cartesian_jacobian( const vectorfield & vf, MatrixX & jacobian );

// This is the second derivatives of the coordinate transformation from (unit-)spherical to cartesian
// i.e. d^2_cartesian/d^2_spherical
void spherical_to_cartesian_hessian( const vectorfield & vf, MatrixX & gamma_x, MatrixX & gamma_y, MatrixX & gamma_z );

//
void spherical_to_cartesian_coordinate_basis( const vectorfield & vf, MatrixX & basis );

//
void spherical_coordinate_christoffel_symbols( const vectorfield & vf, MatrixX & gamma_theta, MatrixX & gamma_phi );

// Calculate Hessian for a vectorfield constrained to unit length, at any extremum (i.e. where vectors || gradient)
void hessian_bordered(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & tangent_basis,
    MatrixX & hessian_out );
// Calculate tangential derivatives and correction terms according to the projector approach
void hessian_projected(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & tangent_basis,
    MatrixX & hessian_out );
// Calculate tangential derivatives and correction terms according to the projector and Weingarten map approach
void hessian_weingarten(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & tangent_basis,
    MatrixX & hessian_out );
// Calculate spherical derivatives using coordinate transformations via jacobians
void hessian_spherical(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & hessian_out );
// Calculate spherical derivates and correction terms using jacobians and christoffel symbols
void hessian_covariant(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & hessian_out );

// Geodesic distance between two vectorfields
scalar dist_geodesic( const vectorfield & v1, const vectorfield & v2 );

// Calculate the "tangent" vectorfields pointing between a set of configurations
void Tangents(
    std::vector<std::shared_ptr<vectorfield>> configurations, const std::vector<scalar> & energies,
    std::vector<vectorfield> & tangents );

} // namespace Manifoldmath
} // namespace Engine

#endif