#include <utility/Configurations.hpp>
#include <data/Spin_System.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <Eigen/Dense>

#include <fmt/format.h>

#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using Utility::Constants::Pi;

namespace Utility
{
    namespace Configurations
    {
        void filter_to_mask(const vectorfield & spins, const vectorfield & positions, filterfunction filter, intfield & mask)
        {
            int nos = spins.size();
            mask = intfield(nos, 0);

            for (unsigned int iatom = 0; iatom < mask.size(); ++iatom)
            {
                if (filter(spins[iatom], positions[iatom]))
                {
                    mask[iatom] = 1;
                }
            }
        }

        void Move(vectorfield& configuration, const Data::Geometry & geometry, int da, int db, int dc)
        {
            int delta = geometry.n_cell_atoms*da + geometry.n_cell_atoms*geometry.n_cells[0] * db + geometry.n_cell_atoms*geometry.n_cells[0] * geometry.n_cells[1] * dc;
            if (delta < 0)
                delta += geometry.nos;
            std::rotate(configuration.begin(), configuration.begin() + delta, configuration.end());
        }

        void Insert(Data::Spin_System &s, const vectorfield& configuration, int shift, filterfunction filter)
        {
            auto& spins = *s.spins;
            auto& positions = s.geometry->positions;
            int nos = s.nos;
            if (shift < 0) shift += nos;

            if (nos != configuration.size())
            {
                Log(Log_Level::Warning, Log_Sender::All, "Tried to insert spin configuration with NOS != NOS_system");
                return;
            }

            for (int iatom = 0; iatom < nos; ++iatom)
            {
                if (filter(spins[iatom], positions[iatom]))
                {
                    spins[iatom] = configuration[(iatom + shift) % nos];
                }
            }
        }

        void Domain(Data::Spin_System & s, Vector3 v, filterfunction filter)
        {
            if (v.norm() < 1e-8)
            {
                Log( Log_Level::Warning, Log_Sender::All, fmt::format("Homogeneous vector was ({}, {}, {}) and got set to (0, 0, 1)", v[0], v[1], v[2]) );
                v[0] = 0.0; v[1] = 0.0; v[2] = 1.0;		// if vector is zero -> set vector to 0,0,1 (posZdir)
            }
            else
            {
                v.normalize();
            }

            auto& spins = *s.spins;
            auto& positions = s.geometry->positions;

            for (int iatom = 0; iatom < s.nos; ++iatom)
            {
                if (filter(spins[iatom], positions[iatom]))
                {
                    spins[iatom] = v;
                }
            }
        }

        void Random(Data::Spin_System & s, filterfunction filter, bool external)
        {
            auto& spins = *s.spins;
            auto& positions = s.geometry->positions;

            auto distribution = std::uniform_real_distribution<scalar>(-1, 1);
            if (!external) {
                for (int iatom = 0; iatom < s.nos; ++iatom)
                {
                    if (filter(spins[iatom], positions[iatom]))
                    {
                        Engine::Vectormath::get_random_vector_unitsphere(distribution, s.llg_parameters->prng, spins[iatom]);
                    }
                }
            }
            else {
                std::mt19937 prng = std::mt19937(123456789);
                for (int iatom = 0; iatom < s.nos; ++iatom)
                {
                    if (filter(spins[iatom], positions[iatom]))
                    {
                        Engine::Vectormath::get_random_vector_unitsphere(distribution, s.llg_parameters->prng, spins[iatom]);
                    }
                }
            }
        }


        void Add_Noise_Temperature(Data::Spin_System & s, scalar temperature, int delta_seed, filterfunction filter)
        {
            if (temperature == 0.0) return;

            auto& spins = *s.spins;
            auto& positions = s.geometry->positions;
            vectorfield xi(spins.size());
            intfield mask;

            filter_to_mask(spins, positions, filter, mask);

            scalar epsilon = std::sqrt(temperature*Constants::k_B);

            std::mt19937 * prng;
            if (delta_seed!=0) prng = new std::mt19937(123456789+delta_seed);
            else prng = &s.llg_parameters->prng;

            auto distribution = std::uniform_real_distribution<scalar>(-1, 1);

            Engine::Vectormath::get_random_vectorfield_unitsphere(*prng, xi);
            Engine::Vectormath::scale(xi, epsilon);
            Engine::Vectormath::add_c_a(1, xi, *s.spins, mask);
            Engine::Vectormath::normalize_vectors(*s.spins);
        }

        void Hopfion(Data::Spin_System & s, Vector3 pos, scalar r, int order, filterfunction filter)
        {
            using std::pow;
            using std::sqrt;
            using std::acos;
            using std::sin;
            using std::cos;
            using std::atan;
            using std::atan2;

            auto& spins = *s.spins;
            auto& positions = s.geometry->positions;

            if (r != 0.0)
            {
                scalar tmp;
                scalar d, T, t, F, f;
                for (int n = 0; n<s.nos; n++)
                {
                    // Distance of spin from center
                    if (filter(spins[n], positions[n]))
                    {
                        d = (positions[n] - pos).norm();

                        // Theta
                        if (d == 0)
                        {
                            T = 0;
                        }
                        else
                        {
                            T = (positions[n][2] - pos[2]) / d; // angle with respect to the main axis of toroid [0,0,1]
                        }
                        T = acos(T);
                        // ...
                        t = d / r;	// r is a big radius of the torus
                        t = 1.0 + 4.22 / (t*t);
                        tmp = Pi*(1.0 - 1.0 / sqrt(t));
                        t = sin(tmp)*sin(T);
                        t = acos(1.0 - 2.0*t*t);
                        // ...
                        F = atan2(positions[n][1] - pos[1], positions[n][0] - pos[0]);
                        if (T > Pi / 2.0)
                        {
                            f = F + atan(1.0 / (tan(tmp)*cos(T)));
                        }
                        else
                        {
                            f = F + atan(1.0 / (tan(tmp)*cos(T))) + Pi;
                        }
                        // Spin orientation
                        spins[n][0] = sin(t)*cos(order * f);
                        spins[n][1] = sin(t)*sin(order * f);
                        spins[n][2] = cos(t);
                    }
                }
            }
        }

        void Skyrmion(Data::Spin_System & s, Vector3 pos, scalar r, scalar order, scalar phase, bool upDown, bool achiral, bool rl, bool experimental, filterfunction filter)
        {

			//bool experimental uses Method similar to PHYSICAL REVIEW B 67, 020401(R) (2003)

			auto& spins = *s.spins;
			auto& positions = s.geometry->positions;

			// skaled to fit with
			scalar r_new = r;
			if (experimental) { r_new = r * 1.2; }
			int iatom, ksi = ((int)rl) * 2 - 1, dir = ((int)upDown) * 2 - 1;
			scalar distance, phi_i, theta_i;
			for (iatom = 0; iatom < s.nos; ++iatom)
			{
				distance = std::sqrt(std::pow(s.geometry->positions[iatom][0] - pos[0], 2) + std::pow(s.geometry->positions[iatom][1] - pos[1], 2));
				distance = distance / r_new;
				//if (filter(spins[iatom], positions[iatom]))
				//{
					double x = (s.geometry->positions[iatom][0] - pos[0]) / distance / r_new;
					double y = (s.geometry->positions[iatom][1] - pos[1]) / distance / r_new;
					phi_i = std::atan2(y, x);
					if (distance == 0) { phi_i = 0; }
					if (experimental) { theta_i = Pi - 4 * std::asin(std::tanh(distance)); }
					else { theta_i = Pi - Pi * distance; }

					spins[iatom][0] = -std::sin(order * phi_i);
					spins[iatom][1] = std::cos(order * (phi_i));
					spins[iatom][2] = 0;
				//}
			}
			for (auto& v : spins) v.normalize();
			//bool experimental uses Method similar to PHYSICAL REVIEW B 67, 020401(R) (2003)
/*
			auto& spins = *s.spins;
			auto& positions = s.geometry->positions;

			// skaled to fit with 
			scalar r_new = r;
			if (experimental) { r_new = r * 1.2; }
			int iatom, ksi = ((int)rl) * 2 - 1, dir = ((int)upDown) * 2 - 1;
			scalar distance, phi_i, theta_i;
			for (iatom = 0; iatom < s.nos; ++iatom)
			{
				distance = std::sqrt(std::pow(s.geometry->positions[iatom][0] - pos[0], 2) + std::pow(s.geometry->positions[iatom][1] - pos[1], 2));
				distance = distance / r_new;
				if (filter(spins[iatom], positions[iatom]))
				{
					double x = (s.geometry->positions[iatom][0] - pos[0]) / distance / r_new;
					phi_i = std::acos(std::max(-1.0, std::min(1.0, x)));
					if (distance == 0) { phi_i = 0; }
					if (s.geometry->positions[iatom][1] - pos[1] < 0.0) { phi_i = -phi_i; }
					phi_i += phase / 180 * Pi;
					if (experimental) { theta_i = Pi - 4 * std::asin(std::tanh(distance)); }
					else { theta_i = Pi - Pi * distance; }

					spins[iatom][0] = ksi * std::sin(theta_i) * std::cos(order * phi_i);
					spins[iatom][1] = ksi * std::sin(theta_i) * std::sin(order * (phi_i + achiral * Pi));
					spins[iatom][2] = std::cos(theta_i) * -dir;
				}
			}
			for (auto& v : spins) v.normalize();*/
        }
        // end Skyrmion

        void SpinSpiral(Data::Spin_System & s, std::string direction_type, Vector3 q, Vector3 axis, scalar theta, filterfunction filter)
        {
            scalar phase;
            Vector3 vx{ 1,0,0 }, vy{ 0,1,0 }, vz{ 0,0,1 };
            Vector3 e1, e2;

            Vector3 a1 = s.geometry->bravais_vectors[0];
            Vector3 a2 = s.geometry->bravais_vectors[1];
            Vector3 a3 = s.geometry->bravais_vectors[2];

            // -------------------- Preparation --------------------
            axis.normalize();

            /*
            if axis_z=0 its in the xy-plane
                axis, vz, (axis x vz)
            else its either above or below the xy-plane.
            if its above the xy-plane, it points in z-direction
                axis, vx, -vy
            if its below the xy-plane, it points in -z-direction
                axis, vx, vy
            */

            // Choose orthogonalisation basis for Grahm-Schmidt
            //		We will need two vectors with which the axis always forms the
            //		same orientation (händigkeit des vektor-dreibeins)
            // If axis_z=0 its in the xy-plane
            //		the vectors should be: axis, vz, (axis x vz)
            if (axis[2] == 0)
            {
                e2 = axis.cross(vz);
                e1 = vz;
            }
            // Else its either above or below the xy-plane.
            //		if its above the xy-plane, it points in z-direction
            //		the vectors should be: axis, vx, -vy
            else if (axis[2] > 0)
            {
                e1 = vx;
                e2 = vy;
            }
            //		if its below the xy-plane, it points in -z-direction
            //		the vectors should be: axis, vx, vy
            else if (axis[2] < 0)
            {
                e1 = vx;
                e2 = -vy;
            }

            // Some normalisations
            theta = theta / 180.0 * Pi;
            scalar qnorm = q.norm();

            // Grahm-Schmidt orthogonalization: two vectors orthogonal to an axis
            Vector3 v1, v2;
            //u1 = axis
            //u2 = v1 = vx - vx*axis/|axis|^2 * axis
            //u3 = v2 = vy - vy*axis/|axis|^2 * axis - vy*v1/|v1|^2 * v1
            // Projections
            scalar proj1 = e1.dot(axis);
            scalar proj2 = e2.dot(axis);

            // First vector
            v1 = (e1 - proj1 * axis).normalized();

            // One more projection
            scalar proj3 = e2.dot(v1);

            // Second vector
            v2 = (e2 - proj2 * axis - proj3*v1).normalized();

            // -------------------- Spin Spiral creation --------------------
            auto& spins = *s.spins;
            auto& positions = s.geometry->positions;
            if (direction_type == "Reciprocal Lattice")
            {
                // bi = 2*pi*(aj x ak) / (ai * (aj x ak))
                Vector3 b1, b2, b3;
                b1 = 2.0 * Pi * a2.cross(a3) / (a1.dot(a2.cross(a3)));
                b2 = 2.0 * Pi * a3.cross(a1) / (a2.dot(a3.cross(a1)));
                b3 = 2.0 * Pi * a1.cross(a2) / (a3.dot(a1.cross(a2)));
                // The q-vector is specified in units of the reciprocal lattice
                Vector3 projBQ = q[0]*b1 + q[1]*b2 + q[2]*b3;
                q = projBQ;
            }
            else if (direction_type == "Real Lattice")
            {
                // The q-vector is specified in units of the real lattice
                Vector3 projBQ = { q.dot(a1), q.dot(a2), q.dot(a3) };
                q = projBQ;
            }
            else if (direction_type == "Real Space")
            {
                // The q-vector is specified in units of (x, y, z)
            }
            else
            {
                Log(Log_Level::Warning, Log_Sender::All, "Got passed invalid type for SS: " + direction_type);
            }
            for (int iatom = 0; iatom < s.nos; ++iatom)
            {
                if (filter(spins[iatom], positions[iatom]))
                {
                    // Phase is scalar product of spin position and q
                    phase = s.geometry->positions[iatom].dot(q);
                    //phase = phase / 180.0 * Pi;// / period;
                    // The opening angle determines how far from the axis the spins rotate around it.
                    //		The rotation is done by alternating between v1 and v2 periodically
                    scalar norms = 0.0;
                    spins[iatom] = axis * std::cos(theta)
                        + v1 * std::cos(phase) * std::sin(theta)
                        + v2 * std::sin(phase) * std::sin(theta);
                    spins[iatom].normalize();
                }
            }// endfor iatom
        }

        void SpinSpiral(Data::Spin_System & s, std::string direction_type, Vector3 q1, Vector3 q2, Vector3 axis, scalar theta, filterfunction filter)
        {
            Vector3 vx{ 1,0,0 }, vy{ 0,1,0 }, vz{ 0,0,1 };
            Vector3 e1, e2;
            Vector3 qm, qk;

            Vector3 a1 = s.geometry->bravais_vectors[0];
            Vector3 a2 = s.geometry->bravais_vectors[1];
            Vector3 a3 = s.geometry->bravais_vectors[2];

            // -------------------- Preparation --------------------
            axis.normalize();

            /*
            if axis_z=0 its in the xy-plane
                axis, vz, (axis x vz)
            else its either above or below the xy-plane.
            if its above the xy-plane, it points in z-direction
                axis, vx, -vy
            if its below the xy-plane, it points in -z-direction
                axis, vx, vy
            */

            // Choose orthogonalisation basis for Grahm-Schmidt
            //		We will need two vectors with which the axis always forms the
            //		same orientation (händigkeit des vektor-dreibeins)
            // If axis_z=0 its in the xy-plane
            //		the vectors should be: axis, vz, (axis x vz)
            if (axis[2] == 0)
            {
                e2 = axis.cross(vz);
                e1 = vz;
            }
            // Else its either above or below the xy-plane.
            //		if its above the xy-plane, it points in z-direction
            //		the vectors should be: axis, vx, -vy
            else if (axis[2] > 0)
            {
                e1 = vx;
                e2 = -vy;
            }
            //		if its below the xy-plane, it points in -z-direction
            //		the vectors should be: axis, vx, vy
            else if (axis[2] < 0)
            {
                e1 = vx;
                e2 = vy;
            }

            // Some normalisations
            theta = theta / 180.0 * Pi;
            scalar q1norm = q1.norm();
            scalar q2norm = q2.norm();
            scalar axnorm = axis.norm();
            axis.normalize();

            // Grahm-Schmidt orthogonalization: two vectors orthogonal to an axis
            Vector3 v1, v2;
            //u1 = axis
            //u2 = v1 = vx - vx*axis/|axis|^2 * axis
            //u3 = v2 = vy - vy*axis/|axis|^2 * axis - vy*v1/|v1|^2 * v1
            scalar proj1 = 0, proj2 = 0, proj3 = 0, proj1a=0, proj2a=0, proj3a=0, proj1b=0, proj2b=0, proj3b=0;
            // Projections
            proj1a = e1.dot(axis);
            proj2a = e2.dot(axis);
            proj1b = axis.dot(axis);
            proj2b = axis.dot(axis);
            proj1 += proj1a / proj1b;
            proj2 += proj2a / proj2b;

            // First vector
            v1 = e1 - proj1 * axis;

            // One more projection
            proj3a = e2.dot(v1);
            proj3b = v1.dot(v1);
            proj3 = proj3a / proj3b;

            // Second vector
            v2 = e2 - proj2 * axis - proj3*v1;

            // -------------------- Spin Spiral creation --------------------
            auto& spins = *s.spins;
            auto& positions = s.geometry->positions;
            if (direction_type == "Reciprocal Lattice")
            {
                // bi = 2*pi*(aj x ak) / (ai * (aj x ak))
                Vector3 b1, b2, b3;
                b1 = 2.0 * Pi * a2.cross(a3) / (a1.dot(a2.cross(a3)));
                b2 = 2.0 * Pi * a3.cross(a1) / (a2.dot(a3.cross(a1)));
                b3 = 2.0 * Pi * a1.cross(a2) / (a3.dot(a1.cross(a2)));

                // The q-vectors are specified in units of the reciprocal lattice
                Vector3 projBQ = q1[0]*b1 + q1[1]*b2 + q1[2]*b3;
                q1 = projBQ;
                projBQ = q2[0]*b1 + q2[1]*b2 + q2[2]*b3;
                q2 = projBQ;
                qm = (q1+q2)*0.5;
                qk = (q1-q2)*0.5;

            }
            else if (direction_type == "Real Lattice")
            {
                // The q-vector is specified in units of the real lattice
                Vector3 projBQ = { q1.dot(a1), q1.dot(a2), q1.dot(a3) };
                q1 = projBQ;
                projBQ = { q2.dot(a1), q2.dot(a2), q2.dot(a3) };
                q2 = projBQ;
            }
            else if (direction_type == "Real Space")
            {
                // The q-vector is specified in units of (x, y, z)
            }
            else
            {
                Log(Log_Level::Warning, Log_Sender::All, "Got passed invalid type for SS: " + direction_type);
            }

            for (int iatom = 0; iatom < s.nos; ++iatom)
            {
                if (filter(spins[iatom], positions[iatom]))
                {
                    // Phase is scalar product of spin position and q
                    auto& r = s.geometry->positions[iatom];
                    //phase = phase / 180.0 * Pi;// / period;
                    // The opening angle determines how far from the axis the spins rotate around it.
                    //		The rotation is done by alternating between v1 and v2 periodically
                    scalar norms = 0.0;
                    spins[iatom] =	axis * std::sin(r.dot(qm))
                        + v1 * std::cos(r.dot(qm)) * std::sin(r.dot(qk))
                        + v2 * std::cos(r.dot(qm)) * std::cos(r.dot(qk));
                    spins[iatom].normalize();
                }
            }// endfor iatom
        }

        void Set_Atom_Types(Data::Spin_System & s, int atom_type, filterfunction filter)
        {
            auto& spins = *s.spins;
            auto& geometry = s.geometry;
            auto& positions = geometry->positions;

            for (int iatom = 0; iatom < s.nos; ++iatom)
            {
                if (filter(spins[iatom], positions[iatom]))
                {
                    geometry->atom_types[iatom] = atom_type;
                    if(atom_type < 0)
                        geometry->mu_s[iatom] = 0.0;
                }
            }
        }

        void Set_Pinned(Data::Spin_System & s, bool pinned, filterfunction filter)
        {
            auto& spins = *s.spins;
            auto& geometry = s.geometry;
            auto& positions = geometry->positions;

            int unpinned = (int)!pinned;
            for (int iatom = 0; iatom < s.nos; ++iatom)
            {
                if (filter(spins[iatom], positions[iatom]))
                {
                    geometry->mask_unpinned[iatom] = unpinned;
                    geometry->mask_pinned_cells[iatom] = spins[iatom];
                }
            }
        }
    }//end namespace Spin_Setters
}//end namespace Utility
