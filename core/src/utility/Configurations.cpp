#include <utility/Configurations.hpp>
#include <data/Spin_System.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <Eigen/Dense>

#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Utility
{
	namespace Configurations
	{
		void Insert(Data::Spin_System &s, const vectorfield& configuration, filterfunction filter)
		{
			auto& spins = *s.spins;
			auto& spin_pos = s.geometry->spin_pos;

			if (s.nos != configuration.size())
			{
				Log(Log_Level::Warning, Log_Sender::All, "Tried to insert spin configuration with NOS != NOS_system");
				return;
			}

			for (int iatom = 0; iatom < s.nos; ++iatom)
			{
				if (filter(spins[iatom], spin_pos[iatom]))
				{
					spins[iatom] = configuration[iatom];
				}
			}
		}

		void Domain(Data::Spin_System & s, Vector3 v, filterfunction filter)
		{
			try
			{
				v.normalize();
			}
			catch (Exception ex)
			{
				if (ex == Exception::Division_by_zero) 
				{
					std::string message = "Homogeneous vector was (" + std::to_string(v[0]) + ", " + std::to_string(v[1]) + ", " + std::to_string(v[2]) + ") and got set to (0, 0, 1)";
					Log(Log_Level::Warning, Log_Sender::All, message);
					v[0] = 0.0; v[1] = 0.0; v[2] = 1.0;		// if vector is zero -> set vector to 0,0,1 (posZdir)
				}
				else { throw(ex); }
			}

			auto& spins = *s.spins;
			auto& spin_pos = s.geometry->spin_pos;

			for (int iatom = 0; iatom < s.nos; ++iatom)
			{
				if (filter(spins[iatom], spin_pos[iatom]))
				{
					spins[iatom] = v;
				}
			}
		}

		void Random(Data::Spin_System & s, filterfunction filter, bool external)
		{
			auto& spins = *s.spins;
			auto& spin_pos = s.geometry->spin_pos;

			if (!external) {
				for (int iatom = 0; iatom < s.nos; ++iatom)
				{
					if (filter(spins[iatom], spin_pos[iatom]))
					{
						Random(s, iatom, s.llg_parameters->prng);
					}
				}
			}
			else {
				std::mt19937 prng = std::mt19937(123456789);
				for (int iatom = 0; iatom < s.nos; ++iatom)
				{
					if (filter(spins[iatom], spin_pos[iatom]))
					{
						Random(s, iatom, prng);
					}
				}
			}
		}

		void Random(Data::Spin_System & s, int no, std::mt19937 & prng)
		{
			auto& spins = *s.spins;
			Vector3 v = { 0, 0, 0 };			// declare v= 0,0,0
			while (true)
			{
				for (int dim = 0; dim < 3; ++dim) {		// use spin_system's PRNG
					v[dim] = s.llg_parameters->distribution_minus_plus_one(prng);		// roll random for v in 3 dimensions
				}
				try {
					v.normalize();	// try normalizing v
					spins[no] = v;	// copy normalized v into spins array
					return;			// normalizing worked -> return function
				}
				catch (Exception ex) {
					if (ex != Exception::Division_by_zero) { throw(ex); }				// throw everything except division by zero
				}
			}
		}// end Random


		void Add_Noise_Temperature(Data::Spin_System & s, scalar temperature, int delta_seed, filterfunction filter)
		{
			if (temperature == 0.0) return;

			auto& spins = *s.spins;
			auto& spin_pos = s.geometry->spin_pos;

			Vector3 v = { 0, 0, 0 };
			auto epsilon = std::sqrt(2.0*s.llg_parameters->damping / (1.0 + std::pow(s.llg_parameters->damping, 2))*temperature*Constants::k_B);
			
			std::mt19937 * prng;
			if (delta_seed!=0) prng = new std::mt19937(123456789+delta_seed);
			else prng = &s.llg_parameters->prng;

			for (int i = 0; i < s.nos; ++i)
			{
				if (filter(spins[i], spin_pos[i]))
				{
					while (true)
					{
						for (int dim = 0; dim < 3; ++dim)
						{		// use spin_system's PRNG
							v[dim] = s.llg_parameters->distribution_minus_plus_one(*prng) * epsilon;		// roll random for v in 3 dimensions
						}
						try
						{
							scalar l = v.norm();
							//Vectormath::Normalize(v);			// try normalizing v
							(*s.spins)[i] += v;// copy normalized v into spins array
							break;									// normalizing worked -> return function
						}
						catch (Exception ex)
						{
							if (ex != Exception::Division_by_zero) throw(ex);				// throw everything except division by zero
						}
					}
				}
			}
			for (auto& v : *s.spins) v.normalize();
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
			auto& spin_pos = s.geometry->spin_pos;

			if (r != 0.0)
			{
				scalar tmp;
				scalar d, T, t, F, f;
				for (int n = 0; n<s.nos; n++)
				{
					// Distance of spin from center
					if (filter(spins[n], spin_pos[n]))
					{
						d = (spin_pos[n] - pos).norm();
					
						// Theta
						if (d == 0)
						{
							T = 0;
						}
						else
						{
							T = (spin_pos[n][2] - pos[2]) / d; // angle with respect to the main axis of toroid [0,0,1]
						}
						T = acos(T);
						// ...
						t = d / r;	// r is a big radius of the torus
						t = 1.0 + 4.22 / (t*t);
						tmp = M_PI*(1.0 - 1.0 / sqrt(t));
						t = sin(tmp)*sin(T);
						t = acos(1.0 - 2.0*t*t);
						// ...
						F = atan2(spin_pos[n][1] - pos[1], spin_pos[n][0] - pos[0]);
						if (T > M_PI / 2.0)
						{
							f = F + atan(1.0 / (tan(tmp)*cos(T)));
						}
						else
						{
							f = F + atan(1.0 / (tan(tmp)*cos(T))) + M_PI;
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
			auto& spin_pos = s.geometry->spin_pos;

			// skaled to fit with 
			scalar r_new = r;
			if (experimental) { r_new = r*1.2; }
			int iatom, ksi = ((int)rl) * 2 - 1, dir = ((int)upDown) * 2 - 1;
			scalar distance, phi_i, theta_i;
			for (iatom = 0; iatom < s.nos; ++iatom)
			{
				distance = std::sqrt(std::pow(s.geometry->spin_pos[iatom][0] - pos[0], 2) + std::pow(s.geometry->spin_pos[iatom][1] - pos[1], 2));
				distance = distance / r_new;
				if (filter(spins[iatom], spin_pos[iatom]))
				{
					double x = (s.geometry->spin_pos[iatom][0] - pos[0]) / distance / r_new;
					phi_i = std::acos(std::max(-1.0, std::min(1.0, x)));
					if (distance == 0) { phi_i = 0; }
					if (s.geometry->spin_pos[iatom][1] - pos[1] < 0.0) { phi_i = - phi_i ; }
					phi_i += phase / 180 * M_PI;
					if (experimental) { theta_i = M_PI - 4 * std::asin(std::tanh(distance)); }
					else { theta_i = M_PI - M_PI *distance; }

					spins[iatom][0] = ksi * std::sin(theta_i) * std::cos(order * phi_i);
					spins[iatom][1] = ksi * std::sin(theta_i) * std::sin(order * (phi_i + achiral * M_PI));
					spins[iatom][2] = std::cos(theta_i) * -dir;
				}
			}
			for (auto& v : spins) v.normalize();	
		}
		// end Skyrmion

		void SpinSpiral(Data::Spin_System & s, std::string direction_type, Vector3 q, Vector3 axis, scalar theta, filterfunction filter)
		{
			scalar phase;
			Vector3 vx{ 1,0,0 }, vy{ 0,1,0 }, vz{ 0,0,1 };
			Vector3 e1, e2;
			
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
			theta = theta / 180.0 * M_PI;
			q *= 2.0 * M_PI;
			scalar qnorm = q.norm();
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
			auto& spin_pos = s.geometry->spin_pos;
			if (direction_type == "Real Lattice")
			{
				// NOTE this is not yet the correct function!!
				for (int iatom = 0; iatom < s.nos; ++iatom)
				{
					if (filter(spins[iatom], spin_pos[iatom]))
					{
						// Phase is scalar product of spin position and q
						phase = s.geometry->spin_pos[iatom].dot(q);
						//phase = phase / 180.0 * M_PI;// / period;
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
			else if (direction_type == "Reciprocal Lattice")
			{
				Log(Log_Level::Error, Log_Sender::All, "The reciprocal lattice spin spiral is not yet implemented!");
				// Not yet implemented!
				// bi = 2*pi*(aj x ak) / (ai * (aj x ak))
			}
			else if (direction_type == "Real Space")
			{
				Log(Log_Level::Error, Log_Sender::All, "The real space spin spiral is not yet implemented!");
			}
			else
			{
				Log(Log_Level::Warning, Log_Sender::All, "Got passed invalid type for SS: " + direction_type);
			}
		}

		//void SpinSpiral(Data::Spin_System & s, std::string direction_type, scalar q[3], scalar axis[3], scalar theta)
		//{
		//	scalar gamma, rho, r1[3], r2[3], r3[3];

		//	scalar cross[3];
		//	scalar sabs;

		//	if (direction_type == "Real Lattice")
		//	{
		//		// Renormalise input
		//		theta = theta / 180.0 * M_PI;
		//		for (int dim = 0; dim < 3; ++dim)
		//		{
		//			q[dim] = q[dim] * 2.0 * M_PI;
		//		}
		//		// NOTE this is not yet the correct function!!
		//		for (int iatom = 0; iatom < s.nos; ++iatom)
		//		{
		//			gamma = 0;
		//			for (int dim = 0; dim < 3; ++dim)
		//			{
		//				gamma += q[dim] * s.geometry.spin_pos[dim][iatom];
		//			}
		//			rho = std::sqrt(std::pow(axis[0], 2) + std::pow(axis[1], 2));

		//			r1[0] = axis[0] * axis[2] / rho;
		//			r1[1] = -axis[1] / rho;
		//			r1[2] = axis[0];

		//			r2[0] = axis[1] * axis[2] / rho;
		//			r2[1] = axis[0] / rho;
		//			r2[2] = axis[1];

		//			r3[0] = -rho;
		//			r3[1] = 0;
		//			r3[2] = axis[2];

		//			cross[0] = std::sin(theta)*std::cos(gamma);
		//			cross[1] = std::sin(theta)*std::sin(gamma);
		//			cross[2] = std::cos(theta);

		//			for (int dim = 0; dim < 3; ++dim)
		//			{
		//				s.spins[dim * s.nos + iatom] = 0;
		//			}
		//			for (int dim = 0; dim < 3; ++dim)
		//			{
		//				//s.spins[dim * s.nos + iatom] = cross[dim];

		//				s.spins[0 * s.nos + iatom] += r1[dim] * cross[dim];
		//				s.spins[1 * s.nos + iatom] += r2[dim] * cross[dim];
		//				s.spins[2 * s.nos + iatom] += r3[dim] * cross[dim];
		//			}

		//			/*sabs = std::sqrt(std::pow(s.spins[0 * s.nos + iatom], 2) + std::pow(s.spins[1 * s.nos + iatom], 2) + std::pow(s.spins[2 * s.nos + iatom], 2));
		//			for (int dim = 0; dim < 3; ++dim)
		//			{
		//			s.spins[dim * s.nos + iatom] = s.spins[dim * s.nos + iatom] / sabs;
		//			}
		//			int x = 0;*/
		//		}// endfor iatom
		//	}
		//	else if (direction_type == "Reciprocal Lattice")
		//	{
		//		// Not yet implemented!
		//		// bi = 2*pi*(aj x ak) / (ai * (aj x ak))
		//	}
		//	else if (direction_type == "Real Space")
		//	{
		//		//for (int iatom = 0; iatom < s.nos; ++iatom)
		//		//{
		//		//	distance = s.geometry.spin_pos[0][iatom] * direction[0]
		//		//		+ s.geometry.spin_pos[1][iatom] * direction[1]
		//		//		+ s.geometry.spin_pos[2][iatom] * direction[2];
		//		//	distance = distance / period;
		//		//	if (distance >= 0.0)
		//		//	{
		//		//		Vectormath::Cross_Product(axis, z, cross);
		//		//		for (int dim = 0; dim < 3; ++dim)
		//		//		{
		//		//			s.spins[dim * s.nos + iatom] = z[dim] * std::cos(M_PI*distance) + cross[dim] * std::sin(M_PI*distance);
		//		//		}
		//		//	}// end 0 <= distance
		//		//}// endfor iatom
		//	}
		//	else
		//	{
		//		std::cout << "Got passed invalid type for SS: " << direction_type << std::endl;
		//	}
		//}

	}//end namespace Spin_Setters
}//end namespace Utility




 /*
 // Set homogeneous spins to +z-direction
 for (iatom = 0; iatom < s->nos; ++iatom){
 s->spins[2][iatom] = - 1.0;
 //if ((fmod(iatom, 31) == 14 || fmod(iatom, 31) == 15 || fmod(iatom, 31) == 16) && iatom > 400 && iatom < 700) {
 //s->spins[2][iatom] = 1.0;
 //}
 }
 */