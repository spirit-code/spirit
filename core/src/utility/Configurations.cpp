#include "Configurations.h"
#include "Spin_System.h"
#include "Vectormath.h"
#include "Logging.h"
#include "Exception.h"

#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//extern Utility::LoggingHandler Log;

namespace Utility
{
	namespace Configurations
	{
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		void DomainWall(Data::Spin_System & s, const double pos[3], double v[3], bool greater)
		{
			try {
				Vectormath::Normalize(v, 3);				// try normalizing v
			}
			catch (Exception ex) {
				if (ex == Exception::Division_by_zero) 
				{
					std::string message = "Homogeneous vector was (" + std::to_string(v[0]) + ", " + std::to_string(v[1]) + ", " + std::to_string(v[2]) + ") and got set to (0, 0, 1)";
					Log.Send(Log_Level::WARNING, Log_Sender::ALL, message);
					v[0] = 0.0; v[1] = 0.0; v[2] = 1.0;		// if vector is zero -> set vector to 0,0,1 (posZdir)
				}
				else { throw(ex); }
			}

			int dim, iatom, nos = s.nos;
			auto& spins = *s.spins;

			if (greater) {
				for (dim = 0; dim < 3; ++dim) {
					for (iatom = 0; iatom < nos; ++iatom) {
						if (s.geometry->spin_pos[0][iatom] >= pos[0]) {
							if (s.geometry->spin_pos[1][iatom] >= pos[1]) {
								if (s.geometry->spin_pos[2][iatom] >= pos[2]) {
									spins[dim*s.nos + iatom] = v[dim];
								}
							}
						}
					}//endfor iatom
				}//endfor dim
			}//endif greater
			else {
				for (dim = 0; dim < 3; ++dim) {
					for (iatom = 0; iatom < nos; ++iatom) {
						if (s.geometry->spin_pos[0][iatom] <= pos[0]) {
							if (s.geometry->spin_pos[1][iatom] <= pos[1]) {
								if (s.geometry->spin_pos[2][iatom] <= pos[2]) {
									spins[dim*s.nos + iatom] = v[dim];
								}
							}
						}
					}//endfor iatom
				}//endfor dim
			}//endelse greater

		}//endfor DomainWall

		void Homogeneous(Data::Spin_System & s, double v[3])
		{
			double pos[3] = { -1.0E+20, -1.0E+20, -1.0E+20 };
			DomainWall(s, pos, v, true);
		}//endfor Homogeneous

		void PlusZ(Data::Spin_System & s)
		{
			double v[3] = { 0.0 , 0.0, 1.0 };
			Homogeneous(s, v);
		}//endfor PlusZ

		void MinusZ(Data::Spin_System & s)
		{
			double v[3] = { 0.0 , 0.0, -1.0 };
			Homogeneous(s, v);
		}//endfor MinusZ

		void Random(Data::Spin_System & s, bool external)
		{
			if (!external) {
				for (int iatom = 0; iatom < s.nos; ++iatom) {
					Random(s, iatom, s.llg_parameters->prng);
				}
			}
			else {
				std::mt19937 prng = std::mt19937(123456789);
				for (int iatom = 0; iatom < s.nos; ++iatom) {
					Random(s, iatom, prng);
				}
			}
		}

		void Random(Data::Spin_System & s, int no, std::mt19937 &prng)
		{
			auto& spins = *s.spins;
			std::vector<double> v = { 0.0, 0.0, 0.0 };			// declare v= 0,0,0
			while (true) {
				for (int dim = 0; dim < 3; ++dim) {		// use spin_system's PRNG
					v[dim] = s.llg_parameters->distribution_minus_plus_one(prng);		// roll random for v in 3 dimensions
				}
				try {
					Vectormath::Normalize(v);			// try normalizing v
					for (int dim = 0; dim < 3; ++dim) {
						spins[dim*s.nos + no] = v[dim];// copy normalized v into spins array
					}
					return;									// normalizing worked -> return function
				}
				catch (Exception ex) {
					if (ex != Exception::Division_by_zero) { throw(ex); }				// throw everything except division by zero
				}
			}
		}// end Random
		void Skyrmion(Data::Spin_System & s, std::vector<double> pos, double r, double order, double phase, bool upDown, bool achiral, bool rl, bool experimental)
		{
			//bool experimental uses Method similar to PHYSICAL REVIEW B 67, 020401(R) (2003)
			auto& spins = *s.spins;
			// skaled to fit with 
			double r_new = r;
			if (experimental) { r_new = r*1.2; }
			int iatom, ksi = ((int)rl) * 2 - 1, dir = ((int)upDown) * 2 - 1;
			double distance, phi_i, theta_i;
			for (iatom = 0; iatom < s.nos; ++iatom) {
				distance = std::sqrt(std::pow(s.geometry->spin_pos[0][iatom] - pos[0], 2) + std::pow(s.geometry->spin_pos[1][iatom] - pos[1], 2));
				distance = distance / r_new;
				if (distance >= 0.0 && distance * r_new <= r) {
					double x = (s.geometry->spin_pos[0][iatom] - pos[0]) / distance / r_new;
					phi_i = std::acos(std::max(-1.0, std::min(1.0, x)));
					if (distance == 0) { phi_i = 0; }
					if (s.geometry->spin_pos[1][iatom] - pos[1] < 0.0) { phi_i = - phi_i ; }
					phi_i += phase / 180 * M_PI;
					if(experimental){ theta_i = M_PI - 4 * std::asin(std::tanh(distance)); }
					else{ theta_i = M_PI - M_PI *distance; }
					spins[2 * s.nos + iatom] = std::cos(theta_i) * -dir;
					spins[1 * s.nos + iatom] = ksi * std::sin(theta_i) * std::sin(order * (phi_i + achiral * M_PI));
					spins[iatom] = ksi * std::sin(theta_i) * std::cos(order * phi_i);
				}
			}
			Utility::Vectormath::Normalize_3Nos(spins);	
		}
		// end Skyrmion

		void SpinSpiral(Data::Spin_System & s, std::string direction_type, double q[3], double axis[3], double theta)
		{
			double phase;
			double vx[3] = { 1,0,0 }, vy[3] = { 0,1,0 }, vz[3] = { 0, 0, 1 };
			double e1[3], e2[3];
			
			// -------------------- Preparation --------------------
			Vectormath::Normalize(axis, 3);
			
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
				Vectormath::Cross_Product(axis, vz, e2);
				for (int dim = 0; dim < 3; ++dim) { e1[dim] = vz[dim]; }
			}
			// Else its either above or below the xy-plane.
			//		if its above the xy-plane, it points in z-direction
			//		the vectors should be: axis, vx, -vy
			else if (axis[2] > 0)
			{
				for (int dim = 0; dim < 3; ++dim)
				{
					e1[dim] = vx[dim];
					e2[dim] = -vy[dim];
				}
			}
			//		if its below the xy-plane, it points in -z-direction
			//		the vectors should be: axis, vx, vy
			else if (axis[2] < 0)
			{
				for (int dim = 0; dim < 3; ++dim)
				{
					e1[dim] = vx[dim];
					e2[dim] = vy[dim];
				}
			}

			// Some normalisations
			theta = theta / 180.0 * M_PI;
			for (int dim = 0; dim < 3; ++dim)
			{
				q[dim] = q[dim] * 2.0 * M_PI;
			}
			double qnorm = std::sqrt(std::pow(q[0], 2) + std::pow(q[1], 2) + std::pow(q[2], 2));
			double axnorm = std::sqrt(std::pow(axis[0],2) + std::pow(axis[1], 2) + std::pow(axis[2], 2));
			for (int dim = 0; dim < 3; ++dim)
			{
				axis[dim] = axis[dim] / axnorm;
			}

			// Grahm-Schmidt orthogonalization: two vectors orthogonal to an axis
			double v1[3], v2[3];
			//u1 = axis
			//u2 = v1 = vx - vx*axis/|axis|^2 * axis
			//u3 = v2 = vy - vy*axis/|axis|^2 * axis - vy*v1/|v1|^2 * v1
			double proj1 = 0, proj2 = 0, proj3 = 0, proj1a=0, proj2a=0, proj3a=0, proj1b=0, proj2b=0, proj3b=0;
			// Projections
			for (int dim = 0; dim < 3; ++dim)
			{
				proj1a += e1[dim] * axis[dim];
				proj2a += e2[dim] * axis[dim];
			}
			for (int dim = 0; dim < 3; ++dim)
			{
				proj1b += (axis[dim] * axis[dim]);
				proj2b += (axis[dim] * axis[dim]);
			}
			proj1 += proj1a / proj1b;
			proj2 += proj2a / proj2b;

			// First vector
			for (int dim = 0; dim < 3; ++dim)
			{
				v1[dim] = e1[dim] - proj1 * axis[dim];
			}

			// One more projection
			for (int dim = 0; dim < 3; ++dim)
			{
				proj3a += e2[dim] * v1[dim];
				proj3b += v1[dim] * v1[dim];
			}
			proj3 = proj3a / proj3b;

			// Second vector
			for (int dim = 0; dim < 3; ++dim)
			{
				v2[dim] = e2[dim] - proj2 * axis[dim] - proj3*v1[dim];
			}

			// -------------------- Spin Spiral creation --------------------
			auto& spins = *s.spins;
			if (direction_type == "Real Lattice")
			{
				// NOTE this is not yet the correct function!!
				for (int iatom = 0; iatom < s.nos; ++iatom)
				{
					// Phase is scalar product of spin position and q
					phase = (s.geometry->spin_pos[0][iatom] * q[0])
						+ (s.geometry->spin_pos[1][iatom] * q[1])
						+ (s.geometry->spin_pos[2][iatom] * q[2]);
					//phase = phase / 180.0 * M_PI;// / period;
					// The opening angle determines how far from the axis the spins rotate around it.
					//		The rotation is done by alternating between v1 and v2 periodically
					double norms = 0.0;
					for (int dim = 0; dim < 3; ++dim)
					{
						spins[dim * s.nos + iatom] = axis[dim] * std::cos(theta)
													+ v1[dim] * std::cos(phase) * std::sin(theta)
													+ v2[dim] * std::sin(phase) * std::sin(theta);
						norms += std::pow(spins[dim * s.nos + iatom], 2);
					}
					norms = std::sqrt(norms);
					
					// Write to spin
					for (int dim = 0; dim < 3; ++dim)
					{
						spins[dim * s.nos + iatom] = spins[dim * s.nos + iatom] / norms;
					}
				}// endfor iatom
			}
			else if (direction_type == "Reciprocal Lattice")
			{
				Log.Send(Log_Level::L_ERROR, Log_Sender::ALL, "The reciprocal lattice spin spiral is not yet implemented!");
				// Not yet implemented!
				// bi = 2*pi*(aj x ak) / (ai * (aj x ak))
			}
			else if (direction_type == "Real Space")
			{
				Log.Send(Log_Level::L_ERROR, Log_Sender::ALL, "The real space spin spiral is not yet implemented!");
			}
			else
			{
				Log.Send(Log_Level::WARNING, Log_Sender::ALL, "Got passed invalid type for SS: " + direction_type);
			}
		}

		//void SpinSpiral(Data::Spin_System & s, std::string direction_type, double q[3], double axis[3], double theta)
		//{
		//	double gamma, rho, r1[3], r2[3], r3[3];

		//	double cross[3];
		//	double sabs;

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