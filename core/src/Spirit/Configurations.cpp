#include <Spirit/Configurations.h>
#include <Spirit/State.h>
#include "Spirit_Defines.h"
#include <data/State.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Configurations.hpp>

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


std::function< bool(const Vector3&, const Vector3&) > get_filter(Vector3 position, const float r_cut_rectangular[3], float r_cut_cylindrical, float r_cut_spherical, bool inverted)
{
	bool no_cut_rectangular_x = r_cut_rectangular[0] < 0;
	bool no_cut_rectangular_y = r_cut_rectangular[1] < 0;
	bool no_cut_rectangular_z = r_cut_rectangular[2] < 0;
	bool no_cut_cylindrical   = r_cut_cylindrical    < 0;
	bool no_cut_spherical     = r_cut_spherical      < 0;

	std::function< bool(const Vector3&, const Vector3&) > filter;
	if (!inverted)
	{
		filter =
			[position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, no_cut_rectangular_x, no_cut_rectangular_y, no_cut_rectangular_z, no_cut_cylindrical, no_cut_spherical]
			(const Vector3& spin, const Vector3& spin_pos)
		{
			Vector3 r_rectangular = spin_pos - position;
			scalar r_cylindrical = std::sqrt(std::pow(spin_pos[0] - position[0], 2) + std::pow(spin_pos[1] - position[1], 2));
			scalar r_spherical   = (spin_pos-position).norm();
			if (   ( no_cut_rectangular_x || std::abs(r_rectangular[0]) < r_cut_rectangular[0] )
				&& ( no_cut_rectangular_y || std::abs(r_rectangular[1]) < r_cut_rectangular[1] )
				&& ( no_cut_rectangular_z || std::abs(r_rectangular[2]) < r_cut_rectangular[2] )
				&& ( no_cut_cylindrical   || r_cylindrical    < r_cut_cylindrical )
				&& ( no_cut_spherical     || r_spherical      < r_cut_spherical )
				) return true;
			return false;
		};
	}
	else
	{
		filter =
			[position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, no_cut_rectangular_x, no_cut_rectangular_y, no_cut_rectangular_z, no_cut_cylindrical, no_cut_spherical]
			(const Vector3& spin, const Vector3& spin_pos)
		{
			Vector3 r_rectangular = spin_pos - position;
			scalar r_cylindrical = std::sqrt(std::pow(spin_pos[0] - position[0], 2) + std::pow(spin_pos[1] - position[1], 2));
			scalar r_spherical   = (spin_pos-position).norm();
			if (!( ( no_cut_rectangular_x || std::abs(r_rectangular[0]) < r_cut_rectangular[0] )
				&& ( no_cut_rectangular_y || std::abs(r_rectangular[1]) < r_cut_rectangular[1] )
				&& ( no_cut_rectangular_z || std::abs(r_rectangular[2]) < r_cut_rectangular[2] )
				&& ( no_cut_cylindrical   || r_cylindrical    < r_cut_cylindrical )
				&& ( no_cut_spherical     || r_spherical      < r_cut_spherical )
				)) return true;
			return false;
		};
	}

	return filter;
}

std::string filter_to_string(const float position[3], const float r_cut_rectangular[3], float r_cut_cylindrical, float r_cut_spherical, bool inverted)
{
	std::string ret = "";
	if ( position[0]!=0 || position[1]!=0 || position[2]!=0 )
		ret += "Position: (" + std::to_string(position[0]) + "," + std::to_string(position[1]) + "," + std::to_string(position[2]) + ").";
	if ( r_cut_rectangular[0] <= 0 && r_cut_rectangular[1] <= 0 && r_cut_rectangular[2] <= 0 &&
		 r_cut_cylindrical <= 0 &&
		 r_cut_spherical <= 0 &&
		 !inverted )
	{
		if (ret != "") ret += " ";
		ret += "Entire space.";
	}
	else
	{
		if ( r_cut_rectangular[0] > 0 || r_cut_rectangular[1] > 0 || r_cut_rectangular[2] > 0 )
		{
			if (ret != "") ret += " ";
			ret += "Rectangular region: (" + std::to_string(r_cut_rectangular[0]) + "," + std::to_string(r_cut_rectangular[1]) + "," + std::to_string(r_cut_rectangular[2]) + ").";
		}
		if ( r_cut_cylindrical > 0 )
		{
			if (ret != "") ret += " ";
			ret += "Cylindrical region, r=" + std::to_string(r_cut_cylindrical) + ".";
		}
		if ( r_cut_spherical > 0 )
		{
			if (ret != "") ret += " ";
			ret += "Spherical region, r=" + std::to_string(r_cut_spherical) + ".";
		}
		if ( inverted )
		{
			if (ret != "") ret += " ";
			ret += "Inverted.";
		}

	}
	return ret;
}



void Configuration_To_Clipboard(State *state, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	state->clipboard_spins = std::shared_ptr<vectorfield>(new vectorfield(*image->spins));
	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
		"Copied spin configuration to clipboard.", idx_image, idx_chain);
}

void Configuration_From_Clipboard(State *state, const float position[3], const float r_cut_rectangular[3], float r_cut_cylindrical, float r_cut_spherical, bool inverted, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Get relative position
	Vector3 _pos{ position[0], position[1], position[2] };
	Vector3 vpos = _pos; // image->geometry->center + _pos;

	// Create position filter
	auto filter = get_filter(vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);

	// Apply configuration
	image->Lock();
	Utility::Configurations::Insert(*image, *state->clipboard_spins, 0, filter);
	image->Unlock();

	auto filterstring = filter_to_string(position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
		"Set spin configuration from clipboard. " + filterstring, idx_image, idx_chain);
}

bool Configuration_From_Clipboard_Shift(State *state, const float position_initial[3], const float position_final[3], const float r_cut_rectangular[3], float r_cut_cylindrical, float r_cut_spherical, bool inverted, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Get relative position
	Vector3 pos_initial{ position_initial[0], position_initial[1], position_initial[2] };
	Vector3 pos_final{ position_final[0], position_final[1], position_final[2] };
	Vector3 shift = pos_initial - pos_final;

	Vector3 decomposed = Engine::Vectormath::decompose(shift, image->geometry->basis);
	
	int da = (int)std::round(decomposed[0]);
	int db = (int)std::round(decomposed[1]);
	int dc = (int)std::round(decomposed[2]);

	if (da == 0 && db == 0 && dc == 0)
		return false;

	auto& geometry = *image->geometry;
	int delta = geometry.n_spins_basic_domain*da + geometry.n_spins_basic_domain*geometry.n_cells[0] * db + geometry.n_spins_basic_domain*geometry.n_cells[0] * geometry.n_cells[1] * dc;

	// Create position filter
	auto filter = get_filter(pos_final, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);

	// Apply configuration
	image->Lock();
	Utility::Configurations::Insert(*image, *state->clipboard_spins, delta, filter);
	image->Unlock();

	auto filterstring = filter_to_string(position_final, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
		"Set shifted spin configuration from clipboard. " + filterstring, idx_image, idx_chain);
	return true;
}


void Configuration_Domain(State *state, const float direction[3], const float position[3], const float r_cut_rectangular[3], float r_cut_cylindrical, float r_cut_spherical, bool inverted, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Get relative position
	Vector3 _pos{ position[0], position[1], position[2] };
	Vector3 vpos = image->geometry->center + _pos;

	// Create position filter
	auto filter = get_filter(vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);

	// Apply configuration
	Vector3 vdir{ direction[0], direction[1], direction[2] };
	image->Lock();
	Utility::Configurations::Domain(*image, vdir, filter);
	image->Unlock();

	auto filterstring = filter_to_string(position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
		"Set domain configuration (" + std::to_string(direction[0]) + "," + std::to_string(direction[1]) + "," + std::to_string(direction[2]) + "). " + filterstring, idx_image, idx_chain);
}


// void Configuration_DomainWall(State *state, const float pos[3], float v[3], bool greater, int idx_image, int idx_chain)
// {
// 	std::shared_ptr<Data::Spin_System> image;
// 	std::shared_ptr<Data::Spin_System_Chain> chain;
// 	from_indices(state, idx_image, idx_chain, image, chain);

// 	// Create position filter
// 	Vector3 vpos{pos[0], pos[1], pos[2]};
// 	std::function< bool(const Vector3&, const Vector3&) > filter = [vpos](const Vector3& spin, const Vector3& position)
// 	{
// 		scalar r = std::sqrt(std::pow(position[0] - vpos[0], 2) + std::pow(position[1] - vpos[1], 2));
// 		if ( r < 3) return true;
// 		return false;
// 	};
// 	// Apply configuration
// 	Utility::Configurations::Domain(*image, Vector3{ v[0],v[1],v[2] }, filter);
// }


void Configuration_PlusZ(State *state, const float position[3], const float r_cut_rectangular[3], float r_cut_cylindrical, float r_cut_spherical, bool inverted, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Get relative position
	Vector3 _pos{ position[0], position[1], position[2] };
	Vector3 vpos = image->geometry->center + _pos;

	// Create position filter
	auto filter = get_filter(vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);

	// Apply configuration
	Vector3 vdir{ 0,0,1 };
	image->Lock();
	Utility::Configurations::Domain(*image, vdir, filter);
	image->Unlock();

	auto filterstring = filter_to_string(position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
		"Set PlusZ configuration. " + filterstring, idx_image, idx_chain);
}

void Configuration_MinusZ(State *state, const float position[3], const float r_cut_rectangular[3], float r_cut_cylindrical, float r_cut_spherical, bool inverted, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Get relative position
	Vector3 _pos{ position[0], position[1], position[2] };
	Vector3 vpos = image->geometry->center + _pos;

	// Create position filter
	auto filter = get_filter(vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);

	// Apply configuration
	Vector3 vdir{ 0,0,-1 };
	image->Lock();
	Utility::Configurations::Domain(*image, vdir, filter);
	image->Unlock();

	auto filterstring = filter_to_string(position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
		"Set MinusZ configuration. " + filterstring, idx_image, idx_chain);
}

void Configuration_Random(State *state, const float position[3], const float r_cut_rectangular[3], float r_cut_cylindrical, float r_cut_spherical, bool inverted, bool external, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Get relative position
	Vector3 _pos{ position[0], position[1], position[2]};
	Vector3 vpos = image->geometry->center + _pos;

	// Create position filter
	auto filter = get_filter(vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);

	// Apply configuration
	image->Lock();
    Utility::Configurations::Random(*image, filter, external);
	image->Unlock();

	auto filterstring = filter_to_string(position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
		"Set random configuration. " + filterstring, idx_image, idx_chain);
}

void Configuration_Add_Noise_Temperature(State *state, float temperature, const float position[3], const float r_cut_rectangular[3], float r_cut_cylindrical, float r_cut_spherical, bool inverted, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Get relative position
	Vector3 _pos{ position[0], position[1], position[2] };
	Vector3 vpos = image->geometry->center + _pos;

	// Create position filter
	auto filter = get_filter(vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);

	// Apply configuration
	image->Lock();
    Utility::Configurations::Add_Noise_Temperature(*image, temperature, 0, filter);
	image->Unlock();

	auto filterstring = filter_to_string(position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
		"Added noise with temperature T="+std::to_string(temperature)+". " + filterstring, idx_image, idx_chain);
}

void Configuration_Hopfion(State *state, float r, int order, const float position[3], const float r_cut_rectangular[3], float r_cut_cylindrical, float r_cut_spherical, bool inverted, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Get relative position
	Vector3 _pos{ position[0], position[1], position[2] };
	Vector3 vpos = image->geometry->center + _pos;

	// Set cutoff radius
	if (r_cut_spherical < 0) r_cut_spherical = r*M_PI;

	// Create position filter
	auto filter = get_filter(vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);

	// Apply configuration
	image->Lock();
	Utility::Configurations::Hopfion(*image, vpos, r, order, filter);
	image->Unlock();

	auto filterstring = filter_to_string(position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
	std::string parameterstring = "r=" + std::to_string(r);
	if (order != 1) parameterstring += ", order=" + std::to_string(order);
	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
		"Set hopfion configuration, " + parameterstring + ". " + filterstring, idx_image, idx_chain);
}

void Configuration_Skyrmion(State *state, float r, float order, float phase, bool upDown, bool achiral, bool rl, const float position[3], const float r_cut_rectangular[3], float r_cut_cylindrical, float r_cut_spherical, bool inverted, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Get relative position
	Vector3 _pos{ position[0], position[1], position[2] };
	Vector3 vpos = image->geometry->center + _pos;

	// Set cutoff radius
	if (r_cut_cylindrical < 0) r_cut_cylindrical = r;

	// Create position filter
	auto filter = get_filter(vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);

    // Apply configuration
	image->Lock();
    Utility::Configurations::Skyrmion(*image, vpos, r, order, phase, upDown, achiral, rl, false, filter);
	image->Unlock();

	auto filterstring = filter_to_string(position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
	std::string parameterstring = "r=" + std::to_string(r);
	if (order != 1) parameterstring += ", order=" + std::to_string(order);
	if (phase != 0) parameterstring += ", phase=" + std::to_string(phase);
	if (upDown != 0) parameterstring += ", upDown=" + std::to_string(upDown);
	if (achiral != 0) parameterstring += ", achiral";
	if (rl != 0) parameterstring += ", rl=" + std::to_string(rl);
	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
		"Set skyrmion configuration, " + parameterstring + ". " + filterstring, idx_image, idx_chain);
}

void Configuration_SpinSpiral(State *state, const char * direction_type, float q[3], float axis[3], float theta, const float position[3], const float r_cut_rectangular[3], float r_cut_cylindrical, float r_cut_spherical, bool inverted, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Get relative position
	Vector3 _pos{ position[0], position[1], position[2] };
	Vector3 vpos = image->geometry->center + _pos;

	// Create position filter
	auto filter = get_filter(vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);

    // Apply configuration
    std::string dir_type(direction_type);
	Vector3 vq{ q[0], q[1], q[2] };
	Vector3 vaxis{ axis[0], axis[1], axis[2] };
	image->Lock();
	Utility::Configurations::SpinSpiral(*image, dir_type, vq, vaxis, theta, filter);
	image->Unlock();

	auto filterstring = filter_to_string(position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
	std::string parameterstring = "W.r.t. " + std::string(direction_type)
		+ ", q=(" + std::to_string(q[0]) + "," + std::to_string(q[1]) + "," + std::to_string(q[2]) + ")"
		+ ", axis=(" + std::to_string(axis[0]) + "," + std::to_string(axis[1]) + "," + std::to_string(axis[2]) + ")"
		+ ", theta=" + std::to_string(theta);
	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
		"Set spin spiral configuration. " + parameterstring + ". " +  filterstring, idx_image, idx_chain);
}