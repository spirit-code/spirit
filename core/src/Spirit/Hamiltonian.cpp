#include <Spirit/Hamiltonian.h>

#include <data/State.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Hamiltonian_Heisenberg_Neighbours.hpp>
#include <engine/Hamiltonian_Heisenberg_Pairs.hpp>
#include <engine/Hamiltonian_Gaussian.hpp>
#include <utility/Constants.hpp>

using namespace Utility;

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

void Hamiltonian_Set_Boundary_Conditions(State *state, const bool * periodical, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    image->Lock();

    image->hamiltonian->boundary_conditions[0] = periodical[0];
    image->hamiltonian->boundary_conditions[1] = periodical[1];
    image->hamiltonian->boundary_conditions[2] = periodical[2];

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set boundary conditions to " + std::to_string(periodical[0]) + " " + std::to_string(periodical[1]) + " " + std::to_string(periodical[2]), idx_image, idx_chain);

	image->Unlock();
}

void Hamiltonian_Set_mu_s(State *state, float mu_s, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	image->Lock();

    if (image->hamiltonian->Name() == "Heisenberg (Neighbours)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Neighbours*)image->hamiltonian.get();
        for (auto& m : ham->mu_s) m = mu_s;
        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Set mu_s to " + std::to_string(mu_s), idx_image, idx_chain);
    }
    else if (image->hamiltonian->Name() == "Heisenberg (Pairs)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Pairs*)image->hamiltonian.get();
        for (auto& m : ham->mu_s) m = mu_s;
        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Set mu_s to " + std::to_string(mu_s), idx_image, idx_chain);
    }
    else
        Log(Utility::Log_Level::Warning, Utility::Log_Sender::API,
            "mu_s cannot be set on " + image->hamiltonian->Name(), idx_image, idx_chain);

	image->Unlock();
}

void Hamiltonian_Set_Field(State *state, float magnitude, const float * normal, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	// Lock mutex because simulations may be running
	image->Lock();

	// Set
    if (image->hamiltonian->Name() == "Heisenberg (Neighbours)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Neighbours*)image->hamiltonian.get();
        int nos = image->nos;

        // Indices and Magnitudes
        intfield new_indices(nos);
        scalarfield new_magnitudes(nos);
        for (int i=0; i<nos; ++i)
        {
            new_indices[i] = i;
            new_magnitudes[i] = magnitude *  ham->mu_s[i] * Constants::mu_B;
        }
        // Normals
        Vector3 new_normal{normal[0], normal[1], normal[2]};
        new_normal.normalize();
        vectorfield new_normals(nos, new_normal);
        
        // Into the Hamiltonian
        ham->external_field_indices = new_indices;
        ham->external_field_magnitudes = new_magnitudes;
        ham->external_field_normals = new_normals;

        // Update Energies
        ham->Update_Energy_Contributions();

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Set external field to " + std::to_string(magnitude) + ", direction (" + std::to_string(normal[0]) + "," + std::to_string(normal[1]) + "," + std::to_string(normal[2]) + ")", idx_image, idx_chain);
    }
    else if (image->hamiltonian->Name() == "Heisenberg (Pairs)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Pairs*)image->hamiltonian.get();
        int nos = image->nos;

        // Indices and Magnitudes
        intfield new_indices(nos);
        scalarfield new_magnitudes(nos);
        for (int i=0; i<nos; ++i)
        {
            new_indices[i] = i;
            new_magnitudes[i] = magnitude *  ham->mu_s[i] * Constants::mu_B;
        }
        // Normals
        Vector3 new_normal{normal[0], normal[1], normal[2]};
        new_normal.normalize();
        vectorfield new_normals(nos, new_normal);
        
        // Into the Hamiltonian
        ham->external_field_indices = new_indices;
        ham->external_field_magnitudes = new_magnitudes;
        ham->external_field_normals = new_normals;

        // Update Energies
        ham->Update_Energy_Contributions();

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Set external field to " + std::to_string(magnitude) + ", direction (" + std::to_string(normal[0]) + "," + std::to_string(normal[1]) + "," + std::to_string(normal[2]) + ")", idx_image, idx_chain);

    }
    else
        Log(Utility::Log_Level::Warning, Utility::Log_Sender::API,
            "External field cannot be set on " + image->hamiltonian->Name(), idx_image, idx_chain);
    
	// Unlock mutex
	image->Unlock();
}

void Hamiltonian_Set_Anisotropy(State *state, float magnitude, const float * normal, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	image->Lock();

    if (image->hamiltonian->Name() == "Heisenberg (Neighbours)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Neighbours*)image->hamiltonian.get();
        int nos = image->nos;

		// Indices and Magnitudes
		intfield new_indices(nos);
		scalarfield new_magnitudes(nos);
		for (int i = 0; i<nos; ++i)
		{
			new_indices[i] = i;
			new_magnitudes[i] = magnitude;
		}
		// Normals
		Vector3 new_normal{ normal[0], normal[1], normal[2] };
		new_normal.normalize();
		vectorfield new_normals(nos, new_normal);

		// Into the Hamiltonian
		ham->anisotropy_indices = new_indices;
		ham->anisotropy_magnitudes = new_magnitudes;
		ham->anisotropy_normals = new_normals;

		// Update Energies
		ham->Update_Energy_Contributions();

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Set anisotropy to " + std::to_string(magnitude) + ", direction (" + std::to_string(normal[0]) + "," + std::to_string(normal[1]) + "," + std::to_string(normal[2]) + ")", idx_image, idx_chain);
    }
    else if (image->hamiltonian->Name() == "Heisenberg (Pairs)")
    {
		auto ham = (Engine::Hamiltonian_Heisenberg_Pairs*)image->hamiltonian.get();
		int nos = image->nos;

		// Indices and Magnitudes
		intfield new_indices(nos);
		scalarfield new_magnitudes(nos);
		for (int i = 0; i<nos; ++i)
		{
			new_indices[i] = i;
			new_magnitudes[i] = magnitude;
		}
		// Normals
		Vector3 new_normal{ normal[0], normal[1], normal[2] };
		new_normal.normalize();
		vectorfield new_normals(nos, new_normal);

		// Into the Hamiltonian
		ham->anisotropy_indices = new_indices;
		ham->anisotropy_magnitudes = new_magnitudes;
		ham->anisotropy_normals = new_normals;

		// Update Energies
		ham->Update_Energy_Contributions();

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Set anisotropy to " + std::to_string(magnitude) + ", direction (" + std::to_string(normal[0]) + "," + std::to_string(normal[1]) + "," + std::to_string(normal[2]) + ")", idx_image, idx_chain);
    }
    else
        Log(Utility::Log_Level::Warning, Utility::Log_Sender::API,
            "Anisotropy cannot be set on " + image->hamiltonian->Name(), idx_image, idx_chain);

	image->Unlock();
}

void Hamiltonian_Set_Exchange(State *state, int n_shells, const float* jij, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	image->Lock();

    if (image->hamiltonian->Name() == "Heisenberg (Neighbours)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Neighbours*)image->hamiltonian.get();

		ham->exchange_magnitudes = scalarfield(n_shells);
        for (int i=0; i<n_shells; ++i)
        {
            ham->exchange_magnitudes[i] = jij[i];
        }
		ham->exchange_neighbours = Engine::Neighbours::Get_Neighbours_in_Shells(*image->geometry, n_shells);

        ham->Update_Energy_Contributions();

		std::string message = "Set exchange to " + std::to_string(n_shells) + " shells";
		if (n_shells > 0) message += " Jij[0] = " + std::to_string(jij[0]);
        Log(Utility::Log_Level::Info, Utility::Log_Sender::API, message, idx_image, idx_chain);
    }
    else if (image->hamiltonian->Name() == "Heisenberg (Pairs)")
    {
		// Get the necessary pairs list
		auto neighbours = Engine::Neighbours::Get_Neighbours_in_Shells(*image->geometry, n_shells);
		pairfield pairs(0);
		scalarfield magnitudes(0);
		for (auto& neigh : neighbours)
		{
			pairs.push_back({ neigh.iatom, neigh.ineigh, {neigh.translations[0], neigh.translations[1], neigh.translations[2]} });
			magnitudes.push_back({ (scalar)0.5*jij[neigh.idx_shell] });
		}

		// Set Hamiltonian's arrays
		auto ham = (Engine::Hamiltonian_Heisenberg_Pairs*)image->hamiltonian.get();
		ham->exchange_pairs = pairs;
		ham->exchange_magnitudes = magnitudes;

		// Update the list of different contributions
		ham->Update_Energy_Contributions();

		std::string message = "Set exchange to " + std::to_string(n_shells) + " shells";
		if (n_shells > 0) message += " Jij[0] = " + std::to_string(jij[0]);
        Log(Utility::Log_Level::Info, Utility::Log_Sender::API, message, idx_image, idx_chain);
    }
    else
        Log(Utility::Log_Level::Warning, Utility::Log_Sender::API,
            "Exchange cannot be set on " + image->hamiltonian->Name(), idx_image, idx_chain);


	image->Unlock();
}

void Hamiltonian_Set_DMI(State *state, int n_shells, const float * dij, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	image->Lock();

    if (image->hamiltonian->Name() == "Heisenberg (Neighbours)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Neighbours*)image->hamiltonian.get();

		ham->dmi_magnitudes = scalarfield(n_shells);
        for (int i=0; i<n_shells; ++i)
        {
            ham->dmi_magnitudes[i] = dij[i];
        }
		ham->dmi_neighbours = Engine::Neighbours::Get_Neighbours_in_Shells(*image->geometry, n_shells);
		ham->dmi_normals = vectorfield(0);
		for (unsigned int ineigh = 0; ineigh < ham->dmi_neighbours.size(); ++ineigh)
		{
			ham->dmi_normals.push_back(Engine::Neighbours::DMI_Normal_from_Pair(*image->geometry, { ham->dmi_neighbours[ineigh].iatom, ham->dmi_neighbours[ineigh].ineigh, {ham->dmi_neighbours[ineigh].translations[0], ham->dmi_neighbours[ineigh].translations[1], ham->dmi_neighbours[ineigh].translations[2]} }, 1));
		}

        ham->Update_Energy_Contributions();

		std::string message = "Set dmi to " + std::to_string(n_shells) + " shells";
		if (n_shells > 0) message += " Dij[0] = " + std::to_string(dij[0]);
        Log(Utility::Log_Level::Info, Utility::Log_Sender::API, message, idx_image, idx_chain);
    }
    else if (image->hamiltonian->Name() == "Heisenberg (Pairs)")
    {
		// Get the necessary pairs list
		auto neighbours = Engine::Neighbours::Get_Neighbours_in_Shells(*image->geometry, n_shells);
		pairfield pairs(0);
		scalarfield magnitudes(0);
		vectorfield normals(0);
		for (auto& neigh : neighbours)
		{
			pairs.push_back({ neigh.iatom, neigh.ineigh, {neigh.translations[0], neigh.translations[1], neigh.translations[2]} });
			magnitudes.push_back({ (scalar)0.5*dij[neigh.idx_shell] });
			normals.push_back({ Engine::Neighbours::DMI_Normal_from_Pair(*image->geometry, pairs.back(), 1) });
		}

		// Set Hamiltonian's arrays
		auto ham = (Engine::Hamiltonian_Heisenberg_Pairs*)image->hamiltonian.get();
		ham->dmi_pairs = pairs;
		ham->dmi_magnitudes = magnitudes;
		ham->dmi_normals = normals;

		// Update the list of different contributions
		ham->Update_Energy_Contributions();

		std::string message = "Set dmi to " + std::to_string(n_shells) + " shells";
		if (n_shells > 0) message += " Dij[0] = " + std::to_string(dij[0]);
        Log(Utility::Log_Level::Info, Utility::Log_Sender::API, message, idx_image, idx_chain);
    }
    else
        Log(Utility::Log_Level::Warning, Utility::Log_Sender::API, "DMI cannot be set on " + image->hamiltonian->Name(), idx_image, idx_chain);

	image->Unlock();
}

void Hamiltonian_Set_DDI(State *state, float radius, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	image->Lock();

    if (image->hamiltonian->Name() == "Heisenberg (Neighbours)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Neighbours*)image->hamiltonian.get();

        ham->ddi_radius = radius;
		auto neighbours = Engine::Neighbours::Get_Neighbours_in_Radius(*image->geometry, radius);
		scalarfield magnitudes(0);
		vectorfield normals(0);
		scalar magnitude;
		Vector3 normal;
		for (unsigned int i=0; i<neighbours.size(); ++i)
		{
		    Engine::Neighbours::DDI_from_Pair(*image->geometry, {neighbours[i].iatom, neighbours[i].ineigh, {neighbours[i].translations[0], neighbours[i].translations[1], neighbours[i].translations[2]}}, magnitude, normal);
			magnitudes.push_back(magnitude);
			normals.push_back(normal);
		}
        ham->ddi_neighbours = neighbours;
        ham->ddi_magnitudes = magnitudes;
        ham->ddi_normals = normals;

		// Update the list of different contributions
		ham->Update_Energy_Contributions();

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API, "Set ddi radius to " + std::to_string(radius), idx_image, idx_chain);
    }
    else if (image->hamiltonian->Name() == "Heisenberg (Pairs)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Pairs*)image->hamiltonian.get();

		auto pairs = Engine::Neighbours::Get_Pairs_in_Radius(*image->geometry, radius);
		scalarfield magnitudes(0);
		vectorfield normals(0);
        scalar magnitude;
        Vector3 normal;
		for (auto& pair : pairs)
		{
		    Engine::Neighbours::DDI_from_Pair(*image->geometry, pair, magnitude, normal);
			magnitudes.push_back(magnitude);
			normals.push_back(normal);
		}
		ham->ddi_pairs = pairs;
		ham->ddi_magnitudes = magnitudes;
		ham->ddi_normals = normals;

		// Update the list of different contributions
		ham->Update_Energy_Contributions();

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API, "Set ddi radius to " + std::to_string(radius), idx_image, idx_chain);
    }
    else
        Log(Utility::Log_Level::Warning, Utility::Log_Sender::API, "DDI cannot be set on " + image->hamiltonian->Name(), idx_image, idx_chain);


    image->Unlock();
}

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

const char * Hamiltonian_Get_Name(State * state, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	return image->hamiltonian->Name().c_str();
}

void Hamiltonian_Get_Boundary_Conditions(State *state, bool * periodical, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    periodical[0] = image->hamiltonian->boundary_conditions[0];
    periodical[1] = image->hamiltonian->boundary_conditions[1];
    periodical[2] = image->hamiltonian->boundary_conditions[2];
}

void Hamiltonian_Get_mu_s(State *state, float * mu_s, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Heisenberg (Neighbours)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Neighbours*)image->hamiltonian.get();
        for (int i=0; i<image->geometry->n_spins_basic_domain; ++i)
            mu_s[i] = (float)ham->mu_s[i];
    }
    else if (image->hamiltonian->Name() == "Heisenberg (Pairs)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Pairs*)image->hamiltonian.get();
        for (int i=0; i<image->geometry->n_spins_basic_domain; ++i)
            mu_s[i] = (float)ham->mu_s[i];
    }
}

void Hamiltonian_Get_Field(State *state, float * magnitude, float * normal, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Heisenberg (Neighbours)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Neighbours*)image->hamiltonian.get();

        if (ham->external_field_indices.size() > 0)
        {
            // Magnitude
            *magnitude = (float)(ham->external_field_magnitudes[0] / ham->mu_s[0] / Constants::mu_B);
            
            // Normal
            normal[0] = (float)ham->external_field_normals[0][0];
            normal[1] = (float)ham->external_field_normals[0][1];
            normal[2] = (float)ham->external_field_normals[0][2];
        }
		else
		{
			*magnitude = 0;
			normal[0] = 0;
			normal[1] = 0;
			normal[2] = 1;
		}
    }
    else if (image->hamiltonian->Name() == "Heisenberg (Pairs)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Pairs*)image->hamiltonian.get();

        if (ham->external_field_indices.size() > 0)
        {
            // Magnitude
            *magnitude = (float)(ham->external_field_magnitudes[0] / ham->mu_s[0] / Constants::mu_B);

            // Normal
            normal[0] = (float)ham->external_field_normals[0][0];
            normal[1] = (float)ham->external_field_normals[0][1];
            normal[2] = (float)ham->external_field_normals[0][2];
        }
		else
		{
			*magnitude = 0;
			normal[0] = 0;
			normal[1] = 0;
			normal[2] = 1;
		}
    }
}

void Hamiltonian_Get_Anisotropy(State *state, float * magnitude, float * normal, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Heisenberg (Neighbours)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Neighbours*)image->hamiltonian.get();

        if (ham->anisotropy_indices.size() > 0)
        {
            // Magnitude
            *magnitude = (float)ham->anisotropy_magnitudes[0];
            
            // Normal
            normal[0] = (float)ham->anisotropy_normals[0][0];
            normal[1] = (float)ham->anisotropy_normals[0][1];
            normal[2] = (float)ham->anisotropy_normals[0][2];
        }
		else
		{
			*magnitude = 0;
			normal[0] = 0;
			normal[1] = 0;
			normal[2] = 1;
		}
    }
    else if (image->hamiltonian->Name() == "Heisenberg (Pairs)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Pairs*)image->hamiltonian.get();
        
        if (ham->anisotropy_indices.size() > 0)
        {
            // Magnitude
            *magnitude = (float)ham->anisotropy_magnitudes[0];

            // Normal
            normal[0] = (float)ham->anisotropy_normals[0][0];
            normal[1] = (float)ham->anisotropy_normals[0][1];
            normal[2] = (float)ham->anisotropy_normals[0][2];
        }
		else
		{
			*magnitude = 0;
			normal[0] = 0;
			normal[1] = 0;
			normal[2] = 1;
		}
    }
}

void Hamiltonian_Get_Exchange(State *state, int * n_shells, float * jij, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Heisenberg (Neighbours)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Neighbours*)image->hamiltonian.get();

        *n_shells = ham->exchange_magnitudes.size();

        // Note the array needs to be correctly allocated beforehand!
        for (int i=0; i<*n_shells; ++i)
        {
            jij[i] = (float)ham->exchange_magnitudes[i];
        }
    }
    else if (image->hamiltonian->Name() == "Heisenberg (Pairs)")
    {
        // TODO
    }
}

void Hamiltonian_Get_DMI(State *state, int * n_shells, float * dij, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Heisenberg (Neighbours)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Neighbours*)image->hamiltonian.get();

        *n_shells = ham->dmi_magnitudes.size();

        for (int i=0; i<*n_shells; ++i)
        {
            dij[i] = (float)ham->dmi_magnitudes[i];
        }
    }
    else if (image->hamiltonian->Name() == "Heisenberg (Pairs)")
    {
        // TODO
    }
}

void Hamiltonian_Get_DDI(State *state, float * radius, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Heisenberg (Neighbours)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Neighbours*)image->hamiltonian.get();

        *radius = (float)ham->ddi_radius;
    }
    else if (image->hamiltonian->Name() == "Heisenberg (Pairs)")
    {
        auto ham = (Engine::Hamiltonian_Heisenberg_Neighbours*)image->hamiltonian.get();

        *radius = (float)ham->ddi_radius;
    }
}