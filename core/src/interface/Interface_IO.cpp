#include <interface/Interface_IO.h>
#include <interface/Interface_Configurations.h>
#include <data/State.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <utility/IO.hpp>

#include <memory>
#include <string>

/////// TODO: make use of file format specifications
/////// TODO: implement remaining functions

/*------------------------------------------------------------------------------------------------------ */
/*--------------------------------- From Config File --------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

int IO_System_From_Config(State * state, const char * file, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Create System
	std::shared_ptr<Data::Spin_System> system = Utility::IO::Spin_System_from_Config(std::string(file));

	// Filter for unacceptable differences to other systems in the chain
	for (int i = 0; i < chain->noi; ++i)
	{
		if (state->active_chain->images[i]->nos != system->nos) return 0;
		// Currently the SettingsWidget does not support different images being isotropic AND anisotropic at the same time
		if (state->active_chain->images[i]->hamiltonian->Name() != system->hamiltonian->Name()) return 0;
	}

	state->collection->chains[idx_chain]->images[idx_image] = system;
	Configuration_Random(state, false, idx_image, idx_chain);
	return 1;
}


/*------------------------------------------------------------------------------------------------------ */
/*-------------------------------------- Images -------------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

void IO_Image_Read(State * state, const char * file, int format, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    // Read the data
	Utility::IO::Read_Spin_Configuration(image, std::string(file), Utility::IO::VectorFileFormat(format));
}

void IO_Image_Write(State * state, const char * file, int format, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Write the data
	Utility::IO::Append_Spin_Configuration(image, 0, std::string(file));
}

void IO_Image_Append(State * state, const char * file, int iteration, int format, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Write the data
	Utility::IO::Append_Spin_Configuration(image, 0, std::string(file));
}


/*------------------------------------------------------------------------------------------------------ */
/*-------------------------------------- Chains -------------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

void IO_Chain_Read(State * state, const char * file, int idx_image, int idx_chain)
{

}

void IO_Chain_Write(State * state, const char * file, int idx_image, int idx_chain)
{

}


/*------------------------------------------------------------------------------------------------------ */
/*------------------------------------ Collection ------------------------------------------------------ */
/*------------------------------------------------------------------------------------------------------ */

void IO_Collection_Read(State * state, const char * file, int idx_image, int idx_chain)
{

}

void IO_Collection_Write(State * state, const char * file, int idx_image, int idx_chain)
{

}


/*------------------------------------------------------------------------------------------------------ */
/*--------------------------------------- Data --------------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

void IO_Energies_Save(State * state, const char * file, int idx_chain)
{
	int idx_image = -1;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Write the data
	Utility::IO::Save_Energies(*chain, 0, std::string(file));
}


void IO_Energies_Spins_Save(State * state, const char * file, int idx_chain)
{
	int idx_image = -1;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Write the data
	Utility::IO::Save_Energies_Spins(*state->active_chain, std::string(file));
}

void IO_Energies_Interpolated_Save(State * state, const char * file, int idx_chain)
{
	int idx_image = -1;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Write the data
	Utility::IO::Save_Energies_Interpolated(*state->active_chain, std::string(file));
}