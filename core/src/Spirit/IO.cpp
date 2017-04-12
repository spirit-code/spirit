#include <Spirit/State.h>
#include <Spirit/IO.h>
#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>

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

	float defaultPos[3] = {0,0,0}; 
	float defaultRect[3] = {-1,-1,-1}; 
	state->collection->chains[idx_chain]->images[idx_image] = system;
	Configuration_Random(state, defaultPos, defaultRect, -1, -1, false, false, idx_image, idx_chain);
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
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Read the data
	Utility::IO::Read_SpinChain_Configuration(chain, std::string(file));

	// Update state
	State_Update(state);

	// Update array lengths
	Chain_Setup_Data(state, idx_chain);

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
		"Read chain from file " + std::string(file), -1, idx_chain);
}

void IO_Chain_Write(State * state, const char * file, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Read the data
	Utility::IO::Save_SpinChain_Configuration(chain, std::string(file));
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

//IO_Energies_Spins_Save
void IO_Write_System_Energy_per_Spin(State * state, const char * file, int idx_chain)
{
	int idx_image = -1;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Write the data
	Utility::IO::Write_System_Energy_per_Spin(*image, std::string(file));
}

//IO_Energy_Spins_Save
void IO_Write_System_Energy(State * state, const char * file, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Write the data
	Utility::IO::Write_System_Energy(*image, std::string(file));
}

//IO_Energies_Save
void IO_Write_Chain_Energies(State * state, const char * file, int idx_chain)
{
	int idx_image = -1;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Write the data
	Utility::IO::Write_Chain_Energies(*chain, idx_chain, std::string(file));
}

//IO_Energies_Interpolated_Save
void IO_Write_Chain_Energies_Interpolated(State * state, const char * file, int idx_chain)
{
	int idx_image = -1;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Write the data
	Utility::IO::Write_Chain_Energies_Interpolated(*chain, std::string(file));
}