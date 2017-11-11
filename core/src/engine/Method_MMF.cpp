#include <Spirit_Defines.h>
#include <engine/Method_MMF.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <io/IO.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
//#include <unsupported/Eigen/CXX11/Tensor>
#include <GenEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>
#include <GenEigsRealShiftSolver.h>

#include <fmt/format.h>

using Utility::Log_Level;
using Utility::Log_Sender;

namespace Engine
{
	template <Solver solver>
    Method_MMF<solver>::Method_MMF(std::shared_ptr<Data::Spin_System_Chain_Collection> collection, int idx_chain) :
        Method_Solver<solver>(collection->parameters, -1, idx_chain), collection(collection)
    {
		int noc = collection->noc;
		int nos = collection->chains[0]->images[0]->nos;
		switched1 = false;
		switched2 = false;
		this->SenderName = Utility::Log_Sender::MMF;
        
        // The systems we use are the last image of each respective chain
        for (int ichain = 0; ichain < noc; ++ichain)
		{
			this->systems.push_back(this->collection->chains[ichain]->images.back());
		}

		// History
        this->history = std::map<std::string, std::vector<scalar>>{
			{"max_torque_component", {this->force_max_abs_component}} };

		// We assume that the systems are not converged before the first iteration
		this->force_max_abs_component = this->collection->parameters->force_convergence + 1.0;

		this->hessian = std::vector<MatrixX>(noc, MatrixX(3*nos, 3*nos));	// [noc][3nos]
		// Forces
		this->gradient   = std::vector<vectorfield>(noc, vectorfield(nos));	// [noc][3nos]
		this->minimum_mode = std::vector<vectorfield>(noc, vectorfield(nos));	// [noc][3nos]
		this->xi = vectorfield(this->nos, {0,0,0});

		// Last iteration
		this->spins_last = std::vector<vectorfield>(noc, vectorfield(nos));	// [noc][3nos]
		this->spins_last[0] = *this->systems[0]->spins;
		this->Rx_last = 0.0;

		// Force function
		// ToDo: move into parameters
		this->mm_function = "Spectra Matrix"; // "Spectra Matrix" "Spectra Prefactor" "Lanczos"
    }
	

	template <Solver solver>
    void Method_MMF<solver>::Calculate_Force(const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces)
    {
		if (this->mm_function == "Spectra Matrix")
		{
			this->Calculate_Force_Spectra_Matrix(configurations, forces);
		}
		/*else if (this->mm_function == "Spectra Prefactor")
		{
			this->Calculate_Force_Spectra_Prefactor(configurations, forces);
		}
		else if (this->mm_function == "Lanczos")
		{
			this->Calculate_Force_Lanczos(configurations, forces);
		}*/
		#ifdef SPIRIT_ENABLE_PINNING
			Vectormath::set_c_a(1, forces[0], forces[0], this->parameters->pinning->mask_unpinned);
		#endif // SPIRIT_ENABLE_PINNING
    }

	MatrixX projector(vectorfield & image)
	{
		int nos = image.size();
		int size = 3*nos;

		// Get projection matrix M=1-S, blockwise S=x*x^T
		MatrixX proj = MatrixX::Identity(size, size);
		for (int i = 0; i < nos; ++i)
		{
			proj.block<3, 3>(3*i, 3*i) -= image[i] * image[i].transpose();
		}

		return proj;
	}

	template <Solver solver>
	void Method_MMF<solver>::Calculate_Force_Spectra_Matrix(const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces)
	{
		const int nos = configurations[0]->size();
		// std::cerr << "mmf iteration" << std::endl;
		
		// Loop over chains and calculate the forces
		for (int ichain = 0; ichain < this->collection->noc; ++ichain)
		{
			auto& image = *configurations[ichain];
			auto& hess = hessian[ichain];

			// Copy std::vector<Eigen::Vector3> into one single Eigen::VectorX
			Eigen::Ref<VectorX> x = Eigen::Map<VectorX>(image[0].data(), 3 * nos);
			
			// The gradient (unprojected)
			this->systems[ichain]->hamiltonian->Gradient(image, gradient[ichain]);

			// The Hessian (unprojected)
			this->systems[ichain]->hamiltonian->Hessian(image, hess);

			// The projection matrix
			auto P = projector(image);

			// std::cerr << "------------------------" << std::endl;
			// std::cerr << "x:             " << x.transpose() << std::endl;
			// std::cerr << "gradient:      " << gradient.transpose() << std::endl;
			// std::cerr << "hessian:       " << std::endl << hess << std::endl;
			// std::cerr << "projector:     " << std::endl << P << std::endl;
			// std::cerr << "gradient proj: " << std::endl << (P*grad).transpose() << std::endl;
			// std::cerr << "hessian proj:  " << std::endl << P*hess*P << std::endl;

			// Eigen::EigenSolver<Eigen::MatrixXd> estest1(hess);
			// td::cerr << "hessian:    " << std::endl << hess << std::endl;
			// std::cerr << "eigen vals: " << std::endl << estest1.eigenvalues() << std::endl;
			// std::cerr << "eigen vecs: " << std::endl << estest1.eigenvectors().real() << std::endl;


			/*
			/  Remove Hessian's components in the basis of the image (project it into tangent space)
			/      and add the gradient contributions (inner and outer product)
			*/

			// Correction components
			// TODO: write Kernel for this
			//     the following is equivalent to hess = P*hess*P but that could not be parallelized on the GPU
			// for (int i = 0; i < nos; ++i)
			// {
			// 	for (int j = 0; j < nos; ++j)
			// 	{
			// 		hess.block<3, 3>(3*i, 3*j) = P.block<3, 3>(3*i, 3*i) * hess.block<3, 3>(3*i, 3*j) * P.block<3, 3>(3*j, 3*j);
			// 	}
			// }

			// // Hessian projection components (diagonal contribution blocks)
			// // TODO: write Kernel for this
			// for (int i = 0; i < nos; ++i)
			// {
			// 	hess.block<3, 3>(3*i, 3*i) -= P.block<3, 3>(3*i, 3*i)*(image[i].dot(gradient[ichain][i])) + (P.block<3, 3>(3*i, 3*i)*gradient[ichain][i])*image[i].transpose();
			// }
			
			// // std::cerr << "hessian final: " << std::endl << hess << std::endl;


    		// // Eigen::EigenSolver<Eigen::MatrixXd> estest1(hess);
			// // Eigen::VectorXd estestvals = estest1.eigenvalues().real();
			// // std::sort(estestvals.data(),estestvals.data()+estestvals.size());
			// // std::cerr << "eigen vals (" << estestvals.size() << "): " << std::endl << estestvals.transpose() << std::endl;
			// // int n_zero=0;
			// // for (int _i=0; _i<estestvals.size(); ++_i)
			// // {
			// // 	if (std::abs(estestvals[_i]) < 1e-7) ++n_zero;
			// // }
			// // std::cerr << std::endl << "number of zero eigenvalues: " << n_zero << std::endl;

			// // Get the lowest Eigenvector
			// //		Create a Spectra solver
			// Spectra::DenseGenMatProd<scalar> op(hess);
			// Spectra::GenEigsSolver< scalar, Spectra::SMALLEST_REAL, Spectra::DenseGenMatProd<scalar> > hessian_spectrum(&op, 1, 3*nos);
			// hessian_spectrum.init();
			// //		Compute the specified spectrum
			// int nconv = hessian_spectrum.compute();
			// if (hessian_spectrum.info() == Spectra::SUCCESSFUL)
			// {
			// 	// Calculate the Force
			// 	// 		Retrieve the Eigenvalues
			// 	VectorX evalues = hessian_spectrum.eigenvalues().real();
				
			// 	//std::cerr << "spectra val: " << std::endl << hessian_spectrum.eigenvalues() << std::endl;
			// 	//std::cerr << "spectra vec: " << std::endl << -reorder_vector(hessian_spectrum.eigenvectors().col(0).real()) << std::endl;
				
			// 	// 		Check if the lowest eigenvalue is negative
			// 	if (evalues[0] < -1e-5)// || switched2)
			// 	{
			// 		// std::cerr << "negative region " << evalues.transpose() << std::endl;
			// 		if (switched1)
			// 			switched2 = true;
			// 		// Retrieve the Eigenvectors
			// 		MatrixX evectors = hessian_spectrum.eigenvectors().real();
			// 		Eigen::Ref<VectorX> evec = evectors.col(0);

			// 		// The following line does not seem to work with Eigen3.3
			// 		// this->minimum_mode[ichain] = vectorfield(evec.data(), evec.data() + evec.rows()*evec.cols());
			// 		// TODO: use Eigen's Map or Ref instead
			// 		for (int n=0; n<nos; ++n)
			// 		{
			// 			this->minimum_mode[ichain][n] = {evec[3*n], evec[3*n+1], evec[3*n+2]};
			// 		}

			// 		scalar check = Vectormath::dot(minimum_mode[ichain], image);
			// 		scalar checknorm = Manifoldmath::norm(minimum_mode[ichain]);
			// 		if (std::abs(check) > 1e-8 || std::abs(checknorm) < 1e-8)
			// 		{
			// 			std::cerr << "-------------------------" << std::endl;
			// 			std::cerr << "BAD MODE! evalue = " << evalues[0] << std::endl;
			// 			std::cerr << "-------------------------" << std::endl;
			// 		}


			// 		//for (int _i = 0; _i < forces[ichain].size(); ++_i) minimum_mode[ichain][_i] = -minimum_mode[ichain][_i];
			// 		// 		Normalize the mode vector in 3N dimensions
			// 		Engine::Manifoldmath::normalize(this->minimum_mode[ichain]);
					
					
			// 		// Invert the gradient force along the minimum mode
			// 		Engine::Manifoldmath::invert_parallel(gradient[ichain], minimum_mode[ichain]);
			// 		// Copy out the forces
			// 		for (unsigned int _i = 0; _i < forces[ichain].size(); ++_i) forces[ichain][_i] = -gradient[ichain][_i];
			// 		//forces[ichain] = gradient[ichain];
					
			// 		// std::cerr << "grad      " << gradient[ichain][0] << " " << gradient[ichain][1] << " " << gradient[ichain][2] << std::endl;
			// 		// std::cerr << "min mode: " << std::endl << evec << std::endl;
			// 		// std::cerr << "mode      " << minimum_mode[ichain][0] << " " << minimum_mode[ichain][1] << " " << minimum_mode[ichain][2] << std::endl;
			// 		// std::cerr << "force     " << forces[ichain][0] << " " << forces[ichain][1] << " " << forces[ichain][2] << std::endl;
			// 	}
			// 	//		Otherwise we seek for the lowest nonzero eigenvalue
			// 	else
			// 	{
			// 		switched1 = true;
			// 		// std::cerr << "positive region " << evalues.transpose() << std::endl;
			// 		///////////////////////////////////
			// 		//// The Eigen way... calculate them all... Inefficient! Do it the spectra way instead!
			// 		//Eigen::EigenSolver<Eigen::MatrixXd> estest1(hess);
			// 		//std::cerr << "hessian:    " << std::endl << hess << std::endl;
			// 		//std::cerr << "eigen vals: " << std::endl << estest1.eigenvalues() << std::endl;
			// 		//std::cerr << "eigen vecs: " << std::endl << reorder_matrix(estest1.eigenvectors().real()) << std::endl;
			// 		///////////////////////////////////

			// 		// // Create the Minimum Mode
			// 		// Spectra::DenseGenRealShiftSolve<scalar> op_pos(hess);
			// 		// Spectra::GenEigsRealShiftSolver< scalar, Spectra::SMALLEST_REAL, Spectra::DenseGenRealShiftSolve<scalar> > hessian_spectrum_pos(&op_pos, 1, 3 * nos, 1.0e-5);
			// 		// hessian_spectrum_pos.init();
			// 		// //		Compute the specified spectrum
			// 		// int nconv = hessian_spectrum_pos.compute();
			// 		// // 		Retrieve the Eigenvectors
			// 		// //std::cerr << "-----" << std::endl;
			// 		// //std::cerr << "spectra:" << std::endl;
			// 		// //std::cerr << hessian_spectrum_pos.eigenvalues() << std::endl;
			// 		// //std::cerr << hessian_spectrum_pos.eigenvectors().real() << std::endl;
			// 		// //std::cerr << "-----" << std::endl;
			// 		// Eigen::MatrixXd evectors = hessian_spectrum_pos.eigenvectors().real();
			// 		// //auto evec = evectors.col(0);
			// 		// Eigen::Ref<Eigen::VectorXd> evec = evectors.col(0);

			// 		// for (int n=0; n<nos; ++n)
			// 		// {
			// 		// 	this->minimum_mode[ichain][n] = {evec[3*n], evec[3*n+1], evec[3*n+2]};
			// 		// }

			// 		// Engine::Manifoldmath::normalize(this->minimum_mode[ichain]);


			// 		// //this->minimum_mode[ichain] = std::vector<scalar>(evec.data(), evec.data() + evec.rows()*evec.cols());
			// 		// ////for (int _i = 0; _i < forces[ichain].size(); ++_i) minimum_mode[ichain][_i] = -minimum_mode[ichain][_i];
			// 		// //// 		Normalize the mode vector in 3N dimensions
			// 		// //Utility::Vectormath::Normalize(this->minimum_mode[ichain]);

			// 		// ////// 		Copy via assignment
			// 		// ////this->minimum_mode[ichain] = std::vector<scalar>(evec_min.data(), evec_min.data() + evec_min.rows()*evec_min.cols());
			// 		// ////// 		Normalize the mode vector in 3N dimensions
			// 		// ////Utility::Vectormath::Normalize(this->minimum_mode[ichain]);
			// 		// ////// We are too close to the local minimum so we have to use a strong force
			// 		// //////		We apply the force against the minimum mode of the positive eigenvalue
			// 		// scalar v1v2 = 0.0;
			// 		// for (int i = 0; i < nos; ++i)
			// 		// {
			// 		// 	v1v2 += gradient[ichain][i].dot(minimum_mode[ichain][i]);
			// 		// }
			// 		// // Take out component in direction of v2
			// 		// for (int i = 0; i < nos; ++i)
			// 		// {
			// 		// 	gradient[ichain][i] =  -v1v2 * minimum_mode[ichain][i]; // -gradient[ichain][i]
			// 		// }

			// 		// for (int i = 0; i < nos; ++i)
			// 		// {
			// 		// 	forces[ichain][i] =  -gradient[ichain][i];
			// 		// }

			// 		// Copy out the forces
			// 		forces[ichain] = gradient[ichain];
			// 	}
			// 	//std::cerr << "------------------------" << std::endl;
			// }
			// else
			// {
			// 	Log(Log_Level::Error, Log_Sender::MMF, "Failed to calculate eigenvectors of the Hessian!");
			// 	Log(Log_Level::Info, Log_Sender::MMF, "Zeroing the MMF force...");
			// 	for (Vector3 x : forces[ichain]) x.setZero();
			// }
		}
		// std::cerr << "mmf iteration done" << std::endl;
	}

	

	void printmatrix(MatrixX & m)
	{
		std::cerr << m << std::endl;
	}

		
    // Check if the Forces are converged
	template <Solver solver>
    bool Method_MMF<solver>::Converged()
    {
		if (this->force_max_abs_component < this->collection->parameters->force_convergence) return true;
		return false;
    }

	template <Solver solver>
    void Method_MMF<solver>::Hook_Pre_Iteration()
    {

	}

	template <Solver solver>
    void Method_MMF<solver>::Hook_Post_Iteration()
    {
        // --- Convergence Parameter Update
		this->force_max_abs_component = 0;
		for (int ichain = 0; ichain < collection->noc; ++ichain)
		{
			scalar fmax = this->Force_on_Image_MaxAbsComponent(*(this->systems[ichain]->spins), gradient[ichain]);
			if (fmax > this->force_max_abs_component) this->force_max_abs_component = fmax;
		}

        // --- Update the chains' last images
		for (auto system : this->systems) system->UpdateEnergy();
		for (auto chain : collection->chains)
		{
			int i = chain->noi - 1;
			if (i>0) chain->Rx[i] = chain->Rx[i - 1] + Engine::Manifoldmath::dist_geodesic(*chain->images[i]->spins, *chain->images[i-1]->spins);
		}
    }

	template <Solver solver>
    void Method_MMF<solver>::Save_Current(std::string starttime, int iteration, bool initial, bool final)
	{
		// History save
        this->history["max_torque_component"].push_back(this->force_max_abs_component);

		// File save
		if (this->parameters->output_any)
		{
			// Convert indices to formatted strings
            std::string s_img = fmt::format("{:0>2}", this->idx_image);

			std::string preSpinsFile;
			std::string preEnergyFile;
            std::string fileTag;
            
			if (this->collection->parameters->output_file_tag == "<time>")
                fileTag = starttime + "_";
            else if (this->collection->parameters->output_file_tag != "")
                fileTag = this->collection->parameters->output_file_tag + "_";
            else
                fileTag = "";
            
			preSpinsFile = this->parameters->output_folder + "/" + fileTag + "Spins_" + s_img;
			preEnergyFile = this->parameters->output_folder + "/" + fileTag + "Energy_" + s_img;
			
			// Function to write or append image and energy files
			auto writeOutputConfiguration = [this, preSpinsFile, preEnergyFile, iteration](std::string suffix, bool append)
			{
				// File name and comment
				std::string spinsFile = preSpinsFile + suffix + ".txt";
                std::string comment = std::to_string( iteration );
				// Spin Configuration
                IO::Write_Spin_Configuration( *( this->systems[0] )->spins, 
                                              *( this->systems[0] )->geometry, spinsFile, 
                                              IO::VF_FileFormat::SPIRIT_WHITESPACE_SPIN, 
                                              comment, append );
			};

			auto writeOutputEnergy = [this, preSpinsFile, preEnergyFile, iteration](std::string suffix, bool append)
			{
				int base = (int)log10(this->parameters->n_iterations);
				std::string s_iter = fmt::format("{:0>"+fmt::format("{}",base)+"}", iteration);
				bool normalize = this->systems[0]->llg_parameters->output_energy_divide_by_nspins;

				// File name
				std::string energyFile = preEnergyFile + suffix + ".txt";
				std::string energyFilePerSpin = preEnergyFile + suffix + "_perSpin.txt";

				// Energy
				// Check if Energy File exists and write Header if it doesn't
				std::ifstream f(energyFile);
				if (!f.good()) IO::Write_Energy_Header(*this->systems[0], energyFile);
				// Append Energy to File
				//IO::Append_Image_Energy(*this->systems[0], iteration, energyFile, normalize);

				//
				scalar Rx = Rx_last + Engine::Manifoldmath::dist_geodesic(spins_last[0], *this->systems[0]->spins);
				spins_last[0] = *this->systems[0]->spins;
				Rx_last = Rx;
				//
				scalar nd = 1.0;
				if (this->collection->parameters->output_energy_divide_by_nspins) nd /= this->systems[0]->nos; // nos divide
				std::string output_to_file = s_iter + fmt::format("    {18.10f}    {18.10f}\n", Rx, this->systems[0]->E * nd);
				IO::Append_String_to_File(output_to_file, energyFile);
			};


			// Initial image before simulation
			if (initial && this->parameters->output_initial)
			{
				writeOutputConfiguration("_initial", false);
				writeOutputEnergy("_initial", false);
			}
			// Final image after simulation
			else if (final && this->parameters->output_final)
			{
				writeOutputConfiguration("_final", false);
				writeOutputEnergy("_final", false);
			}

			// Single file output
            int base = (int)log10(this->parameters->n_iterations);
            std::string s_iter = fmt::format("{:0>"+fmt::format("{}",base)+"}", iteration);
			if (this->systems[0]->llg_parameters->output_configuration_step)
			{
				writeOutputConfiguration("_" + s_iter, false);
			}
			if (this->systems[0]->llg_parameters->output_energy_step)
			{
				writeOutputEnergy("_" + s_iter, false);
			}

			// Archive file output (appending)
			if (this->systems[0]->llg_parameters->output_configuration_archive)
			{
				writeOutputConfiguration("_archive", true);
			}
			if (this->systems[0]->llg_parameters->output_energy_archive)
			{
				writeOutputEnergy("_archive", true);
			}


			//if (initial && this->parameters->output_initial) return;

			// Insert copies of the current systems into their corresponding chains
			// - this way we will be able to look at the history of the optimizations
			// for (int ichain=0; ichain<collection->noc; ++ichain)
			// {
			//     // Copy the image
			//	   this->systems[ichain]->Lock()
			//     auto copy = std::shared_ptr<Data::Spin_System>(new Data::Spin_System(*this->systems[ichain]));
			//	   this->systems[ichain]->Unlock()
				
			//     // Insert into chain
			//     auto chain = collection->chains[ichain];
			//     chain->noi++;
			//     chain->images.insert(chain->images.end(), copy);
			//     chain->climbing_image.insert(chain->climbing_image.end(), false);
			//     chain->falling_image.insert(chain->falling_image.end(), false);
			// }

			// Reallocate and recalculate the chains' Rx, E and interpolated values for their last two images

			// Append Each chain's new image to it's corresponding archive

			// In the final save, we save all chains to file?
			//if (final && this->parameters->output_final)
			//{

			//}

			//auto writeoutput = [this, starttime, iteration](std::string suffix)
			//{
			//	// Convert indices to formatted strings
			//	auto s_img = IO::int_to_formatted_string(this->idx_image, 2);
			//	auto s_iter = IO::int_to_formatted_string(iteration, 6);

			//	// if (this->parameters->output_archive)
			//	// {
			//		// Append Spin configuration to Spin_Archieve_File
			//		std::string spinsFile, energyFile;
			//		if (this->collection->parameters->output_tag_time)
			//			spinsFile = this->parameters->output_folder + "/" + starttime + "_Spins_" + s_img + suffix + ".txt";
			//		else
			//			spinsFile = this->parameters->output_folder + "/Spins_" + s_img + suffix + ".txt";
			//		IO::Write_Spin_Configuration( *(this->systems[0])->spins, iteration, spinsFile, 
            //                                     true );
			//		
			//		if (this->collection->parameters->output_energy)
			//		{
			//			// Append iteration, Rx and E to Energy file
			//			scalar nd = 1.0 / this->systems[0]->nos; // nos divide
			//			const int buffer_length = 200;
			//			std::string output_to_file = "";
			//			output_to_file.reserve(int(1E+08));
			//			char buffer_string_conversion[buffer_length + 2];
			//			if (this->collection->parameters->output_tag_time)
			//				energyFile = this->parameters->output_folder + "/" + starttime + "_Energy_" + s_img + suffix + ".txt";
			//			else
			//				energyFile = this->parameters->output_folder + "/Energy_" + s_img + suffix + ".txt";
			//			//
			//			scalar Rx = Rx_last + Engine::Manifoldmath::dist_geodesic(spins_last[0], *this->systems[0]->spins);
			//			//
			//			snprintf(buffer_string_conversion, buffer_length, "    %18.10f    %18.10f\n",
			//				Rx, this->systems[0]->E * nd);
			//			//
			//			spins_last[0] = *this->systems[0]->spins;
			//			Rx_last = Rx;
			//			//
			//			output_to_file += s_iter;
			//			output_to_file.append(buffer_string_conversion);
			//			IO::Append_String_to_File(output_to_file, energyFile);
			//		}
			//	// }

			//	//// Do it manually to avoid the adding of header
			//	//auto s = this->systems[0];
			//	//const int buffer_length = 80;
			//	//std::string output_to_file = "";
			//	//output_to_file.reserve(int(1E+08));
			//	//char buffer_string_conversion[buffer_length + 2];
			//	////------------------------ End Init ----------------------------------------

			//	//for (int iatom = 0; iatom < s->nos; ++iatom) {
			//	//	snprintf(buffer_string_conversion, buffer_length, "\n %18.10f %18.10f %18.10f",
			//	//		(*s->spins)[0 * s->nos + iatom], (*s->spins)[1 * s->nos + iatom], (*s->spins)[2 * s->nos + iatom]);
			//	//	output_to_file.append(buffer_string_conversion);
			//	//}
			//	//output_to_file.append("\n");
			//	//IO::Append_String_to_File(output_to_file, spinsFile);

			//};

			//std::string suffix = "_archive";
			//writeoutput(suffix);
		}
    }

	template <Solver solver>
    void Method_MMF<solver>::Finalize()
    {
        this->collection->iteration_allowed=false;
    }

	template <Solver solver>
    bool Method_MMF<solver>::Iterations_Allowed()
	{
		return this->collection->iteration_allowed;
	}

    // Method name as string
	template <Solver solver>
    std::string Method_MMF<solver>::Name() { return "MMF"; }

	// Template instantiations
	template class Method_MMF<Solver::SIB>;
	template class Method_MMF<Solver::Heun>;
	template class Method_MMF<Solver::Depondt>;
	template class Method_MMF<Solver::NCG>;
	template class Method_MMF<Solver::VP>;
}