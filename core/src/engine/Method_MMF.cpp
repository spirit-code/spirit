#include "Method_MMF.h"
#include "Manifoldmath.h"
#include "Vectormath.h"
#include "Interface_Log.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
//#include <unsupported/Eigen/CXX11/Tensor>
#include <GenEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>

namespace Engine
{
    Method_MMF::Method_MMF(std::shared_ptr<Data::Spin_System_Chain_Collection> collection, int idx_chain) :
        Method(collection->parameters, -1, idx_chain), collection(collection)
    {
		int noc = collection->noc;
		int nos = collection->chains[0]->images[0]->nos;

		this->SenderName = Utility::Log_Sender::MMF;
        
        // The systems we use are the last image of each respective chain
        for (int ichain = 0; ichain < noc; ++ichain)
		{
			this->systems.push_back(this->collection->chains[ichain]->images.back());
		}

		// We assume that the systems are not converged before the first iteration
		this->force_maxAbsComponent = this->collection->parameters->force_convergence + 1.0;

		this->hessian = std::vector<std::vector<double>>(noc, std::vector<double>(9 * nos*nos));	// [noi][3nos]
		// Forces
		this->F_gradient   = std::vector<std::vector<double>>(noc, std::vector<double>(3 * nos));	// [noi][3nos]
		this->minimum_mode = std::vector<std::vector<double>>(noc, std::vector<double>(3 * nos));	// [noi][3nos]
    }

	Eigen::MatrixXd projector(std::vector<double> & image)
	{
		int size = image.size();
		//		Get the image as Eigen vector
		Eigen::VectorXd e_image = Eigen::Map<Eigen::VectorXd>(image.data(), size);
		// 		Get basis change matrix M=1-S, S=x*x^T
		//Log(Log_Level::Debug, Log_Sender::MMF, "before basis matrix creation");
		Eigen::MatrixXd proj = Eigen::MatrixXd::Identity(size, size) - e_image*e_image.transpose();
		// 		Change the basis of the Hessian: H -> H - SHS
		//Log(Log_Level::Debug, Log_Sender::MMF, "after basis matrix creation");
		return proj;
	}

	std::vector<Eigen::MatrixXd> gamma(std::vector<double> & image)
	{
		Eigen::Matrix3d A_x, A_y, A_z;
		A_x << -2*image[0], -image[1], -image[2],
				-image[1], 0, 0,
				-image[2], 0, 0;
		A_y <<  0, -image[0], 0,
				-image[0], -2*image[1], -image[2],
				0, -image[2], 0;
		A_z << 0, 0, -image[0],
				0, 0, -image[1],
				-image[0], -image[1], -2*image[2];
		return std::vector<Eigen::MatrixXd>({ A_x, A_y, A_z });
	}

    void Method_MMF::Calculate_Force(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces)
    {
		const int nos = configurations[0]->size() / 3;

        // Loop over chains and calculate the forces
		for (int ichain = 0; ichain < collection->noc; ++ichain)
		{
			auto& image = *configurations[ichain];

			// The gradient force (unprojected) is simply minus the effective field
			this->systems[ichain]->hamiltonian->Effective_Field(image, F_gradient[ichain]);
			Eigen::VectorXd e_gradient = Eigen::Map<Eigen::VectorXd>(F_gradient[ichain].data(), 3 * nos);
			e_gradient = -e_gradient;

			// Get the unprojected Hessian
			this->systems[ichain]->hamiltonian->Hessian(image, hessian[ichain]);
			Eigen::MatrixXd e_hessian = Eigen::Map<Eigen::MatrixXd>(hessian[ichain].data(), 3 * nos, 3 * nos);

			// std::cerr << "------------------------" << std::endl;
			// std::cerr << "gradient:      " << std::endl << e_gradient.transpose() << std::endl;
			// std::cerr << "hessian:       " << std::endl << e_hessian << std::endl;
			
			// Remove Hessian's components in the basis of the image (project it into tangent space)
			auto e_projector = projector(image);
			e_hessian = e_projector.transpose()*e_hessian*e_projector;
			//Log(Log_Level::Debug, Log_Sender::MMF, "after basis change");


			/*std::cerr << "projector:     " << std::endl << e_projector << std::endl;
			std::cerr << "gradient proj: " << std::endl << (e_projector*e_gradient).transpose() << std::endl;
			std::cerr << "hessian proj:  " << std::endl << e_hessian << std::endl;*/

			// Calculate contribution of gradient
			auto e_gamma = gamma(image);
			//for (int g = 0; g < 3; ++g) std::cerr << "gamma " << g << ":" << std::endl << e_gamma[g] << std::endl;
			for (int i = 0; i < 3; ++i)
			{
				for (int j = 0; j < 3; ++j)
				{
					double temp = 0;
					for (int k = 0; k < 3; ++k)
					{
						for (int m = 0; m < 3; ++m)
						{
							temp += e_projector(i, k) * e_gamma[k](j, m) * e_gradient[m];
							//std::cerr << i << " " << j << " " << k << " " << m << " - g - " << e_gamma[k](j, m) << std::endl;
						}
					}
					e_hessian(i, j) += temp;
				}
			}

			// std::cerr << "hessian final: " << std::endl << e_hessian << std::endl;
			// std::cerr << "gradient:  " << std::endl << grad << std::endl;
			// std::cerr << "------------------------" << std::endl;

			// Get the lowest Eigenvector
			//		Create a Spectra solver
			Spectra::DenseGenMatProd<double> op(e_hessian);
			Spectra::GenEigsSolver< double, Spectra::SMALLEST_REAL, Spectra::DenseGenMatProd<double> > hessian_spectrum(&op, 1, 3);
			hessian_spectrum.init();
			//		Compute the specified spectrum
			int nconv = hessian_spectrum.compute();
			if (hessian_spectrum.info() == Spectra::SUCCESSFUL)
			{
				// Calculate the Force
				// 		Retrieve the Eigenvalues
				Eigen::VectorXd evalues = hessian_spectrum.eigenvalues().real();
				// std::cerr << "spectra val: " << std::endl << hessian_spectrum.eigenvalues() << std::endl;
				// std::cerr << "spectra vec: " << std::endl << hessian_spectrum.eigenvectors().real() << std::endl;
				// 		Check if the lowest eigenvalue is negative
				if (evalues[0] < -10e-7)
				{
					// Create the Minimum Mode
					// 		Retrieve the Eigenvectors
					Eigen::MatrixXd evectors = hessian_spectrum.eigenvectors().real();
					Eigen::Ref<Eigen::VectorXd> x = evectors.col(0);
					// We have found the mode towards a saddle point
					// 		Copy via assignment
					this->minimum_mode[ichain] = std::vector<double>(x.data(), x.data() + x.rows()*x.cols());
					// 		Normalize the mode vector in 3N dimensions
					Utility::Vectormath::Normalize(this->minimum_mode[ichain]);
					// std::cerr << "grad " << F_gradient[ichain][0] << " " << F_gradient[ichain][1] << " " << F_gradient[ichain][2] << std::endl;
					// std::cerr << "mode " << minimum_mode[ichain][0] << " " << minimum_mode[ichain][1] << " " << minimum_mode[ichain][2] << std::endl;
					// Invert the gradient force along the minimum mode
					Utility::Manifoldmath::Project_Reverse(F_gradient[ichain], minimum_mode[ichain]);
					// Copy out the forces
					forces[ichain] = F_gradient[ichain];
					// std::cerr << "force " << forces[ichain][0] << " " << forces[ichain][1] << " " << forces[ichain][2] << std::endl;
				}
				//		Otherwise we seek for the lowest nonzero eigenvalue
				else
				{
					/////////////////////////////////
					// The Eigen way... calculate them all... Inefficient! Do it the spectra way instead!
					Eigen::EigenSolver<Eigen::MatrixXd> estest(e_hessian);
					auto evals = estest.eigenvalues().real();
					// std::cerr << "eigen vals: " << std::endl << estest.eigenvalues() << std::endl;
					// std::cerr << "eigen vecs: " << std::endl << estest.eigenvectors().real() << std::endl;
					/////////////////////////////////

					// Find lowest nonzero eigenvalue
					double eval_min = 10e-7;
					int idx_eval_min = 0;
					for (int ival=0; ival<evals.size(); ++ival)
					{
						if (evals[ival] > 10e-7 && evals[ival] < eval_min)
						{
							eval_min = evals[ival];
							idx_eval_min = ival;
							break;
						}
					}
					// std::cerr << "lowest eval: " << eval_min << std::endl;
					// std::cerr << "lowest evec: " << estest.eigenvectors().col(idx_eval_min).real() << std::endl;
					// Corresponding eigenvector
					auto evec_min = estest.eigenvectors().col(idx_eval_min).real();

					// 		Copy via assignment
					this->minimum_mode[ichain] = std::vector<double>(evec_min.data(), evec_min.data() + evec_min.rows()*evec_min.cols());
					// 		Normalize the mode vector in 3N dimensions
					Utility::Vectormath::Normalize(this->minimum_mode[ichain]);
					// We are too close to the local minimum so we have to use a strong force
					//		We apply the force against the minimum mode of the positive eigenvalue
					double v1v2 = 0.0;
					for (int i = 0; i < 3 * nos; ++i)
					{
						v1v2 += F_gradient[ichain][i] * minimum_mode[ichain][i];
					}
					// Take out component in direction of v2
					for (int i = 0; i < 3 * nos; ++i)
					{
						F_gradient[ichain][i] = - v1v2 * minimum_mode[ichain][i];
					}

					// Copy out the forces
					forces[ichain] = F_gradient[ichain];
				}
			}
			else
			{
				Log(Log_Level::Error, Log_Sender::MMF, "Failed to calculate eigenvectors of the Hessian!");
				Log(Log_Level::Info, Log_Sender::MMF, "Zeroing the MMF force...");
				for (double x : forces[ichain]) x = 0;
			}
        }
    }
		
    // Check if the Forces are converged
    bool Method_MMF::Force_Converged()
    {
		if (this->force_maxAbsComponent < this->collection->parameters->force_convergence) return true;
		return false;
    }

    void Method_MMF::Hook_Pre_Iteration()
    {

	}

    void Method_MMF::Hook_Post_Iteration()
    {
        // --- Convergence Parameter Update
		this->force_maxAbsComponent = 0;
		for (int ichain = 0; ichain < collection->noc; ++ichain)
		{
			double fmax = this->Force_on_Image_MaxAbsComponent(*(systems[ichain]->spins), F_gradient[ichain]);
			if (fmax > this->force_maxAbsComponent) this->force_maxAbsComponent = fmax;
		}

        // --- Update the chains' last images
		for (auto system : systems) system->UpdateEnergy();
		for (auto chain : collection->chains)
		{
			int i = chain->noi - 1;
			if (i>0) chain->Rx[i] = chain->Rx[i - 1] + Utility::Manifoldmath::Dist_Geodesic(*chain->images[i]->spins, *chain->images[i-1]->spins);
		}
    }

    void Method_MMF::Save_Current(std::string starttime, int iteration, bool initial, bool final)
	{
        if (initial) return;
        // Insert copies of the current systems into their corresponding chains
        // - this way we will be able to look at the history of the optimizations
        // for (int ichain=0; ichain<collection->noc; ++ichain)
        // {
        //     // Copy the image
        //     auto copy = std::shared_ptr<Data::Spin_System>(new Data::Spin_System(*this->systems[ichain]));
            
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
        if (final)
        {

        }
    }

    void Method_MMF::Finalize()
    {
        this->collection->iteration_allowed=false;
    }

    bool Method_MMF::Iterations_Allowed()
	{
		return this->collection->iteration_allowed;
	}

    // Optimizer name as string
    std::string Method_MMF::Name() { return "MMF"; }
}