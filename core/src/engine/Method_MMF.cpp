#include "Method_MMF.hpp"
#include "Manifoldmath.hpp"
#include "Vectormath.hpp"
#include "Logging.hpp"
#include "IO.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
//#include <unsupported/Eigen/CXX11/Tensor>
#include <GenEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>
#include <GenEigsRealShiftSolver.h>

using Utility::Log_Level;
using Utility::Log_Sender;

namespace Engine
{
    Method_MMF::Method_MMF(std::shared_ptr<Data::Spin_System_Chain_Collection> collection, int idx_chain) :
        Method(collection->parameters, -1, idx_chain), collection(collection)
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

		// We assume that the systems are not converged before the first iteration
		this->force_maxAbsComponent = this->collection->parameters->force_convergence + 1.0;

		this->hessian = std::vector<std::vector<double>>(noc, std::vector<double>(9 * nos*nos));	// [noc][3nos]
		// Forces
		this->F_gradient   = std::vector<std::vector<double>>(noc, std::vector<double>(3 * nos));	// [noc][3nos]
		this->minimum_mode = std::vector<std::vector<double>>(noc, std::vector<double>(3 * nos));	// [noc][3nos]

		// Last iteration
		this->spins_last = std::vector<std::vector<double>>(noc, std::vector<double>(3 * nos));	// [noc][3nos]
		this->spins_last[0] = *this->systems[0]->spins;
		this->Rx_last = 0.0;

		// Force function
		// ToDo: move into parameters
		this->mm_function = "Spectra Matrix"; // "Spectra Matrix" "Spectra Prefactor" "Lanczos"
    }
	

    void Method_MMF::Calculate_Force(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces)
    {
		if (this->mm_function == "Spectra Matrix")
		{
			this->Calculate_Force_Spectra_Matrix(configurations, forces);
		}
		else if (this->mm_function == "Spectra Prefactor")
		{
			this->Calculate_Force_Spectra_Prefactor(configurations, forces);
		}
		else if (this->mm_function == "Lanczos")
		{
			this->Calculate_Force_Lanczos(configurations, forces);
		}
    }


	Eigen::MatrixXd deorder_matrix(Eigen::MatrixXd m)
	{
		Eigen::MatrixXd r(m.rows(), m.rows());
		int nos = m.rows() / 3;

		for (int i = 0; i < nos; ++i)
		{
			for (int dim1 = 0; dim1 < 3; ++dim1)
			{
				for (int j = 0; j < nos; ++j)
				{
					for (int dim2 = 0; dim2 < 3; ++dim2)
					{
						 r(i + dim1*nos, j + dim2*nos) = m(dim1 + 3 * i, dim2 + 3 * j);
					}
				}
			}
		}
		return r;
	}

	Eigen::MatrixXd projector(std::vector<double> & image)
	{
		int size = image.size();
		int nos = size / 3;
		//		Get the image as Eigen vector
		//Eigen::VectorXd e_image = Eigen::Map<Eigen::VectorXd>(image.data(), size);
		// 		Get basis change matrix M=1-S, S=x*x^T
		//Log(Log_Level::Debug, Log_Sender::MMF, "before basis matrix creation");

		Eigen::MatrixXd proj = Eigen::MatrixXd::Identity(size, size);// -e_image*e_image.transpose();
		// TODO: do this with stride, instead of deordering later
		Eigen::Vector3d spin;
		for (int i = 0; i < nos; ++i)
		{
			for (int dim = 0; dim < 3; ++dim)
			{
				spin[dim] = image[i+dim*nos];
			}
			proj.block<3, 3>(3*i, 3*i) -= spin*spin.transpose();
		}
		//std::cerr << "projector orig: " << std::endl << proj << std::endl;
		// 		Change the basis of the Hessian: H -> H - SHS
		//Log(Log_Level::Debug, Log_Sender::MMF, "after basis matrix creation");
		return deorder_matrix(proj);
	}

	Eigen::MatrixXd reorder_matrix(Eigen::MatrixXd m)
	{
		Eigen::MatrixXd r(m.rows(), m.rows());
		int nos = m.rows() / 3;

		for (int i = 0; i < nos; ++i)
		{
			for (int dim1 = 0; dim1 < 3; ++dim1)
			{
				for (int j = 0; j < nos; ++j)
				{
					for (int dim2 = 0; dim2 < 3; ++dim2)
					{
						r(dim1 + 3 * i, dim2 + 3 * j) = m(i + dim1*nos, j + dim2*nos);
					}
				}
			}
		}
		return r;
	}

	Eigen::VectorXd reorder_vector(Eigen::VectorXd m)
	{
		Eigen::VectorXd r(m.rows());
		int nos = m.rows() / 3;

		for (int i = 0; i < nos; ++i)
		{
			for (int dim = 0; dim < 3; ++dim)
			{
				r(dim + 3 * i) = m(i + dim*nos);
			}
		}
		return r;
	}

	void Method_MMF::Calculate_Force_Spectra_Matrix(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces)
	{
		//std::cerr << "calculating force" << std::endl;
		const int nos = configurations[0]->size() / 3;

		// Loop over chains and calculate the forces
		for (int ichain = 0; ichain < collection->noc; ++ichain)
		{
			auto& image = *configurations[ichain];
			Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(configurations[ichain]->data(), 3 * nos);

			// The gradient force (unprojected) is simply minus the effective field
			this->systems[ichain]->hamiltonian->Effective_Field(image, F_gradient[ichain]);
			Eigen::VectorXd grad = Eigen::Map<Eigen::VectorXd>(F_gradient[ichain].data(), 3 * nos);
			//std::cerr << F_gradient[0][0] << std::endl;
			//std::cerr << grad[0] << std::endl;
			grad = -grad;
			//std::cerr << F_gradient[0][0] << std::endl;
			//std::cerr << grad[0] << std::endl;

			// Get the unprojected Hessian
			this->systems[ichain]->hamiltonian->Hessian(image, hessian[ichain]);
			Eigen::MatrixXd H = Eigen::Map<Eigen::MatrixXd>(hessian[ichain].data(), 3 * nos, 3 * nos);


			// Remove Hessian's components in the basis of the image (project it into tangent space)
			//		and add the gradient contributions (inner and outer product)
			auto P = projector(image);

			//std::cerr << "------------------------" << std::endl;
			//std::cerr << "gradient:      " << reorder_vector(grad).transpose() << std::endl;
			//std::cerr << "hessian:       " << std::endl << reorder_matrix(H) << std::endl;
			//std::cerr << "projector:     " << std::endl << reorder_matrix(P) << std::endl;
			//std::cerr << "hessian proj:  " << std::endl << reorder_matrix(P.transpose()*H*P) << std::endl;

			H = P.transpose()*H*P - P*(x.dot(grad)) - (P*grad)*x.transpose(); //- P*(x.dot(grad));// -(P*grad)*x.transpose();

			//Log(Log_Level::Debug, Log_Sender::MMF, "after basis change");
			
			//std::cerr << "gradient proj: " << std::endl << reorder_vector(P*grad).transpose() << std::endl;
			//std::cerr << "hessian final: " << std::endl << reorder_matrix(H) << std::endl;
			

			//Eigen::EigenSolver<Eigen::MatrixXd> estest1(reorder_matrix(H));
			//std::cerr << "hessian:    " << std::endl << reorder_matrix(H) << std::endl;
			//std::cerr << "eigen vals: " << std::endl << estest1.eigenvalues() << std::endl;
			//std::cerr << "eigen vecs: " << std::endl << estest1.eigenvectors().real() << std::endl;
			//Eigen::EigenSolver<Eigen::MatrixXd> estest2(H);
			//std::cerr << "eigen vals: " << std::endl << estest2.eigenvalues() << std::endl;
			//std::cerr << "eigen vecs: " << std::endl << (estest2.eigenvectors().real()) << std::endl;
			// Get the lowest Eigenvector
			//		Create a Spectra solver
			Spectra::DenseGenMatProd<double> op(H);
			Spectra::GenEigsSolver< double, Spectra::SMALLEST_REAL, Spectra::DenseGenMatProd<double> > hessian_spectrum(&op, 1, 3*nos);
			hessian_spectrum.init();
			//		Compute the specified spectrum
			int nconv = hessian_spectrum.compute();
			if (hessian_spectrum.info() == Spectra::SUCCESSFUL)
			{
				// Calculate the Force
				// 		Retrieve the Eigenvalues
				Eigen::VectorXd evalues = hessian_spectrum.eigenvalues().real();
				//std::cerr << "spectra val: " << std::endl << hessian_spectrum.eigenvalues() << std::endl;
				//std::cerr << "spectra vec: " << std::endl << -reorder_vector(hessian_spectrum.eigenvectors().col(0).real()) << std::endl;
				// 		Check if the lowest eigenvalue is negative
				if (evalues[0] < -1e-2)// || switched2)
				{
					if (switched1)
						switched2 = true;
					// Create the Minimum Mode
					// 		Retrieve the Eigenvectors
					Eigen::MatrixXd evectors = -hessian_spectrum.eigenvectors().real();
					Eigen::Ref<Eigen::VectorXd> evec = evectors.col(0);
					// We have found the mode towards a saddle point
					// 		Copy via assignment
					this->minimum_mode[ichain] = std::vector<double>(evec.data(), evec.data() + evec.rows()*evec.cols());
					//for (int _i = 0; _i < forces[ichain].size(); ++_i) minimum_mode[ichain][_i] = -minimum_mode[ichain][_i];
					// 		Normalize the mode vector in 3N dimensions
					Utility::Vectormath::Normalize(this->minimum_mode[ichain]);
					//std::cerr << "grad " << F_gradient[ichain][0] << " " << F_gradient[ichain][1] << " " << F_gradient[ichain][2] << std::endl;
					//std::cerr << "mode " << minimum_mode[ichain][0] << " " << minimum_mode[ichain][1] << " " << minimum_mode[ichain][2] << std::endl;
					// Invert the gradient force along the minimum mode
					Utility::Manifoldmath::Project_Reverse(F_gradient[ichain], minimum_mode[ichain]);
					// Copy out the forces
					for (unsigned int _i = 0; _i < forces[ichain].size(); ++_i) forces[ichain][_i] = +F_gradient[ichain][_i];
					//forces[ichain] = F_gradient[ichain];
					//std::cerr << "force " << forces[ichain][0] << " " << forces[ichain][1] << " " << forces[ichain][2] << std::endl;
				}
				//		Otherwise we seek for the lowest nonzero eigenvalue
				else
				{
					switched1 = true;
					//std::cerr << "sticky region " << evalues[0] << std::endl;
					///////////////////////////////////
					//// The Eigen way... calculate them all... Inefficient! Do it the spectra way instead!
					//Eigen::EigenSolver<Eigen::MatrixXd> estest1(H);
					//std::cerr << "hessian unreordered:" << std::endl << H << std::endl;
					//std::cerr << "eigen vals: " << std::endl << estest1.eigenvalues() << std::endl;
					//std::cerr << "eigen vecs: " << std::endl << reorder_matrix(estest1.eigenvectors().real()) << std::endl;
					//Eigen::EigenSolver<Eigen::MatrixXd> estest2(reorder_matrix(H));
					//std::cerr << "hessian:    " << std::endl << reorder_matrix(H) << std::endl;
					//std::cerr << "eigen vals: " << std::endl << estest2.eigenvalues() << std::endl;
					//std::cerr << "eigen vecs: " << std::endl << estest2.eigenvectors().real() << std::endl << std::endl;
					///////////////////////////////////

					//// Create the Minimum Mode
					//Spectra::DenseGenRealShiftSolve<double> op_pos(H);
					//Spectra::GenEigsRealShiftSolver< double, Spectra::LARGEST_REAL, Spectra::DenseGenRealShiftSolve<double> > hessian_spectrum_pos(&op_pos, 1, 3 * nos, 1.0e-9);
					//hessian_spectrum_pos.init();
					////		Compute the specified spectrum
					//int nconv = hessian_spectrum_pos.compute();
					//// 		Retrieve the Eigenvectors
					//std::cerr << "-----" << std::endl;
					//std::cerr << "spectra:" << std::endl;
					//std::cerr << hessian_spectrum_pos.eigenvalues() << std::endl;
					//std::cerr << hessian_spectrum_pos.eigenvectors().real() << std::endl;
					//std::cerr << "-----" << std::endl;
					//Eigen::MatrixXd evectors = hessian_spectrum_pos.eigenvectors().real();
					////auto evec = evectors.col(0);
					//Eigen::Ref<Eigen::VectorXd> evec = evectors.col(0);


					//this->minimum_mode[ichain] = std::vector<double>(evec.data(), evec.data() + evec.rows()*evec.cols());
					////for (int _i = 0; _i < forces[ichain].size(); ++_i) minimum_mode[ichain][_i] = -minimum_mode[ichain][_i];
					//// 		Normalize the mode vector in 3N dimensions
					//Utility::Vectormath::Normalize(this->minimum_mode[ichain]);

					////// 		Copy via assignment
					////this->minimum_mode[ichain] = std::vector<double>(evec_min.data(), evec_min.data() + evec_min.rows()*evec_min.cols());
					////// 		Normalize the mode vector in 3N dimensions
					////Utility::Vectormath::Normalize(this->minimum_mode[ichain]);
					////// We are too close to the local minimum so we have to use a strong force
					//////		We apply the force against the minimum mode of the positive eigenvalue
					//double v1v2 = 0.0;
					//for (int i = 0; i < 3 * nos; ++i)
					//{
					//	v1v2 += F_gradient[ichain][i] * minimum_mode[ichain][i];
					//}
					// Take out component in direction of v2
					for (int i = 0; i < 3 * nos; ++i)
					{
						F_gradient[ichain][i] = -F_gradient[ichain][i]; //-v1v2 * minimum_mode[ichain][i]; //
					}

					// Copy out the forces
					forces[ichain] = F_gradient[ichain];
				}
				//std::cerr << "------------------------" << std::endl;
			}
			else
			{
				Log(Log_Level::Error, Log_Sender::MMF, "Failed to calculate eigenvectors of the Hessian!");
				for (int _i = 0; _i < 6; ++_i) std::cerr << minimum_mode[ichain][_i] << " "; std::cerr << std::endl;
				for (int _i = 0; _i < 6; ++_i) std::cerr << F_gradient[ichain][_i] << " "; std::cerr << std::endl;
				std::cerr << reorder_matrix(H) << std::endl;
				Log(Log_Level::Info, Log_Sender::MMF, "Zeroing the MMF force...");
				for (double x : forces[ichain]) x = 0;

				///////////////////////////////////
				//// The Eigen way... calculate them all... Inefficient! Do it the spectra way instead!
				//Eigen::EigenSolver<Eigen::MatrixXd> estest(H);
				//auto evals = estest.eigenvalues().real();
				////std::cerr << "eigen vals: " << std::endl << estest.eigenvalues() << std::endl;
				////std::cerr << "eigen vecs: " << std::endl << estest.eigenvectors() << std::endl;
				///////////////////////////////////

				//// Find lowest nonzero eigenvalue
				//double eval_min = 10e15;
				//int idx_eval_min = 0;
				//for (int ival = 0; ival<evals.size(); ++ival)
				//{
				//	if (std::abs(evals[ival]) > 10e-9 && evals[ival] < eval_min)
				//	{
				//		eval_min = evals[ival];
				//		idx_eval_min = ival;
				//		break;
				//	}
				//}
				//// std::cerr << "lowest eval: " << eval_min << std::endl;
				//// std::cerr << "lowest evec: " << estest.eigenvectors().col(idx_eval_min).real() << std::endl;
				//// Corresponding eigenvector
				//auto evec_min = estest.eigenvectors().col(idx_eval_min).real();

				//// 		Copy via assignment
				//this->minimum_mode[ichain] = std::vector<double>(evec_min.data(), evec_min.data() + evec_min.rows()*evec_min.cols());
				//// 		Normalize the mode vector in 3N dimensions
				//Utility::Vectormath::Normalize(this->minimum_mode[ichain]);
				//// We are too close to the local minimum so we have to use a strong force
				////		We apply the force against the minimum mode of the positive eigenvalue
				//double v1v2 = 0.0;
				//for (int i = 0; i < 3 * nos; ++i)
				//{
				//	v1v2 += F_gradient[ichain][i] * minimum_mode[ichain][i];
				//}
				//// Take out component in direction of v2
				//for (int i = 0; i < 3 * nos; ++i)
				//{
				//	F_gradient[ichain][i] = -v1v2 * minimum_mode[ichain][i];
				//}

				//// Copy out the forces
				//forces[ichain] = F_gradient[ichain];
			}
		}
	}

	std::vector<Eigen::MatrixXd> gamma(std::vector<double> & image)
	{
		int nos = image.size() / 3;

		Eigen::MatrixXd a_x(3 * nos, 3 * nos), a_y(3 * nos, 3 * nos), a_z(3 * nos, 3 * nos);
		Eigen::Matrix3d A_x, A_y, A_z;
		for (int i = 0; i < nos; ++i)
		{
			A_x << -2 * image[i + 0 * nos], -image[i + 1 * nos], -image[i + 2 * nos],
				-image[i + 1 * nos], 0, 0,
				-image[i + 2 * nos], 0, 0;
			A_y << 0, -image[i + 0 * nos], 0,
				-image[i + 0 * nos], -2 * image[i + 1 * nos], -image[i + 2 * nos],
				0, -image[i + 2 * nos], 0;
			A_z << 0, 0, -image[i + 0 * nos],
				0, 0, -image[i + 1 * nos],
				-image[i + 0 * nos], -image[i + 1 * nos], -2 * image[i + 2 * nos];

			a_x.block<3, 3>(3*i, 3*i) = A_x;
			a_y.block<3, 3>(3*i, 3*i) = A_y;
			a_z.block<3, 3>(3*i, 3*i) = A_z;
		}
		return std::vector<Eigen::MatrixXd>({ a_x, a_y, a_z });
	}

	void printmatrix(Eigen::MatrixXd & m)
	{
		std::cerr << m << std::endl;
	}

	void Method_MMF::Calculate_Force_Spectra_Prefactor(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces)
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

			for (int i = 0; i < 3 * nos; ++i)
			{
				for (int j = 0; j < 3 * nos; ++j)
				{
					double temp = 0;
					for (int k = 0; k < 3; ++k)
					{
						for (int m = 0; m < 3 * nos; ++m)
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
				if (evalues[0] < -10e-9)
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
					double eval_min = 10e-9;
					int idx_eval_min = 0;
					for (int ival=0; ival<evals.size(); ++ival)
					{
						if (evals[ival] > 10e-9 && evals[ival] < eval_min)
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
				/*Log(Log_Level::Info, Log_Sender::MMF, "Zeroing the MMF force...");
				for (double x : forces[ichain]) x = 0;*/
				/////////////////////////////////
				// The Eigen way... calculate them all... Inefficient! Do it the spectra way instead!
				Eigen::EigenSolver<Eigen::MatrixXd> estest(e_hessian);
				auto evals = estest.eigenvalues().real();
				// std::cerr << "eigen vals: " << std::endl << estest.eigenvalues() << std::endl;
				// std::cerr << "eigen vecs: " << std::endl << estest.eigenvectors().real() << std::endl;
				/////////////////////////////////

				// Find lowest nonzero eigenvalue
				double eval_min = 10e-9;
				int idx_eval_min = 0;
				for (int ival = 0; ival<evals.size(); ++ival)
				{
					if (evals[ival] > 10e-9 && evals[ival] < eval_min)
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
					F_gradient[ichain][i] = -v1v2 * minimum_mode[ichain][i];
				}

				// Copy out the forces
				forces[ichain] = F_gradient[ichain];
			}
		}
	}

	void Method_MMF::Calculate_Force_Lanczos(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces)
	{

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
        //if (initial) return;

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

		auto writeoutput = [this, starttime, iteration](std::string suffix)
		{
			// Convert indices to formatted strings
			auto s_img = Utility::IO::int_to_formatted_string(this->idx_image, 2);
			auto s_iter = Utility::IO::int_to_formatted_string(iteration, 6);

			// Append Spin configuration to Spin_Archieve_File
			auto spinsFile = this->parameters->output_folder + "/" + starttime + "_" + "Spins_" + s_img + suffix + ".txt";
			Utility::IO::Append_Spin_Configuration(this->systems[0], iteration, spinsFile);

			// Append iteration, Rx and E to Energy file
			double nd = 1.0 / this->systems[0]->nos; // nos divide
			const int buffer_length = 200;
			std::string output_to_file = "";
			output_to_file.reserve(int(1E+08));
			char buffer_string_conversion[buffer_length + 2];
			auto energyFile = this->parameters->output_folder + "/" + starttime + "_" + "Energy_" + s_img + suffix + ".txt";
			//
			double Rx = Rx_last + Utility::Manifoldmath::Dist_Geodesic(spins_last[0], *this->systems[0]->spins);
			//
			snprintf(buffer_string_conversion, buffer_length, "    %18.10f    %18.10f\n",
				Rx, this->systems[0]->E * nd);
			//
			spins_last[0] = *this->systems[0]->spins;
			Rx_last = Rx;
			//
			output_to_file += s_iter;
			output_to_file.append(buffer_string_conversion);
			Utility::IO::Append_String_to_File(output_to_file, energyFile);


			//// Do it manually to avoid the adding of header
			//auto s = this->systems[0];
			//const int buffer_length = 80;
			//std::string output_to_file = "";
			//output_to_file.reserve(int(1E+08));
			//char buffer_string_conversion[buffer_length + 2];
			////------------------------ End Init ----------------------------------------

			//for (int iatom = 0; iatom < s->nos; ++iatom) {
			//	snprintf(buffer_string_conversion, buffer_length, "\n %18.10f %18.10f %18.10f",
			//		(*s->spins)[0 * s->nos + iatom], (*s->spins)[1 * s->nos + iatom], (*s->spins)[2 * s->nos + iatom]);
			//	output_to_file.append(buffer_string_conversion);
			//}
			//output_to_file.append("\n");
			//Utility::IO::Append_String_to_File(output_to_file, spinsFile);

		};

		std::string suffix = "_archive";
		writeoutput(suffix);
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