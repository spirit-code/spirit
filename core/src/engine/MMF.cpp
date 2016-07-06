#include "MMF.h"
#include "Eff_Field.h"
#include "Energy.h"
#include "Force.h"
#include "Manifoldmath.h"
#include "Cubic_Hermite_Spline.h"
#include "IO.h"
#include <iostream>

using namespace Utility;

namespace Engine
{
	namespace MMF
	{
		// Iteratively apply the GNEB method to a Spin System Chain
		void Iterate(std::shared_ptr<Data::Spin_System_Chain> c, int n_iterations, int log_steps)
		{
			int i, step=0;
			std::vector<std::string> output_strings(int(n_iterations / log_steps + 2));

			std::cout << std::endl << "Iterating MMF with stepsize of " << log_steps << " iterations per step";

			for (i = 0; i < n_iterations && c->iteration_allowed; ++i) {
				// Do one single Iteration
				MMF::Iteration(c);

				if (0 == fmod(i, log_steps)) {
					step += 1;
					std::cout.precision(20);
					std::cout << std::endl << "Iteration step " << step;

					//output_strings[step - 1] = IO::Spins_to_String(c->images[0].get());
				}// endif log_steps
			}// endif i

			std::cout << std::endl << "MMF Iteration terminating after " << i << " of " << n_iterations << " steps.";
			IO::Filedump(output_strings, "spin_archieve.dat", c->images[0]->debug_parameters->output_notification, step);
		}

		// Apply one iteration of the GNEB method to a Spin System Chain
		void Iteration(std::shared_ptr<Data::Spin_System_Chain> c)
		{
			
		}


		bool isConverged(std::vector<std::vector<double>> & force)
		{
			return false;
		}

	}
}