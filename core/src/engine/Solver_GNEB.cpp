#include <iostream>
#include <math.h>

#include "Force.h"
#include "Force_GNEB.h"
#include "Manifoldmath.h"
#include "Cubic_Hermite_Spline.h"
#include "IO.h"
#include "Timing.h"

#include "Optimizer_Heun.h"
#include "Optimizer_SIB.h"
#include "Solver_GNEB.h"
#include "Vectormath.h"

#include"Logging.h"
using namespace Utility;

namespace Engine
{
    Solver_GNEB::Solver_GNEB(std::shared_ptr<Data::Spin_System_Chain> c, std::shared_ptr<Optimizer> optim) : Solver(c, optim)
	{
		// Solver child-class specific instructions
		this->force_call = std::shared_ptr<Engine::Force>(new Force_GNEB(this->c));
		this->systems = c->images;

		// Configure the Optimizer
		this->optimizer->Configure(systems, force_call);
	}

	// Iteratively apply the GNEB method to a Spin System Chain
	void Solver_GNEB::Iterate()
	{
		//========================= Init local vars ================================
		int n_iterations = c->gneb_parameters->n_iterations;
		int log_steps = c->gneb_parameters->log_steps;
		int i, step = 0, n_log = n_iterations/log_steps;
		this->starttime = Timing::CurrentDateTime();
		std::string suffix = "";
		//------------------------ End Init ----------------------------------------

		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "-------------- Started GNEB Simulation --------------");
		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "Iterating with stepsize of " + std::to_string(log_steps) + " iterations per step");
		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "Optimizer: Heun");
		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "-----------------------------------------------------");

		auto t_start = system_clock::now();
		auto t_current = system_clock::now();
		auto t_last = system_clock::now();
		for (i = 0; i < n_iterations && c->iteration_allowed && !this->force_call->IsConverged() && !this->StopFilePresent(); ++i)
		{
			// Do one single Iteration
			this->Iteration();

			// Recalculate FPS
			this->t_iterations.pop_front();
			this->t_iterations.push_back(system_clock::now());

			// Log Output every log_steps steps
			if (0 == fmod(i, log_steps))
			{
				step += 1;

				t_last = t_current;
				t_current = system_clock::now();

				Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "GNEB Iteration step          " + std::to_string(step) + " / " + std::to_string(n_log));
				Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "                           = " + std::to_string(i) + " / " + std::to_string(n_iterations));
				Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "    Time since last step:    " + std::to_string(Timing::SecondsPassed(t_last, t_current)) + " seconds.");
				Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "    Iterations / sec:        " + std::to_string(log_steps / Timing::SecondsPassed(t_last, t_current)));
				Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "    Maximum force component: " + std::to_string(this->force_call->maxAbsComponent));

				Save_Step(0, i, suffix);

				//output_strings[step - 1] = IO::Spins_to_String(c->images[0].get());
			}// endif log_steps
		}// endif i
		auto t_end = system_clock::now();

		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "-------------- Finished GNEB Simulation --------------");
		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "Terminated at                   " + std::to_string(i) + " / " + std::to_string(n_iterations) + " iterations.");
		if (this->force_call->IsConverged())
			Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "    The transition has converged to a maximum force component of " + std::to_string(this->force_call->maxAbsComponent));
		else
			Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "    Maximum force component:    " + std::to_string(this->force_call->maxAbsComponent));
		if (this->StopFilePresent())
			Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "    A STOP file has been found.");
		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "    GNEB Simulation ran for     " + std::to_string(Timing::MinutesPassed(t_start, t_end)) + " minutes.");
		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::GNEB, "------------------------------------------------------");

		//suffix = "_" + IO::int_to_formatted_string(i, (int)log10(n_iterations)) + "_final";
		suffix = "_final";
		Save_Step(0, i, suffix);
		//IO::Dump_to_File(output_strings, "spin_archieve.dat", c->images[0]->debug_parameters->output_notification, step);
	}

	// Apply one iteration of the GNEB method to a Spin System Chain
	void Solver_GNEB::Iteration()
	{
		int nos = c->images[0]->nos;

		this->optimizer->Step();

		// Calculate and interpolate energies and store in the spin systems and spin system chain
		std::vector<double> E(c->noi, 0);
		std::vector<double> dE_dRx(c->noi, 0);
		// Calculate the inclinations at the data points
		for (int i = 0; i < c->noi; ++i)
		{
			// x
			if (i > 0) c->Rx[i] = c->Rx[i - 1] + Utility::Manifoldmath::Dist_Geodesic(c->images[i - 1]->spins, c->images[i]->spins);
			// y
			E[i] = c->images[i]->E;
			// dy/dx
			for (int j = 0; j < 3 * nos; ++j)
			{
				dE_dRx[i] += c->images[i]->effective_field[j] * c->tangents[i][j];
			}
		}
		// Actual Interpolation
		std::vector<std::vector<double>> interp = Utility::Cubic_Hermite_Spline::Interpolate(c->Rx, E, dE_dRx, c->gneb_parameters->n_E_interpolations);
		c->Rx_interpolated = interp[0];
		c->E_interpolated = interp[1];
	}


	void Solver_GNEB::Save_Step(int image, int iteration, std::string suffix)
	{
		// always formatting to 6 digits may be problematic!
		auto s_iter = IO::int_to_formatted_string(iteration, 6);

		// Save current Image Chain
		auto imagesFile = this->c->gneb_parameters->output_folder + "/" + this->starttime + "_Images_" + s_iter + suffix + ".txt";
		Utility::IO::Save_SpinChain_Configuration(this->c, imagesFile);

		// Save current Energies with reaction coordinates
		auto energiesFile = this->c->gneb_parameters->output_folder + "/" + this->starttime + "_E_Images_" + s_iter + suffix + ".txt";
		//		Check if Energy File exists and write Header if it doesn't
		std::ifstream f(energiesFile);
		if (!f.good()) Utility::IO::Write_Energy_Header(energiesFile);
		//		Save
		Utility::IO::Save_Energies(*this->c, iteration, energiesFile);

		// Save interpolated Energies
		auto energiesInterpFile = this->c->gneb_parameters->output_folder + "/" + this->starttime + "_E_interp_Images_" + s_iter + suffix + ".txt";
		Utility::IO::Save_Energies_Interpolated(*this->c, energiesInterpFile);

		// Save Log
		Log.Append_to_File();
	}
}