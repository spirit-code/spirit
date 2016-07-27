
#include "Method_LLG.h"

#include "Force.h"
#include "Force_LLG.h"
#include "Optimizer_Heun.h"

#include "Spin_System.h"
#include "Spin_System_Chain.h"
#include "Vectormath.h"
#include "IO.h"
#include "Configurations.h"
#include "Timing.h"
#include "Exception.h"

#include <iostream>
#include <ctime>
#include <math.h>

#include"Logging.h"

using namespace Utility;

namespace Engine
{
    Method_LLG::Method_LLG(std::shared_ptr<Data::Spin_System_Chain> c, std::shared_ptr<Optimizer> optim) : Method(c, optim)
	{
		// Method child-class specific instructions
		// currently only support a single image being iterated:
		this->force_call = std::shared_ptr<Engine::Force>(new Force_LLG(this->c));
		this->systems.push_back(c->images[c->idx_active_image]);

		// Configure the Optimizer
		this->optimizer->Configure(systems, force_call);
	}
    
    void Method_LLG::Iterate()
    {
		//========================= Init local vars ================================
        auto s = c->images[c->idx_active_image];
		int n = s->llg_parameters->n_iterations;
		int log_steps = s->llg_parameters->log_steps;
        int i, step = 0, image = c->idx_active_image, n_log = n / log_steps;
		this->starttime = Timing::CurrentDateTime();
		std::string suffix = "_archive";
        //------------------------ End Init ----------------------------------------
        //epsilon = std::sqrt(2.0*s->llg_parameters->damping / (1.0 + (s->llg_parameters->damping * s->llg_parameters->damping))*s->llg_parameters->temperature);
		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "-------------- Started LLG Simulation --------------");
		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "Iterating with stepsize of " + std::to_string(log_steps) + " iterations per step");
		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "Optimizer: SIB"); // Use Optimizer->Name for this print!
		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "----------------------------------------------------");
        
        auto t_start = system_clock::now();
        auto t_current = system_clock::now();
        auto t_last = system_clock::now();

        for (i = 0; i < n && s->iteration_allowed && !this->force_call->IsConverged() && !this->StopFilePresent(); ++i) {
            // Do one single Iteration
            this->Iteration();

			// Recalculate FPS
			this->t_iterations.pop_front();
			this->t_iterations.push_back(system_clock::now());

			// Periodical output
            if (0 == fmod(i, log_steps)) {
                step += 1;
                s->UpdateEnergy();
				
				t_last = t_current;
				t_current = system_clock::now();

				Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "LLG Iteration step           " + std::to_string(step) + " / " + std::to_string(n_log));
				Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "                           = " + std::to_string(i) + " / " + std::to_string(n));
				Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "    Time since last step:    " + std::to_string(Timing::SecondsPassed(t_last, t_current)) + " seconds.");
				Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "    Iterations / sec:        " + std::to_string(log_steps / Timing::SecondsPassed(t_last, t_current)));
				Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "    Total Energy:            " + std::to_string(s->E / s->nos));
				Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "    Maximum force component: " + std::to_string(this->force_call->maxAbsComponent));

				Save_Step(image, i, suffix);
                // temporarily removed
                // output_strings[step - 1] = IO::Spins_to_String(s);
            }// endif log_steps
        }// endif i
        auto t_end = system_clock::now();
		
		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "-------------- Finished LLG Simulation --------------");
		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "Terminated at                   " + std::to_string(i) + " / " + std::to_string(n) + " iterations.");
        Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "    Total Energy:               " + std::to_string(s->E / s->nos));
		if (this->force_call->IsConverged())
			Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "    The configuration has converged to a maximum force component of " + std::to_string(this->force_call->maxAbsComponent));
        else
            Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "    Maximum force component:    " + std::to_string(this->force_call->maxAbsComponent));
        if (this->StopFilePresent())
            Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "    A STOP file has been found.");
		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "    LLG Simulation ran for      " + std::to_string(Timing::MinutesPassed(t_start, t_end)) + " minutes.");
		Log.Send(Utility::Log_Level::ALL, Utility::Log_Sender::LLG, "-----------------------------------------------------");

		suffix = "_" + IO::int_to_formatted_string(i, (int)log10(n)) + "_final";
		Save_Step(image, i, suffix);
        //IO::Dump_to_File(output_strings, "spin_archieve.dat", s->debug_parameters->output_notification, step);
    }// end Iterate
    
    
    void Method_LLG::Iteration()
    {
		this->optimizer->Step();

        if (this->c->images[this->c->idx_active_image]->llg_parameters->renorm_sd) {
            try {
                //Vectormath::Normalize(3, s->nos, s->spins);
            }
            catch (Exception ex)
			{
                if (ex == Exception::Division_by_zero)
				{
					Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::LLG, "During Iteration Spin = (0,0,0) was detected. Using Random Spin Array");
                    //Utility::Configurations::Random(s, false);
                }
                else { throw(ex); }
            }

        }//endif renorm_sd
    }
	
	void Method_LLG::Save_Step(int image, int iteration, std::string suffix)
	{
		// Convert image to a formatted string
		auto s_img = IO::int_to_formatted_string(image, 2);
		auto s_iter = IO::int_to_formatted_string(iteration, 6);

		// Append Spin configuration to File
		auto spinsFile = this->c->images[image]->llg_parameters->output_folder + "/" + this->starttime + "_" + "Spins_" + s_img + suffix + ".txt";
		Utility::IO::Append_Spin_Configuration(this->c->images[image], iteration, spinsFile);

		// Save Spin configuration to new File
		auto spinsIterFile = this->c->images[image]->llg_parameters->output_folder + "/" + this->starttime + "_" + "Spins_" + s_img + "_" + s_iter + ".txt";
		Utility::IO::Append_Spin_Configuration(this->c->images[image], iteration, spinsIterFile);

		// Check if Energy File exists and write Header if it doesn't
		auto energyFile = this->c->images[image]->llg_parameters->output_folder + "/" + this->starttime + "_Energy_" + s_img + suffix + ".txt";
		std::ifstream f(energyFile);
		if (!f.good()) Utility::IO::Write_Energy_Header(energyFile);
		// Append Energy to File
		Utility::IO::Append_Energy(*(this->c->images[image]), iteration, energyFile);

		// Save Log
		Log.Append_to_File();
	}
}