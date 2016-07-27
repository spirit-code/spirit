#include "Optimizer.h"

namespace Engine
{
    void Optimizer::Configure(std::vector<std::shared_ptr<Data::Spin_System>> systems, std::shared_ptr<Engine::Force> force_call)
    {
        this->systems = systems;

        this->noi = systems.size();
        this->nos = systems[0]->nos;

        this->configurations = std::vector<std::vector<double>>(this->noi, std::vector<double>(3 * this->nos));
        for (int i = 0; i < this->noi; ++i)
        {
            this->configurations[i] = systems[i]->spins;
        }

        this->force = std::vector<std::vector<double>>(this->noi, std::vector<double>(3 * this->nos, 0));	// [noi][3*nos]

        this->force_call = force_call;
        // Calculate forces once, so that the Solver does not think it's converged
        this->force_call->Calculate(this->configurations, this->force);
    }

    // One step in the optimization
    void Optimizer::Step()
    {
        // Not Implemented!
        Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Optimizer::Step() of the Optimizer base class!"));
    }

    // Optimizer name as string
    std::string Optimizer::Name()
    {
        // Not Implemented!
        Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Optimizer::Name() of the Optimizer base class!"));
        return "--";
    }
    std::string Optimizer::Fullname()
    {
        // Not Implemented!
        Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Optimizer::Fullname() of the Optimizer base class!"));
        return "--";
    }
}