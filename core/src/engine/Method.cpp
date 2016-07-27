#include "Method.h"

namespace Engine
{
    Method::Method(std::shared_ptr<Data::Spin_System_Chain> c, std::shared_ptr<Optimizer> optimizer)
    {
        this->c = c;
        this->optimizer = optimizer;
        this->starttime = Utility::Timing::CurrentDateTime();

        this->t_iterations.push_back(system_clock::now());
        this->t_iterations.push_back(system_clock::now());
        this->t_iterations.push_back(system_clock::now());
        this->t_iterations.push_back(system_clock::now());
        this->t_iterations.push_back(system_clock::now());
        this->t_iterations.push_back(system_clock::now());
        this->t_iterations.push_back(system_clock::now());
        this->ips = 0;
    }

    // Iterate for n iterations
    void Method::Iterate()
    {
        // Not Implemented!
        Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Solver::Iterate() of the Solver base class!"));
        throw Utility::Exception::Not_Implemented;
    }

    void Method::Iteration()
    {
        // Not Implemented!
        Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Solver::Iteration() of the Solver base class!"));
        throw Utility::Exception::Not_Implemented;
    }

    double Method::getIterationsPerSecond()
    {
        double l_ips = 0.0;
        for (unsigned int i = 0; i < t_iterations.size() - 1; ++i)
        {
            l_ips += Utility::Timing::SecondsPassed(t_iterations[i], t_iterations[i+1]);
        }
        this->ips = 1.0 / (l_ips / (t_iterations.size() - 1));
        return this->ips;
    }


    void Method::Save_Step(int image, int iteration, std::string suffix)
    {
        // Not Implemented!
        Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Solver::Save_Step() of the Solver base class!"));
        throw Utility::Exception::Not_Implemented;
    }

    bool Method::StopFilePresent()
    {
        std::ifstream f("STOP");
        return f.good();
    }
}