#include <Optimizer_CG.h>


namespace Engine
{
    Optimizer_CG::Optimizer_CG(std::vector<std::shared_ptr<Data::Spin_System>> systems, std::shared_ptr<Engine::Method> method) :
        Optimizer(systems, method)
    {

    }

    void Optimizer_CG::Iteration()
    {
        // Not Implemented!
        Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Optimizer::Step() of the Optimizer base class!"));
    }

    // Optimizer name as string
    std::string Optimizer_CG::Name() { return "CG"; }
    std::string Optimizer_CG::FullName() { return "Conjugate Gradient"; }
}