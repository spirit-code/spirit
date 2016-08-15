#include <Optimizer_CG.h>


namespace Engine
{
    Optimizer_CG::Optimizer_CG(std::shared_ptr<Engine::Method> method) :
        Optimizer(method)
    {

    }

    void Optimizer_CG::Iteration()
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Optimizer::Step() of the Optimizer base class!"));
    }

    // Optimizer name as string
    std::string Optimizer_CG::Name() { return "CG"; }
    std::string Optimizer_CG::FullName() { return "Conjugate Gradient"; }
}