#include <Optimizer_QM.h>


namespace Engine
{
    Optimizer_QM::Optimizer_QM(std::vector<std::shared_ptr<Data::Spin_System>> systems, std::shared_ptr<Engine::Method> method) :
        Optimizer(systems, method)
    {

    }

    void Optimizer_QM::Iteration()
    {
        // Not Implemented!
        Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Optimizer::Step() of the Optimizer base class!"));
    }
    
    // Optimizer name as string
    std::string Optimizer_QM::Name() { return "QM"; }
    std::string Optimizer_QM::FullName() { return "Quick-Min"; }
}