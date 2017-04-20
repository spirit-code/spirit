#include <engine/Method.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

namespace Engine
{
    Method::Method(std::shared_ptr<Data::Parameters_Method> parameters, int idx_img, int idx_chain) :
        parameters(parameters), idx_image(idx_img), idx_chain(idx_chain)
    {
        this->SenderName = Utility::Log_Sender::All;
		this->force_maxAbsComponent = parameters->force_convergence + 1.0;
		this->history = std::map<std::string, std::vector<scalar>>{
            {"max_torque_component", {this->force_maxAbsComponent}} };
    }

    void Method::Calculate_Force(std::vector<std::shared_ptr<vectorfield>> configurations, std::vector<vectorfield> & forces)
    {

    }

    bool Method::Force_Converged()
    {
        bool converged = false;
        if ( this->force_maxAbsComponent < this->parameters->force_convergence ) converged = true;
        return converged;
    }

    bool Method::ContinueIterating()
    {
        return this->Iterations_Allowed() && !this->Force_Converged();
    }

    bool Method::Iterations_Allowed()
    {
        return this->systems[0]->iteration_allowed;
    }

    void Method::Save_Current(std::string starttime, int iteration, bool initial, bool final)
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Method::Save_Current() of the Method base class!"));
        throw Utility::Exception::Not_Implemented;
    }

    void Method::Hook_Pre_Iteration()
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Method::Hook_Pre_Iteration() of the Method base class!"));
        throw Utility::Exception::Not_Implemented;
    }

    void Method::Hook_Post_Iteration()
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Method::Hook_Post_Iteration() of the Method base class!"));
        throw Utility::Exception::Not_Implemented;
    }

    void Method::Finalize()
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Method::Finalize() of the Method base class!"));
        throw Utility::Exception::Not_Implemented;
    }

    // Return the maximum of absolute values of force components for an image
    scalar  Method::Force_on_Image_MaxAbsComponent(const vectorfield & image, vectorfield & force)
    {
        // Take out component in direction of v2
        Manifoldmath::project_tangential(force, image);

        // We want the Maximum of Absolute Values of all force components on all images
        return Vectormath::max_abs_component(force);
    }

	void Method::Lock()
	{
		for (auto& system : this->systems) system->Lock();
	}

	void Method::Unlock()
	{
		for (auto& system : this->systems) system->Unlock();
	}

    std::string Method::Name()
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Method::Name() of the Method base class!"));
        return "--";
    }
}