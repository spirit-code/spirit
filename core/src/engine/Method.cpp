#include <engine/Method.hpp>

#include <algorithm>

namespace Engine
{
    Method::Method(std::shared_ptr<Data::Parameters_Method> parameters, int idx_img, int idx_chain) :
        parameters(parameters), idx_image(idx_img), idx_chain(idx_chain)
    {
        this->SenderName = Utility::Log_Sender::All;
		this->force_maxAbsComponent = parameters->force_convergence + 1.0;
    }

    void Method::Calculate_Force(std::vector<std::shared_ptr<std::vector<Vector3>>> configurations, std::vector<std::vector<Vector3>> & forces)
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

	std::pair<scalar, scalar> minmax_component(std::vector<Vector3> v1)
	{
		scalar min=1e6, max=-1e6;
		std::pair<scalar, scalar> minmax;
		for (unsigned int i = 0; i < v1.size(); ++i)
		{
			for (int dim = 0; dim < 3; ++dim)
			{
				if (v1[i][dim] < min) min = v1[i][dim];
				if (v1[i][dim] > max) max = v1[i][dim];
			}
		}
		minmax.first = min;
		minmax.second = max;
		return minmax;
	}

    // Return the maximum of absolute values of force components for an image
    scalar  Method::Force_on_Image_MaxAbsComponent(const std::vector<Vector3> & image, std::vector<Vector3> force)
    {
        int nos = image.size();
        // We project the force orthogonal to the SPIN
        //Utility::Manifoldmath::Project_Orthogonal(F_gradient[img], this->c->tangents[img]);
        
        // Take out component in direction of v2
        for (int i = 0; i < nos; ++i)
        {
			force[i] -= force[i].dot(image[i]) * image[i];
        }

        // We want the Maximum of Absolute Values of all force components on all images
        scalar absmax = 0;
        // Find minimum and maximum values
        std::pair<scalar,scalar> minmax = minmax_component(force);
        // Mamimum of absolute values
        absmax = std::max(absmax, std::abs(minmax.first));
        absmax = std::max(absmax, std::abs(minmax.second));
        // Return
        return absmax;
    }

    std::string Method::Name()
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Method::Name() of the Method base class!"));
        return "--";
    }
}