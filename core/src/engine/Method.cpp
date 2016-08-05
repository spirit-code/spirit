#include "Method.h"
//#include "Manifoldmath.h"
#include <algorithm>

namespace Engine
{
    Method::Method(std::shared_ptr<Data::Parameters_Method> parameters, int idx_img, int idx_chain) : parameters(parameters)
    {
        this->SenderName = Utility::Log_Sender::ALL;
    }

    void Method::Calculate_Force(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces)
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
        return this->systems[0]->iteration_allowed && !this->Force_Converged(); // && c->iteration_allowed;
    }

    void Method::Save_Step(int iteration, bool final)
    {
        // Not Implemented!
        Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Method::Save_Step() of the Method base class!"));
        throw Utility::Exception::Not_Implemented;
    }

    void Method::Hook_Pre_Step()
    {
        // Not Implemented!
        Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Method::Hook_Pre_Step() of the Method base class!"));
        throw Utility::Exception::Not_Implemented;
    }

    void Method::Hook_Post_Step()
    {
        // Not Implemented!
        Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Method::Hook_Post_Step() of the Method base class!"));
        throw Utility::Exception::Not_Implemented;
    }

    // Return the maximum of absolute values of force components for an image
    double  Method::Force_on_Image_MaxAbsComponent(const std::vector<double> & image, std::vector<double> force)
    {
        int nos = image.size()/3;
        // We project the force orthogonal to the SPIN
        //Utility::Manifoldmath::Project_Orthogonal(F_gradient[img], this->c->tangents[img]);
        // Get the scalar product of the vectors
        double v1v2 = 0.0;
        int dim;
        // Take out component in direction of v2
        for (int i = 0; i < nos; ++i)
        {
            v1v2 = 0.0;
            for (dim = 0; dim < 3; ++dim)
            {
                v1v2 += force[i + dim*nos] * image[i + dim*nos];
            }
            for (dim = 0; dim < 3; ++dim)
            {
                force[i + dim*nos] = force[i + dim*nos] - v1v2 * image[i + dim*nos];
            }
        }

        // We want the Maximum of Absolute Values of all force components on all images
        double absmax = 0;
        // Find minimum and maximum values
        auto minmax = std::minmax_element(force.begin(), force.end());
        // Mamimum of absolute values
        absmax = std::max(absmax, std::abs(*minmax.first));
        absmax = std::max(absmax, std::abs(*minmax.second));
        // Return
        return absmax;
    }

    std::string Method::Name()
    {
        // Not Implemented!
        Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Method::Name() of the Method base class!"));
        return "--";
    }
}