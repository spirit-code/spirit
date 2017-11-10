#ifndef VFRENDERING_UTILITIES_HXX
#define VFRENDERING_UTILITIES_HXX

#include <vector>

#include <glm/glm.hpp>

#include <VFRendering/Options.hxx>

namespace VFRendering {
using Options = Utilities::Options;

namespace Utilities {

class OpenGLException : public std::runtime_error {
public:
    OpenGLException(const std::string& message);
};

unsigned int createProgram(const std::string& vertex_shader_source,
                           const std::string& fragment_shader_source,
                           const std::vector<std::string>& attributes);

enum class Colormap {
    DEFAULT,
    BLUERED,
    BLUEGREENRED,
    BLUEWHITERED,
    HSV,
    BLACK,
    WHITE
};

std::string getColormapImplementation(const Colormap& colormap);

std::pair<glm::mat4, glm::mat4> getMatrices(const VFRendering::Options& options, float aspect_ratio);

}
}

#endif
