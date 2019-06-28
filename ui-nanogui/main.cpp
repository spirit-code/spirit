#include <nanogui/tabheader.h>
#include <nanogui/glcanvas.h>
#include <nanogui/opengl.h>
#include <nanogui/glutil.h>
#include <nanogui/screen.h>
#include <nanogui/window.h>
#include <nanogui/layout.h>
#include <nanogui/label.h>
#include <nanogui/checkbox.h>
#include <nanogui/button.h>
#include <nanogui/toolbutton.h>
#include <nanogui/popupbutton.h>
#include <nanogui/combobox.h>
#include <nanogui/progressbar.h>
#include <nanogui/entypo.h>
#include <nanogui/messagedialog.h>
#include <nanogui/textbox.h>
#include <nanogui/slider.h>
#include <nanogui/imagepanel.h>
#include <nanogui/imageview.h>
#include <nanogui/vscrollpanel.h>
#include <nanogui/colorwheel.h>
#include <nanogui/colorpicker.h>
#include <nanogui/graph.h>
#include <nanogui/tabwidget.h>

#include <iostream>
#include <string>
#include <vector>
#include <memory>

// Includes for the GLTexture class.
#include <cstdint>
#include <utility>

#if defined(__GNUC__)
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#if defined(_WIN32)
#  pragma warning(push)
#  pragma warning(disable: 4457 4456 4005 4312)
#endif

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#if defined(_WIN32)
#  pragma warning(pop)
#endif
#if defined(_WIN32)
#  if defined(APIENTRY)
#    undef APIENTRY
#  endif
#  include <windows.h>
#endif

#include <GLCanvas.hpp>
#include <ConfigurationsWindow.hpp>
#include <AdvancedGraph.hpp>

#include <Spirit/State.h>
#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/Transitions.h>
#include <Spirit/Simulation.h>
#include <Spirit/Version.h>
#include <Spirit/Log.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

class MainWindow : public nanogui::Screen
{
public:
    MainWindow(int width, int height, std::shared_ptr<State> state)
        : nanogui::Screen(Eigen::Vector2i{width, height}, std::string("NanoGUI - Spirit v")+Spirit_Version(), true),
        state(state)
    {
        using namespace nanogui;

        gl_canvas = new VFGLCanvas(this, state);
        gl_canvas->setSize(this->size());

        ConfigurationsWindow *configurations = new ConfigurationsWindow(this, state);
        configurations->setPosition(Vector2i(-100, 100));
        configurations->setLayout(new GroupLayout());
        configurations->setUpdateCallback([&] {this->gl_canvas->updateData();});

        auto w = new Window(this, "Example Graph");
        w->setLayout(new GroupLayout());
        w->setSize({300,200});
        auto graph = new AdvancedGraph(w, Marker::SQUARE, Color(0,0,255,255), 1.4);
        graph->setSize({300, 200});
        graph->setPosition({0,0});
        graph->setGrid(true);
        graph->setMarginBot(40);
        graph->setXLabel("Reaction Coordinate");
        graph->setYLabel("Energy [meV]");
        graph->setXMin(0);
        graph->setXMax(9.5);
        graph->setNTicksX(10);
        graph->setYMin(1e6); 
        graph->setYMax(5e6);
        graph->setLine(true);
        graph->setValues( {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {1.2e6, 2.2e6, 3e6, 4.02e6, 2e6, 4.5e6, 3.7e6, 3.8e6, 2.9e6, 10e6} );
        performLayout();
    }

    virtual bool keyboardEvent(int key, int scancode, int action, int modifiers) {
        if( Screen::keyboardEvent(key, scancode, action, modifiers) )
            return true;
        if( key == GLFW_KEY_ESCAPE && action == GLFW_PRESS )
        {
            setVisible(false);
            return true;
        }
        return false;
    }

    virtual bool resizeEvent(const Eigen::Vector2i & size)
    {
        nanogui::Screen::resizeEvent(size);
        this->gl_canvas->setSize(size);
        this->gl_canvas->drawGL();
        return true;
    }

    virtual void draw(NVGcontext *ctx)
    {
        Screen::draw(ctx);
    }

    virtual void drawContents()
    {
        Screen::drawContents();
        this->gl_canvas->drawGL();
    }

private:
    std::shared_ptr<State> state;
    VFGLCanvas *gl_canvas;
    nanogui::TabHeader *header;
};

int main(int /* argc */, char ** /* argv */)
try
{
    std::string cfgfile = "input/input.cfg";
    bool quiet = false;
    auto state = std::shared_ptr<State>(State_Setup(cfgfile.c_str(), quiet), State_Delete);
    Configuration_PlusZ(state.get());
    Configuration_Skyrmion(state.get(), 5, 1, -90, false, false, false);

    #ifdef _OPENMP
        int nt = omp_get_max_threads() - 1;
        Log_Send(state.get(), Log_Level_Info, Log_Sender_UI, ("Using OpenMP with n=" + std::to_string(nt) + " threads").c_str());
    #endif

    nanogui::init();

    /* scoped variables */
    {
        nanogui::ref<MainWindow> app = new MainWindow(800, 600, state);
        app->drawAll();
        app->setVisible(true);
        nanogui::mainloop();
    }

    nanogui::shutdown();
}
catch (const std::runtime_error &e)
{
    std::string error_msg = std::string("Caught a fatal error: ") + std::string(e.what());
    #if defined(_WIN32)
        MessageBoxA(nullptr, error_msg.c_str(), NULL, MB_ICONERROR | MB_OK);
    #else
        std::cerr << error_msg << std::endl;
    #endif
    return -1;
}