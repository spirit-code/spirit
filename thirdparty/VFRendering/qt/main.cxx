#include <fstream>
#include <QSurfaceFormat>
#include <QApplication>
#include "VFRenderingWidget.hxx"

int main(int argc, char **argv) {

    QApplication app(argc, argv);
  
    QSurfaceFormat format;
    format.setMajorVersion(3);
    format.setMinorVersion(3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format);
    VFRenderingWidget window;

    std::vector<glm::vec3> directions;
    VFRendering::Geometry geometry = VFRendering::Geometry::cartesianGeometry({21, 21, 21}, {-20, -20, -20}, {20, 20, 20});
    for (const auto& position : geometry.positions()) {
        directions.push_back(glm::normalize(position));
    }
    window.update(geometry, directions);

    VFRendering::Options options;
    options.set<VFRendering::View::Option::CAMERA_POSITION>({0, 0, 30});
    options.set<VFRendering::View::Option::CENTER_POSITION>({0, 0, 0});
    options.set<VFRendering::View::Option::UP_VECTOR>({0, 1, 0});
    options.set<VFRendering::View::Option::SYSTEM_CENTER>((geometry.min() + geometry.max()) * 0.5f);
    options.set<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>(VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::HSV));
    window.updateOptions(options);

    window.resize(1000, 1000);
    window.setWindowTitle("VFRenderingWidgetExample");
    window.show();

    return app.exec();
}
