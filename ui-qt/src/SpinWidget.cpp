#include <fstream>
#include <sstream>
#include "SpinWidget.hpp"

#include <QTimer>
#include <QMouseEvent>

#include <VFRendering/CombinedRenderer.hxx>
#include <VFRendering/ArrowRenderer.hxx>
#include <VFRendering/IsosurfaceRenderer.hxx>
#include <VFRendering/BoundingBoxRenderer.hxx>
#include <VFRendering/VectorSphereRenderer.hxx>
#include <VFRendering/CoordinateSystemRenderer.hxx>

#include <glm/gtc/type_ptr.hpp>

#include "Interface_Geometry.h"
#include "Interface_System.h"
#include "Interface_Simulation.h"


SpinWidget::SpinWidget(std::shared_ptr<State> state, QWidget *parent) : QOpenGLWidget(parent)
{
    this->state = state;
    setFocusPolicy(Qt::StrongFocus);

    QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    sizePolicy.setHorizontalStretch(0);
    sizePolicy.setVerticalStretch(0);
    this->setSizePolicy(sizePolicy);

    this->setMinimumSize(200,200);
    this->setBaseSize(600,600);
  
    setColormap(Colormap::HSV);
    
    m_view.setOption<VFRendering::ArrowRenderer::Option::CONE_RADIUS>(0.125f);
    m_view.setOption<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>(0.3f);
    m_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_RADIUS>(0.0625f);
    m_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_HEIGHT>(0.35f);

	this->m_coordinatecross_position = { 0.8f, 0, 0.2f, 0.2f };
	this->m_miniview_position = { 0, 0, 0.2f, 0.2f };
	this->show_arrows = true;
	this->show_boundingbox = true;
	this->show_isosurface = false;
	this->show_surface = false;
	this->show_miniview = true;
	this->show_coordinatesystem = true;
    
    setZRange({-1, 1});
}

void SpinWidget::initializeGL()
{
    // Get GL context
    makeCurrent();
    // Initialize the visualisation options
    std::vector<int> n_cells(3);
    Geometry_Get_N_Cells(state.get(), n_cells.data());
    // Initial camera position
    _reset_camera = true;
    // Fetch data and update GL arrays
    this->updateData();
  
	// Create renderers
	//	System
	this->m_renderer_arrows = std::make_shared<VFRendering::ArrowRenderer>(m_view);
	this->m_renderer_boundingbox = std::make_shared<VFRendering::BoundingBoxRenderer>(m_view);
	this->m_renderer_surface = std::make_shared<VFRendering::IsosurfaceRenderer>(m_view);
	this->m_renderer_isosurface = std::make_shared<VFRendering::IsosurfaceRenderer>(m_view);
	std::vector<std::shared_ptr<VFRendering::RendererBase>> renderers = {
		m_renderer_arrows,
		m_renderer_boundingbox
	};
	this->m_system = std::make_shared<VFRendering::CombinedRenderer>(m_view, renderers);

	// Isosurface options
	m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>([](const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
		(void)position;
		return direction.z;
	});
	m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(0.0);

	//	Sphere
	this->m_sphere = std::make_shared<VFRendering::VectorSphereRenderer>(m_view);

	//	Coordinate cross
	this->m_coordinatecross = std::make_shared<VFRendering::CoordinateSystemRenderer>(m_view);

	// Setup the View
	this->m_mainview = this->m_system;
	this->m_miniview = this->m_sphere;
	this->setupRenderers();

    updateData();
}

void SpinWidget::teardownGL() {
	// GLSpins::terminate();
}

void SpinWidget::resizeGL(int width, int height) {
  m_view.setFramebufferSize(width*devicePixelRatio(), height*devicePixelRatio());
  //update();
	QTimer::singleShot(1, this, SLOT(update()));
}

void SpinWidget::updateData()
{
	int nos = System_Get_NOS(state.get());
	std::vector<glm::vec3> positions = std::vector<glm::vec3>(nos);
	std::vector<glm::vec3> directions = std::vector<glm::vec3>(nos);
  std::vector<std::array<VFRendering::Geometry::index_type, 4>> tetrahedra_indices;

	// ToDo: Update the pointer to our Data instead of copying Data?
	// Positions and directions
	//		get pointer
	scalar *spins, *spin_pos;
	bool keep_magnitudes = false;
	if (true)
	{
		spins = System_Get_Spin_Directions(state.get());
	}
	else
	{
		spins = System_Get_Effective_Field(state.get());
		keep_magnitudes = true;
	}
	spin_pos = Geometry_Get_Spin_Positions(state.get());
	//		copy
	for (int i = 0; i < nos; ++i)
	{
		positions[i] = glm::vec3(spin_pos[0 * nos + i], spin_pos[1 * nos + i], spin_pos[2 * nos + i]);
	}
	for (int i = 0; i < nos; ++i)
	{
		directions[i] = glm::vec3(spins[i], spins[nos + i], spins[2 * nos + i]);
	}
	//    normalize if needed
	if (keep_magnitudes)
	{
		float max_length = 0;
		for (auto direction : directions)
		{
			max_length = std::max(max_length, glm::length(direction));
		}
		if (max_length > 0)
		{
			for (auto& direction : directions)
			{
				direction /= max_length;
			}
		}
	}
	else
	{
		for (auto& direction : directions)
		{
			direction = glm::normalize(direction);
		}
	}

	// Triangles and Tetrahedra
	//		get tetrahedra
	if (Geometry_Get_Dimensionality(state.get()) == 3)
	{
		const std::array<VFRendering::Geometry::index_type, 4> *tetrahedra_indices_ptr = nullptr;
		int num_tetrahedra = Geometry_Get_Triangulation(state.get(), reinterpret_cast<const int **>(&tetrahedra_indices_ptr));
		tetrahedra_indices = std::vector<std::array<VFRendering::Geometry::index_type, 4>>(tetrahedra_indices_ptr, tetrahedra_indices_ptr + num_tetrahedra);
	}
  
	//		get bounds
	float b_min[3], b_max[3];
	Geometry_Get_Bounds(state.get(), b_min, b_max);
	glm::vec3 bounds_min = glm::make_vec3(b_min);
	glm::vec3 bounds_max = glm::make_vec3(b_max);
    glm::vec3 center = (bounds_min + bounds_max) * 0.5f;
    m_view.setOption<VFRendering::View::Option::SYSTEM_CENTER>(center);
	if (_reset_camera)
	{
		setCameraToDefault();
    _reset_camera = false;
	}
  
  bool is_2d = (Geometry_Get_Dimensionality(state.get()) < 3);
  VFRendering::Geometry geometry(positions, {}, tetrahedra_indices, is_2d);
  m_view.update(geometry, directions);
}

void SpinWidget::paintGL() {
	// ToDo: This does not catch the case that we have no simulation running
	//		 but we switched between images or chains...
	if (Simulation_Running_Any(state.get()))
	{
		this->updateData();
	}
  
    m_view.draw();
	QTimer::singleShot(1, this, SLOT(update()));
}

void SpinWidget::mousePressEvent(QMouseEvent *event) {
  m_previous_mouse_position = event->pos();
}

void SpinWidget::mouseMoveEvent(QMouseEvent *event) {
  glm::vec2 current_mouse_position = glm::vec2(event->pos().x(), event->pos().y()) * (float)devicePixelRatio();
  glm::vec2 previous_mouse_position = glm::vec2(m_previous_mouse_position.x(), m_previous_mouse_position.y()) * (float)devicePixelRatio();
  m_previous_mouse_position = event->pos();
  
  if (event->buttons() & Qt::LeftButton || event->buttons() & Qt::RightButton) {
    auto movement_mode = VFRendering::CameraMovementModes::ROTATE;
    if ((event->modifiers() & Qt::AltModifier) == Qt::AltModifier || event->buttons() & Qt::RightButton) {
      movement_mode = VFRendering::CameraMovementModes::TRANSLATE;
    }
    m_view.mouseMove(previous_mouse_position, current_mouse_position, movement_mode);
    ((QWidget *)this)->update();
  }
}

void SpinWidget::setCameraToDefault() {
    float camera_distance = 30.0f;
    auto center_position = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
    auto camera_position = center_position + camera_distance * glm::vec3(0, 0, 1);
    auto up_vector = glm::vec3(0, 1, 0);

    VFRendering::Options options;
    options.set<VFRendering::View::Option::CAMERA_POSITION>(camera_position);
    options.set<VFRendering::View::Option::CENTER_POSITION>(center_position);
    options.set<VFRendering::View::Option::UP_VECTOR>(up_vector);
    m_view.updateOptions(options);
}

void SpinWidget::setCameraToX() {
    float camera_distance = glm::length(options().get<VFRendering::View::Option::CENTER_POSITION>()-options().get<VFRendering::View::Option::CAMERA_POSITION>());
    auto center_position = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
    auto camera_position = center_position + camera_distance * glm::vec3(1, 0, 0);
    auto up_vector = glm::vec3(0, 0, 1);
    
    VFRendering::Options options;
    options.set<VFRendering::View::Option::CAMERA_POSITION>(camera_position);
    options.set<VFRendering::View::Option::CENTER_POSITION>(center_position);
    options.set<VFRendering::View::Option::UP_VECTOR>(up_vector);
    m_view.updateOptions(options);
}

void SpinWidget::setCameraToY() {
    float camera_distance = glm::length(options().get<VFRendering::View::Option::CENTER_POSITION>()-options().get<VFRendering::View::Option::CAMERA_POSITION>());
    auto center_position = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
    auto camera_position = center_position + camera_distance * glm::vec3(0, -1, 0);
    auto up_vector = glm::vec3(0, 0, 1);
    
    VFRendering::Options options;
    options.set<VFRendering::View::Option::CAMERA_POSITION>(camera_position);
    options.set<VFRendering::View::Option::CENTER_POSITION>(center_position);
    options.set<VFRendering::View::Option::UP_VECTOR>(up_vector);
    m_view.updateOptions(options);
}

void SpinWidget::setCameraToZ() {
    float camera_distance = glm::length(options().get<VFRendering::View::Option::CENTER_POSITION>()-options().get<VFRendering::View::Option::CAMERA_POSITION>());
    auto center_position = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
    auto camera_position = center_position + camera_distance * glm::vec3(0, 0, 1);
    auto up_vector = glm::vec3(0, 1, 0);
    
    VFRendering::Options options;
    options.set<VFRendering::View::Option::CAMERA_POSITION>(camera_position);
    options.set<VFRendering::View::Option::CENTER_POSITION>(center_position);
    options.set<VFRendering::View::Option::UP_VECTOR>(up_vector);
    m_view.updateOptions(options);
}

void SpinWidget::setCameraPositon(const glm::vec3& camera_position)
{
    m_view.setOption<VFRendering::View::Option::CAMERA_POSITION>(camera_position);
}

void SpinWidget::setCameraFocus(const glm::vec3& center_position)
{
    m_view.setOption<VFRendering::View::Option::CENTER_POSITION>(center_position);
}

void SpinWidget::setCameraUpVector(const glm::vec3& up_vector)
{
    m_view.setOption<VFRendering::View::Option::UP_VECTOR>(up_vector);
}

glm::vec3 SpinWidget::getCameraPositon()
{
    return options().get<VFRendering::View::Option::CAMERA_POSITION>();
}

glm::vec3 SpinWidget::getCameraFocus()
{
    return options().get<VFRendering::View::Option::CENTER_POSITION>();
}

glm::vec3 SpinWidget::getCameraUpVector()
{
	return options().get<VFRendering::View::Option::UP_VECTOR>();
}

float SpinWidget::getFramesPerSecond() const {
  return m_view.getFramerate();
}

void SpinWidget::wheelEvent(QWheelEvent *event) {
  float wheel_delta = event->angleDelta().y();
  m_view.mouseScroll(wheel_delta * 0.1);
  ((QWidget *)this)->update();
}

const VFRendering::Options& SpinWidget::options() const {
  return m_view.options();

}

float SpinWidget::verticalFieldOfView() const {
  return m_view.options().get<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>();
}

void SpinWidget::setVerticalFieldOfView(float vertical_field_of_view) {
	makeCurrent();
    m_view.setOption<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>(vertical_field_of_view);
}

glm::vec3 SpinWidget::backgroundColor() const {
  return options().get<VFRendering::View::Option::BACKGROUND_COLOR>();
}

void SpinWidget::setBackgroundColor(glm::vec3 background_color) {
    makeCurrent();
    m_view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(background_color);
}

glm::vec3 SpinWidget::boundingBoxColor() const {
    return m_view.options().get<VFRendering::BoundingBoxRenderer::Option::COLOR>();
}

void SpinWidget::setBoundingBoxColor(glm::vec3 bounding_box_color) {
    makeCurrent();
    m_view.setOption<VFRendering::BoundingBoxRenderer::Option::COLOR>(bounding_box_color);
}

bool SpinWidget::isCoordinateSystemEnabled() const {
  // TODO: return options().get<GLSpins::Option::SHOW_COORDINATE_SYSTEM>();
  return false;
}

void SpinWidget::enableCoordinateSystem(bool enabled) {
	makeCurrent();
  // TODO: auto option = Options<GLSpins>::withOption<GLSpins::Option::SHOW_COORDINATE_SYSTEM>(enabled);
  // TODO: gl_spins->updateOptions(option);
}

bool SpinWidget::isBoundingBoxEnabled() const {
  // TODO: return options().get<GLSpins::Option::SHOW_BOUNDING_BOX>();
  return false;
}

void SpinWidget::enableBoundingBox(bool enabled) {
	makeCurrent();
  // TODO: auto option = Options<GLSpins>::withOption<GLSpins::Option::SHOW_BOUNDING_BOX>(enabled);
  // TODO: gl_spins->updateOptions(option);
}

GLSpins::WidgetLocation SpinWidget::coordinateSystemPosition() const {
  // TODO: return options().get<GLSpins::Option::COORDINATE_SYSTEM_LOCATION>();
  return GLSpins::WidgetLocation::BOTTOM_LEFT;
}

void SpinWidget::setCoordinateSystemPosition(GLSpins::WidgetLocation coordinatesystem_position) {
	makeCurrent();
  // TODO: auto option = Options<GLSpins>::withOption<GLSpins::Option::COORDINATE_SYSTEM_LOCATION>(coordinatesystem_position);
  // TODO: gl_spins->updateOptions(option);
}



void SpinWidget::setupRenderers()
{
	makeCurrent();

	// Create renderers vector
	std::vector<std::pair<std::shared_ptr<VFRendering::RendererBase>, std::array<float, 4>>> renderers;
	renderers.push_back({ this->m_mainview,{ 0, 0, 1, 1 } });
	if (show_miniview)
		renderers.push_back({ this->m_miniview, this->m_miniview_position });
	if (show_coordinatesystem)
		renderers.push_back({ this->m_coordinatecross, this->m_coordinatecross_position });

	// Update View
	m_view.renderers(renderers);
}

void SpinWidget::setVisualizationMode(GLSpins::VisualizationMode visualization_mode)
{
	if (visualization_mode == GLSpins::VisualizationMode::SYSTEM)
	{
		this->m_mainview = this->m_system;
		this->m_miniview = this->m_sphere;
	}
	else if (visualization_mode == GLSpins::VisualizationMode::SPHERE)
	{
		this->m_mainview = this->m_sphere;
		this->m_miniview = this->m_system;
	}

	this->setupRenderers();
}


void SpinWidget::setVisualizationSystem(bool arrows, bool boundingbox, bool surface, bool isosurface)
{
	bool system_is_mainview = false;
	if (this->m_system == this->m_mainview) system_is_mainview = true;

	this->show_arrows = arrows;
	this->show_boundingbox = boundingbox;
	this->show_surface = surface;
	this->show_isosurface = isosurface;

	// Create System
	std::vector<std::shared_ptr<VFRendering::RendererBase>> system(0);
	if (show_arrows)
		system.push_back(this->m_renderer_arrows);
	if (show_boundingbox)
		system.push_back(this->m_renderer_boundingbox);
	if (show_surface)
		system.push_back(this->m_renderer_surface);
	if (show_isosurface)
		system.push_back(this->m_renderer_isosurface);
	this->m_system = std::make_shared<VFRendering::CombinedRenderer>(m_view, system);
	//*this->m_system = VFRendering::CombinedRenderer(m_view, system);

	if (system_is_mainview) this->m_mainview = this->m_system;
	else this->m_miniview = this->m_system;

	this->setupRenderers();
}


void SpinWidget::setVisualizationMiniview(bool show, std::array<float, 4> position)
{
	this->show_miniview = show;
	this->m_miniview_position = position;
	this->setupRenderers();
}

void SpinWidget::setVisualizationCoordinatesystem(bool show, std::array<float, 4> position)
{
	this->show_coordinatesystem = show;
	this->m_coordinatecross_position = position;
	this->setupRenderers();
}


glm::vec2 SpinWidget::zRange() const {
    return m_z_range;
}

void SpinWidget::setZRange(glm::vec2 z_range) {
    m_z_range = z_range;
    std::string is_visible_implementation;
    if (z_range.x <= -1 && z_range.y >= 1) {
        is_visible_implementation = "bool is_visible(vec3 position, vec3 direction) { return true; }";
    } else if (z_range.x <= -1) {
        std::ostringstream sstream;
        sstream << "bool is_visible(vec3 position, vec3 direction) { float z_max = ";
        sstream << z_range.y;
        sstream << "; return normalize(direction).z <= z_max; }";
        is_visible_implementation = sstream.str();
    }  else if (z_range.y >= 1) {
        std::ostringstream sstream;
        sstream << "bool is_visible(vec3 position, vec3 direction) { float z_min = ";
        sstream << z_range.x;
        sstream << "; return normalize(direction).z >= z_min; }";
        is_visible_implementation = sstream.str();
    } else {
        std::ostringstream sstream;
        sstream << "bool is_visible(vec3 position, vec3 direction) { float z_min = ";
        sstream << z_range.x;
        sstream << "; float z_max = ";
        sstream << z_range.y;
        sstream << "; float z = normalize(direction).z;  return z >= z_min && z <= z_max; }";
        is_visible_implementation = sstream.str();
    }
    makeCurrent();
    m_view.setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(is_visible_implementation);
}

float SpinWidget::isovalue() const {
  return options().get<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>();
}

void SpinWidget::setIsovalue(float isovalue) {
  makeCurrent();
  m_view.setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(isovalue);
}


SpinWidget::Colormap SpinWidget::colormap() const {
  return m_colormap;
}

void SpinWidget::setColormap(Colormap colormap) {
  m_colormap = colormap;
  
  std::string colormap_implementation;
  switch (colormap) {
    case Colormap::HSV:
      colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::HSV);
      break;
    case Colormap::BLUE_RED:
      colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::BLUERED);
	  break;
    case Colormap::BLUE_GREEN_RED:
      colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::BLUEGREENRED);
      break;
    case Colormap::BLUE_WHITE_RED:
      colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::BLUEWHITERED);
      break;
    // Custom color maps not included in VFRendering:
    case Colormap::HSV_NO_Z:
      colormap_implementation = R"(
      float atan2(float y, float x) {
        return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
      }
      vec3 hsv2rgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
      }
      vec3 colormap(vec3 direction) {
        vec2 xy = normalize(direction.xy);
        float hue = atan2(xy.x, xy.y) / 3.14159 / 2.0;
        return hsv2rgb(vec3(hue, 1.0, 1.0));
      }
      )";
      break;
    default:
      colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::HSV);
      break;
  }
  makeCurrent();
  m_view.setOption<VFRendering::View::COLORMAP_IMPLEMENTATION>(colormap_implementation);
}

glm::vec2 SpinWidget::spherePointSizeRange() const {
  return options().get<VFRendering::VectorSphereRenderer::Option::POINT_SIZE_RANGE>();
}

void SpinWidget::setSpherePointSizeRange(glm::vec2 sphere_point_size_range) {
	makeCurrent();
    m_view.setOption<VFRendering::VectorSphereRenderer::Option::POINT_SIZE_RANGE>(sphere_point_size_range);
}
