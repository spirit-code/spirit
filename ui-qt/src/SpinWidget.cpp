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
  
    setColormap(GLSpins::Colormap::HSV);
    
    default_options.set<VFRendering::ArrowRenderer::Option::CONE_RADIUS>(0.125f);
    default_options.set<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>(0.3f);
    default_options.set<VFRendering::ArrowRenderer::Option::CYLINDER_RADIUS>(0.0625f);
    default_options.set<VFRendering::ArrowRenderer::Option::CYLINDER_HEIGHT>(0.35f);
    
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

    m_view = std::make_shared<VFRendering::View>();
    m_view->options(default_options);
  
    auto isosurface_renderer_ptr = std::make_shared<VFRendering::IsosurfaceRenderer>(*m_view);
    isosurface_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>([] (const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
        (void)position;
        return direction.z;
    });
    isosurface_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(0.0);
    auto arrow_renderer_ptr = std::make_shared<VFRendering::ArrowRenderer>(*m_view);

    std::vector<std::shared_ptr<VFRendering::RendererBase>> renderers = {
        isosurface_renderer_ptr,
        arrow_renderer_ptr
    };
    m_view->renderers({
        {std::make_shared<VFRendering::CombinedRenderer>(*m_view, renderers), {{0, 0, 1, 1}}},
        {std::make_shared<VFRendering::VectorSphereRenderer>(*m_view), {{0, 0, 0.2, 0.2}}},
        {std::make_shared<VFRendering::CoordinateSystemRenderer>(*m_view), {{0.8, 0, 0.2, 0.2}}}
    });
    updateData();
}

void SpinWidget::teardownGL() {
	// GLSpins::terminate();
}

void SpinWidget::resizeGL(int width, int height) {
  if (m_view) {
    m_view->setFramebufferSize(width*devicePixelRatio(), height*devicePixelRatio());
  }
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
    if (m_view) {
        m_view->setOption<VFRendering::View::Option::SYSTEM_CENTER>(center);
    } else {
        default_options.set<VFRendering::View::Option::SYSTEM_CENTER>(center);
    }
	if (_reset_camera)
	{
		setCameraToDefault();
    _reset_camera = false;
	}
  
  if (m_view) {
    bool is_2d = (Geometry_Get_Dimensionality(state.get()) < 3);
    VFRendering::Geometry geometry(positions, {}, tetrahedra_indices, is_2d);
    m_view->update(geometry, directions);
  }
}

void SpinWidget::paintGL() {
	// ToDo: This does not catch the case that we have no simulation running
	//		 but we switched between images or chains...
	if (Simulation_Running_Any(state.get()))
	{
		this->updateData();
	}
  
  if (m_view) {
    m_view->draw();
  }
	QTimer::singleShot(1, this, SLOT(update()));
}

void SpinWidget::mousePressEvent(QMouseEvent *event) {
  m_previous_mouse_position = event->pos();
}

void SpinWidget::mouseMoveEvent(QMouseEvent *event) {
  glm::vec2 current_mouse_position = glm::vec2(event->pos().x(), event->pos().y()) * (float)devicePixelRatio();
  glm::vec2 previous_mouse_position = glm::vec2(m_previous_mouse_position.x(), m_previous_mouse_position.y()) * (float)devicePixelRatio();
  m_previous_mouse_position = event->pos();
  
  if (m_view) {
    if (event->buttons() & Qt::LeftButton) {
      auto movement_mode = VFRendering::CameraMovementModes::ROTATE;
      if ((event->modifiers() & Qt::AltModifier) == Qt::AltModifier) {
        movement_mode = VFRendering::CameraMovementModes::TRANSLATE;
      }
      m_view->mouseMove(previous_mouse_position, current_mouse_position, movement_mode);
      ((QWidget *)this)->update();
    }
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
    if (m_view) {
        m_view->updateOptions(options);
    } else {
        default_options.update(options);
    }
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
    if (m_view) {
        m_view->updateOptions(options);
    } else {
        default_options.update(options);
    }
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
    if (m_view) {
        m_view->updateOptions(options);
    } else {
        default_options.update(options);
    }
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
    if (m_view) {
        m_view->updateOptions(options);
    } else {
        default_options.update(options);
    }
}

void SpinWidget::setCameraPositonTo(float x, float y, float z)
{
    if (m_view) {
        m_view->setOption<VFRendering::View::Option::CAMERA_POSITION>({x, y, z});
    } else {
        default_options.set<VFRendering::View::Option::CAMERA_POSITION>({x, y, z});
    }
}

void SpinWidget::setCameraFocusTo(float x, float y, float z)
{
    if (m_view) {
        m_view->setOption<VFRendering::View::Option::CENTER_POSITION>({x, y, z});
    } else {
        default_options.set<VFRendering::View::Option::CENTER_POSITION>({x, y, z});
    }
}

void SpinWidget::setCameraUpvectorTo(float x, float y, float z)
{
    if (m_view) {
        m_view->setOption<VFRendering::View::Option::UP_VECTOR>({x, y, z});
    } else {
        default_options.set<VFRendering::View::Option::UP_VECTOR>({x, y, z});
    }
}

std::vector<float> SpinWidget::getCameraPositon()
{
    glm::vec3 camera_position = options().get<VFRendering::View::Option::CAMERA_POSITION>();
    return {camera_position.x, camera_position.y, camera_position.z};
}

std::vector<float> SpinWidget::getCameraFocus()
{
    glm::vec3 center_position = options().get<VFRendering::View::Option::CENTER_POSITION>();
    return {center_position.x, center_position.y, center_position.z};
}

std::vector<float> SpinWidget::getCameraUpvector()
{
	glm::vec3 up_vector = options().get<VFRendering::View::Option::UP_VECTOR>();
    return {up_vector.x, up_vector.y, up_vector.z};
}

float SpinWidget::getFramesPerSecond() const {
  if (m_view) {
    return m_view->getFramerate();
  }
  return 0;
}

void SpinWidget::wheelEvent(QWheelEvent *event) {
  float wheel_delta = event->angleDelta().y();
  if (m_view) {
    m_view->mouseScroll(wheel_delta * 0.1);
    ((QWidget *)this)->update();
  }
}

const VFRendering::Options& SpinWidget::options() const {
  if (m_view) {
    return m_view->options();
  }
  return default_options;
}

float SpinWidget::verticalFieldOfView() const {
  if (m_view) {
    return m_view->options().get<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>();
  } else {
    return VFRendering::Options().get<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>();
  }
}

void SpinWidget::setVerticalFieldOfView(float vertical_field_of_view) {
	makeCurrent();
  if (m_view) {
    m_view->setOption<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>(vertical_field_of_view);
  } else {
      default_options.set<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>(vertical_field_of_view);
  }
}

glm::vec3 SpinWidget::backgroundColor() const {
  return options().get<VFRendering::View::Option::BACKGROUND_COLOR>();
}

void SpinWidget::setBackgroundColor(glm::vec3 background_color) {
  makeCurrent();
  if (m_view) {
    m_view->setOption<VFRendering::View::Option::BACKGROUND_COLOR>(background_color);
  } else {
    default_options.set<VFRendering::View::Option::BACKGROUND_COLOR>(background_color);
  }
}

glm::vec3 SpinWidget::boundingBoxColor() const {
  if (m_view) {
    return m_view->options().get<VFRendering::BoundingBoxRenderer::Option::COLOR>();
  } else {
    return VFRendering::Options().get<VFRendering::BoundingBoxRenderer::Option::COLOR>();
  }
}

void SpinWidget::setBoundingBoxColor(glm::vec3 bounding_box_color) {
  makeCurrent();
  if (m_view) {
    m_view->setOption<VFRendering::BoundingBoxRenderer::Option::COLOR>(bounding_box_color);
  } else {
    default_options.set<VFRendering::BoundingBoxRenderer::Option::COLOR>(bounding_box_color);
  }
}

bool SpinWidget::isMiniviewEnabled() const {
  // TODO: return options().get<GLSpins::Option::SHOW_MINIVIEW>();
  return false;
}

void SpinWidget::enableMiniview(bool enabled) {
	makeCurrent();
  // TODO: auto option = Options<GLSpins>::withOption<GLSpins::Option::SHOW_MINIVIEW>(enabled);
  // TODO: gl_spins->updateOptions(option);
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

GLSpins::WidgetLocation SpinWidget::miniviewPosition() const {
  // TODO: return options().get<GLSpins::Option::MINIVIEW_LOCATION>();
  return GLSpins::WidgetLocation::BOTTOM_LEFT;
}

void SpinWidget::setMiniviewPosition(GLSpins::WidgetLocation miniview_position) {
	makeCurrent();
  // TODO: auto option = Options<GLSpins>::withOption<GLSpins::Option::MINIVIEW_LOCATION>(miniview_position);
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

GLSpins::VisualizationMode SpinWidget::visualizationMode() const {
  // TODO: return options().get<GLSpins::Option::VISUALIZATION_MODE>();
  return GLSpins::VisualizationMode::ARROWS;
}

void SpinWidget::setVisualizationMode(GLSpins::VisualizationMode visualization_mode) {
	makeCurrent();
  // TODO: auto option = Options<GLSpins>::withOption<GLSpins::Option::VISUALIZATION_MODE>(visualization_mode);
  // TODO: gl_spins->updateOptions(option);
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
    if (m_view) {
        makeCurrent();
        m_view->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(is_visible_implementation);
    } else {
        default_options.set<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(is_visible_implementation);
    }
}

float SpinWidget::isovalue() const {
  return options().get<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>();
}

void SpinWidget::setIsovalue(float isovalue) {
  makeCurrent();
  if (m_view) {
    m_view->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(isovalue);
  } else {
    default_options.set<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(isovalue);
  }
}


GLSpins::Colormap SpinWidget::colormap() const {
  return m_colormap;
}

void SpinWidget::setColormap(GLSpins::Colormap colormap) {
  m_colormap = colormap;
  
  std::string colormap_implementation;
  switch (colormap) {
    case GLSpins::Colormap::HSV:
      colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::HSV);
      break;
    case GLSpins::Colormap::BLUE_RED:
      colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::BLUERED);
	  break;
    case GLSpins::Colormap::BLUE_GREEN_RED:
      colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::BLUEGREENRED);
      break;
    case GLSpins::Colormap::BLUE_WHITE_RED:
      colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::BLUEWHITERED);
      break;
    // Custom color maps not included in VFRendering:
    case GLSpins::Colormap::HSV_NO_Z:
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
  if (m_view) {
    makeCurrent();
    m_view->setOption<VFRendering::View::COLORMAP_IMPLEMENTATION>(colormap_implementation);
  } else {
    default_options.set<VFRendering::View::COLORMAP_IMPLEMENTATION>(colormap_implementation);
  }
}

glm::vec2 SpinWidget::spherePointSizeRange() const {
  return options().get<VFRendering::VectorSphereRenderer::Option::POINT_SIZE_RANGE>();
}

void SpinWidget::setSpherePointSizeRange(glm::vec2 sphere_point_size_range) {
	makeCurrent();
  if (m_view) {
    m_view->setOption<VFRendering::VectorSphereRenderer::Option::POINT_SIZE_RANGE>(sphere_point_size_range);
  } else {
    default_options.set<VFRendering::VectorSphereRenderer::Option::POINT_SIZE_RANGE>(sphere_point_size_range);
  }
}
