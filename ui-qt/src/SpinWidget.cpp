#include <fstream>
#include <sstream>
#include "SpinWidget.hpp"

#include <QTimer>
#include <QMouseEvent>
#include <QtWidgets>

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
#include "Interface_Hamiltonian.h"


SpinWidget::SpinWidget(std::shared_ptr<State> state, QWidget *parent) : QOpenGLWidget(parent)
{
    this->state = state;

	// QT Widget Settings
    setFocusPolicy(Qt::StrongFocus);
    QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    sizePolicy.setHorizontalStretch(0);
    sizePolicy.setVerticalStretch(0);
    this->setSizePolicy(sizePolicy);
    this->setMinimumSize(200,200);
    this->setBaseSize(600,600);

	// Default VFRendering Settings

    setColormap(Colormap::HSV);
    
    m_view.setOption<VFRendering::ArrowRenderer::Option::CONE_RADIUS>(0.125f);
    m_view.setOption<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>(0.3f);
    m_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_RADIUS>(0.0625f);
    m_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_HEIGHT>(0.35f);
	
    setZRange({-1, 1});

	this->m_source = 0;
	this->visMode = VisualizationMode::SYSTEM;
	this->m_location_coordinatesystem = WidgetLocation::BOTTOM_RIGHT;
	this->m_location_miniview = WidgetLocation::BOTTOM_LEFT;
	this->show_arrows = true;
	this->show_boundingbox = true;
	this->show_isosurface = false;
	this->m_isocomponent = 2;
	this->m_isosurfaceshadows = false;
	this->show_surface = false;
	this->show_miniview = true;
	this->show_coordinatesystem = true;

	// 		Initial camera position
	this->_reset_camera = false;

	// 		Setup Arrays
	this->updateData();

	// 		Read persistent settings
	this->readSettings();
}

void SpinWidget::initializeGL()
{
    // Get GL context
    makeCurrent();
    // Initialize the visualisation options

	float b_min[3], b_max[3];
	Geometry_Get_Bounds(state.get(), b_min, b_max);
	glm::vec3 bounds_min = glm::make_vec3(b_min);
	glm::vec3 bounds_max = glm::make_vec3(b_max);
	glm::vec2 x_range{bounds_min[0], bounds_max[0]};
	glm::vec2 y_range{bounds_min[1], bounds_max[1]};
	glm::vec2 z_range{bounds_min[2], bounds_max[2]};
	glm::vec3 bounding_box_center = { (bounds_min[0] + bounds_max[0]) / 2, (bounds_min[1] + bounds_max[1]) / 2, (bounds_min[2] + bounds_max[2]) / 2 };
	glm::vec3 bounding_box_side_lengths = { bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], bounds_max[2] - bounds_min[2] };

	// Create renderers
	//	System
	this->m_renderer_arrows = std::make_shared<VFRendering::ArrowRenderer>(m_view);

	float indi_length = glm::length(bounds_max - bounds_min)*0.05;
	int   indi_dashes = 5;
	float indi_dashes_per_length = (float)indi_dashes / indi_length;

	bool periodical[3];
	Hamiltonian_Get_Boundary_Conditions(this->state.get(), periodical);
	glm::vec3 indis{ indi_length*periodical[0], indi_length*periodical[1], indi_length*periodical[2] };

	this->m_renderer_boundingbox = std::make_shared<VFRendering::BoundingBoxRenderer>(VFRendering::BoundingBoxRenderer::forCuboid(m_view, bounding_box_center, bounding_box_side_lengths, indis, indi_dashes_per_length));
	if (Geometry_Get_Dimensionality(this->state.get()) == 2)
	{
		this->m_renderer_surface_2D = std::make_shared<VFRendering::SurfaceRenderer>(m_view);
		this->m_renderer_surface = m_renderer_surface_2D;
	}
	else if (Geometry_Get_Dimensionality(this->state.get()) == 3)
	{
		this->m_renderer_surface_3D = std::make_shared<VFRendering::IsosurfaceRenderer>(m_view);
		this->m_renderer_surface = m_renderer_surface_3D;
		this->m_renderer_isosurface = std::make_shared<VFRendering::IsosurfaceRenderer>(m_view);
	}
	std::vector<std::shared_ptr<VFRendering::RendererBase>> renderers = {
		m_renderer_arrows,
		m_renderer_boundingbox
	};
	this->m_system = std::make_shared<VFRendering::CombinedRenderer>(m_view, renderers);

	if (Geometry_Get_Dimensionality(this->state.get()) == 2)
	{
		// 2D Surface options
		// No options yet...
	}
	else if (Geometry_Get_Dimensionality(this->state.get()) == 3)
	{
		// 3D Surface options
		this->m_renderer_surface_3D->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>([x_range, y_range, z_range](const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type
		{
			if (position.x < x_range.x || position.x > x_range.y || position.y < y_range.x || position.y > y_range.y || position.z < z_range.x || position.z > z_range.y) return 1;
			else if (position.x == x_range.x || position.x == x_range.y || position.y == y_range.x || position.y == y_range.y || position.z == z_range.x || position.z == z_range.y) return 0;
			else return -1;
		});
		this->m_renderer_surface_3D->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(0.0);
		// Isosurface options
		if (this->m_isocomponent == 0)
		{
			m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>([](const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
				(void)position;
				return direction.x;
			});
		}
		else if (this->m_isocomponent == 1)
		{
			m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>([](const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
				(void)position;
				return direction.y;
			});
		}
		else if (this->m_isocomponent == 2)
		{
			m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>([](const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
				(void)position;
				return direction.z;
			});
		}
		if (this->m_isosurfaceshadows)
		{
			m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>("float lighting(vec3 position, vec3 normal) { return abs(normal.z); }");
		}
		else
		{
			m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>("float lighting(vec3 position, vec3 normal) { return 1.0; }");
		}
		m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(0.0);
	}

	//	Sphere
	this->m_sphere = std::make_shared<VFRendering::VectorSphereRenderer>(m_view);

	//	Coordinate cross
	this->m_coordinatesystem = std::make_shared<VFRendering::CoordinateSystemRenderer>(m_view);
    this->m_coordinatesystem->setOption<VFRendering::CoordinateSystemRenderer::Option::NORMALIZE>(true);

	// Setup the View
	this->setVisualizationMode(this->visMode);

	// Configure System (Setup the renderers
	this->enableSystem(this->show_arrows, this->show_boundingbox, this->show_surface, this->show_isosurface);
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

	// ToDo: Update the pointer to our Data instead of copying Data?
	// Positions and directions
	//		get pointer
	scalar *spins, *spin_pos;
	spin_pos = Geometry_Get_Spin_Positions(state.get());
	if (this->m_source == 0)
		spins = System_Get_Spin_Directions(state.get());
	else if (this->m_source == 1)
		spins = System_Get_Effective_Field(state.get());
	//		copy
	/*positions.assign(spin_pos, spin_pos + 3*nos);
	directions.assign(spins, spins + 3*nos);*/
	for (int i = 0; i < nos; ++i)
	{
		positions[i] = glm::vec3(spin_pos[3*i], spin_pos[1 + 3*i], spin_pos[2 + 3*i]);
		directions[i] = glm::vec3(spins[3*i], spins[1 + 3*i], spins[2 + 3*i]);
	}
	//		rescale if effective field
	if (this->m_source == 1)
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

	// Triangles and Tetrahedra
	VFRendering::Geometry geometry;
	//		get tetrahedra
	if (Geometry_Get_Dimensionality(state.get()) == 3)
	{
		const std::array<VFRendering::Geometry::index_type, 4> *tetrahedra_indices_ptr = nullptr;
		int num_tetrahedra = Geometry_Get_Triangulation(state.get(), reinterpret_cast<const int **>(&tetrahedra_indices_ptr));
		std::vector<std::array<VFRendering::Geometry::index_type, 4>>  tetrahedra_indices(tetrahedra_indices_ptr, tetrahedra_indices_ptr + num_tetrahedra);
		geometry = VFRendering::Geometry(positions, {}, tetrahedra_indices, false);
	}
	else if (Geometry_Get_Dimensionality(state.get()) == 2)
	{
		bool is_2d = (Geometry_Get_Dimensionality(state.get()) < 3);
		int n_cells[3];
		Geometry_Get_N_Cells(state.get(), n_cells);
		//geometry = VFRendering::Geometry(positions, {}, {}, true);
		std::vector<float> xs(n_cells[0]), ys(n_cells[1]), zs(n_cells[2]);
		for (int i = 0; i < n_cells[0]; ++i) xs[i] = positions[i].x;
		for (int i = 0; i < n_cells[1]; ++i) ys[i] = positions[i*n_cells[0]].y;
		for (int i = 0; i < n_cells[2]; ++i) zs[i] = positions[i*n_cells[0] * n_cells[1]].z;
		geometry = VFRendering::Geometry::rectilinearGeometry(xs, ys, zs);
	}
	else if (Geometry_Get_Dimensionality(state.get()) < 2)
	{
		geometry = VFRendering::Geometry(positions, {}, {}, true);
		// std::cerr << std::endl << std::endl << "-------xxxxxxxx--------" << std::endl << std::endl;
	}
  
	//		get bounds
	float b_min[3], b_max[3];
	Geometry_Get_Bounds(state.get(), b_min, b_max);
	glm::vec3 bounds_min = glm::make_vec3(b_min);
	glm::vec3 bounds_max = glm::make_vec3(b_max);
    glm::vec3 center = (bounds_min + bounds_max) * 0.5f;
    m_view.setOption<VFRendering::View::Option::SYSTEM_CENTER>(center);
	if (this->_reset_camera)
	{
		setCameraToDefault();
    	this->_reset_camera = false;
	}

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

void SpinWidget::setVisualisationSource(int source)
{
	this->m_source = source;
}

void SpinWidget::mousePressEvent(QMouseEvent *event) {
  m_previous_mouse_position = event->pos();
}

void SpinWidget::mouseMoveEvent(QMouseEvent *event)
{
	float scale = 1;

	if (event->modifiers() & Qt::ShiftModifier)
	{
		scale = 10;
	}

	glm::vec2 current_mouse_position = glm::vec2(event->pos().x(), event->pos().y()) * (float)devicePixelRatio() * scale;
	glm::vec2 previous_mouse_position = glm::vec2(m_previous_mouse_position.x(), m_previous_mouse_position.y()) * (float)devicePixelRatio() * scale;
	m_previous_mouse_position = event->pos();
  
	if (event->buttons() & Qt::LeftButton || event->buttons() & Qt::RightButton)
	{
		auto movement_mode = VFRendering::CameraMovementModes::ROTATE;
		if ((event->modifiers() & Qt::AltModifier) == Qt::AltModifier || event->buttons() & Qt::RightButton)
		{
			movement_mode = VFRendering::CameraMovementModes::TRANSLATE;
		}
		m_view.mouseMove(previous_mouse_position, current_mouse_position, movement_mode);
		((QWidget *)this)->update();
	}
}



float SpinWidget::getFramesPerSecond() const
{
	return m_view.getFramerate();
}

void SpinWidget::wheelEvent(QWheelEvent *event)
{
	float scale = 1;

	if (event->modifiers() & Qt::ShiftModifier)
	{
		scale = 0.1;
	}

	float wheel_delta = event->angleDelta().y();
	m_view.mouseScroll(wheel_delta * 0.1 * scale);
	((QWidget *)this)->update();
}

const VFRendering::Options& SpinWidget::options() const
{
	return m_view.options();
}



void SpinWidget::moveCamera(float backforth, float rightleft, float updown)
{
	float scale = 1.0;
	//if (keyboardModifiers)

	auto movement_mode = VFRendering::CameraMovementModes::TRANSLATE;
	m_view.mouseMove({ 0,0 }, { rightleft, updown }, movement_mode);
	m_view.mouseScroll(backforth * 0.1);
	((QWidget *)this)->update();
}

void SpinWidget::rotateCamera(float theta, float phi)
{
	auto movement_mode = VFRendering::CameraMovementModes::ROTATE;
	m_view.mouseMove({ 0,0 }, { phi, theta }, movement_mode);
	((QWidget *)this)->update();
}


//////////////////////////////////////////////////////////////////////////////////////
///// --- Mode ---
void SpinWidget::setVisualizationMode(SpinWidget::VisualizationMode visualization_mode)
{
	if (visualization_mode == SpinWidget::VisualizationMode::SYSTEM)
	{
		this->visMode = VisualizationMode::SYSTEM;
		this->m_mainview = this->m_system;
		this->m_miniview = this->m_sphere;
	}
	else if (visualization_mode == SpinWidget::VisualizationMode::SPHERE)
	{
		this->visMode = VisualizationMode::SPHERE;
		this->m_mainview = this->m_sphere;
		this->m_miniview = this->m_system;
	}

	this->setupRenderers();
}

SpinWidget::VisualizationMode SpinWidget::visualizationMode()
{
	return this->visMode;
}

//////////////////////////////////////////////////////////////////////////////////////
///// --- MiniView ---
void SpinWidget::setVisualizationMiniview(bool show, SpinWidget::WidgetLocation location)
{
	enableMiniview(show);
	setMiniviewPosition(location);
}

bool SpinWidget::isMiniviewEnabled() const {
	return this->show_miniview;
}

void SpinWidget::enableMiniview(bool enabled) {
	this->show_miniview = enabled;
	setupRenderers();
}

SpinWidget::WidgetLocation SpinWidget::miniviewPosition() const {
	return this->m_location_miniview;
}

void SpinWidget::setMiniviewPosition(SpinWidget::WidgetLocation location) {
	this->m_location_miniview = location;
	this->setupRenderers();
}


//////////////////////////////////////////////////////////////////////////////////////
///// --- Coordinate System ---
void SpinWidget::setVisualizationCoordinatesystem(bool show, SpinWidget::WidgetLocation location)
{
	enableCoordinateSystem(show);
	setCoordinateSystemPosition(location);
}

bool SpinWidget::isCoordinateSystemEnabled() const {
	return this->show_coordinatesystem;
}

void SpinWidget::enableCoordinateSystem(bool enabled) {
	this->show_coordinatesystem = enabled;
	setupRenderers();
}

SpinWidget::WidgetLocation SpinWidget::coordinateSystemPosition() const {
	return this->m_location_coordinatesystem;
}

void SpinWidget::setCoordinateSystemPosition(SpinWidget::WidgetLocation location) {
	this->m_location_coordinatesystem = location;
	this->setupRenderers();
}

//////////////////////////////////////////////////////////////////////////////////////
///// --- System ---

/////	enable
void SpinWidget::enableSystem(bool arrows, bool boundingbox, bool surface, bool isosurface)
{
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
	if (show_surface && (Geometry_Get_Dimensionality(this->state.get()) == 2 || Geometry_Get_Dimensionality(this->state.get()) == 3))
		system.push_back(this->m_renderer_surface);
	if (show_isosurface)
		system.push_back(this->m_renderer_isosurface);
	this->m_system = std::make_shared<VFRendering::CombinedRenderer>(m_view, system);
	//*this->m_system = VFRendering::CombinedRenderer(m_view, system);

	if (this->visMode == VisualizationMode::SYSTEM) this->m_mainview = this->m_system;
	else this->m_miniview = this->m_system;

	this->setupRenderers();
}

/////	Arrows
void SpinWidget::setArrows(float size, int lod)
{
	if (lod < 3) lod = 3;
	
	// defaults
	float coneradius = 0.25f;
	float coneheight = 0.6f;
	float cylinderradius = 0.125f;
	float cylinderheight = 0.7f;

	makeCurrent();
	m_view.setOption<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>(coneheight * size);
	m_view.setOption<VFRendering::ArrowRenderer::Option::CONE_RADIUS>(coneradius * size);
	m_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_HEIGHT>(cylinderheight* size);
	m_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_RADIUS>(cylinderradius * size);
	m_view.setOption<VFRendering::ArrowRenderer::Option::LEVEL_OF_DETAIL>(lod);
}

float SpinWidget::arrowSize() const {
	float size = options().get<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>() / 0.6f;
	return size;
}

int SpinWidget::arrowLOD() const {
	int LOD = options().get<VFRendering::ArrowRenderer::Option::LEVEL_OF_DETAIL>();
	return LOD;
}

/////	Z Range (Arrows?)
glm::vec2 SpinWidget::zRange() const {
	return m_z_range;
}
void SpinWidget::setZRange(glm::vec2 z_range) {
	m_z_range = z_range;
	std::string is_visible_implementation;
	if (z_range.x <= -1 && z_range.y >= 1) {
		is_visible_implementation = "bool is_visible(vec3 position, vec3 direction) { return true; }";
	}
	else if (z_range.x <= -1) {
		std::ostringstream sstream;
		sstream << "bool is_visible(vec3 position, vec3 direction) { float z_max = ";
		sstream << z_range.y;
		sstream << "; return normalize(direction).z <= z_max; }";
		is_visible_implementation = sstream.str();
	}
	else if (z_range.y >= 1) {
		std::ostringstream sstream;
		sstream << "bool is_visible(vec3 position, vec3 direction) { float z_min = ";
		sstream << z_range.x;
		sstream << "; return normalize(direction).z >= z_min; }";
		is_visible_implementation = sstream.str();
	}
	else {
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

/////   Surface
void SpinWidget::setSurface(glm::vec2 x_range, glm::vec2 y_range, glm::vec2 z_range)
{
    makeCurrent();
	if (Geometry_Get_Dimensionality(this->state.get()) == 2)
	{
		// 2D Surface options
		// No options, yet...
	}
	else if (Geometry_Get_Dimensionality(this->state.get()) == 3)
	{
		// 3D Surface options
		if ((x_range.x >= x_range.y) || (y_range.x >= y_range.y) || (z_range.x >= z_range.y)) {
			this->m_renderer_surface_3D->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>([x_range, y_range, z_range](const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type
			{
				/* The selected cuboid does not exist */
				return 1;
			});
		}
		else {
			this->m_renderer_surface_3D->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>([x_range, y_range, z_range](const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type
			{
				(void)direction;

				/* Transform position in selected cuboid to position in unit cube [-1,1]^3 */
				glm::vec3 min = { x_range.x, y_range.x, z_range.x };
				glm::vec3 max = { x_range.y, y_range.y, z_range.y };
				glm::vec3 normalized_position = 2.0f * (position - min) / (max - min) - 1.0f;

				/* Calculate maximum metric / Chebyshev distance */
				glm::vec3 absolute_normalized_position = glm::abs(normalized_position);
				float max_norm = glm::max(glm::max(absolute_normalized_position.x, absolute_normalized_position.y), absolute_normalized_position.z);

				/* Translate so that the selected cuboid surface has an isovalue of 0 */
				return max_norm - 1.0f;
			});
		}
	}
	//this->setupRenderers();
}

/////	Isosurface
float SpinWidget::isovalue() const
{
	return options().get<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>();
}
void SpinWidget::setIsovalue(float isovalue)
{
	makeCurrent();
	m_view.setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(isovalue);
}
float SpinWidget::isocomponent() const
{
	return this->m_isocomponent;
}
void SpinWidget::setIsocomponent(int component)
{
	makeCurrent();
	this->m_isocomponent = component;
	if (this->m_isocomponent == 0)
	{
		m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>([](const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
			(void)position;
			return direction.x;
		});
	}
	else if (this->m_isocomponent == 1)
	{
		m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>([](const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
			(void)position;
			return direction.y;
		});
	}
	else if (this->m_isocomponent == 2)
	{
		m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>([](const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
			(void)position;
			return direction.z;
		});
	}
}
bool SpinWidget::isosurfaceshadows() const
{
	return this->m_isosurfaceshadows;
}
void SpinWidget::setIsosurfaceshadows(bool show)
{
	this->m_isosurfaceshadows = show;
	if (this->m_isosurfaceshadows)
	{
		m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>("float lighting(vec3 position, vec3 normal) { return abs(normal.z); }");
	}
	else
	{
		m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>("float lighting(vec3 position, vec3 normal) { return 1.0; }");
	}
	// The following setupRenderers should maybe not be necessary?
	this->setupRenderers();
}

//////////////////////////////////////////////////////////////////////////////////////
///// --- Sphere ---
glm::vec2 SpinWidget::spherePointSizeRange() const {
	return options().get<VFRendering::VectorSphereRenderer::Option::POINT_SIZE_RANGE>();
}
void SpinWidget::setSpherePointSizeRange(glm::vec2 sphere_point_size_range) {
	makeCurrent();
	m_view.setOption<VFRendering::VectorSphereRenderer::Option::POINT_SIZE_RANGE>(sphere_point_size_range);
}


//////////////////////////////////////////////////////////////////////////////////////
///// --- Renderer Setup ---
void SpinWidget::setupRenderers()
{
	makeCurrent();

	// Get positions
	std::array<float, 4> position_miniview;
	if (this->m_location_miniview == SpinWidget::WidgetLocation::BOTTOM_LEFT)
		position_miniview = { 0, 0, 0.2f, 0.2f };
	else if (this->m_location_miniview == SpinWidget::WidgetLocation::BOTTOM_RIGHT)
		position_miniview = { 0.8f, 0, 0.2f, 0.2f };
	else if (this->m_location_miniview == SpinWidget::WidgetLocation::TOP_LEFT)
		position_miniview = { 0, 0.8f, 0.2f, 0.2f };
	else if (this->m_location_miniview == SpinWidget::WidgetLocation::TOP_RIGHT)
		position_miniview = { 0.8f, 0.8f, 0.2f, 0.2f };

	std::array<float, 4> position_coordinatesystem;
	if (this->m_location_coordinatesystem == SpinWidget::WidgetLocation::BOTTOM_LEFT)
		position_coordinatesystem = { 0, 0, 0.2f, 0.2f };
	else if (this->m_location_coordinatesystem == SpinWidget::WidgetLocation::BOTTOM_RIGHT)
		position_coordinatesystem = { 0.8f, 0, 0.2f, 0.2f };
	else if (this->m_location_coordinatesystem == SpinWidget::WidgetLocation::TOP_LEFT)
		position_coordinatesystem = { 0, 0.8f, 0.2f, 0.2f };
	else if (this->m_location_coordinatesystem == SpinWidget::WidgetLocation::TOP_RIGHT)
		position_coordinatesystem = { 0.8f, 0.8f, 0.2f, 0.2f };

	// Create renderers vector
	std::vector<std::pair<std::shared_ptr<VFRendering::RendererBase>, std::array<float, 4>>> renderers;
	renderers.push_back({ this->m_mainview, { 0, 0, 1, 1 } });
	if (show_miniview)
		renderers.push_back({ this->m_miniview, position_miniview });
	if (show_coordinatesystem)
		renderers.push_back({ this->m_coordinatesystem, position_coordinatesystem });

	// Update View
	m_view.renderers(renderers);
}


//////////////////////////////////////////////////////////////////////////////////////
///// --- Colors ---
SpinWidget::Colormap SpinWidget::colormap() const
{
  return m_colormap;
}

void SpinWidget::setColormap(Colormap colormap)
{
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
    case Colormap::WHITE:
		colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::WHITE);
		break;
    case Colormap::GRAY:
		colormap_implementation = R"(
			vec3 colormap(vec3 direction) {
				return vec3(0.5, 0.5, 0.5);
			}
      	)";
      break;
    case Colormap::BLACK:
      	colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::BLACK);
      	break;
    default:
      	colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::HSV);
      	break;
  }
  makeCurrent();
  m_view.setOption<VFRendering::View::COLORMAP_IMPLEMENTATION>(colormap_implementation);
}

SpinWidget::Color SpinWidget::backgroundColor() const
{
	glm::vec3 color = m_view.options().get<VFRendering::View::Option::BACKGROUND_COLOR>();
	if (color == glm::vec3{ 0, 0, 0 }) return Color::BLACK;
	else if (color == glm::vec3{ 0.5, 0.5, 0.5 }) return Color::GRAY;
	else if (color == glm::vec3{ 1, 1, 1 }) return Color::WHITE;
	else return Color::OTHER;
}

void SpinWidget::setBackgroundColor(Color background_color)
{
	glm::vec3 color;
	if (background_color == Color::BLACK) color = { 0, 0, 0 };
	else if (background_color == Color::GRAY) color = { 0.5, 0.5, 0.5 };
	else if (background_color == Color::WHITE) color = { 1, 1, 1 };
	makeCurrent();
	m_view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(color);
}

SpinWidget::Color SpinWidget::boundingBoxColor() const
{
	glm::vec3 color = m_view.options().get<VFRendering::BoundingBoxRenderer::Option::COLOR>();
	if (color == glm::vec3{ 0, 0, 0 }) return Color::BLACK;
	else if (color == glm::vec3{ 0.5, 0.5, 0.5 }) return Color::GRAY;
	else if (color == glm::vec3{ 1, 1, 1 }) return Color::WHITE;
	else return Color::OTHER;
}

void SpinWidget::setBoundingBoxColor(Color bounding_box_color)
{
	glm::vec3 color;
	if (bounding_box_color == Color::BLACK) color = { 0, 0, 0 };
	else if (bounding_box_color == Color::GRAY) color = { 0.5, 0.5, 0.5 };
	else if (bounding_box_color == Color::WHITE) color = { 1, 1, 1 };
	makeCurrent();
	m_view.setOption<VFRendering::BoundingBoxRenderer::Option::COLOR>(color);
}

void SpinWidget::updateBoundingBoxIndicators()
{
	bool periodical[3];
	float b_min[3], b_max[3];
	Geometry_Get_Bounds(state.get(), b_min, b_max);
	glm::vec3 bounds_min = glm::make_vec3(b_min);
	glm::vec3 bounds_max = glm::make_vec3(b_max);
	glm::vec2 x_range{ bounds_min[0], bounds_max[0] };
	glm::vec2 y_range{ bounds_min[1], bounds_max[1] };
	glm::vec2 z_range{ bounds_min[2], bounds_max[2] };
	glm::vec3 bounding_box_center = { (bounds_min[0] + bounds_max[0]) / 2, (bounds_min[1] + bounds_max[1]) / 2, (bounds_min[2] + bounds_max[2]) / 2 };
	glm::vec3 bounding_box_side_lengths = { bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], bounds_max[2] - bounds_min[2] };

	float indi_length = glm::length(bounds_max - bounds_min)*0.05;
	int   indi_dashes = 5;
	float indi_dashes_per_length = (float)indi_dashes / indi_length;

	Hamiltonian_Get_Boundary_Conditions(this->state.get(), periodical);
	glm::vec3 indis{ indi_length*periodical[0], indi_length*periodical[1], indi_length*periodical[2] };

	this->m_renderer_boundingbox = std::make_shared<VFRendering::BoundingBoxRenderer>(VFRendering::BoundingBoxRenderer::forCuboid(m_view, bounding_box_center, bounding_box_side_lengths, indis, indi_dashes_per_length));
	//setupRenderers();
	//this->setVisualizationMode(this->visualizationMode());
	this->enableSystem(this->show_arrows, this->show_boundingbox, this->show_surface, this->show_isosurface);
}


//////////////////////////////////////////////////////////////////////////////////////
///// --- Camera ---
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
	float camera_distance = glm::length(options().get<VFRendering::View::Option::CENTER_POSITION>() - options().get<VFRendering::View::Option::CAMERA_POSITION>());
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
	float camera_distance = glm::length(options().get<VFRendering::View::Option::CENTER_POSITION>() - options().get<VFRendering::View::Option::CAMERA_POSITION>());
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
	float camera_distance = glm::length(options().get<VFRendering::View::Option::CENTER_POSITION>() - options().get<VFRendering::View::Option::CAMERA_POSITION>());
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
	auto system_center = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
	m_view.setOption<VFRendering::View::Option::CAMERA_POSITION>(system_center + camera_position);
}

void SpinWidget::setCameraFocus(const glm::vec3& center_position)
{
	auto system_center = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
	m_view.setOption<VFRendering::View::Option::CENTER_POSITION>(system_center + center_position);
}

void SpinWidget::setCameraUpVector(const glm::vec3& up_vector)
{
	m_view.setOption<VFRendering::View::Option::UP_VECTOR>(up_vector);
}

glm::vec3 SpinWidget::getCameraPositon()
{
	auto system_center = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
	return options().get<VFRendering::View::Option::CAMERA_POSITION>() - system_center;
}

glm::vec3 SpinWidget::getCameraFocus()
{
	auto system_center = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
	return options().get<VFRendering::View::Option::CENTER_POSITION>() - system_center;
}

glm::vec3 SpinWidget::getCameraUpVector()
{
	return options().get<VFRendering::View::Option::UP_VECTOR>();
}

float SpinWidget::verticalFieldOfView() const
{
	return m_view.options().get<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>();
}

void SpinWidget::setVerticalFieldOfView(float vertical_field_of_view)
{
	makeCurrent();
	m_view.setOption<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>(vertical_field_of_view);
}


// -----------------------------------------------------------------------------------
// --------------------- Persistent Settings -----------------------------------------
// -----------------------------------------------------------------------------------


void SpinWidget::writeSettings()
{
	QSettings settings("Spirit Code", "Spirit");

	settings.beginGroup("General");
	// VisMode
	settings.setValue("Mode", (int)(this->visualizationMode()));
	// Projection
	settings.setValue("FOV", (int)(this->verticalFieldOfView() * 100));
	// Sphere Point Size
	settings.setValue("SpherePointSize1", (int)(this->spherePointSizeRange().x * 100));
	settings.setValue("SpherePointSize2", (int)(this->spherePointSizeRange().y * 100));
	// System
	settings.setValue("Show Arrows", this->show_arrows);
	settings.setValue("Show Bounding Box", this->show_boundingbox);
	settings.setValue("Show Surface", this->show_surface);
	settings.setValue("Show Isosurface", this->show_isosurface);
	// MiniView
	settings.setValue("Show MiniView", this->show_miniview);
	settings.setValue("MiniView Position", (int)this->m_location_miniview);
	// Coordinate System
	settings.setValue("Show Coordinate System", this->show_coordinatesystem);
	settings.setValue("Coordinate System Position", (int)this->m_location_coordinatesystem);
	settings.endGroup();

	// Arrows
	settings.beginGroup("Arrows");
	settings.setValue("Size", (int)(this->arrowSize() * 100));
	settings.setValue("LOD", this->arrowLOD());
	settings.endGroup();

	// Isosurface
	settings.beginGroup("Isosurface");
	settings.setValue("Component", this->isocomponent());
	settings.setValue("Draw Shadows", this->isosurfaceshadows());
	settings.endGroup();

	// Colors
	settings.beginGroup("Colors");
	settings.setValue("Background Color", (int)backgroundColor());
	settings.setValue("Colormap", (int)colormap());
	settings.endGroup();

	// Camera
	settings.beginGroup("Camera");
    auto camera_position = this->getCameraPositon();
	auto center_position = this->getCameraFocus();
	auto up_vector = this->getCameraUpVector();
	settings.beginWriteArray("position");
	for(int dim=0; dim<3; ++dim)
	{
		settings.setArrayIndex(dim);
		settings.setValue("vecp", (int)(100*camera_position[dim]));
	}
	settings.endArray();
	settings.beginWriteArray("center");
	for(int dim=0; dim<3; ++dim)
	{
		settings.setArrayIndex(dim);
		settings.setValue("vecc", (int)(100*center_position[dim]));
	}
	settings.endArray();
	settings.beginWriteArray("up");
	for(int dim=0; dim<3; ++dim)
	{
		settings.setArrayIndex(dim);
		settings.setValue("vecu", (int)(100*up_vector[dim]));
	}
	settings.endArray();
	settings.endGroup();
}

void SpinWidget::readSettings()
{
	makeCurrent();
	QSettings settings("Spirit Code", "Spirit");

	if (settings.childGroups().contains("General"))
	{
		settings.beginGroup("General");
		// VisMode
		this->visMode = VisualizationMode(settings.value("Mode").toInt());
		// Projection
		this->setVerticalFieldOfView((float)(settings.value("FOV").toInt() / 100.0f));
		// Sphere Point Size
		this->setSpherePointSizeRange({ (settings.value("SpherePointSize1").toInt() / 100.0f), (settings.value("SpherePointSize2").toInt() / 100.0f) });
		// System
		this->show_arrows = settings.value("Show Arrows").toBool();
		this->show_boundingbox = settings.value("Show Bounding Box").toBool();
		this->show_surface = settings.value("Show Surface").toBool();
		this->show_isosurface = settings.value("Show Isosurface").toBool();
		// MiniView
		this->show_miniview = settings.value("Show MiniView").toBool();
		this->m_location_miniview = WidgetLocation(settings.value("MiniView Position").toInt());
		// Coordinate System
		this->show_coordinatesystem = settings.value("Show Coordinate System").toBool();
		this->m_location_coordinatesystem = WidgetLocation(settings.value("Coordinate System Position").toInt());
		settings.endGroup();
	}

	// Arrows
	if (settings.childGroups().contains("Arrows"))
	{
		settings.beginGroup("Arrows");
		// Projection
		this->setArrows((float)(settings.value("Size").toInt() / 100.0f), settings.value("LOD").toInt());
		settings.endGroup();
	}

	// Isosurface
	if (settings.childGroups().contains("Isosurface"))
	{
		settings.beginGroup("Isosurface");
		this->m_isocomponent = settings.value("Component").toInt();
		this->m_isosurfaceshadows = settings.value("Draw Shadows").toBool();
		settings.endGroup();
	}

	// Colors
	if (settings.childGroups().contains("Colors"))
	{
		settings.beginGroup("Colors");
		int background_color = settings.value("Background Color").toInt();
		this->setBackgroundColor((Color)background_color);
		if (background_color == 2) this->setBoundingBoxColor((Color)0);
		else this->setBoundingBoxColor((Color)2);
		int map = settings.value("Colormap").toInt();
		this->setColormap((Colormap)map);
		settings.endGroup();
	}

	// Camera
	if (settings.childGroups().contains("Camera"))
	{
		settings.beginGroup("Camera");
		glm::vec3 camera_position, center_position, up_vector;
		settings.beginReadArray("position");
		for(int dim=0; dim<3; ++dim)
		{
			settings.setArrayIndex(dim);
			camera_position[dim] = (float)(settings.value("vecp").toInt()/100.0f);
		}
		settings.endArray();
		this->setCameraPositon(camera_position);
		settings.beginReadArray("center");
		for(int dim=0; dim<3; ++dim)
		{
			settings.setArrayIndex(dim);
			center_position[dim] = (float)(settings.value("vecc").toInt()/100.0f);
		}
		settings.endArray();
		this->setCameraFocus(center_position);

		settings.beginReadArray("up");
		for(int dim=0; dim<3; ++dim)
		{
			settings.setArrayIndex(dim);
			up_vector[dim] = (float)(settings.value("vecu").toInt()/100.0f);
		}
		settings.endArray();
		this->setCameraUpVector(up_vector);
		settings.endGroup();
	}
}


void SpinWidget::closeEvent(QCloseEvent *event)
{
	writeSettings();
	event->accept();
}
