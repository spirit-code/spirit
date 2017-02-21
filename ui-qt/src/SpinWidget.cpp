// #include <fstream>
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

#include <locale>
#include <glm/gtc/type_ptr.hpp>

#include "Spirit/Geometry.h"
#include "Spirit/System.h"
#include "Spirit/Simulation.h"
#include "Spirit/Hamiltonian.h"


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
	setColormapRotationInverted(0, false, false);
    
    m_view.setOption<VFRendering::ArrowRenderer::Option::CONE_RADIUS>(0.125f);
    m_view.setOption<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>(0.3f);
    m_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_RADIUS>(0.0625f);
    m_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_HEIGHT>(0.35f);

	setOverallDirectionRange({ -1, 1 }, { -1, 1 }, { -1, 1 });

	float b_min[3], b_max[3];
	Geometry_Get_Bounds(state.get(), b_min, b_max);
	glm::vec3 bounds_min = glm::make_vec3(b_min);
	glm::vec3 bounds_max = glm::make_vec3(b_max);
	glm::vec2 x_range{ bounds_min[0], bounds_max[0] };
	glm::vec2 y_range{ bounds_min[1], bounds_max[1] };
	glm::vec2 z_range{ bounds_min[2], bounds_max[2] };
	setOverallPositionRange(x_range, y_range, z_range);
	
	this->m_source = 0;
	this->visMode = VisualizationMode::SYSTEM;
	this->m_location_coordinatesystem = WidgetLocation::BOTTOM_RIGHT;
	this->m_location_miniview = WidgetLocation::BOTTOM_LEFT;
	this->show_arrows = true;
	this->show_boundingbox = true;
	this->show_isosurface = false;

	idx_cycle=0;
	slab_displacements = glm::vec3{0,0,0};

	this->m_isocomponent = 2;
	this->m_isosurfaceshadows = false;
	this->show_surface = false;
	this->show_miniview = true;
	this->show_coordinatesystem = true;

	// 		Initial camera position
	this->_reset_camera = false;
	this->m_camera_rotate_free = false;

	// 		Setup Arrays
	this->updateData();

	// 		Read persistent settings
	this->readSettings();
	this->show_arrows = this->user_show_arrows;
	this->show_surface = this->user_show_surface;
	this->show_isosurface = this->user_show_isosurface;
	this->show_boundingbox = this->user_show_boundingbox;
	this->setVerticalFieldOfView(this->user_fov);
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
	
	std::vector<std::shared_ptr<VFRendering::RendererBase>> renderers = {
		m_renderer_arrows,
		m_renderer_boundingbox
	};
	

	if (Geometry_Get_Dimensionality(this->state.get()) == 2)
	{
		// 2D Surface options
		// No options yet...
		this->m_renderer_surface_2D = std::make_shared<VFRendering::SurfaceRenderer>(m_view);
		this->m_renderer_surface = m_renderer_surface_2D;
	}
	else if (Geometry_Get_Dimensionality(this->state.get()) == 3)
	{

		// 3D Surface options
		this->m_renderer_surface_3D = std::make_shared<VFRendering::IsosurfaceRenderer>(m_view);
		this->m_renderer_surface_3D->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(0.0);
		auto mini_diff = glm::vec2{0.00001f, -0.00001f};
		setSurface(x_range + mini_diff, y_range + mini_diff, z_range + mini_diff);

		// Isosurface options
		this->m_renderer_isosurface = std::make_shared<VFRendering::IsosurfaceRenderer>(m_view);
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
			m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>(
				"uniform vec3 uLightPosition;"
				"float lighting(vec3 position, vec3 normal)"
				"{"
				"    vec3 lightDirection = -normalize(uLightPosition-position);"
				"    float diffuse = 0.7*max(0.0, dot(normal, lightDirection));"
				"	 float ambient = 0.2;"
				"    return diffuse+ambient;"
				"}");
		}
		else
		{
			m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>("float lighting(vec3 position, vec3 normal) { return 1.0; }");
		}
		m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(0.0);

		this->m_renderer_surface = m_renderer_surface_3D;
	}

	this->m_system = std::make_shared<VFRendering::CombinedRenderer>(m_view, renderers);

	//	Sphere
	this->m_sphere = std::make_shared<VFRendering::VectorSphereRenderer>(m_view);

	//	Coordinate cross
	this->m_coordinatesystem = std::make_shared<VFRendering::CoordinateSystemRenderer>(m_view);
    this->m_coordinatesystem->setOption<VFRendering::CoordinateSystemRenderer::Option::NORMALIZE>(true);

	// Setup the View
	this->setVisualizationMode(this->visMode);

	// Configure System (Setup the renderers
	this->setSystemCycle(this->idx_cycle);
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

void SpinWidget::screenShot(std::string filename)
{
	auto pixmap = this->grab();
	pixmap.save((filename+".bmp").c_str());
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
		scale = 0.1;
	}

	glm::vec2 current_mouse_position = glm::vec2(event->pos().x(), event->pos().y()) * (float)devicePixelRatio() * scale;
	glm::vec2 previous_mouse_position = glm::vec2(m_previous_mouse_position.x(), m_previous_mouse_position.y()) * (float)devicePixelRatio() * scale;
	m_previous_mouse_position = event->pos();
  
	if (event->buttons() & Qt::LeftButton || event->buttons() & Qt::RightButton)
	{
		VFRendering::CameraMovementModes movement_mode = VFRendering::CameraMovementModes::ROTATE_BOUNDED;
		if (this->m_camera_rotate_free) movement_mode = VFRendering::CameraMovementModes::ROTATE_FREE;
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
		scale = 0.1f;
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
	auto movement_mode = VFRendering::CameraMovementModes::TRANSLATE;
	m_view.mouseMove({ 0,0 }, { rightleft, updown }, movement_mode);
	m_view.mouseScroll(backforth * 0.1);
	((QWidget *)this)->update();
}

void SpinWidget::rotateCamera(float theta, float phi)
{
	VFRendering::CameraMovementModes movement_mode = VFRendering::CameraMovementModes::ROTATE_BOUNDED;
	if (this->m_camera_rotate_free) movement_mode = VFRendering::CameraMovementModes::ROTATE_FREE;
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


void SpinWidget::setSlabRanges()
{
	float f_center[3], bounds_min[3], bounds_max[3];
	Geometry_Get_Bounds(state.get(), bounds_min, bounds_max);
	Geometry_Get_Center(state.get(), f_center);
	glm::vec2 x_range(bounds_min[0]+1e-5, bounds_max[0]-1e-5);
	glm::vec2 y_range(bounds_min[1]+1e-5, bounds_max[1]-1e-5);
	glm::vec2 z_range(bounds_min[2]+1e-5, bounds_max[2]-1e-5);
	glm::vec3 center(f_center[0], f_center[1], f_center[2]);
	center += this->slab_displacements;

	switch(this->idx_cycle)
	{
		case 2:
		{
			x_range = {center[0]-1, center[0]+1};
			break;
		}
		case 3:
		{
			y_range = {center[1]-1, center[1]+1};
			break;
		}
		case 4:
		{
			z_range = {center[2]-1, center[2]+1};
			break;
		}
	}

	this->setSurface(x_range, y_range, z_range);
}

void SpinWidget::setSystemCycle(int idx)
{
	this->idx_cycle = idx;
	
	switch(idx)
	{
		case 0:
		{
			// User settings
			this->show_arrows = this->user_show_arrows;
			this->show_surface = this->user_show_surface;
			this->show_isosurface = this->user_show_isosurface;
			this->show_boundingbox = this->user_show_boundingbox;
			this->setVerticalFieldOfView(this->user_fov);
			// Camera
			break;
		}
		case 1:
		{
			// Isosurface
			this->show_arrows = false;
			this->show_surface = false;
			this->show_isosurface = true;
			this->setVerticalFieldOfView(this->user_fov);
			break;
		}
		case 2:
		{
			// Slab x
			this->show_arrows = false;
			this->show_surface = true;
			this->show_isosurface = false;
			// camera
			// this->setCameraToX();
			// this->setVerticalFieldOfView(0);
			break;
		}
		case 3:
		{
			// Slab y
			this->show_arrows = false;
			this->show_surface = true;
			this->show_isosurface = false;
			// camera
			// this->setCameraToY();
			// this->setVerticalFieldOfView(0);
			break;
		}
		case 4:
		{
			// Slab z
			this->show_arrows = false;
			this->show_surface = true;
			this->show_isosurface = false;
			// camera
			// this->setCameraToZ();
			// this->setVerticalFieldOfView(0);
			break;
		}
	}
	this->setSlabRanges();
}

void SpinWidget::cycleSystem(bool forward)
{
	// save possible user settings
	if (this->idx_cycle == 0)
	{
		this->user_show_arrows = this->show_arrows;
		this->user_show_surface = this->show_surface;
		this->user_show_isosurface = this->show_isosurface;
		this->user_show_boundingbox = this->show_boundingbox;
		this->user_fov = this->verticalFieldOfView();
	}

	if (forward)
	{
		++ this->idx_cycle;
	}
	else
	{
		-- this->idx_cycle;
	}
	if (this->idx_cycle < 0) idx_cycle += 5;
	this->idx_cycle = this->idx_cycle % 5;

	this->setSystemCycle(this->idx_cycle);

	this->enableSystem(this->show_arrows, this->show_boundingbox, this->show_surface, this->show_isosurface);
}


glm::vec2 SpinWidget::surfaceXRange() const
{
	return m_surface_x_range;
}

glm::vec2 SpinWidget::surfaceYRange() const
{
	return m_surface_y_range;
}

glm::vec2 SpinWidget::surfaceZRange() const
{
	return m_surface_z_range;
}


/////	enable
void SpinWidget::enableSystem(bool arrows, bool boundingbox, bool surface, bool isosurface)
{
	this->show_arrows = arrows;
	this->show_boundingbox = boundingbox;
	this->show_surface = surface;
	this->show_isosurface = isosurface;

	if (idx_cycle == 0)
	{
		this->user_show_arrows = this->show_arrows;
		this->user_show_surface = this->show_surface;
		this->user_show_isosurface = this->show_isosurface;
		this->user_show_boundingbox = this->show_boundingbox;
		this->user_fov = this->verticalFieldOfView();
	}

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

void SpinWidget::moveSlab(int amount)
{
	float f_center[3], bounds_min[3], bounds_max[3];
	Geometry_Get_Bounds(state.get(), bounds_min, bounds_max);
	Geometry_Get_Center(state.get(), f_center);
	glm::vec3 center(f_center[0], f_center[1], f_center[2]);
	glm::vec3 pos = center +this->slab_displacements;

	float cell_bounds_min[3], cell_bounds_max[3];
	Geometry_Get_Cell_Bounds(state.get(), cell_bounds_min, cell_bounds_max);
	if (this->idx_cycle == 2)
	{
		// X
		amount *= cell_bounds_max[0] - cell_bounds_min[0];
		if (bounds_min[0] < pos[0]+amount && pos[0]+amount < bounds_max[0])
			this->slab_displacements[0] += amount;
	}
	else if (this->idx_cycle == 3)
	{
		// Y
		amount *= cell_bounds_max[1] - cell_bounds_min[1];
		if (bounds_min[1] < pos[1]+amount && pos[1]+amount < bounds_max[1])
			this->slab_displacements[1] += amount;
	}
	else if (this->idx_cycle == 4)
	{
		// Z
		amount *= cell_bounds_max[2] - cell_bounds_min[2];
		if (bounds_min[2] < pos[2]+amount && pos[2]+amount < bounds_max[2])
			this->slab_displacements[2] += amount;
	}

	this->setSlabRanges();
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


/////	Overall Range Directions
glm::vec2 SpinWidget::xRangeDirection() const {
	return m_x_range_direction;
}
glm::vec2 SpinWidget::yRangeDirection() const {
	return m_y_range_direction;
}
glm::vec2 SpinWidget::zRangeDirection() const {
	return m_z_range_direction;
}

void SpinWidget::setOverallDirectionRange(glm::vec2 x_range, glm::vec2 y_range, glm::vec2 z_range) {
	std::ostringstream sstream;
	std::string is_visible_implementation;
	sstream << "bool is_visible(vec3 position, vec3 direction) {";
	// X
	m_x_range_direction = x_range;
	if (x_range.x <= -1 && x_range.y >= 1) {
		sstream << "bool is_visible_x = true;";
	}
	else if (x_range.x <= -1) {
		sstream << "float x_max = ";
		sstream << x_range.y;
		sstream << "; bool is_visible_x = normalize(direction).x <= x_max;";
	}
	else if (x_range.y >= 1) {
		sstream << "float x_min = ";
		sstream << x_range.x;
		sstream << "; bool is_visible_x = normalize(direction).x >= x_min;";
	}
	else {
		sstream << "float x_min = ";
		sstream << x_range.x;
		sstream << "; float x_max = ";
		sstream << x_range.y;
		sstream << "; float x = normalize(direction).x; bool is_visible_x = x >= x_min && x <= x_max;";
	}
	// Y
	m_y_range_direction = y_range;
	if (y_range.x <= -1 && y_range.y >= 1) {
		sstream << "bool is_visible_y = true;";
	}
	else if (y_range.x <= -1) {
		sstream << "float y_max = ";
		sstream << y_range.y;
		sstream << "; bool is_visible_y = normalize(direction).y <= y_max;";
	}
	else if (y_range.y >= 1) {
		sstream << "float y_min = ";
		sstream << y_range.x;
		sstream << "; bool is_visible_y = normalize(direction).y >= y_min;";
	}
	else {
		sstream << "float y_min = ";
		sstream << y_range.x;
		sstream << "; float y_max = ";
		sstream << y_range.y;
		sstream << "; float y = normalize(direction).y;  bool is_visible_y = y >= y_min && y <= y_max;";
	}
	// Z
	m_z_range_direction = z_range;
	if (z_range.x <= -1 && z_range.y >= 1) {
		sstream << "bool is_visible_z = true;";
	}
	else if (z_range.x <= -1) {
		sstream << "float z_max = ";
		sstream << z_range.y;
		sstream << "; bool is_visible_z = normalize(direction).z <= z_max;";
	}
	else if (z_range.y >= 1) {
		sstream << "float z_min = ";
		sstream << z_range.x;
		sstream << "; bool is_visible_z = normalize(direction).z >= z_min;";
	}
	else {
		sstream << "float z_min = ";
		sstream << z_range.x;
		sstream << "; float z_max = ";
		sstream << z_range.y;
		sstream << "; float z = normalize(direction).z;  bool is_visible_z = z >= z_min && z <= z_max;";
	}
	//
	sstream << " return is_visible_x && is_visible_y && is_visible_z; }";
	is_visible_implementation = sstream.str();
	makeCurrent();
	m_view.setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(is_visible_implementation);
}

/////	Overall Range Position
glm::vec2 SpinWidget::xRangePosition() const {
	return m_x_range_position;
}
glm::vec2 SpinWidget::yRangePosition() const {
	return m_y_range_position;
}
glm::vec2 SpinWidget::zRangePosition() const {
	return m_z_range_position;
}

void SpinWidget::setOverallPositionRange(glm::vec2 x_range, glm::vec2 y_range, glm::vec2 z_range) {
	std::ostringstream sstream;
	std::string is_visible_implementation;
	sstream << "bool is_visible(vec3 position, vec3 direction) {";
	// X
	m_x_range_position = x_range;
	if (x_range.x >= x_range.y) {
		sstream << "bool is_visible_x = true;";
	}
	else {
		sstream << "float x_min = ";
		sstream << x_range.x;
		sstream << "; float x_max = ";
		sstream << x_range.y;
		sstream << "; bool is_visible_x = position.x <= x_max && position.x >= x_min;";
	}
	// Y
	m_y_range_position = y_range;
	if (y_range.x >= y_range.y) {
		sstream << "bool is_visible_y = true;";
	}
	else {
		sstream << "float y_min = ";
		sstream << y_range.x;
		sstream << "; float y_max = ";
		sstream << y_range.y;
		sstream << "; bool is_visible_y = position.y <= y_max && position.y >= y_min;";
	}
	// Z
	m_z_range_position = z_range;
	if (x_range.x >= x_range.y) {
		sstream << "bool is_visible_z = true;";
	}
	else {
		sstream << "float z_min = ";
		sstream << z_range.x;
		sstream << "; float z_max = ";
		sstream << z_range.y;
		sstream << "; bool is_visible_z = position.z <= z_max && position.z >= z_min;";
	}
	//
	sstream << " return is_visible_x && is_visible_y && is_visible_z; }";
	is_visible_implementation = sstream.str();
	makeCurrent();
	m_view.setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(is_visible_implementation);
}


/////   Surface
void SpinWidget::setSurface(glm::vec2 x_range, glm::vec2 y_range, glm::vec2 z_range)
{
	this->m_surface_x_range = x_range;
	this->m_surface_y_range = y_range;
	this->m_surface_z_range = z_range;

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
		m_renderer_isosurface->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>(
			"uniform vec3 uLightPosition;"
			"float lighting(vec3 position, vec3 normal)"
			"{"
			"    vec3 lightDirection = -normalize(uLightPosition-position);"
			"    float diffuse = 0.7*max(0.0, dot(normal, lightDirection));"
			"	 float ambient = 0.2;"
			"    return diffuse+ambient;"
			"}");
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
  setColormapRotationInverted(colormap_rotation(), colormap_inverted()[0], colormap_inverted()[1]);
}

float SpinWidget::colormap_rotation()
{
	return this->m_colormap_rotation;
}

std::array<bool, 2> SpinWidget::colormap_inverted()
{
	return std::array<bool, 2>{this->m_colormap_invert_z, this->m_colormap_invert_xy};
}

void SpinWidget::setColormapRotationInverted(int phi, bool invert_z, bool invert_xy)
{
	this->m_colormap_rotation = phi;
	this->m_colormap_invert_z = invert_z;
	this->m_colormap_invert_xy = invert_xy;
	int sign_z  = 1 - 2 * (int)invert_z;
	int sign_xy = 1 - 2 * (int)invert_xy;

	float P = glm::radians((float)phi)   / 3.14159;

	// Get strings from floats - For some reason the locale is messed up...
	auto old = std::locale::global(std::locale::classic());
	std::locale::global(old);
	// setlocale(LC_ALL, "en_US");
	char s_phi[50];
	sprintf (s_phi, "%f", P);
	char s_sign_z[50];
	sprintf (s_sign_z, "%i", sign_z);
	char s_sign_xy[50];
	sprintf (s_sign_xy, "%i", sign_xy);
	std::string colormap_implementation;
	switch (m_colormap)
	{
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
		// Custom color maps not included in VFRendering:
		case Colormap::HSV:
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
				float hue = atan2()" + std::string(s_sign_xy) + R"(*xy.x, xy.y) / 3.14159 / 2.0 + )" + std::string(s_phi) + R"(/2.0;
				float saturation = direction.z * )" + std::string(s_sign_z) + R"(;
				if (saturation > 0.0) {
					return hsv2rgb(vec3(hue, 1.0-saturation, 1.0));
				} else {
					return hsv2rgb(vec3(hue, 1.0, 1.0+saturation));
				}
			}
			)";
			break;
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
				float hue = atan2()" + std::string(s_sign_xy) + R"(*xy.x, xy.y) / 3.14159 / 2.0 + )" + std::string(s_phi) + R"(;
				return hsv2rgb(vec3(hue, 1.0, 1.0));
			}
			)";
			break;
		case Colormap::BLUE_RED:
			colormap_implementation = R"(
			vec3 colormap(vec3 direction) {
				float z_sign = direction.z * )" + std::string(s_sign_z) + R"(;
				vec3 color_down = vec3(0.0, 0.0, 1.0);
				vec3 color_up = vec3(1.0, 0.0, 0.0);
				return mix(color_down, color_up, z_sign*0.5+0.5);
			}
			)";
			break;
		case Colormap::BLUE_GREEN_RED:
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
				float hue = 1.0/3.0-normalize(direction).z/3.0* )" + std::string(s_sign_z) + R"(;
				return hsv2rgb(vec3(hue, 1.0, 1.0));
			}
			)";
			break;
		case Colormap::BLUE_WHITE_RED:
			colormap_implementation = R"(
			vec3 colormap(vec3 direction) {
				float z_sign = direction.z * )" + std::string(s_sign_z) + R"(;
				if (z_sign < 0) {
					vec3 color_down = vec3(0.0, 0.0, 1.0);
					vec3 color_up = vec3(1.0, 1.0, 1.0);
					return mix(color_down, color_up, z_sign+1);
				} else {
					vec3 color_down = vec3(1.0, 1.0, 1.0);
					vec3 color_up = vec3(1.0, 0.0, 0.0);
					return mix(color_down, color_up, z_sign);
				}
			}
			)";
			break;
		// Default is regular HSV
		default:
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
				float hue = atan2()" + std::string(s_sign_xy) + R"(*xy.x, xy.y) / 3.14159 / 2.0 + )" + std::string(s_phi) + R"(/2.0;
				float saturation = direction.z * )" + std::string(s_sign_z) + R"(;
				if (saturation > 0.0) {
					return hsv2rgb(vec3(hue, 1.0-saturation, 1.0));
				} else {
					return hsv2rgb(vec3(hue, 1.0, 1.0+saturation));
				}
			}
			)";
			break;
  }

  // Set colormap
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
void SpinWidget::cycleCamera() {
	if (this->verticalFieldOfView() == 0)
	{
		this->setVerticalFieldOfView(this->user_fov);
	}
	else
	{
		this->setVerticalFieldOfView(0);
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

void SpinWidget::setCameraToX(bool inverted)
{
	float camera_distance = glm::length(options().get<VFRendering::View::Option::CENTER_POSITION>() - options().get<VFRendering::View::Option::CAMERA_POSITION>());
	auto center_position = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
	auto camera_position = center_position;
	auto up_vector = glm::vec3(0, 0, 1);

	if (!inverted)
	{
		camera_position +=  camera_distance * glm::vec3(1, 0, 0);
	}
	else
	{
		camera_position -=  camera_distance * glm::vec3(1, 0, 0);
	}

	VFRendering::Options options;
	options.set<VFRendering::View::Option::CAMERA_POSITION>(camera_position);
	options.set<VFRendering::View::Option::CENTER_POSITION>(center_position);
	options.set<VFRendering::View::Option::UP_VECTOR>(up_vector);
	m_view.updateOptions(options);
}

void SpinWidget::setCameraToY(bool inverted) {
	float camera_distance = glm::length(options().get<VFRendering::View::Option::CENTER_POSITION>() - options().get<VFRendering::View::Option::CAMERA_POSITION>());
	auto center_position = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
	auto camera_position = center_position;
	auto up_vector = glm::vec3(0, 0, 1);

	if (!inverted)
	{
		camera_position +=  camera_distance * glm::vec3(0, -1, 0);
	}
	else
	{
		camera_position -=  camera_distance * glm::vec3(0, -1, 0);
	}

	VFRendering::Options options;
	options.set<VFRendering::View::Option::CAMERA_POSITION>(camera_position);
	options.set<VFRendering::View::Option::CENTER_POSITION>(center_position);
	options.set<VFRendering::View::Option::UP_VECTOR>(up_vector);
	m_view.updateOptions(options);
}

void SpinWidget::setCameraToZ(bool inverted) {
	float camera_distance = glm::length(options().get<VFRendering::View::Option::CENTER_POSITION>() - options().get<VFRendering::View::Option::CAMERA_POSITION>());
	auto center_position = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
	auto camera_position = center_position;
	auto up_vector = glm::vec3(0, 1, 0);

	if (!inverted)
	{
		camera_position +=  camera_distance * glm::vec3(0, 0, 1);
	}
	else
	{
		camera_position -=  camera_distance * glm::vec3(0, 0, 1);
	}

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
	// Calculate new camera position
	float scale = 1;
	float fov = m_view.options().get<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>();
	if (fov > 0 && vertical_field_of_view > 0)
	{
		scale = std::tan(glm::radians(fov)/2.0) / std::tan(glm::radians(vertical_field_of_view)/2.0);
		setCameraPositon(getCameraPositon()*scale);
	}

	// Set new FOV
	makeCurrent();
	m_view.setOption<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>(vertical_field_of_view);
}

bool SpinWidget::getCameraRotationType()
{
	return this->m_camera_rotate_free;
}
void SpinWidget::setCameraRotationType(bool free)
{
	this->m_camera_rotate_free = free;
}


/////////////// lighting
glm::vec3 from_spherical(float theta, float phi)
{
	float x = glm::sin(glm::radians(theta)) * glm::cos(glm::radians(phi));
	float y = glm::sin(glm::radians(theta)) * glm::sin(glm::radians(phi));
	float z = glm::cos(glm::radians(theta));
	return glm::vec3{x,y,z};
}
void SpinWidget::setLightPosition(float theta, float phi)
{
	this->m_light_theta = theta;
	this->m_light_phi = phi;
	glm::vec3 v_light = glm::normalize(from_spherical(theta, phi)) * 1000.0f;
	m_view.setOption<VFRendering::View::Option::LIGHT_POSITION>(v_light);
}

std::array<float,2> SpinWidget::getLightPosition()
{
	return std::array<float,2>{m_light_theta,m_light_phi};
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
	settings.setValue("FOV", (int)(this->user_fov * 100));
	// Sphere Point Size
	settings.setValue("SpherePointSize1", (int)(this->spherePointSizeRange().x * 100));
	settings.setValue("SpherePointSize2", (int)(this->spherePointSizeRange().y * 100));
	// System
	settings.setValue("Cycle Index", this->idx_cycle);
	settings.setValue("Show Arrows", this->user_show_arrows);
	settings.setValue("Show Bounding Box", this->user_show_boundingbox);
	settings.setValue("Show Surface", this->user_show_surface);
	settings.setValue("Show Isosurface", this->user_show_isosurface);
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
	settings.setValue("Colormap_invert_z", m_colormap_invert_z);
	settings.setValue("Colormap_invert_xy", m_colormap_invert_xy);
	settings.setValue("Colormap_rotation",   m_colormap_rotation);
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
	settings.setValue("free rotation", m_camera_rotate_free);
	settings.endGroup();

	// Light
	settings.beginGroup("Light");
	settings.setValue("theta", (int)m_light_theta);
	settings.setValue("phi", (int)m_light_phi);
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
		this->user_fov = this->verticalFieldOfView();
		// Sphere Point Size
		this->setSpherePointSizeRange({ (settings.value("SpherePointSize1").toInt() / 100.0f), (settings.value("SpherePointSize2").toInt() / 100.0f) });
		// System
		this->idx_cycle = 0;//settings.value("Cycle Index").toInt();
		this->user_show_arrows = settings.value("Show Arrows").toBool();
		this->user_show_boundingbox = settings.value("Show Bounding Box").toBool();
		this->user_show_surface = settings.value("Show Surface").toBool();
		this->user_show_isosurface = settings.value("Show Isosurface").toBool();
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
		bool invert_z = settings.value("Colormap_invert_z").toInt();
		bool invert_xy = settings.value("Colormap_invert_xy").toInt();
		int phi   = settings.value("Colormap_rotation").toInt();
		this->setColormapRotationInverted(phi, invert_z, invert_xy);
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
		this->m_camera_rotate_free = settings.value("free rotation").toBool();
		settings.endGroup();
	}

	// Light
	if (settings.childGroups().contains("Light"))
	{
		settings.beginGroup("Light");
		this->m_light_theta = settings.value("theta").toInt();
		this->m_light_phi   = settings.value("phi").toInt();
		this->setLightPosition(m_light_theta, m_light_phi);
		settings.endGroup();
	}
}


void SpinWidget::closeEvent(QCloseEvent *event)
{
	writeSettings();
	event->accept();
}
