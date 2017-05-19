// #include <fstream>
#include <sstream>
#include <algorithm> 

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
#include "Spirit/Configurations.h"
#include "Spirit/Simulation.h"
#include "Spirit/Hamiltonian.h"


SpinWidget::SpinWidget(std::shared_ptr<State> state, QWidget *parent) : QOpenGLWidget(parent)
{
    this->state = state;
	this->m_gl_initialized = false;
	this->m_suspended = false;

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
	this->m_surface_x_range = x_range;
	this->m_surface_y_range = y_range;
	this->m_surface_z_range = z_range;
	
	this->m_source = 0;
	this->visMode = VisualizationMode::SYSTEM;
	this->m_location_coordinatesystem = WidgetLocation::BOTTOM_RIGHT;
	this->m_location_miniview = WidgetLocation::BOTTOM_LEFT;
	this->show_arrows = true;
	this->show_boundingbox = true;
	this->show_isosurface = false;

	idx_cycle=0;
	slab_displacements = glm::vec3{0,0,0};

	this->n_cell_step = 1;

	this->show_surface = false;
	this->show_miniview = true;
	this->show_coordinatesystem = true;

	// 		Initial camera position
	this->_reset_camera = false;
	this->m_camera_rotate_free = false;
	this->m_camera_projection_perspective = true;

	//		Initial drag mode settings
	drag_radius = 80;
	this->mouse_decoration = new MouseDecoratorWidget(drag_radius);
	this->mouse_decoration->setMinimumSize(2 * drag_radius, 2 * drag_radius);
	this->mouse_decoration->setMaximumSize(2 * drag_radius, 2 * drag_radius);
	this->mouse_decoration->setParent(this);
	this->m_interactionmode = InteractionMode::REGULAR;
	this->m_timer_drag = new QTimer(this);
	this->m_dragging = false;

	// 		Setup Arrays
	this->updateData();

	// 		Read persistent settings
	this->readSettings();
	this->show_arrows = this->user_show_arrows;
	this->show_surface = this->user_show_surface;
	this->show_isosurface = this->user_show_isosurface;
	this->show_boundingbox = this->user_show_boundingbox;
}

void SpinWidget::setSuspended(bool suspended)
{
	this->m_suspended = suspended;
	if (!suspended)
	{
		this->update();
	}
}

const VFRendering::View * SpinWidget::view()
{
	return &(this->m_view);
}

void SpinWidget::addIsosurface(std::shared_ptr<VFRendering::IsosurfaceRenderer> renderer)
{
	if (Geometry_Get_Dimensionality(this->state.get()) == 3)
	{
		this->m_renderers_isosurface.insert(renderer);
		if (m_gl_initialized)
			this->enableSystem(this->show_arrows, this->show_boundingbox, this->show_surface, this->show_isosurface);
	}
}

void SpinWidget::removeIsosurface(std::shared_ptr<VFRendering::IsosurfaceRenderer> renderer)
{
	this->m_renderers_isosurface.erase(renderer);
	if (m_gl_initialized)
		this->enableSystem(this->show_arrows, this->show_boundingbox, this->show_surface, this->show_isosurface);
}

// Return the relative mouse position [-1,1]
glm::vec2 relative_coords_from_mouse(glm::vec2 mouse_pos, glm::vec2 winsize)
{
	glm::vec2 relative = 2.0f*(mouse_pos - 0.5f*winsize);
	relative.x /= winsize.x;
	relative.y /= winsize.y;
	return relative;
}

glm::vec2 SpinWidget::system_coords_from_mouse(glm::vec2 mouse_pos, glm::vec2 winsize)
{
	auto relative = relative_coords_from_mouse(mouse_pos, winsize);
	glm::vec4 proj_back{ relative.x, relative.y, 0, 0 };

	auto matrices = VFRendering::Utilities::getMatrices(m_view.options(), winsize.x/winsize.y);
	auto model_view = glm::inverse(matrices.first);
	auto projection = glm::inverse(matrices.second);

	proj_back = proj_back*projection;
	proj_back = proj_back*model_view;

	auto camera_position = options().get<VFRendering::View::Option::CAMERA_POSITION>();

	return glm::vec2{ proj_back.x + camera_position.x, -proj_back.y + camera_position.y };
}

float SpinWidget::system_radius_from_relative(float radius, glm::vec2 winsize)
{
	auto r1 = system_coords_from_mouse({ 0.0f, 0.0f }, winsize);
	auto r2 = system_coords_from_mouse({ radius-5, 0.0f }, winsize);
	return r2.x-r1.x;
}

void SpinWidget::dragpaste()
{
	QPoint localCursorPos = this->mapFromGlobal(cursor().pos());
	QSize  widgetSize = this->size();

	glm::vec2 mouse_pos{ localCursorPos.x(), localCursorPos.y() };
	glm::vec2 size{ widgetSize.width(),  widgetSize.height() };

	glm::vec2 coords = system_coords_from_mouse(mouse_pos, size);
	float radius = system_radius_from_relative(this->drag_radius, size);
	float rect[3]{ -1, -1, -1 };
	// std::cerr << "--- r = " << radius << " pos = " << coords.x << "  " << coords.y << std::endl;

	float last_position[3]{ last_drag_coords.x, last_drag_coords.y, 0.0f };
	float current_position[3]{ coords.x, coords.y, 0.0f };
	Configuration_From_Clipboard_Shift(state.get(), last_position, current_position, rect, radius);
}


void SpinWidget::initializeGL()
{
	if (m_interactionmode == InteractionMode::DRAG)
	{
		this->setCursor(Qt::BlankCursor);
	}
	else
	{
		mouse_decoration->hide();
	}


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
		// Determine orthogonality of translation vectors
		float ta[3], tb[3], tc[3];
		Geometry_Get_Translation_Vectors(state.get(), ta, tb, tc);
		float tatb = 0, tatc = 0, tbtc = 0;
		for (int dim = 0; dim<3; ++dim)
		{
			tatb += ta[dim] * tb[dim];
			tatc += ta[dim] * tc[dim];
			tbtc += tb[dim] * tc[dim];
		}
		// Rectilinear with one basis atom
		if (Geometry_Get_N_Basis_Atoms(state.get()) == 1 &&
			std::abs(tatb) < 1e-8 && std::abs(tatc) < 1e-8 && std::abs(tbtc) < 1e-8)
		{
			// 2D Surface options
			// No options yet...
			this->m_renderer_surface_2D = std::make_shared<VFRendering::SurfaceRenderer>(m_view);
			this->m_renderer_surface = m_renderer_surface_2D;
		}
	}
	else if (Geometry_Get_Dimensionality(this->state.get()) == 3)
	{
		// 3D Surface options
		this->m_renderer_surface_3D = std::make_shared<VFRendering::IsosurfaceRenderer>(m_view);
		this->m_renderer_surface_3D->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(0.0);
		auto mini_diff = glm::vec2{0.00001f, -0.00001f};
		setSurface(x_range + mini_diff, y_range + mini_diff, z_range + mini_diff);

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
	this->setSystemCycle(SystemMode(this->idx_cycle));
	this->enableSystem(this->show_arrows, this->show_boundingbox, this->show_surface, this->show_isosurface);

	this->m_gl_initialized = true;
}

void SpinWidget::teardownGL()
{
	// GLSpins::terminate();
}

void SpinWidget::resizeGL(int width, int height)
{
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
	int n_cells[3];
	Geometry_Get_N_Cells(this->state.get(), n_cells);
	int n_basis_atoms = Geometry_Get_N_Basis_Atoms(this->state.get());

	int n_cells_draw[3] = {std::max(1, n_cells[0]/n_cell_step), std::max(1, n_cells[1]/n_cell_step), std::max(1, n_cells[2]/n_cell_step)};
	int nos_draw = n_basis_atoms*n_cells_draw[0]*n_cells_draw[1]*n_cells_draw[2];

	std::vector<glm::vec3> positions = std::vector<glm::vec3>(nos_draw);
	std::vector<glm::vec3> directions = std::vector<glm::vec3>(nos_draw);

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
	int icell = 0;
	for (int cell_c=0; cell_c<n_cells_draw[2]; cell_c++)
	{
		for (int cell_b=0; cell_b<n_cells_draw[1]; cell_b++)
		{
			for (int cell_a=0; cell_a<n_cells_draw[0]; cell_a++)
			{
				for (int ibasis=0; ibasis < n_basis_atoms; ++ibasis)
				{
					int idx = ibasis + n_basis_atoms*cell_a*n_cell_step + n_basis_atoms*n_cells[0]*cell_b*n_cell_step + n_basis_atoms*n_cells[0]*n_cells[1]*cell_c*n_cell_step;
					// std::cerr << idx << " " << icell << std::endl;
					positions[icell] = glm::vec3(spin_pos[3*idx], spin_pos[1 + 3*idx], spin_pos[2 + 3*idx]);
					directions[icell] = glm::vec3(spins[3*idx], spins[1 + 3*idx], spins[2 + 3*idx]);
					++icell;
				}
			}
		}
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
		if (n_cells[0]/n_cell_step < 2 || n_cells[1]/n_cell_step < 2 || n_cells[2]/n_cell_step < 2)
		{
			geometry = VFRendering::Geometry(positions, {}, {}, true);
		}
		else
		{
			const std::array<VFRendering::Geometry::index_type, 4> *tetrahedra_indices_ptr = nullptr;
			int num_tetrahedra = Geometry_Get_Triangulation(state.get(), reinterpret_cast<const int **>(&tetrahedra_indices_ptr), n_cell_step);
			std::vector<std::array<VFRendering::Geometry::index_type, 4>>  tetrahedra_indices(tetrahedra_indices_ptr, tetrahedra_indices_ptr + num_tetrahedra);
			geometry = VFRendering::Geometry(positions, {}, tetrahedra_indices, false);
		}
	}
	else if (Geometry_Get_Dimensionality(state.get()) == 2)
	{
		// Determine orthogonality of translation vectors
		float ta[3], tb[3], tc[3];
		Geometry_Get_Translation_Vectors(state.get(), ta, tb, tc);
		float tatb=0, tatc=0, tbtc=0;
		for (int dim=0; dim<3; ++dim)
		{
			tatb += ta[dim] * tb[dim];
			tatc += ta[dim] * tc[dim];
			tbtc += tb[dim] * tc[dim];
		}
		// Rectilinear with one basis atom
		if (Geometry_Get_N_Basis_Atoms(state.get()) == 1 &&
			std::abs(tatb) < 1e-8 && std::abs(tatc) < 1e-8 && std::abs(tbtc) < 1e-8)
		{
			std::vector<float> xs(n_cells_draw[0]), ys(n_cells_draw[1]), zs(n_cells_draw[2]);
			for (int i = 0; i < n_cells_draw[0]; ++i) xs[i] = positions[i].x;
			for (int i = 0; i < n_cells_draw[1]; ++i) ys[i] = positions[i*n_cells_draw[0]].y;
			for (int i = 0; i < n_cells_draw[2]; ++i) zs[i] = positions[i*n_cells_draw[0] * n_cells_draw[1]].z;
			geometry = VFRendering::Geometry::rectilinearGeometry(xs, ys, zs);
		}
		// All others
		else
		{
			geometry = VFRendering::Geometry(positions, {}, {}, true);
		}

	}
	else
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

void SpinWidget::paintGL()
{
	if (this->m_suspended)
		return;

	if (m_interactionmode == InteractionMode::DRAG)
	{
		auto pos = this->mapFromGlobal(QCursor::pos() - QPoint(drag_radius, drag_radius));
		this->mouse_decoration->move((int)pos.x(), (int)pos.y());
	}

	if ( Simulation_Running_Image(this->state.get())      ||
		 Simulation_Running_Chain(this->state.get())      ||
		 Simulation_Running_Collection(this->state.get()) ||
		 this->m_dragging)
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

void SpinWidget::mousePressEvent(QMouseEvent *event)
{
	if (this->m_suspended)
		return;

	m_previous_mouse_position = event->pos();

	if (m_interactionmode == InteractionMode::DRAG)
	{
		if (event->button() == Qt::LeftButton)
		{
			QPoint localCursorPos = this->mapFromGlobal(cursor().pos());
			QSize  widgetSize = this->size();
			glm::vec2 mouse_pos{ localCursorPos.x(), localCursorPos.y() };
			glm::vec2 size{ widgetSize.width(),  widgetSize.height() };
			last_drag_coords = system_coords_from_mouse(mouse_pos, size);

			m_timer_drag->stop();
			// Copy spin configuration
			Configuration_To_Clipboard(state.get());
			// Set up Update Timers
			connect(m_timer_drag, &QTimer::timeout, this, &SpinWidget::dragpaste);
			float ips = Simulation_Get_IterationsPerSecond(state.get());
			if (ips > 1000)
			{
				m_timer_drag->start(1);
			}
			else if (ips > 0)
			{
				m_timer_drag->start((int)(1000/ips));
			}
			m_dragging = true;
		}
	}
}

void SpinWidget::mouseReleaseEvent(QMouseEvent *event)
{
	if (this->m_suspended)
		return;

	if (m_interactionmode == InteractionMode::DRAG)
	{
		if (event->button() == Qt::LeftButton)
		{
			m_timer_drag->stop();
			m_dragging = false;
		}
		else if (event->button() == Qt::RightButton)
		{
			dragpaste();
			this->updateData();
		}
	}
}

void SpinWidget::mouseMoveEvent(QMouseEvent *event)
{
	if (this->m_suspended)
		return;

	float scale = 1;

	if (event->modifiers() & Qt::ShiftModifier)
	{
		scale = 0.1f;
	}

	if (m_interactionmode == InteractionMode::DRAG)
	{
		dragpaste();
	}
	else
	{
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

	if (event->modifiers() & Qt::ControlModifier)
	{
		float wheel_delta = scale*event->angleDelta().y()/10.0f;
		drag_radius = std::max(1.0f, std::min(500.0f, drag_radius + wheel_delta));
		this->mouse_decoration->setRadius(drag_radius);
		this->mouse_decoration->setMinimumSize(2 * drag_radius, 2 * drag_radius);
		this->mouse_decoration->setMaximumSize(2 * drag_radius, 2 * drag_radius);
	}
	else
	{
		float wheel_delta = event->angleDelta().y();
		m_view.mouseScroll(wheel_delta * 0.1 * scale);
		((QWidget *)this)->update();
	}
}

const VFRendering::Options& SpinWidget::options() const
{
	return m_view.options();
}



void SpinWidget::moveCamera(float backforth, float rightleft, float updown)
{
	if (this->m_suspended)
		return;

	auto movement_mode = VFRendering::CameraMovementModes::TRANSLATE;
	m_view.mouseMove({ 0,0 }, { rightleft, updown }, movement_mode);
	m_view.mouseScroll(backforth * 0.1);
	((QWidget *)this)->update();
}

void SpinWidget::rotateCamera(float theta, float phi)
{
	if (this->m_suspended)
		return;
		
	if (this->m_interactionmode == InteractionMode::DRAG)
	{
		theta = 0;
	}
	VFRendering::CameraMovementModes movement_mode = VFRendering::CameraMovementModes::ROTATE_BOUNDED;
	if (this->m_camera_rotate_free) movement_mode = VFRendering::CameraMovementModes::ROTATE_FREE;
	m_view.mouseMove({ 0,0 }, { phi, theta }, movement_mode);
	((QWidget *)this)->update();
}


//////////////////////////////////////////////////////////////////////////////////////
int SpinWidget::visualisationNCellSteps()
{
	return this->n_cell_step;
}

void SpinWidget::setVisualisationNCellSteps(int n_cell_steps)
{
	float size_before = this->arrowSize();
	this->n_cell_step = n_cell_steps;
	this->setArrows(size_before, this->arrowLOD());
	this->updateData();
}

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

void SpinWidget::setInteractionMode(InteractionMode mode)
{
	if (mode == InteractionMode::DRAG)
	{
		// Save latest regular mode settings
		this->regular_mode_perspective = this->cameraProjection();
		this->regular_mode_cam_pos   = this->getCameraPositon();
		this->regular_mode_cam_focus = this->getCameraFocus();
		this->regular_mode_cam_up    = this->getCameraUpVector();
		// Set cursor
		this->setCursor(Qt::BlankCursor);
		this->mouse_decoration->show();
		// Apply camera changes
		this->setCameraToZ();
		this->setCameraProjection(false);
		// Set mode after changes so that changes are not blocked
		this->m_interactionmode = mode;
	}
	else
	{
		// Unset cursor
		this->unsetCursor();
		this->mouse_decoration->hide();
		// Set mode before changes so that changes are not blocked
		this->m_interactionmode = mode;
		// Apply camera changes
		this->setCameraProjection(this->regular_mode_perspective);
		this->setCameraPosition(this->regular_mode_cam_pos);
		this->setCameraFocus(this->regular_mode_cam_focus);
		this->setCameraUpVector(this->regular_mode_cam_up);
	}
}

SpinWidget::InteractionMode SpinWidget::interactionMode()
{
	return this->m_interactionmode;
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
	glm::vec2 x_range(bounds_min[0], bounds_max[0]);
	glm::vec2 y_range(bounds_min[1], bounds_max[1]);
	glm::vec2 z_range(bounds_min[2], bounds_max[2]);
	glm::vec3 center(f_center[0], f_center[1], f_center[2]);
	center += this->slab_displacements;

	float delta = 0.51;

	switch(this->idx_cycle)
	{
		case 2:
		{
			if ((int)center.x == center.x) { center.x += 0.5; }
			x_range = {center[0] - delta, center[0] + delta };
			break;
		}
		case 3:
		{
			if ((int)center.y == center.y) { center.y += 0.5; }
			y_range = {center[1] - delta, center[1] + delta };
			break;
		}
		case 4:
		{
			if ((int)center.z == center.z) { center.z += 0.5; }
			z_range = {center[2] - delta, center[2] + delta };
			break;
		}
	}

	float mini_shift = 1e-5f;
	x_range.x = std::max(bounds_min[0] + mini_shift, x_range.x );
	x_range.y = std::min(bounds_max[0] - mini_shift, x_range.y);
	y_range.x = std::max(bounds_min[1] + mini_shift, y_range.x);
	y_range.y = std::min(bounds_max[1] - mini_shift, y_range.y);
	z_range.x = std::max(bounds_min[2] + mini_shift, z_range.x);
	z_range.y = std::min(bounds_max[2] - mini_shift, z_range.y);

	this->setSurface(x_range, y_range, z_range);
}

void SpinWidget::setSystemCycle(SystemMode mode)
{
	this->idx_cycle = (int)mode;
	
	switch(mode)
	{
		case SystemMode::CUSTOM:
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
		case SystemMode::ISOSURFACE:
		{
			// Isosurface
			this->show_arrows = false;
			this->show_surface = false;
			this->show_isosurface = true;
			this->setVerticalFieldOfView(this->user_fov);
			break;
		}
		case SystemMode::SLAB_X:
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
		case SystemMode::SLAB_Y:
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
		case SystemMode::SLAB_Z:
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

void SpinWidget::cycleSystem(SystemMode mode)
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

	this->idx_cycle = (int)mode;

	this->setSystemCycle(mode);

	this->enableSystem(this->show_arrows, this->show_boundingbox, this->show_surface, this->show_isosurface);
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

	this->setSystemCycle(SystemMode(this->idx_cycle));

	this->enableSystem(this->show_arrows, this->show_boundingbox, this->show_surface, this->show_isosurface);
}

SpinWidget::SystemMode SpinWidget::systemCycle()
{
	return SystemMode(this->idx_cycle);
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
	{
		for (auto& iso : this->m_renderers_isosurface) system.push_back(iso);
	}
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
	for (int i = 0; i < 3; ++i) if ((int)f_center[i] == f_center[i]) f_center[i] += 0.5;
	glm::vec3 center(f_center[0], f_center[1], f_center[2]);
	glm::vec3 pos = center + this->slab_displacements;

	float cell_bounds_min[3], cell_bounds_max[3];
	Geometry_Get_Cell_Bounds(state.get(), cell_bounds_min, cell_bounds_max);
	glm::vec3 cell_size{ cell_bounds_max[0] - cell_bounds_min[0], cell_bounds_max[1] - cell_bounds_min[1], cell_bounds_max[2] - cell_bounds_min[2] };
	if (this->idx_cycle == 2)
	{
		// X
		amount *= cell_size[0];
		this->slab_displacements[0] = std::min(std::max(bounds_min[0] + 0.5f*cell_size[0], pos[0] + amount), bounds_max[0] - 0.5f*cell_size[0]) - center[0];
	}
	else if (this->idx_cycle == 3)
	{
		// Y
		amount *= cell_size[1];
		this->slab_displacements[1] = std::min(std::max(bounds_min[1] + 0.5f*cell_size[1], pos[1] + amount), bounds_max[1] - 0.5f*cell_size[1]) - center[1];
	}
	else if (this->idx_cycle == 4)
	{
		// Z
		amount *= cell_size[2];
		this->slab_displacements[2] = std::min(std::max(bounds_min[2] + 0.5f*cell_size[2], pos[2] + amount), bounds_max[2]-0.5f*cell_size[2]) - center[2];
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

	float b_min[3], b_max[3];
	Geometry_Get_Bounds(state.get(), b_min, b_max);
	glm::vec3 bounds_min = glm::make_vec3(b_min);
	glm::vec3 bounds_max = glm::make_vec3(b_max);
	glm::vec2 x_range{ bounds_min[0], bounds_max[0] };
	glm::vec2 y_range{ bounds_min[1], bounds_max[1] };
	glm::vec2 z_range{ bounds_min[2], bounds_max[2] };
	glm::vec3 bounding_box_center = { (bounds_min[0] + bounds_max[0]) / 2, (bounds_min[1] + bounds_max[1]) / 2, (bounds_min[2] + bounds_max[2]) / 2 };
	glm::vec3 bounding_box_side_lengths = { bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], bounds_max[2] - bounds_min[2] };

	int n_cells[3];
	Geometry_Get_N_Cells(this->state.get(), n_cells);

	float density = 0.01f;
	if (n_cells[0] > 1) density = std::max(density, n_cells[0] / (bounds_max[0] - bounds_min[0]));
	if (n_cells[1] > 1) density = std::max(density, n_cells[1] / (bounds_max[1] - bounds_min[1]));
	if (n_cells[2] > 1) density = std::max(density, n_cells[2] / (bounds_max[2] - bounds_min[2]));
	density /= n_cell_step;

	makeCurrent();
	m_view.setOption<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>(coneheight * size / density);
	m_view.setOption<VFRendering::ArrowRenderer::Option::CONE_RADIUS>(coneradius * size / density);
	m_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_HEIGHT>(cylinderheight* size / density);
	m_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_RADIUS>(cylinderradius * size / density);
	m_view.setOption<VFRendering::ArrowRenderer::Option::LEVEL_OF_DETAIL>(lod);
}

float SpinWidget::arrowSize() const
{
	float b_min[3], b_max[3];
	Geometry_Get_Bounds(state.get(), b_min, b_max);
	glm::vec3 bounds_min = glm::make_vec3(b_min);
	glm::vec3 bounds_max = glm::make_vec3(b_max);
	glm::vec2 x_range{ bounds_min[0], bounds_max[0] };
	glm::vec2 y_range{ bounds_min[1], bounds_max[1] };
	glm::vec2 z_range{ bounds_min[2], bounds_max[2] };
	glm::vec3 bounding_box_center = { (bounds_min[0] + bounds_max[0]) / 2, (bounds_min[1] + bounds_max[1]) / 2, (bounds_min[2] + bounds_max[2]) / 2 };
	glm::vec3 bounding_box_side_lengths = { bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], bounds_max[2] - bounds_min[2] };

	int n_cells[3];
	Geometry_Get_N_Cells(this->state.get(), n_cells);
	
	float density = 0.01f;
	if (n_cells[0] > 1) density = std::max(density, n_cells[0] / (bounds_max[0] - bounds_min[0]));
	if (n_cells[1] > 1) density = std::max(density, n_cells[1] / (bounds_max[1] - bounds_min[1]));
	if (n_cells[2] > 1) density = std::max(density, n_cells[2] / (bounds_max[2] - bounds_min[2]));
	density /= n_cell_step;

	float size = options().get<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>() / 0.6f * density;
	return size;
}

int SpinWidget::arrowLOD() const
{
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

void SpinWidget::setOverallDirectionRange(glm::vec2 x_range, glm::vec2 y_range, glm::vec2 z_range)
{
	m_x_range_direction = x_range;
	m_y_range_direction = y_range;
	m_z_range_direction = z_range;

	this->updateIsVisibleImplementation();
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

void SpinWidget::setOverallPositionRange(glm::vec2 x_range, glm::vec2 y_range, glm::vec2 z_range)
{
	m_x_range_position = x_range;
	m_y_range_position = y_range;
	m_z_range_position = z_range;

	this->updateIsVisibleImplementation();
}

void SpinWidget::updateIsVisibleImplementation()
{
	std::ostringstream sstream;
	std::string is_visible_implementation;
	sstream << "bool is_visible(vec3 position, vec3 direction) {";
	//		position
	// X
	if (m_x_range_position.x >= m_x_range_position.y)
	{
		sstream << "bool is_visible_x_pos = true;";
	}
	else
	{
		sstream << "float x_min_pos = ";
		sstream << m_x_range_position.x;
		sstream << "; float x_max_pos = ";
		sstream << m_x_range_position.y;
		sstream << "; bool is_visible_x_pos = position.x <= x_max_pos && position.x >= x_min_pos;";
	}
	// Y
	if (m_y_range_position.x >= m_y_range_position.y) {
		sstream << "bool is_visible_y_pos = true;";
	}
	else {
		sstream << "float y_min_pos = ";
		sstream << m_y_range_position.x;
		sstream << "; float y_max_pos = ";
		sstream << m_y_range_position.y;
		sstream << "; bool is_visible_y_pos = position.y <= y_max_pos && position.y >= y_min_pos;";
	}
	// Z
	if (m_z_range_position.x >= m_z_range_position.y) {
		sstream << "bool is_visible_z_pos = true;";
	}
	else {
		sstream << "float z_min_pos = ";
		sstream << m_z_range_position.x;
		sstream << "; float z_max_pos = ";
		sstream << m_z_range_position.y;
		sstream << "; bool is_visible_z_pos = position.z <= z_max_pos && position.z >= z_min_pos;";
	}
	//		direction
	// X
	if (m_x_range_direction.x <= -1 && m_x_range_direction.y >= 1)
	{
		sstream << "bool is_visible_x_dir = true;";
	}
	else if (m_x_range_direction.x <= -1)
	{
		sstream << "float x_max_dir = ";
		sstream << m_x_range_direction.y;
		sstream << "; bool is_visible_x_dir = normalize(direction).x <= x_max_dir;";
	}
	else if (m_x_range_direction.y >= 1)
	{
		sstream << "float x_min_dir = ";
		sstream << m_x_range_direction.x;
		sstream << "; bool is_visible_x_dir = normalize(direction).x >= x_min_dir;";
	}
	else
	{
		sstream << "float x_min_dir = ";
		sstream << m_x_range_direction.x;
		sstream << "; float x_max_dir = ";
		sstream << m_x_range_direction.y;
		sstream << "; float x_dir = normalize(direction).x; bool is_visible_x_dir = x_dir >= x_min_dir && x_dir <= x_max_dir;";
	}
	// Y
	if (m_y_range_direction.x <= -1 && m_y_range_direction.y >= 1) {
		sstream << "bool is_visible_y_dir = true;";
	}
	else if (m_y_range_direction.x <= -1) {
		sstream << "float y_max_dir = ";
		sstream << m_y_range_direction.y;
		sstream << "; bool is_visible_y_dir = normalize(direction).y <= y_max_dir;";
	}
	else if (m_y_range_direction.y >= 1) {
		sstream << "float y_min_dir = ";
		sstream << m_y_range_direction.x;
		sstream << "; bool is_visible_y_dir = normalize(direction).y >= y_min_dir;";
	}
	else {
		sstream << "float y_min_dir = ";
		sstream << m_y_range_direction.x;
		sstream << "; float y_max_dir = ";
		sstream << m_y_range_direction.y;
		sstream << "; float y_dir = normalize(direction).y;  bool is_visible_y_dir = y_dir >= y_min_dir && y_dir <= y_max_dir;";
	}
	// Z
	if (m_z_range_direction.x <= -1 && m_z_range_direction.y >= 1) {
		sstream << "bool is_visible_z_dir = true;";
	}
	else if (m_z_range_direction.x <= -1) {
		sstream << "float z_max_dir = ";
		sstream << m_z_range_direction.y;
		sstream << "; bool is_visible_z_dir = normalize(direction).z <= z_max_dir;";
	}
	else if (m_z_range_direction.y >= 1) {
		sstream << "float z_min_dir = ";
		sstream << m_z_range_direction.x;
		sstream << "; bool is_visible_z_dir = normalize(direction).z >= z_min_dir;";
	}
	else {
		sstream << "float z_min_dir = ";
		sstream << m_z_range_direction.x;
		sstream << "; float z_max_dir = ";
		sstream << m_z_range_direction.y;
		sstream << "; float z_dir = normalize(direction).z;  bool is_visible_z_dir = z_dir >= z_min_dir && z_dir <= z_max_dir;";
	}
	//
	sstream << " return is_visible_x_pos && is_visible_y_pos && is_visible_z_pos && is_visible_x_dir && is_visible_y_dir && is_visible_z_dir; }";
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
  setColormapRotationInverted(colormap_rotation(), m_colormap_invert_z, m_colormap_invert_xy);
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
void SpinWidget::cycleCamera()
{
	if (this->m_camera_projection_perspective)
	{
		this->m_camera_projection_perspective = false;
	}
	else
	{
		this->m_camera_projection_perspective = true;
	}
	this->setVerticalFieldOfView(this->user_fov);
}

void SpinWidget::setCameraToDefault()
{
	if (this->m_interactionmode == InteractionMode::REGULAR)
	{
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
}

void SpinWidget::setCameraToX(bool inverted)
{
	if (this->m_interactionmode == InteractionMode::REGULAR)
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
}

void SpinWidget::setCameraToY(bool inverted)
{
	if (this->m_interactionmode == InteractionMode::REGULAR)
	{
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
}

void SpinWidget::setCameraToZ(bool inverted)
{
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

void SpinWidget::setCameraPosition(const glm::vec3& camera_position)
{
	if (this->m_interactionmode == InteractionMode::REGULAR)
	{
		auto system_center = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
		m_view.setOption<VFRendering::View::Option::CAMERA_POSITION>(system_center + camera_position);
	}
}

void SpinWidget::setCameraFocus(const glm::vec3& center_position)
{
	if (this->m_interactionmode == InteractionMode::REGULAR)
	{
		auto system_center = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
		m_view.setOption<VFRendering::View::Option::CENTER_POSITION>(system_center + center_position);
	}
}

void SpinWidget::setCameraUpVector(const glm::vec3& up_vector)
{
	if (this->m_interactionmode == InteractionMode::REGULAR)
	{
		m_view.setOption<VFRendering::View::Option::UP_VECTOR>(up_vector);
	}
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
	return this->user_fov;
}

void SpinWidget::setVerticalFieldOfView(float vertical_field_of_view)
{
	this->user_fov = vertical_field_of_view;
	if (!this->m_camera_projection_perspective)
	{
		vertical_field_of_view = 0;
	}

	// Calculate new camera position
	float scale = 1;
	float fov = m_view.options().get<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>();
	if (fov > 0 && vertical_field_of_view > 0)
	{
		scale = std::tan(glm::radians(fov)/2.0) / std::tan(glm::radians(vertical_field_of_view)/2.0);
		setCameraPosition(getCameraPositon()*scale);
	}
	else if (fov > 0)
	{
		scale = std::tan(glm::radians(fov) / 2.0);
		setCameraPosition(getCameraPositon()*scale);
	}
	else if (vertical_field_of_view > 0)
	{
		scale = 1.0 / std::tan(glm::radians(vertical_field_of_view) / 2.0);
		setCameraPosition(getCameraPositon()*scale);
	}

	// Set new FOV
	makeCurrent();
	m_view.setOption<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>(vertical_field_of_view);
	enableSystem(show_arrows, show_boundingbox, show_surface, show_isosurface);
}

bool SpinWidget::cameraProjection()
{
	return this->m_camera_projection_perspective;
}

void SpinWidget::setCameraProjection(bool perspective)
{
	if (this->m_interactionmode == InteractionMode::REGULAR)
	{
		this->m_camera_projection_perspective = perspective;
		this->setVerticalFieldOfView(this->user_fov);
	}
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
	glm::vec3 camera_position;
	glm::vec3 center_position;
	glm::vec3 up_vector;
	bool perspective;
	if (this->m_interactionmode == InteractionMode::REGULAR)
	{
		perspective = this->cameraProjection();
		camera_position = this->getCameraPositon();
		center_position = this->getCameraFocus();
		up_vector = this->getCameraUpVector();
	}
	else
	{
		perspective = this->regular_mode_perspective;
		camera_position = this->regular_mode_cam_pos;
		center_position = this->regular_mode_cam_focus;
		up_vector = this->regular_mode_cam_up;
	}
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
	settings.setValue("FOV", (int)(this->user_fov * 100));
	settings.setValue("perspective projection", perspective);
	settings.setValue("free rotation", this->m_camera_rotate_free);
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
		this->user_fov = (float)(settings.value("FOV").toInt() / 100.0f);
		this->m_camera_projection_perspective = settings.value("perspective projection").toBool();
		this->regular_mode_perspective = this->m_camera_projection_perspective;
		this->m_camera_rotate_free = settings.value("free rotation").toBool();
		m_view.setOption<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>(this->m_camera_projection_perspective*this->user_fov);
		glm::vec3 camera_position, center_position, up_vector;
		settings.beginReadArray("position");
		for(int dim=0; dim<3; ++dim)
		{
			settings.setArrayIndex(dim);
			camera_position[dim] = (float)(settings.value("vecp").toInt()/100.0f);
		}
		settings.endArray();
		this->setCameraPosition(camera_position);
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
