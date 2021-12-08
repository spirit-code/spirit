#include "IsosurfaceWidget.hpp"
#include "SpinWidget.hpp"

#include <QtWidgets>

IsosurfaceWidget::IsosurfaceWidget( std::shared_ptr<State> state, SpinWidget * spinWidget )
{
    this->state      = state;
    this->spinWidget = spinWidget;
    setAttribute( Qt::WA_DeleteOnClose );
    // Setup User Interface
    this->setupUi( this );

    // Create renderer pointer
    this->m_renderer
        = std::make_shared<VFRendering::IsosurfaceRenderer>( *spinWidget->view(), *spinWidget->vectorfield() );

    // Defaults
    this->setShowIsosurface( true );
    this->setIsovalue( 0 );
    this->setIsocomponent( 2 );
    this->setDrawShadows( false );

    // Read values
    auto isovalue = this->isovalue();
    horizontalSlider_isovalue->setRange( 0, 100 );
    horizontalSlider_isovalue->setValue( (int)( isovalue + 1 * 50 ) );
    int component = this->isocomponent();
    if( component == 0 )
        this->radioButton_isosurface_x->setChecked( true );
    else if( component == 1 )
        this->radioButton_isosurface_y->setChecked( true );
    else if( component == 2 )
        this->radioButton_isosurface_z->setChecked( true );

    // Add this isosurface to the SpinWidget
    this->spinWidget->addIsosurface( m_renderer );

    // Connect Slots
    this->setupSlots();

    // Input validation
    QRegularExpression re( "[+|-]?[\\d]*[\\.]?[\\d]*" );
    this->number_validator = new QRegularExpressionValidator( re );
    this->setupInputValidators();
}

bool IsosurfaceWidget::showIsosurface()
{
    return this->m_show_isosurface;
}

void IsosurfaceWidget::setShowIsosurface( bool show )
{
    this->m_show_isosurface = show;
    QTimer::singleShot( 1, this->spinWidget, SLOT( update() ) );
}

float IsosurfaceWidget::isovalue()
{
    return this->m_isovalue;
}

void IsosurfaceWidget::setIsovalue( float value )
{
    this->m_isovalue = value;
    m_renderer->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>( m_isovalue );
    QTimer::singleShot( 1, this->spinWidget, SLOT( update() ) );
}

int IsosurfaceWidget::isocomponent()
{
    return this->m_isocomponent;
}

void IsosurfaceWidget::setIsocomponent( int component )
{
    this->m_isocomponent = component;

    if( this->m_isocomponent == 0 )
    {
        m_renderer->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>(
            []( const glm::vec3 & position,
                const glm::vec3 & direction ) -> VFRendering::IsosurfaceRenderer::isovalue_type
            {
                (void)position;
                return direction.x;
            } );
    }
    else if( this->m_isocomponent == 1 )
    {
        m_renderer->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>(
            []( const glm::vec3 & position,
                const glm::vec3 & direction ) -> VFRendering::IsosurfaceRenderer::isovalue_type
            {
                (void)position;
                return direction.y;
            } );
    }
    else if( this->m_isocomponent == 2 )
    {
        m_renderer->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>(
            []( const glm::vec3 & position,
                const glm::vec3 & direction ) -> VFRendering::IsosurfaceRenderer::isovalue_type
            {
                (void)position;
                return direction.z;
            } );
    }

    QTimer::singleShot( 1, this->spinWidget, SLOT( update() ) );
}

bool IsosurfaceWidget::drawShadows()
{
    return this->m_draw_shadows;
}

void IsosurfaceWidget::setDrawShadows( bool show )
{
    this->m_draw_shadows = show;

    if( this->m_draw_shadows )
    {
        m_renderer->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>(
            "uniform vec3 uLightPosition;"
            "float lighting(vec3 position, vec3 normal)"
            "{"
            "    vec3 lightDirection = -normalize(uLightPosition-position);"
            "    float diffuse = 0.7*max(0.0, dot(normal, lightDirection));"
            "    float ambient = 0.2;"
            "    return diffuse+ambient;"
            "}" );
    }
    else
    {
        m_renderer->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>(
            "float lighting(vec3 position, vec3 normal) { return 1.0; }" );
    }

    QTimer::singleShot( 1, this->spinWidget, SLOT( update() ) );
}

void IsosurfaceWidget::setupSlots()
{
    connect( horizontalSlider_isovalue, SIGNAL( valueChanged( int ) ), this, SLOT( slot_setIsovalue_slider() ) );
    connect( lineEdit_isovalue, SIGNAL( returnPressed() ), this, SLOT( slot_setIsovalue_lineedit() ) );
    connect( radioButton_isosurface_x, SIGNAL( toggled( bool ) ), this, SLOT( slot_setIsocomponent() ) );
    connect( radioButton_isosurface_y, SIGNAL( toggled( bool ) ), this, SLOT( slot_setIsocomponent() ) );
    connect( radioButton_isosurface_z, SIGNAL( toggled( bool ) ), this, SLOT( slot_setIsocomponent() ) );

    connect( checkBox_invert_lighting, SIGNAL( stateChanged( int ) ), this, SLOT( slot_setTriangleNormal() ) );

    connect( pushButton_remove, SIGNAL( clicked() ), this, SLOT( close() ) );
}

void IsosurfaceWidget::slot_setIsovalue_slider()
{
    float isovalue = horizontalSlider_isovalue->value() / 50.0f - 1.0f;
    this->lineEdit_isovalue->setText( QString::number( isovalue ) );
    this->setIsovalue( isovalue );
}

void IsosurfaceWidget::slot_setIsovalue_lineedit()
{
    float isovalue = this->lineEdit_isovalue->text().toFloat();
    this->horizontalSlider_isovalue->setValue( (int)( isovalue * 50 + 50 ) );
    this->setIsovalue( isovalue );
}

void IsosurfaceWidget::slot_setIsocomponent()
{
    if( this->radioButton_isosurface_x->isChecked() )
    {
        this->setIsocomponent( 0 );
    }
    else if( this->radioButton_isosurface_y->isChecked() )
    {
        this->setIsocomponent( 1 );
    }
    else if( this->radioButton_isosurface_z->isChecked() )
    {
        this->setIsocomponent( 2 );
    }
}

void IsosurfaceWidget::slot_setTriangleNormal()
{
    m_renderer->setOption<VFRendering::IsosurfaceRenderer::Option::FLIP_NORMALS>(
        this->checkBox_invert_lighting->isChecked() );
    QTimer::singleShot( 1, this->spinWidget, SLOT( update() ) );
}

void IsosurfaceWidget::setupInputValidators()
{
    // Isovalue
    this->lineEdit_isovalue->setValidator( this->number_validator );
}

void IsosurfaceWidget::closeEvent( QCloseEvent * event )
{
    // Remove this isosurface from the SpinWidget
    this->spinWidget->removeIsosurface( m_renderer );
    // Notify others that this widget was closed
    emit closedSignal();
    // Close
    event->accept();
}