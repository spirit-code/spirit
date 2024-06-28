#include "PlotWidget.hpp"

#include "Spirit/Chain.h"
#include "Spirit/Parameters_GNEB.h"
#include "Spirit/System.h"

#include <QGraphicsLayout>
#include <QtGui/QImage>
#include <QtGui/QPainter>

using namespace QtCharts;

static const int SIZE = 30;
static const int INCR = 3;
static const int REGU = 8;
static const int TRIA = 10;

PlotWidget::PlotWidget( std::shared_ptr<State> state, bool plot_image_energies, bool plot_interpolated )
        : plot_image_energies( plot_image_energies ), plot_interpolated( plot_interpolated )
{
    this->state               = state;
    this->plot_interpolated_n = Parameters_GNEB_Get_N_Energy_Interpolations( state.get() );

    // Create Chart
    chart = new QChart();
    chart->legend()->hide();
    chart->setTitle( "" );
    chart->setMargins( { 0, 0, 0, 0 } );
    chart->layout()->setContentsMargins( 0, 0, 0, 0 );
    chart->setBackgroundRoundness( 0 );

    // Use Chart
    this->setChart( chart );
    this->setRenderHint( QPainter::Antialiasing );

    // Create triangle painters
    QRectF rect = QRectF( 0, 0, SIZE, SIZE );
    QPainterPath triangleUpPath;
    triangleUpPath.moveTo( rect.left() + ( rect.width() / 2 ), rect.top() + INCR );
    triangleUpPath.lineTo( rect.right() - INCR, rect.bottom() - INCR );
    triangleUpPath.lineTo( rect.left() + INCR, rect.bottom() - INCR );
    triangleUpPath.lineTo( rect.left() + ( rect.width() / 2 ), rect.top() + INCR );
    QPainterPath triangleDownPath;
    triangleDownPath.moveTo( rect.left() + ( rect.width() / 2 ), rect.bottom() - INCR );
    triangleDownPath.lineTo( rect.left() + INCR, rect.top() + INCR );
    triangleDownPath.lineTo( rect.right() - INCR, rect.top() + INCR );
    triangleDownPath.lineTo( rect.left() + ( rect.width() / 2 ), rect.bottom() - INCR );

    triangleUpRed = QImage( SIZE, SIZE, QImage::Format_ARGB32 );
    triangleUpRed.fill( Qt::transparent );
    triangleDownRed = QImage( SIZE, SIZE, QImage::Format_ARGB32 );
    triangleDownRed.fill( Qt::transparent );
    triangleUpBlue = QImage( SIZE, SIZE, QImage::Format_ARGB32 );
    triangleUpBlue.fill( Qt::transparent );
    triangleDownBlue = QImage( SIZE, SIZE, QImage::Format_ARGB32 );
    triangleDownBlue.fill( Qt::transparent );

    QPainter painter1( &triangleUpRed );
    painter1.setRenderHint( QPainter::Antialiasing );
    painter1.setPen( QColor( "Red" ) );
    painter1.setBrush( painter1.pen().color() );
    painter1.drawPath( triangleUpPath );
    QPainter painter2( &triangleDownRed );
    painter2.setRenderHint( QPainter::Antialiasing );
    painter2.setPen( QColor( "Red" ) );
    painter2.setBrush( painter2.pen().color() );
    painter2.drawPath( triangleDownPath );
    QPainter painter3( &triangleUpBlue );
    painter3.setRenderHint( QPainter::Antialiasing );
    painter3.setPen( QColor( "RoyalBlue" ) );
    painter3.setBrush( painter3.pen().color() );
    painter3.drawPath( triangleUpPath );
    QPainter painter4( &triangleDownBlue );
    painter4.setRenderHint( QPainter::Antialiasing );
    painter4.setPen( QColor( "RoyalBlue" ) );
    painter4.setBrush( painter4.pen().color() );
    painter4.drawPath( triangleDownPath );

    // Create Series
    // Normal images energies
    series_E_normal = new QScatterSeries();
    series_E_normal->setColor( QColor( "RoyalBlue" ) );
    series_E_normal->setMarkerSize( REGU );
    series_E_normal->setMarkerShape( QScatterSeries::MarkerShapeCircle );
    // Climbing images
    series_E_climbing = new QScatterSeries();
    series_E_climbing->setColor( QColor( "RoyalBlue" ) );
    series_E_climbing->setMarkerSize( TRIA );
    series_E_climbing->setMarkerShape( QScatterSeries::MarkerShapeRectangle );
    series_E_climbing->setBrush( triangleUpBlue.scaled( TRIA, TRIA ) );
    series_E_climbing->setPen( QColor( Qt::transparent ) );
    // Falling images
    series_E_falling = new QScatterSeries();
    series_E_falling->setColor( QColor( "RoyalBlue" ) );
    series_E_falling->setMarkerSize( TRIA );
    series_E_falling->setMarkerShape( QScatterSeries::MarkerShapeRectangle );
    series_E_falling->setBrush( triangleDownBlue.scaled( TRIA, TRIA ) );
    series_E_falling->setPen( QColor( Qt::transparent ) );
    // Stationary images
    series_E_stationary = new QScatterSeries();
    series_E_stationary->setColor( QColor( "RoyalBlue" ) );
    series_E_stationary->setMarkerSize( REGU );
    series_E_stationary->setMarkerShape( QScatterSeries::MarkerShapeRectangle );
    // Interpolated energies
    series_E_interp = new QLineSeries();
    series_E_interp->setColor( QColor( "RoyalBlue" ) );

    // Current energy
    series_E_current = new QScatterSeries();
    series_E_current->setColor( QColor( "Red" ) );
    series_E_current->setMarkerSize( REGU );
    series_E_current->setMarkerShape( QScatterSeries::MarkerShapeCircle );

    // Add Series
    chart->addSeries( series_E_interp );
    chart->addSeries( series_E_normal );
    chart->addSeries( series_E_climbing );
    chart->addSeries( series_E_falling );
    chart->addSeries( series_E_stationary );
    chart->addSeries( series_E_current );

    // Create Axes
    this->chart->createDefaultAxes();
    this->chart->axisX()->setTitleText( "Rx" );
    this->chart->axisX()->setMin( -0.04 );
    this->chart->axisX()->setMax( 1.04 );
    this->chart->axisY()->setTitleText( "E" );

    // Fill the Series with initial values
    this->plotEnergies();
}

void PlotWidget::updateData()
{
    this->plotEnergies();
    this->chart->update();
}

void PlotWidget::plotEnergies()
{
    int noi = Chain_Get_NOI( state.get() );
    int nos = System_Get_NOS( state.get() );

    if( this->plot_interpolated
        && this->plot_interpolated_n != Parameters_GNEB_Get_N_Energy_Interpolations( state.get() ) )
        Parameters_GNEB_Set_N_Energy_Interpolations( state.get(), this->plot_interpolated_n );

    int size_interp = noi + ( noi - 1 ) * Parameters_GNEB_Get_N_Energy_Interpolations( state.get() );

    // Allocate arrays
    Rx              = std::vector<scalar>( noi, 0 );
    energies        = std::vector<scalar>( noi, 0 );
    Rx_interp       = std::vector<scalar>( size_interp, 0 );
    energies_interp = std::vector<scalar>( size_interp, 0 );

    // Get Data
    scalar Rx_tot = System_Get_Rx( state.get(), noi - 1 );
    Chain_Get_Rx( state.get(), Rx.data() );
    Chain_Get_Energy( state.get(), energies.data() );
    if( this->plot_interpolated )
    {
        Chain_Get_Rx_Interpolated( state.get(), Rx_interp.data() );
        Chain_Get_Energy_Interpolated( state.get(), energies_interp.data() );
    }

    // Replacement data vectors
    auto empty      = QVector<QPointF>( 0 );
    auto current    = QVector<QPointF>( 0 );
    auto normal     = QVector<QPointF>( 0 );
    auto climbing   = QVector<QPointF>( 0 );
    auto falling    = QVector<QPointF>( 0 );
    auto stationary = QVector<QPointF>( 0 );
    auto interp     = QVector<QPointF>( 0 );

    // Min and max yaxis values
    float ymin = 1e8, ymax = -1e8;

    // Add data to series
    int idx_current = System_Get_Index( state.get() );

    scalar Rx_cur = Rx[idx_current];
    if( renormalize_Rx && Rx_tot > 0 )
        Rx_cur /= Rx_tot;

    scalar E_cur = energies[idx_current];
    if( divide_by_nos )
        E_cur /= nos;

    current.push_back( QPointF( Rx_cur, E_cur ) );

    if( this->plot_image_energies )
    {
        for( int i = 0; i < noi; ++i )
        {
            if( i > 0 && Rx_tot > 0 )
                Rx[i] = Rx[i];
            energies[i] = energies[i];

            if( renormalize_Rx && Rx_tot > 0 )
                Rx[i] /= Rx_tot;

            if( divide_by_nos )
                energies[i] /= nos;

            if( Parameters_GNEB_Get_Climbing_Falling( state.get(), i ) == 0 )
                normal.push_back( QPointF( Rx[i], energies[i] ) );
            else if( Parameters_GNEB_Get_Climbing_Falling( state.get(), i ) == 1 )
                climbing.push_back( QPointF( Rx[i], energies[i] ) );
            else if( Parameters_GNEB_Get_Climbing_Falling( state.get(), i ) == 2 )
                falling.push_back( QPointF( Rx[i], energies[i] ) );
            else if( Parameters_GNEB_Get_Climbing_Falling( state.get(), i ) == 3 )
                stationary.push_back( QPointF( Rx[i], energies[i] ) );

            if( energies[i] < ymin )
                ymin = energies[i];
            if( energies[i] > ymax )
                ymax = energies[i];
        }
    }
    if( this->plot_interpolated )
    {
        for( int i = 0; i < size_interp; ++i )
        {
            if( i > 0 && Rx_tot > 0 )
                Rx_interp[i] = Rx_interp[i];
            energies_interp[i] = energies_interp[i];

            if( renormalize_Rx && Rx_tot > 0 )
                Rx_interp[i] /= Rx_tot;

            if( divide_by_nos )
                energies_interp[i] /= nos;

            interp.push_back( QPointF( Rx_interp[i], energies_interp[i] ) );

            if( energies_interp[i] < ymin )
                ymin = energies_interp[i];
            if( energies_interp[i] > ymax )
                ymax = energies_interp[i];
        }
    }

    // Set marker type for current image
    if( Parameters_GNEB_Get_Climbing_Falling( state.get() ) == 0 )
    {
        series_E_current->setMarkerShape( QScatterSeries::MarkerShapeCircle );
        series_E_current->setMarkerSize( REGU );
        series_E_current->setBrush( QColor( "Red" ) );
    }
    else if( Parameters_GNEB_Get_Climbing_Falling( state.get() ) == 1 )
    {
        series_E_current->setMarkerShape( QScatterSeries::MarkerShapeRectangle );
        series_E_current->setMarkerSize( TRIA );
        series_E_current->setBrush( triangleUpRed.scaled( TRIA, TRIA ) );
        series_E_current->setPen( QColor( Qt::transparent ) );
    }
    else if( Parameters_GNEB_Get_Climbing_Falling( state.get() ) == 2 )
    {
        series_E_current->setMarkerShape( QScatterSeries::MarkerShapeRectangle );
        series_E_current->setMarkerSize( TRIA );
        series_E_current->setBrush( triangleDownRed.scaled( TRIA, TRIA ) );
        series_E_current->setPen( QColor( Qt::transparent ) );
    }
    else if( Parameters_GNEB_Get_Climbing_Falling( state.get() ) == 3 )
    {
        series_E_current->setMarkerShape( QScatterSeries::MarkerShapeRectangle );
        series_E_current->setMarkerSize( REGU );
        series_E_current->setBrush( QColor( "Red" ) );
    }

    // Clear series
    series_E_normal->replace( empty );
    series_E_climbing->replace( empty );
    series_E_falling->replace( empty );
    series_E_stationary->replace( empty );
    series_E_interp->replace( empty );

    // Re-fill Series
    series_E_normal->replace( normal );
    series_E_climbing->replace( climbing );
    series_E_falling->replace( falling );
    series_E_stationary->replace( stationary );
    series_E_interp->replace( interp );

    // Current image - red dot
    series_E_current->replace( empty );
    series_E_current->replace( current );

    // Rescale y axis
    if( auto axes = this->chart->axes( Qt::Vertical ); !axes.empty() )
    {
        float delta = 0.1 * ( ymax - ymin );
        if( delta < 1e-6 )
            delta = 0.1;

        axes[0]->setMin( ymin - delta );
        axes[0]->setMax( ymax + delta );
    }

    // Rescale x axis
    if( auto axes = this->chart->axes( Qt::Horizontal ); !axes.empty() )
    {
        if( !renormalize_Rx && Rx_tot > 0 )
        {
            const float delta = 0.04 * Rx_tot;
            axes[0]->setMin( Rx[0] - delta );
            axes[0]->setMax( Rx[noi - 1] + delta );
        }
        else if( Rx_tot > 0 )
        {
            axes[0]->setMin( -0.04 );
            axes[0]->setMax( 1.04 );
        }
        else
        {
            axes[0]->setMin( -0.04 );
            axes[0]->setMax( 0.04 );
        }
    }
}
