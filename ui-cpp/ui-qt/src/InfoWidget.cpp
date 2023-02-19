#include "InfoWidget.hpp"
#include "ControlWidget.hpp"
#include "SpinWidget.hpp"

#include <QGraphicsBlurEffect>
#include <QtWidgets/QWidget>

#include "Spirit/Chain.h"
#include "Spirit/Geometry.h"
#include "Spirit/Quantities.h"
#include "Spirit/Simulation.h"
#include "Spirit/System.h"

InfoWidget::InfoWidget( std::shared_ptr<State> state, SpinWidget * spinWidget, ControlWidget * controlWidget )
{
    this->state         = state;
    this->spinWidget    = spinWidget;
    this->controlWidget = controlWidget;

    // Setup User Interface
    this->setupUi( this );

    // Mouse events should be passed through to the SpinWidget behind
    setAttribute( Qt::WA_TransparentForMouseEvents, true );

    // Update timer
    m_timer = new QTimer( this );
    connect( m_timer, &QTimer::timeout, this, &InfoWidget::updateData );
    m_timer->start( 200 );
}

void InfoWidget::updateData()
{
    // FPS
    this->m_Label_FPS->setText(
        QString::fromLatin1( "FPS: " ) + QString::number( (int)this->spinWidget->getFramesPerSecond() ) );

    // Number of spins
    int nos = System_Get_NOS( state.get() );
    QString nosqstring;
    if( nos < 1e5 )
        nosqstring = QString::number( nos );
    else
        nosqstring = QString::number( (float)nos, 'E', 2 );

    // Energies
    double E = System_Get_Energy( state.get() );
    this->m_Label_E->setText( QString::fromLatin1( "E      = " ) + QString::number( E, 'f', 10 ) );
    this->m_Label_E_dens->setText( QString::fromLatin1( "E dens = " ) + QString::number( E / nos, 'f', 10 ) );

    // Magnetization
    scalar M[3];
    Quantity_Get_Average_Spin( state.get(), M );
    this->m_Label_Mx->setText( QString::fromLatin1( "Sx: " ) + QString::number( M[0], 'f', 8 ) );
    this->m_Label_My->setText( QString::fromLatin1( "Sy: " ) + QString::number( M[1], 'f', 8 ) );
    this->m_Label_Mz->setText( QString::fromLatin1( "Sz: " ) + QString::number( M[2], 'f', 8 ) );

    // Force
    double f_max = Simulation_Get_MaxTorqueNorm( state.get() );
    this->m_Label_Force_Max->setText( QString::fromLatin1( "F (max):     " ) + QString::number( f_max, 'f', 12 ) );
    this->m_Label_Force_Max_2->setText( QString::fromLatin1( "F (max):     " ) + QString::number( f_max, 'E', 3 ) );

    if( Simulation_Running_On_Chain( state.get() ) )
    {
        scalar * forces = new scalar[Chain_Get_NOI( state.get() )];
        Simulation_Get_Chain_MaxTorqueNorms( state.get(), forces );
        float f_current = forces[System_Get_Index( state.get() )];
        this->m_Label_Force_Current->show();
        this->m_Label_Force_Current->setText(
            QString::fromLatin1( "F (current): " ) + QString::number( f_current, 'f', 12 ) );
        this->m_Label_Force_Current_2->show();
        this->m_Label_Force_Current_2->setText(
            QString::fromLatin1( "F (current): " ) + QString::number( f_current, 'E', 3 ) );
    }
    else
    {
        this->m_Label_Force_Current->hide();
        this->m_Label_Force_Current_2->hide();
    }

    // Dimensions
    this->m_Label_NOI->setText(
        QString::fromLatin1( "NOI: " ) + QString::number( Chain_Get_NOI( this->state.get() ) ) );
    this->m_Label_NOS->setText( QString::fromLatin1( "NOS: " ) + nosqstring );
    this->m_Label_NBasis->setText(
        QString::fromLatin1( "N Basis Atoms: " ) + QString::number( Geometry_Get_N_Cell_Atoms( this->state.get() ) ) );
    int n_cells[3];
    Geometry_Get_N_Cells( this->state.get(), n_cells );
    QString text_Dims = QString::fromLatin1( "Cells: " ) + QString::number( n_cells[0] ) + QString::fromLatin1( " x " )
                        + QString::number( n_cells[1] ) + QString::fromLatin1( " x " ) + QString::number( n_cells[2] );
    int nth = this->spinWidget->visualisationNCellSteps();
    if( nth == 2 )
        text_Dims
            += QString::fromLatin1( "\n (showing every " ) + QString::number( nth ) + QString::fromLatin1( "nd atom)" );
    else if( nth == 3 )
        text_Dims
            += QString::fromLatin1( "\n (showing every " ) + QString::number( nth ) + QString::fromLatin1( "rd atom)" );
    else if( nth > 3 )
        text_Dims
            += QString::fromLatin1( "\n (showing every " ) + QString::number( nth ) + QString::fromLatin1( "th atom)" );
    this->m_Label_Dims->setText( text_Dims );

    // Simulation
    this->m_Label_Method->setText( QString::fromStdString( "Method: " + this->controlWidget->methodName() ) );
    this->m_Label_Solver->setText( QString::fromStdString( "Solver: " + this->controlWidget->solverName() ) );
    int walltime            = Simulation_Get_Wall_Time( state.get() );
    int hours               = walltime / ( 60 * 60 * 1000 );
    int minutes             = ( walltime - 60 * 60 * 1000 * hours ) / ( 60 * 1000 );
    int seconds             = ( walltime - 60 * 60 * 1000 * hours - 60 * 1000 * minutes ) / 1000;
    int miliseconds         = walltime - 60 * 60 * 1000 * hours - 60 * 1000 * minutes - 1000 * seconds;
    QString qs_hours        = QString( "%1" ).arg( hours, 2, 10, QChar( '0' ) );
    QString qs_minutes      = QString( "%1" ).arg( minutes, 2, 10, QChar( '0' ) );
    QString qs_seconds      = QString( "%1" ).arg( seconds, 2, 10, QChar( '0' ) );
    QString qs_milliseconds = QString( "%1" ).arg( miliseconds, 3, 10, QChar( '0' ) );
    QString qs_walltime     = qs_hours + ":" + qs_minutes + ":" + qs_seconds + "." + qs_milliseconds;
    this->m_Label_Wall_Time->setText( qs_walltime );
    float ips        = Simulation_Get_IterationsPerSecond( state.get() );
    int precision    = 0;
    QString qstr_ips = "";
    if( ips < 1 )
        precision = 4;
    else if( ips > 99 )
        precision = 0;
    else
        precision = 2;
    if( ips < 1e5 )
        qstr_ips = QString::number( ips, 'f', precision );
    else
        qstr_ips = QString::fromLatin1( "> 100k" );
    this->m_Label_IPS->setText( QString::fromLatin1( "IPS: " ) + qstr_ips );
    int iter = Simulation_Get_Iteration( this->state.get() );
    QString qs_iter;
    if( iter > 1e6 )
        qs_iter = QString::number( (float)iter, 'E', 4 );
    else
        qs_iter = QString::number( iter );
    this->m_Label_Iteration->setText( QString::fromLatin1( "Iteration: " ) + qs_iter );
    float simulation_time = Simulation_Get_Time( state.get() );
    QString qs_time       = QString::number( simulation_time, 'f', 2 );
    this->m_Label_Time->setText( qs_time );
}