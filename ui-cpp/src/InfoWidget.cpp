#include "InfoWidget.hpp"
#include "SpinWidget.hpp"

#include "Spirit/Chain.h"
#include "Spirit/Geometry.h"
#include "Spirit/System.h"
#include "Spirit/Quantities.h"

#include <QtWidgets/QWidget>
#include <QGraphicsBlurEffect>

InfoWidget::InfoWidget(std::shared_ptr<State> state, SpinWidget *spinWidget)
{
    this->state = state;
    this->spinWidget = spinWidget;
    // Setup User Interface
    this->setupUi(this);

    // Mouse events should be passed through to the SpinWidget behind
    setAttribute(Qt::WA_TransparentForMouseEvents, true);
    
    // Update timer
    m_timer = new QTimer(this);
    connect(m_timer, &QTimer::timeout, this, &InfoWidget::updateData);
    m_timer->start(200);
}

void InfoWidget::updateData()
{
    // FPS
    this->m_Label_FPS->setText(QString::fromLatin1("FPS: ") + QString::number((int)this->spinWidget->getFramesPerSecond()));

    // Number of spins
    int nos = System_Get_NOS(state.get());
    QString nosqstring;
    if (nos < 1e5)
        nosqstring = QString::number(nos);
    else
        nosqstring = QString::number((float)nos, 'E', 2);

    // Energies
    double E = System_Get_Energy(state.get());
    this->m_Label_E->setText(     QString::fromLatin1("E      = ") + QString::number(E, 'f', 10));
    this->m_Label_E_dens->setText(QString::fromLatin1("E dens = ") + QString::number(E/nos, 'f', 10));

    // Magnetization
    float M[3];
    Quantity_Get_Magnetization(state.get(), M);
    this->m_Label_Mx->setText(QString::fromLatin1("Mx: ") + QString::number(M[0], 'f', 8));
    this->m_Label_My->setText(QString::fromLatin1("My: ") + QString::number(M[1], 'f', 8));
    this->m_Label_Mz->setText(QString::fromLatin1("Mz: ") + QString::number(M[2], 'f', 8));

    // Force
    // ...

    // Dimensions
    this->m_Label_NOI->setText(QString::fromLatin1("NOI:  ") + QString::number(Chain_Get_NOI(this->state.get())));
    this->m_Label_NOS->setText(QString::fromLatin1("NOS:  ") + nosqstring + QString::fromLatin1(" "));
    int n_cells[3];
    Geometry_Get_N_Cells(this->state.get(), n_cells);
    QString text_Dims = QString::fromLatin1("Dims: ") + QString::number(n_cells[0]) + QString::fromLatin1("x") +
        QString::number(n_cells[1]) + QString::fromLatin1("x") + QString::number(n_cells[2]);
    int nth = this->spinWidget->visualisationNCellSteps();
    if (nth == 2)
    {
        text_Dims += QString::fromLatin1("    (using every ") + QString::number(nth) + QString::fromLatin1("nd)");
    }
    else if (nth == 3)
    {
        text_Dims += QString::fromLatin1("    (using every ") + QString::number(nth) + QString::fromLatin1("rd)");
    }
    else if (nth > 3)
    {
        text_Dims += QString::fromLatin1("    (using every ") + QString::number(nth) + QString::fromLatin1("th)");
    }
    this->m_Label_Dims->setText(text_Dims);
}