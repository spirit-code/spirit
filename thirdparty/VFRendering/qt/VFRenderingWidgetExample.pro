# Example Qt Project using the VFRenderingWidget class

TEMPLATE = app
TARGET = VFRenderingWidgetExample
QT += widgets
QMAKE_MACOSX_DEPLOYMENT_TARGET = 10.12

INCLUDEPATH += ../include
INCLUDEPATH += ../thirdparty/glm/include
# the libs are set for use with the top level Makefile
LIBS += -L../build ../build/libVFRendering.a -lqhullcpp -L../thirdparty/qhull/lib/ -lqhullcpp -lqhullstatic_r

HEADERS += VFRenderingWidget.hxx
SOURCES += main.cxx VFRenderingWidget.cxx
