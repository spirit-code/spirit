#pragma once
#ifndef VISUALISATIONWIDGET_HPP
#define VISUALISATIONWIDGET_HPP 

#include <nanogui/window.h>
#include <nanogui/button.h>
#include <nanogui/entypo.h>
#include <nanogui/combobox.h>
#include <nanogui/popup.h>
#include <nanogui/popupbutton.h>

#include <RendererWidgets/RendererBaseWidget.hpp>
#include <RendererWidgets/ArrowRendererWidget.hpp>
#include <RendererWidgets/DotRendererWidget.hpp>
#include <RendererWidgets/SphereRendererWidget.hpp>

#include <ListSelect.hpp>

class VisualisationWidget : public nanogui::Window
{
    public:
    VisualisationWidget(nanogui::Widget * parent, VFGLCanvas * gl_canvas) : nanogui::Window(parent, "Visualisation"), gl_canvas(gl_canvas)
    {
        this->setLayout(new nanogui::GroupLayout());

        add_renderer_btn = new nanogui::PopupButton(this, "Add", ENTYPO_ICON_PLUS);
        add_renderer_btn->setIconPosition(nanogui::Button::IconPosition::Right);

        add_renderer_btn->popup()->setLayout(new nanogui::GroupLayout(10));

        auto btn = new nanogui::Button(add_renderer_btn->popup(), "Arrow");
        btn->setCallback(
            [&]{ 
                    addRenderer( new ArrowRendererWidget(this, this->gl_canvas) );
                    add_renderer_btn->setPushed(false);
                }
        );

        btn = new nanogui::Button(add_renderer_btn->popup(), "Dots");
        btn->setCallback(
            [&]{
                addRenderer( new DotRendererWidget(this, this->gl_canvas) );
                add_renderer_btn->setPushed(false);
            }
        );

        btn = new nanogui::Button(add_renderer_btn->popup(), "Sphere");
        btn->setCallback(
            [&]{
                    addRenderer( new SphereRendererWidget(this, this->gl_canvas) );
                    add_renderer_btn->setPushed(false);
            }
        );
        new nanogui::Label(this, "Renderer(s):");
    }

    void addRenderer(RendererBaseWidget * ren)
    {
        this->setHeight( this->preferredSize(this->screen()->nvgContext())[1] );
        this->performLayout(this->screen()->nvgContext());
        ren->close_btn->setCallback(
            [&, ren]
            {
                this->screen()->updateFocus(nullptr); // We need to make sure to remove ren from the focusPath
                this->removeChild(ren);
                this->setHeight( this->preferredSize(this->screen()->nvgContext())[1] );
                this->performLayout(this->screen()->nvgContext());
            }
        );
    }

    private:
    VFGLCanvas * gl_canvas;
    ListSelect * renderer_select;
    nanogui::PopupButton * add_renderer_btn;
};

#endif