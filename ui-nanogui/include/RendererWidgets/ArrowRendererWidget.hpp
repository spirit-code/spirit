#pragma once
#ifndef ARROWRENDERERWIDGET_HPP
#define ARROWRENDERERWIDGET_HPP
#include <RendererWidgets/RendererBaseWidget.hpp>
#include <VFRendering/ArrowRenderer.hxx>
#include <nanogui/textbox.h>
#include <nanogui/slider.h>

class ArrowRendererWidget : public RendererBaseWidget
{
    public:
    ArrowRendererWidget(nanogui::Widget * parent, VFGLCanvas * gl_canvas) : RendererBaseWidget(parent, gl_canvas)
    {
        using VFRendering::ArrowRenderer;
        using Option = VFRendering::ArrowRenderer::Option;

        this->setTitle("Arrows");
        this->renderer = std::shared_ptr<VFRendering::ArrowRenderer>(new VFRendering::ArrowRenderer( gl_canvas->View(), gl_canvas->Vf()));

        this->chk->setChecked(true);
        this->gl_canvas->addRenderer(this->renderer);

        new nanogui::Label(this, "Size");
        auto size = new nanogui::Slider(this);
        size->setRange({0.1,2});
        size->setValue(1);
        size->setCallback
        (
            [&](float val)
            {
               auto ren = static_cast<ArrowRenderer *>(this->renderer.get());
               ren -> setOption<Option::CONE_RADIUS>( val * 0.25);
               ren -> setOption<Option::CONE_HEIGHT>( val * 0.6);
               ren -> setOption<Option::CYLINDER_RADIUS>( val * 0.125);
               ren -> setOption<Option::CYLINDER_HEIGHT>( val * 0.7);
            }
        );
        
        new nanogui::Label(this, "LOD");
        auto lod = new nanogui::IntBox<uint>(this, 10);
        lod->setMinMaxValues(0, 20);
        lod->setEditable(true);
        lod->setCallback(  
            [&](uint val) {
                static_cast<ArrowRenderer *>(this->renderer.get()) -> setOption<Option::LEVEL_OF_DETAIL>(val);
            }
        );
    }
};

#endif