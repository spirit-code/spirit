#pragma once
#ifndef DOTRENDERERWIDGET_HPP
#define DOTRENDERERWIDGET_HPP
#include <RendererWidgets/RendererBaseWidget.hpp>
#include <VFRendering/DotRenderer.hxx>
#include <ListSelect.hpp>

class DotRendererWidget : public RendererBaseWidget
{
    public:
    DotRendererWidget(nanogui::Widget * parent, VFGLCanvas * gl_canvas) : RendererBaseWidget(parent, gl_canvas)
    {
        using Option = VFRendering::DotRenderer::Option;
        using DotStyle = VFRendering::DotRenderer::DotStyle;

        this->setTitle("Dots");
        this->renderer = std::shared_ptr<VFRendering::DotRenderer>(new VFRendering::DotRenderer( gl_canvas->View(), gl_canvas->Vf() ) );
        this->renderer->setOption<VFRendering::DotRenderer::DOT_RADIUS>(1000);
        
        this->chk->setChecked(true);
        this->gl_canvas->addRenderer(this->renderer);

        new nanogui::Label(this, "Style");
        auto style = new ListSelect(this, {"Circle", "Square"});
        style->setCallback
        (
            [&](int i )
            {
                if(i==0)
                    static_cast<VFRendering::DotRenderer *>(this->renderer.get())->setOption<Option::DOT_STYLE>(DotStyle::CIRCLE);
                else if(i==1)
                    static_cast<VFRendering::DotRenderer *>(this->renderer.get())->setOption<Option::DOT_STYLE>(DotStyle::SQUARE);
            }
        );

        new nanogui::Label(this, "Radius");
        auto radius = new nanogui::Slider(this);
        radius->setRange({100,2000});
        radius->setValue(1000);
        radius->setCallback
        (
            [&](float val)
            {
                static_cast<VFRendering::DotRenderer *>(this->renderer.get())->setOption<Option::DOT_RADIUS>(int(val));
            }
        );
      

            
    }
};

#endif