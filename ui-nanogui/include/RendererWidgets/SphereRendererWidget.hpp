#pragma once
#ifndef SPHERERENDERERWIDGET_HPP
#define SPHERERENDERERWIDGET_HPP

#include <RendererWidgets/RendererBaseWidget.hpp>
#include <VFRendering/SphereRenderer.hxx>
#include <ListSelect.hpp>

class SphereRendererWidget : public RendererBaseWidget
{
    public:
    SphereRendererWidget(nanogui::Widget * parent, VFGLCanvas * gl_canvas) : RendererBaseWidget(parent, gl_canvas)
    {
        using Option = VFRendering::SphereRenderer::Option;

        this->setTitle("Spheres");
        this->renderer = std::shared_ptr<VFRendering::SphereRenderer>(new VFRendering::SphereRenderer( gl_canvas->View(), gl_canvas->Vf() ) );
        this->renderer->setOption<VFRendering::SphereRenderer::SPHERE_RADIUS>(0.3);

        this->chk->setChecked(true);
        this->gl_canvas->addRenderer(this->renderer);

        new nanogui::Label(this, "Radius");
        auto radius = new nanogui::Slider(this);
        radius->setRange({0.1,0.8});
        radius->setValue(0.3);
        radius->setCallback
        (
            [&](float val)
            {
                static_cast<VFRendering::SphereRenderer *>(this->renderer.get())->setOption<Option::SPHERE_RADIUS>(val);
            }
        );

        new nanogui::Label(this, "LOD");
        auto lod = new nanogui::IntBox<uint>(this, 1);
        lod->setMinMaxValues(0, 2);
        lod->setEditable(true);
        lod->setCallback(  
            [&](uint val) {
                static_cast<VFRendering::SphereRenderer *>(this->renderer.get()) -> setOption<Option::LEVEL_OF_DETAIL>(val);
            }
        );  
    }
};

#endif
