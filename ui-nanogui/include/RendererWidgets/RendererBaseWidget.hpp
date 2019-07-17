#pragma once
#ifndef RENDERERBASEWIDGET_HPP
#define RENDERERBASEWIDGET_HPP

#include <nanogui/window.h>
#include <nanogui/checkbox.h>
#include <VFRendering/View.hxx>
#include <nanogui/layout.h>
#include <nanogui/label.h>
#include <GLCanvas.hpp>

class RendererBaseWidget : public nanogui::Window
{
    public:
    RendererBaseWidget(nanogui::Widget * parent, VFGLCanvas * gl_canvas) : nanogui::Window(parent, "Base"), gl_canvas(gl_canvas)
    {
        this->setLayout(new nanogui::GridLayout());

        chk = new nanogui::CheckBox(this->buttonPanel(), "Show");
        chk->setCallback(
            [&](bool checked)
            {
                if(checked)
                    this->gl_canvas->addRenderer(renderer);
                else 
                    this->gl_canvas->removeRenderer(renderer);
                }
        );

        this->close_btn = new nanogui::Button(this->buttonPanel(), "", ENTYPO_ICON_CROSS);
    }

    ~RendererBaseWidget()
    {
        if(this->chk->checked() && this->gl_canvas)
            this->gl_canvas->removeRenderer(renderer);
    }

    // We modify the draw code of the standard window a bit (remove drop shadows, left align titles)
    void draw(NVGcontext *ctx) override {
        int ds = 0, cr = mTheme->mWindowCornerRadius;
        int hh = mTheme->mWindowHeaderHeight;

        /* Draw window */
        nvgSave(ctx);
        nvgBeginPath(ctx);
        nvgRoundedRect(ctx, mPos.x(), mPos.y(), mSize.x(), mSize.y(), cr);

        nvgFillColor(ctx, mMouseFocus ? mTheme->mWindowFillFocused
                                    : mTheme->mWindowFillUnfocused);
        nvgFill(ctx);

        nvgSave(ctx);
        nvgResetScissor(ctx);
        nvgBeginPath(ctx);
        nvgRect(ctx, mPos.x()-ds,mPos.y()-ds, mSize.x()+2*ds, mSize.y()+2*ds);
        nvgRoundedRect(ctx, mPos.x(), mPos.y(), mSize.x(), mSize.y(), cr);
        nvgPathWinding(ctx, NVG_HOLE);
        nvgFill(ctx);
        nvgRestore(ctx);

        if (!mTitle.empty()) {
            /* Draw header */
            NVGpaint headerPaint = nvgLinearGradient(
                ctx, mPos.x(), mPos.y(), mPos.x(),
                mPos.y() + hh,
                mTheme->mWindowHeaderGradientTop,
                mTheme->mWindowHeaderGradientTop);

            nvgBeginPath(ctx);
            nvgRoundedRect(ctx, mPos.x(), mPos.y(), mSize.x(), hh, cr);

            nvgFillPaint(ctx, headerPaint);
            nvgFill(ctx);

            nvgBeginPath(ctx);
            nvgRoundedRect(ctx, mPos.x(), mPos.y(), mSize.x(), hh, cr);
            nvgStrokeColor(ctx, mTheme->mWindowHeaderSepTop);

            nvgSave(ctx);
            nvgIntersectScissor(ctx, mPos.x(), mPos.y(), mSize.x(), 0.5f);
            nvgStroke(ctx);
            nvgRestore(ctx);

            nvgBeginPath(ctx);
            nvgMoveTo(ctx, mPos.x() + 0.5f, mPos.y() + hh - 1.5f);
            nvgLineTo(ctx, mPos.x() + mSize.x() - 0.5f, mPos.y() + hh - 1.5);
            nvgStrokeColor(ctx, mTheme->mWindowHeaderSepBot);
            nvgStroke(ctx);

            nvgFontSize(ctx, 18.0f);
            nvgFontFace(ctx, "sans-bold");
            nvgTextAlign(ctx, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);

            nvgFontBlur(ctx, 0);
            nvgFillColor(ctx, mFocused ? mTheme->mWindowTitleFocused
                                    : mTheme->mWindowTitleUnfocused);
            nvgText(ctx, mPos.x() + 5, mPos.y() + hh / 2 - 1,
                    mTitle.c_str(), nullptr);
        }
        nvgRestore(ctx);
        Widget::draw(ctx);
    }

    // We also add some space to the preferred size in x-direction
    virtual Eigen::Vector2i preferredSize(NVGcontext *ctx) const override
    {
        if (mButtonPanel)
            mButtonPanel->setVisible(false);
        Eigen::Vector2i result = Widget::preferredSize(ctx);
        if (mButtonPanel)
            mButtonPanel->setVisible(true);

        nvgFontSize(ctx, 18.0f);
        nvgFontFace(ctx, "sans-bold");
        float bounds[4];
        nvgTextBounds(ctx, 0, 0, mTitle.c_str(), nullptr, bounds);

        return result.cwiseMax(Eigen::Vector2i(
            bounds[2]-bounds[0] + 80, bounds[3]-bounds[1]
        ));
    }

    // This window should not be draggable
    virtual bool mouseDragEvent(const Eigen::Vector2i& /* p */, const Eigen::Vector2i& /* rel */, int /* button */, int /* modifiers*/) override
    { return true; }

    nanogui::Button * close_btn;
    std::shared_ptr<VFRendering::RendererBase> renderer;
    nanogui::CheckBox * chk;
    std::string name;
    nanogui::Label * name_label;
    nanogui::ref<VFGLCanvas> gl_canvas;
};

#endif