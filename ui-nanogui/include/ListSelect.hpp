#pragma once
#ifndef LISTSELECT_HPP
#define LISTSELECT_HPP
#include <nanovg.h>
#include <nanogui/common.h>
#include <nanogui/layout.h>
#include <nanogui/widget.h>
#include <nanogui/window.h>
#include <nanogui/screen.h>
#include <nanogui/label.h>
#include <ListWindow.hpp>

class ListSelect : public nanogui::Widget
{
    using Color = nanogui::Color;

    public:
    ListSelect(Widget *parent, const std::vector<std::string> & items, int init_idx = 0) : nanogui::Widget(parent),
    items(items), mSelectedIndex(init_idx)
    {
        this->listWindow = new ListWindow(this->screen(), items);
        this->listWindow->setVisible(false);
        this->listWindow->setSelectedIndex(init_idx);
        this->screen()->moveWindowToFront( this->listWindow );

        this->listWindow->setItemSelectedCallback( [&](int i)
            {
                this->mSelectedIndex = i;
                listWindow->setVisible(false);
                this->setVisible(true);

                if(mCallback)
                    mCallback(i);
            } 
        );

        this->size_x = preferredSize(this->screen()->nvgContext())[0];
        this->listWindow->setSize({this->size_x, items.size() * fontSize()});
        this->listWindow->performLayout(this->screen()->nvgContext());
    }

    virtual void draw(NVGcontext *ctx) override
    {
        Widget::draw(ctx);

        auto gradTop = mTheme->mButtonGradientTopFocused;
        auto gradBot = mTheme->mButtonGradientBotFocused;
        nvgBeginPath(ctx);

        nvgRoundedRect(ctx, mPos.x() + 1, mPos.y() + 1.0f, mSize.x() - 2,
                   mSize.y() - 2, mTheme->mButtonCornerRadius - 1);

        NVGpaint bg = nvgLinearGradient(ctx, mPos.x(), mPos.y(), mPos.x(),
                                    mPos.y() + mSize.y(), gradTop, gradBot);

        nvgStrokeColor(ctx, mTheme->mTextColor);
        nvgStroke(ctx);
        nvgFillPaint(ctx, bg);
        nvgFill(ctx);

        nvgBeginPath(ctx);
        nvgRoundedRect(ctx, mPos.x() + 1, mPos.y() + 1.0f, mSize.x() - 2, mSize.y() - 2, mTheme->mButtonCornerRadius - 1);
        nvgFontFace(ctx, mFont.c_str());
        nvgFontSize(ctx, fontSize());
        nvgFillColor(ctx, mTheme->mTextColor);

        if (mFixedSize.x() > 0) {
            nvgTextAlign(ctx, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
            nvgTextBox(ctx, mPos.x(), mPos.y(), mFixedSize.x(), this->items[this->mSelectedIndex].c_str(), nullptr);
        } else {
            nvgTextAlign(ctx, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
            nvgTextBox(ctx, mPos.x(), mPos.y(), size_x, this->items[this->mSelectedIndex].c_str(), nullptr);
        }
    }

    // Preferred size is the size of the longest item in the list
    Eigen::Vector2i preferredSize(NVGcontext *ctx) const override
    {
        auto fs = fixedSize();
        int size_x = 0;
        for(const auto & item : items)
        {
            nvgTextAlign(ctx, NVG_ALIGN_MIDDLE | NVG_ALIGN_MIDDLE);
            size_x = std::max( nvgTextBounds(ctx, 0, 0, item.c_str(), nullptr, nullptr) + 8, float(size_x) );
        }
        return Eigen::Vector2i( fs[0]>0 ? fs[0] : size_x, fs[1]>0 ? fs[1] : fontSize() );
    }

    virtual bool mouseButtonEvent(const Eigen::Vector2i &p, int button, bool down, int modifiers) override
    {
        if( this->visible() && !this->listWindow->visible())
        {
            if(button == GLFW_MOUSE_BUTTON_1 && down)
            {
                this->setVisible(false);
                showListWindow();
                return true;
            }
        }
    }

    void showListWindow()
    {
        auto ctx = this->screen()->nvgContext();
        auto as = this->absolutePosition();
        auto ps = this->listWindow->preferredSize(ctx);

        // Compute the position at which to show the list window
        Eigen::Vector2i pos = {as[0], as[1] - mSelectedIndex * fontSize()};

        pos[1] = std::max(pos[1], 0);
        pos[1] = std::min(pos[1], this->screen()->size()[1] - ps[1]);

        this->listWindow->setPosition(pos);
        this->listWindow->setVisible(true);
        this->listWindow->setModal(true);
        this->parent()->setEnabled(false);
        this->screen()->moveWindowToFront( this->listWindow );
    }

    void performLayout(NVGcontext * ctx) override
    {
        Widget::performLayout(ctx);
    }

    void setCallback(const std::function<void(int)> &callback) { mCallback = callback; }

    /// Set the currently active font (2 are available by default: 'sans' and 'sans-bold')
    void setFont(const std::string &font) { mFont = font; }
    /// Get the currently active font
    const std::string &font() const { return mFont; }

    int selectedIndex() {return mSelectedIndex;}
    void setSelectedIndex(int idx){mSelectedIndex = idx;}

    protected:
    const std::vector<std::string> items;
    int mSelectedIndex;

    ListWindow * listWindow;
    bool enter_list_window;
    std::string mFont = "sans";
    int size_x = -1;
    std::function<void(int)> mCallback;

};

#endif