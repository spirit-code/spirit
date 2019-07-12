#pragma once
#ifndef MAINWINDOWLAYOUT_HPP
#define MAINWINDOWLAYOUT_HPP

#include <nanogui/layout.h>

// A version of AdvancedGridLayout that allows unregistered widgets
class MainWindowLayout : public nanogui::AdvancedGridLayout
{
    public:
    MainWindowLayout(const std::vector<int> &cols = {}, const std::vector<int> &rows = {}, int margin = 0) :
    nanogui::AdvancedGridLayout(cols, rows, margin) {};


    // This is a modified version of AdvancedGridLayout::performLayout that simply invokes performLayout for
    // non-registered children and then continues
    virtual void performLayout(NVGcontext *ctx, nanogui::Widget *widget) const override 
    {
        using namespace nanogui;
        std::vector<int> grid[2];
        computeLayout(ctx, widget, grid);

        grid[0].insert(grid[0].begin(), mMargin);
        const Window *window = dynamic_cast<const Window *>(widget);
        if (window && !window->title().empty())
            grid[1].insert(grid[1].begin(), widget->theme()->mWindowHeaderHeight + mMargin/2);
        else
            grid[1].insert(grid[1].begin(), mMargin);

        for (int axis=0; axis<2; ++axis) {
            for (size_t i=1; i<grid[axis].size(); ++i)
                grid[axis][i] += grid[axis][i-1];

            for (Widget *w : widget->children()) {
                // if (!w->visible())
                    // continue;

                Anchor anchor;
                auto it = mAnchor.find(w);
                if (it == mAnchor.end())
                {
                    w->performLayout(ctx);
                    continue;
                }
                    // throw std::runtime_error("Widget was not registered with the grid layout!");
                anchor=it->second;

                int itemPos = grid[axis][anchor.pos[axis]];
                int cellSize  = grid[axis][anchor.pos[axis] + anchor.size[axis]] - itemPos;
                int ps = w->preferredSize(ctx)[axis], fs = w->fixedSize()[axis];
                int targetSize = fs ? fs : ps;

                switch (anchor.align[axis]) {
                    case Alignment::Minimum:
                        break;
                    case Alignment::Middle:
                        itemPos += (cellSize - targetSize) / 2;
                        break;
                    case Alignment::Maximum:
                        itemPos += cellSize - targetSize;
                        break;
                    case Alignment::Fill:
                        targetSize = fs ? fs : cellSize;
                        break;
                }

                Vector2i pos = w->position(), size = w->size();
                pos[axis] = itemPos;
                size[axis] = targetSize;
                w->setPosition(pos);
                w->setSize(size);
                w->performLayout(ctx);
            }
        }
    }
};

#endif