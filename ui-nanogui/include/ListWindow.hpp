#pragma once
#ifndef LISTWINDOW_HPP
#define LISTWINDOW_HPP
#include <nanogui/layout.h>
#include <nanogui/opengl.h>
#include <ListWindow.hpp>
#include <iostream>

class ListWindow : public nanogui::Window
{
    public:
    ListWindow(Widget * parent, const std::vector<std::string> & items) : nanogui::Window(parent, "") 
    {
        this->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Vertical));
        for(const auto item : items)
        {
            new nanogui::Label(this, item);
        }
    }

    virtual bool mouseButtonEvent(const Eigen::Vector2i &p, int button, bool down, int modifiers) override
    {
        if(this->visible())
        {
            if(button == GLFW_MOUSE_BUTTON_1 && down)
            {
                for (auto it = this->mChildren.rbegin(); it != this->mChildren.rend(); ++it) 
                {
                    nanogui::Label * w = dynamic_cast<nanogui::Label *>(*it);
                    if( p[1] > w->absolutePosition()[1] && p[1] < w->absolutePosition()[1] + w->size()[1] )
                    {
                        this->mSelectedIndex=childIndex(w);
                        mItemSelectedCallback(childIndex(w));
                        return true; // return true to stop the event from propagating to the ListSelect
                    }
                }
            }
        }
    }

    bool mouseMotionEvent(const Eigen::Vector2i &p, const Eigen::Vector2i &rel, int button, int modifiers) override
    {
        for (auto it = this->mChildren.rbegin(); it != this->mChildren.rend(); ++it) 
        {
            nanogui::Label * w = dynamic_cast<nanogui::Label *>(*it);
            if( p[1] > w->absolutePosition()[1] && p[1] < w->absolutePosition()[1] + w->size()[1] )
            {
                w->setColor(this->theme()->mTextColor);
            } else {
                w->setColor(this->theme()->mDisabledTextColor);
            }
        }
        return true;
    }

    int selectedIndex() 
    {
        return mSelectedIndex;
    }

    void setSelectedIndex(int idx)
    {
        mSelectedIndex = idx;
    }

    void setItemSelectedCallback(const std::function<void(int)> &callback)
    {
        mItemSelectedCallback = callback;
    }
    virtual void itemSelectedCallback(int item)
    {
        mItemSelectedCallback(item);
    };

    protected:
    std::function<void(int)> mItemSelectedCallback;
    int mSelectedIndex = -1;
    int size_x;

};

#endif