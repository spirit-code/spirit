#pragma once
#ifndef SPIRIT_IMGUI_RENDERER_WIDGET_HPP
#define SPIRIT_IMGUI_RENDERER_WIDGET_HPP

#include <VFRendering/ArrowRenderer.hxx>
#include <VFRendering/BoundingBoxRenderer.hxx>
#include <VFRendering/CombinedRenderer.hxx>
#include <VFRendering/CoordinateSystemRenderer.hxx>
#include <VFRendering/DotRenderer.hxx>
#include <VFRendering/GlyphRenderer.hxx>
#include <VFRendering/IsosurfaceRenderer.hxx>
#include <VFRendering/ParallelepipedRenderer.hxx>
#include <VFRendering/SphereRenderer.hxx>
#include <VFRendering/SurfaceRenderer.hxx>
#include <VFRendering/View.hxx>

#include <memory>

struct State;

namespace ui
{

enum class Colormap
{
    HSV,
    HSV_NO_Z,
    BLUE_WHITE_RED,
    BLUE_GREEN_RED,
    BLUE_RED,
    MONOCHROME
};

class ColormapWidget
{
protected:
    ColormapWidget();
    void showcolormap_input();

    bool colormap_changed = false;
    std::string colormap_implementation_str;

    Colormap colormap       = Colormap::HSV;
    float colormap_rotation = 0;
    bool colormap_invert_z  = false;
    bool colormap_invert_xy = false;
    glm::vec3 colormap_cardinal_a{ 1, 0, 0 };
    glm::vec3 colormap_cardinal_b{ 0, 1, 0 };
    glm::vec3 colormap_cardinal_c{ 0, 0, 1 };
    glm::vec3 colormap_monochrome_color{ 0.5f, 0.5f, 0.5f };
};

struct RendererWidget
{
    std::shared_ptr<State> state;
    bool show_   = true;
    bool remove_ = false;
    std::shared_ptr<VFRendering::RendererBase> renderer;

    virtual void show() = 0;

protected:
    RendererWidget( std::shared_ptr<State> state ) : state( state ) {}
};

struct BoundingBoxRendererWidget : RendererWidget
{
    BoundingBoxRendererWidget(
        std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield );

    void show() override;

    float line_width    = 0;
    int level_of_detail = 10;
    bool draw_shadows   = false;
};

struct CoordinateSystemRendererWidget : RendererWidget
{
    CoordinateSystemRendererWidget( std::shared_ptr<State> state );

    void show() override;

    std::shared_ptr<VFRendering::CoordinateSystemRenderer> renderer;
};

struct DotRendererWidget : RendererWidget, ColormapWidget
{
    DotRendererWidget(
        std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield );

    void show() override;

    float dotsize = 1;
};

struct ArrowRendererWidget : RendererWidget, ColormapWidget
{
    ArrowRendererWidget(
        std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield );

    void show() override;

    float arrow_size = 1;
    int arrow_lod    = 10;
};

struct ParallelepipedRendererWidget : RendererWidget, ColormapWidget
{
    ParallelepipedRendererWidget(
        std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield );

    void show() override;
};

struct SphereRendererWidget : RendererWidget, ColormapWidget
{
    SphereRendererWidget(
        std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield );

    void show() override;
};

struct SurfaceRendererWidget : RendererWidget, ColormapWidget
{
    SurfaceRendererWidget(
        std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield );

    void show() override;
};

struct IsosurfaceRendererWidget : RendererWidget, ColormapWidget
{
    IsosurfaceRendererWidget(
        std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield );

    void show() override;

    float isovalue    = 0;
    int isocomponent  = 2;
    bool draw_shadows = true;
    bool flip_normals = false;
};

} // namespace ui

#endif