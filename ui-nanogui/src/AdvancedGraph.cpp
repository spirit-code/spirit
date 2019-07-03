#include "AdvancedGraph.hpp"
#include <nanogui/theme.h>
#include <nanogui/opengl.h>
#include <nanovg.h>
#include <nanogui/serializer/core.h>
#include <sstream>

AdvancedGraph::AdvancedGraph(Widget * parent, const Marker & default_marker, const Color & default_marker_color, float default_marker_scale)
    : Widget(parent), x_values({}), y_values({}), markers({}), marker_scale({}), marker_colors({}), default_marker(default_marker),
        default_marker_color(default_marker_color), default_marker_scale(default_marker_scale)
{ };

void AdvancedGraph::setValues(const std::vector<float> & x_values, const std::vector<float> & y_values)
{
    if(x_values.size() != y_values.size())
        throw std::runtime_error("Value vectors x and y need to have the same length!");

    this->x_values      = x_values;
    this->y_values      = y_values;

    if(x_values.size() != markers.size())
    {
        this->markers.resize(x_values.size(), this->default_marker);
        this->marker_scale.resize(x_values.size(), this->default_marker_scale);
        this->marker_colors.resize(x_values.size(), this->default_marker_color);
    }
}

void AdvancedGraph::setValues(const std::vector<float> & x_values, const std::vector<float> & y_values, const std::vector<Marker> & markers, const std::vector<Color> marker_colors, const std::vector<float> & marker_scales)
{
    if(x_values.size() != y_values.size())
        throw std::runtime_error("Value vectors x and y need to have the same length!");
    if(x_values.size() != markers.size())
        throw std::runtime_error("Value vectors and marker vector need to have the same length!");
    if(x_values.size() != marker_colors.size())
        throw std::runtime_error("Value vectors and marker_colors vector need to have the same length!");
    if(x_values.size() != marker_scales.size())
        throw std::runtime_error("Value vectors and marker_scales vector need to have the same length!");

    this->x_values      = x_values;
    this->y_values      = y_values;
    this->markers       = markers;
    this->marker_colors = marker_colors;
    this->marker_scale  = marker_scales;
}

// Inserts a value at the given index (the index of the value in the resulting new arrays will be 'idx')
void AdvancedGraph::insertValue(int idx, float x_value, float y_value)
{
    this->x_values.insert( this->x_values.begin() + idx, x_value );
    this->y_values.insert( this->y_values.begin() + idx, y_value );
    this->markers.insert(  this->markers.begin() + idx, this->default_marker);
    this->marker_colors.insert( this->marker_colors.begin() + idx, this->default_marker_color);
    this->marker_scale.insert( this->marker_scale.begin() + idx, this->default_marker_scale);
}

// Inserts a value at the given index
void AdvancedGraph::insertValue(int idx, float x_value, float y_value, Marker marker, Color marker_color, float marker_scale)
{
    this->x_values.insert( x_values.begin() + idx, x_value );
    this->y_values.insert( y_values.begin() + idx, y_value );
    this->markers.insert(  markers.begin()  + idx,  marker );
    this->marker_colors.insert(marker_colors.begin() + idx, marker_color);
    this->marker_scale.insert(this->marker_scale.begin() + idx, marker_scale);
}

void AdvancedGraph::appendValue(float x_value, float y_value)
{
    this->insertValue(this->x_values.size(), x_value, y_value);
}
void AdvancedGraph::appendValue(float x_value, float y_value, Marker marker, Color marker_color, float marker_scale)
{
    this->insertValue(this->x_values.size(), x_value, y_value, marker, marker_color, marker_scale);
}

void AdvancedGraph::removeValue(int idx)
{
    this->x_values.erase( this->x_values.begin() + idx);
    this->y_values.erase( this->y_values.begin() + idx);
    this->markers.erase( this->markers.begin() + idx);
    this->marker_colors.erase( this->marker_colors.begin() + idx);
    this->marker_scale.erase( this->marker_scale.begin() + idx);
}

void AdvancedGraph::dataToPixel(const nanogui::Vector2f & data, nanogui::Vector2i & pixel)
{
    pixel[0] = mPos.x() + (data.x() - mx_min) * (mSize.x() - (margin_left + margin_right)) / (mx_max - mx_min) + margin_left;
    pixel[1] = mPos.y() + (my_max - data.y()) * (mSize.y() - (margin_top + margin_bot)) / (my_max - my_min) + margin_top;
}

void AdvancedGraph::pixelToData(const nanogui::Vector2i & pixel, nanogui::Vector2f & data)
{
    data[0] =  (pixel.x() - mPos.x() - margin_left) * (mx_max - mx_min) / (mSize.x() - (margin_left + margin_right)) + mx_min;
    data[1] = -(pixel.y() - mPos.y() - margin_top) * (my_max - my_min) / (mSize.y() - (margin_top + margin_bot)) + my_max;
}

// Helper functions for equilateral triangle with center at (x,y) and side length l
void nvgTriangDown(NVGcontext* ctx, float x, float y, float l)
{
    constexpr float h = 0.8660254038; // sqrt(3)/2
    nvgMoveTo(ctx, x, y + h*l*2.f/3.f);     // lower corner
    nvgLineTo(ctx, x - l/2.f, y - h*l/3.f); // upper left corner
    nvgLineTo(ctx, x + l/2.f, y - h*l/3.f); // upper right corner
    nvgLineTo(ctx, x, y + h*l*2.f/3.f);     // back to lower corner
}

void nvgTriangUp(NVGcontext* ctx, float x, float y, float l)
{
    constexpr float h = 0.8660254038; // sqrt(3)/2
    nvgMoveTo(ctx, x, y - h*l*2.f/3.f);     // upper corner
    nvgLineTo(ctx, x - l/2.f, y + h*l/3.f); // lower left corner
    nvgLineTo(ctx, x + l/2.f, y + h*l/3.f); // lower right corner
    nvgLineTo(ctx, x, y - h*l*2.f/3.f);     // back to upper corner
}

void AdvancedGraph::drawMarkers(NVGcontext * ctx)
{
    nanogui::Vector2i cur_pos = {0,0};

    // Note: We add some hard-coded additional scaling to the markers such that 
    // they look roughly the same size for the same value of marker scale
    for (size_t i = 0; i < (size_t) y_values.size(); i++)
    {
        // Only draw markers withing the min/max constraints
        if (x_values[i] >= mx_min && x_values[i] <= mx_max && y_values[i] >= my_min && y_values[i] <= my_max)
        {
            dataToPixel({x_values[i], y_values[i]}, cur_pos);
            int cur_marker_scale = _internal_marker_scale * this->marker_scale[i];

            if(this->markers[i] != Marker::NONE)
            {
                nvgBeginPath(ctx);
                nvgMoveTo(ctx, cur_pos.x(), cur_pos.y());
                nvgFillColor(ctx, marker_colors[i]);

                if(this->markers[i] == Marker::CIRCLE)
                {
                    cur_marker_scale *= 0.56;
                    nvgCircle(ctx, cur_pos.x(), cur_pos.y(), cur_marker_scale);
                } else if (this->markers[i] == Marker::SQUARE)
                {
                    // Need to offset the origin to the left and up, because it denotes the top/left corner of the rectangle
                    nvgRect(ctx, cur_pos.x()-cur_marker_scale/2, cur_pos.y()-cur_marker_scale/2, cur_marker_scale, cur_marker_scale);
                } else if (this->markers[i] == Marker::TRIANG_UP)
                {
                    cur_marker_scale *= 1.3;
                    nvgTriangUp(ctx, cur_pos.x(), cur_pos.y(), cur_marker_scale);
                } else if (this->markers[i] == Marker::TRIANG_DOWN)
                {
                    cur_marker_scale *= 1.3;
                    nvgTriangDown(ctx, cur_pos.x(), cur_pos.y(), cur_marker_scale);
                }

                nvgFill(ctx);
            }
        }
    }
}

void AdvancedGraph::drawTicks(NVGcontext * ctx)
{
    nanogui::Vector2i line_begin = {0,0};
    nanogui::Vector2f data_pt = {0,0};
    std::ostringstream streamObj; // Is there a better way for number to string conversion?
    nvgBeginPath(ctx);
    nvgFontSize(ctx, this->fontSize);
    nvgStrokeColor(ctx, this->mTextColor);
    nvgFillColor(ctx, this->mTextColor);

    // x ticks
    nvgTextAlign(ctx, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
    for( int i = 0; i < n_ticks_x; i++ )
    {
        data_pt = {mx_min + (mx_max - mx_min) * i/n_ticks_x, my_min};
        dataToPixel(data_pt, line_begin);
        nvgMoveTo(ctx, line_begin.x(), line_begin.y());
        nvgLineTo(ctx, line_begin.x(), line_begin.y() - tick_length);

        std::ostringstream().swap(streamObj);
        streamObj << data_pt.x();
        nvgText(ctx, line_begin.x(), line_begin.y(), streamObj.str().c_str(), NULL);
    }

    // y ticks
    nvgTextAlign(ctx, NVG_ALIGN_CENTER | NVG_ALIGN_CENTER);
    for( int i = 0; i < n_ticks_y; i++ )
    {
        data_pt = {mx_min, my_min + (my_max - my_min) * i/n_ticks_y};
        dataToPixel(data_pt, line_begin);
        nvgMoveTo(ctx, line_begin.x(), line_begin.y());
        nvgLineTo(ctx, line_begin.x() + tick_length, line_begin.y());
        std::ostringstream().swap(streamObj);
        streamObj << data_pt.y();

        // TODO: kind of an ugly hack so that the numbers dont overlap the y-axis
        nvgText(ctx, (mPos.x() + line_begin.x())/2, line_begin.y(), streamObj.str().c_str(), NULL);
    }
    nvgStrokeWidth(ctx, this->tick_thickness);
    nvgStroke(ctx);
}

void AdvancedGraph::drawLabels(NVGcontext * ctx)
{
    nvgBeginPath(ctx);
    nvgStrokeColor(ctx, this->mTextColor);
    nvgFillColor(ctx, this->mTextColor);
    nvgFontSize(ctx, this->fontSize);

    // x_label
    nvgTextAlign(ctx, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
    nvgText(ctx, mPos.x() + mSize.x()/2, mPos.y() + mSize.y() - margin_bot/2, x_label.c_str(), NULL);

    // y_label
    nvgTextAlign(ctx, NVG_ALIGN_LEFT | NVG_ALIGN_CENTER);
    nvgText(ctx, mPos.x() + margin_left/8, mPos.y() + margin_top/2, y_label.c_str(), NULL);

    nvgStroke(ctx);
}

void AdvancedGraph::drawBox(NVGcontext * ctx)
{
    nvgBeginPath(ctx);
    nvgRect(ctx, mPos.x() + margin_left, mPos.y() + margin_top, mSize.x() - (margin_left + margin_right), mSize.y() - (margin_top + margin_bot));
    nvgStrokeColor(ctx, this->boxColor);
    nvgStrokeWidth(ctx, this->box_thickness);
    nvgStroke(ctx);
}

void AdvancedGraph::drawGrid(NVGcontext * ctx)
{
    nanogui::Vector2i line_begin = {0,0};
    nanogui::Vector2i line_end   = {0,0};
    nvgBeginPath(ctx);

    // Vertical lines
    for( int i = 0; i < n_ticks_x; i++ )
    {   
        dataToPixel({mx_min + (mx_max - mx_min) * i/n_ticks_x, my_min}, line_begin);
        dataToPixel({mx_min + (mx_max - mx_min) * i/n_ticks_x, my_max}, line_end);
        nvgMoveTo(ctx, line_begin.x(), line_begin.y());
        nvgLineTo(ctx, line_end.x(), line_end.y());
    }

    // Horizontal lines
    for( int i = 0; i < n_ticks_y; i++ )
    {
        dataToPixel({mx_min, my_min + (my_max - my_min) * i/n_ticks_y}, line_begin);
        dataToPixel({mx_max, my_min + (my_max - my_min) * i/n_ticks_y}, line_end);
        nvgMoveTo(ctx, line_begin.x(), line_begin.y());
        nvgLineTo(ctx, line_end.x(), line_end.y());
    }

    nvgStrokeColor(ctx, this->gridColor);
    nvgStrokeWidth(ctx, this->grid_thickness);
    nvgStroke(ctx);
}

// Note: We kind of require the x_values to be ordered ...
void AdvancedGraph::drawLineSegments(NVGcontext * ctx)
{

    nanogui::Vector2i cur_pos = {0,0};
    nanogui::Vector2i prev_pos = {0,0};
    nanogui::Vector2f inter1 = {0,0};
    nanogui::Vector2f inter2 = {0,0};

    nvgBeginPath(ctx);
    dataToPixel({x_values[0], y_values[0]}, cur_pos);
    nvgMoveTo(ctx, cur_pos.x(), cur_pos.y()); // Begin path at first data point

    for( size_t i = 1; i < (size_t) y_values.size(); i++ ) 
    {
        // If the previous and the current value lie within the boundary box we can simply draw the segment
        if( checkBoundary(x_values[i-1], y_values[i-1]) & checkBoundary(x_values[i], y_values[i]) )
        {
            dataToPixel({x_values[i], y_values[i]}, cur_pos);    
            nvgLineTo(ctx, cur_pos.x(), cur_pos.y());
        }
        else // Otherwise we compute the intersections
        {
            if( computeLineIntersection({x_values[i-1], y_values[i-1]}, {x_values[i], y_values[i]}, inter1, inter2) )
            {
                dataToPixel( inter1, cur_pos );
                nvgMoveTo(ctx, cur_pos[0], cur_pos[1]); // Move to left intersection
                dataToPixel( inter2, cur_pos ); // cur_pos now holds pixel coords of data point
                nvgLineTo(ctx, cur_pos[0], cur_pos[1]); // Move to left intersection
            }
        }
    }

    nvgStrokeColor(ctx, lineColor);
    nvgStrokeWidth(ctx, line_thickness);
    nvgStroke(ctx);
    // nvgFillColor(ctx, mForegroundColor);
}

void AdvancedGraph::draw(NVGcontext *ctx)
{
    Widget::draw(ctx);
    nvgBeginPath(ctx);

    // Fille the background
    nvgRect(ctx, mPos.x(), mPos.y(), mSize.x(), mSize.y());
    nvgFillColor(ctx, mBackgroundColor);
    nvgFill(ctx);

    if( y_values.size() < 2 )
        return;

    if( show_grid )
        drawGrid(ctx);

    if( show_box )
        drawBox(ctx);

    if( show_line )
        drawLineSegments(ctx);

    drawMarkers(ctx);

    drawTicks(ctx);
    drawLabels(ctx);

    nvgFontFace(ctx, "sans");

    nvgBeginPath(ctx);
    nvgRect(ctx, mPos.x(), mPos.y(), mSize.x(), mSize.y());
    nvgStrokeColor(ctx, Color(100, 255));
    nvgStroke(ctx);
}