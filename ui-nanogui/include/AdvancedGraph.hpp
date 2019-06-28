#include <nanogui/common.h>
#include <nanogui/graph.h>
#include <iostream>
#include <stdexcept>

enum Marker
{
    NONE        = 0, 
    CIRCLE      = 1,
    SQUARE      = 2,
    TRIANG_UP   = 3,
    TRIANG_DOWN = 4
};

// TODO: multiple graphs per widget
class AdvancedGraph : public nanogui::Widget
{
    using Color = nanogui::Color;

    public: // TODO: maybe write getters/setters for these ...

    virtual void drawMarkers(NVGcontext * ctx);
    virtual void drawTicks(NVGcontext * ctx);
    virtual void drawLabels(NVGcontext * ctx);
    virtual void drawLineSegments(NVGcontext * ctx);
    virtual void drawGrid(NVGcontext * ctx);
    virtual void drawBox(NVGcontext * ctx);

    // Conversion between pixels in widget and data points, respecting margins
    // pixel coordinates originate at top-left corner
    // data coordinates originate at bottom-left corner
    virtual void dataToPixel(const nanogui::Vector2f & data, nanogui::Vector2i & pixel);
    virtual void pixelToData(const nanogui::Vector2i & pixel, nanogui::Vector2f & data);

    public:
    // Construct the Widget, the default values are used for initialization, and when adding data without specifying the marker etc
    AdvancedGraph(Widget * parent, const Marker & default_marker = Marker::CIRCLE, const Color & default_marker_color = Color(0,0,0,255), float default_marker_scale = 1.0 );

    // Sets x and y values, if there are more values than previously default styles are appended
    virtual void setValues(const std::vector<float> & x_values, const std::vector<float> & y_values);
    // Sets x and y values and sets styles as specified
    virtual void setValues(const std::vector<float> & x_values, const std::vector<float> & y_values, const std::vector<Marker> & markers, const std::vector<Color> marker_colors, const std::vector<float> & marker_scales);

    // Inserts a value at the given index, using default style
    // (the index of the value in the resulting new arrays will be 'idx')
    virtual void insertValue(int idx, float x_value, float y_value);
    // Inserts a value at the given index, using specified style
    virtual void insertValue(int idx, float x_value, float y_value, Marker marker, Color marker_color, float marker_scale);
    // Appending at the end
    virtual void appendValue(float x_value, float y_value);
    virtual void appendValue(float x_value, float y_value, Marker marker, Color marker_color, float marker_scale);
    virtual void removeValue(int idx);
    virtual int getNValues() { return this->x_values.size(); };

    const std::vector<float> &XValues() const { return x_values; }
    const std::vector<float> &YValues() const { return y_values; }

    // Set/Get minimum x value
    void setXMin(float x_min) {mx_min = x_min;}
    float getXMin() { return mx_min; }
    // Set/Get maximal x value
    void setXMax(float x_max) { mx_max = x_max; }
    float getXMax() { return mx_max; }

    // Get/Set minimum y value
    void setYMin(float y_min) { my_min = y_min; }
    float getYMin() { return my_min; }
    // Get/Set maximal y value
    void setYMax(float y_max) { my_max = y_max; }
    float getYMax() { return my_max; }

    // Set the color of the marker at idx
    void setMarkerColor(int idx, const Color & color) { this->marker_colors[idx] = color; }
    // Set the colors of the markers to colors
    void setMarkerColor(const std::vector<Color> & colors) 
    { 
        if(this->y_values.size() == colors.size()) 
        {
            this->marker_colors = colors;
        } else {
            throw std::runtime_error("Color array has wrong size!");
        }
    }
    // Set the color of all markers
    void setMarkerColor(const Color & color) 
    {  
        for(auto & c : marker_colors)
        {
            c = color;
        }
    }
    const Color & getMarkerColor(int idx) { return this->marker_colors[idx]; }

    // Set the marker at idx
    void setMarker(int idx, Marker marker) { this->markers[idx] = marker; }
    // Set all markers to marker
    void setMarker(Marker marker)
    {
        for(auto & m : markers)
        {
            m = marker;
        }
    }
    // Set markers according to the vector markers
    void setMarker(const std::vector<Marker> & markers)
    {
        if( this->x_values.size() == markers.size() )
            this->markers = markers;
        else
            throw std::runtime_error("Marker array has wrong size!");
    }
    Marker getMarker(int idx) { return this->markers[idx]; }

    // Set the marker scale at idx
    void setMarkerScale(int idx, float scale) {this->marker_scale[idx] = scale;};
    // Set the marker scale for all markers
    void setMarkerScale(float scale) 
    {
        for(auto & s : marker_scale)
        {
            s = scale;
        }  
    }
    // Set the marker scales according to vector markers
    void setMarkerScale(const std::vector<float> & scales)
    {
         if( this->x_values.size() == scales.size() )
            this->marker_scale = scales;
        else
            throw std::runtime_error("Scales array has wrong size!");
    }
    float getMarkerScale(int idx) { return this->marker_scale[idx]; }

    // Axis Labels
    void setXLabel(const std::string & label){ this->x_label = label; }
    std::string getXLabel(){ return this->x_label; }
    void setYLabel(const std::string & label){ this->y_label = label; }
    std::string getYLabel(){ return this->y_label; }

    // Ticks
    void setNTicksX(int n_ticks_x) { this->n_ticks_x = n_ticks_x; }
    int getNTicksX() { return n_ticks_x; }
    void setNTicksY(int n_ticks_y) { this->n_ticks_y = n_ticks_y; }
    int getNTicksY() { return n_ticks_y; }
    void setTickColor(const Color & color) { this->tickColor = color; }
    Color & getTickColor() { return this->tickColor;}

    // Grid properties
    void setGrid(bool show_grid){ this->show_grid = show_grid; }
    bool getGrid(){ return this->show_grid; }
    void setGridThickness(int grid_thickness){ this->grid_thickness = grid_thickness; }
    int getGridThickness() { return this->grid_thickness; }
    void setGridColor(const Color & color) { this->gridColor = color; }
    Color & getGridColor() { return this->gridColor;}

    // Margins
    void setMarginLeft(int margin_left){ this->margin_left = margin_left; }
    int getMarginLeft(){ return this->margin_left; }
    void setMarginRight(int margin_right){ this->margin_right = margin_right; }
    int getMarginRight(){ return this->margin_right; }
    void setMarginTop(int margin_top){ this->margin_top = margin_top; }
    int getMarginTop(){ return this->margin_top; }
    void setMarginBot(int margin_bot){ this->margin_bot = margin_bot; }
    int getMarginBot(){ return this->margin_bot; }
  
    // Font for ticks and labels
    void setFontSize(int font_size){ this->fontSize = font_size; }
    int getFontSize(){ return this->fontSize; }
    void setTextColor(const Color & color) { this->mTextColor = color;}
    Color & getTextColor() { return this->mTextColor; }

    // Line segment properties
    void setLine(bool show_line){ this->show_line = show_line; }
    bool getLine() { return this->show_line; }
    void setLineThickness(int thickness) { this->line_thickness = thickness; }
    int  getLineThickness() { return this->line_thickness; }
    void setLineColor(const Color & color) { this->lineColor = color; }
    Color & getLineColor() { return this->lineColor; } 

    // Boundary Box properties
    void setBox(bool show_box){ this->show_box = show_box; }
    bool getBox(bool show_box){ return this->show_box; }
    void setBoxThickness(int thickness) { this->box_thickness = thickness; }
    int  getBoxThickness() { return this->box_thickness; }
    void setBoxColor(const Color & color) { this->boxColor = color; }
    Color & getBoxColor() { return this->boxColor; } 

    // Background colors
    void setBackgroundColor(const Color & color) { this->mBackgroundColor = color;}
    Color & getBackgroundColor() { return this->mBackgroundColor; }

    // Conveniency method that sets the transparency of all colors
    virtual void setTransparency(float alpha ) // alpha = 0.0 - 1.0
    {
        getBoxColor().w()  = alpha;
        getTextColor().w() = alpha;
        getTickColor().w() = alpha;
        getBackgroundColor().w() = alpha;
        getLineColor().w() = alpha;
        getBoxColor().w() = alpha;
        getGridColor().w() = alpha;
        for( auto & c : marker_colors )
        {
            c.w() = alpha;
        }
    }

    virtual void draw(NVGcontext *ctx) override;

    protected:
    Color mBackgroundColor = Color(255,255); // White
    Color mTextColor       = Color(0,255); // Black

    std::vector<float> y_values;
    std::vector<float> x_values;

    // Marker Data
    std::vector<Marker> markers;
    Marker default_marker;

    std::vector<Color>  marker_colors;
    Color default_marker_color;

    std::vector<float> marker_scale;
    float default_marker_scale;

    // Minimum and maximum values
    float mx_min, mx_max, my_min, my_max;

    float _internal_marker_scale = 5;
    int fontSize = 16;

    // X and y axis labels;
    std::string x_label = "", y_label = "";

    // Distance of widget border to beginning of plot (leaves space for labels, ticks etc)
    int margin_left = 55, margin_right = 20, margin_top = 30, margin_bot = 40;

    // Number of Ticks on the x-axis
    int n_ticks_x = 10, n_ticks_y = 10, tick_length = 5, tick_thickness = 2;
    Color tickColor = Color(0,255);

    // If a grid should be drawn
    bool show_grid = false;
    // Color in which the grid is drawn
    Color gridColor = Color(200, 255); // Gray
    // Thickness of grid
    int grid_thickness = 1;

    int line_thickness = 1;
    bool show_line     = true;
    Color lineColor    = Color(0,255); // Black

    int box_thickness = 1;
    bool show_box     = true;
    Color boxColor    = Color(0,255); // Black

    // Check if a point lies within the boundaries
    bool checkBoundary(const float x, const float y){ return (x >= mx_min && x <= mx_max && y >= my_min && y <= my_max); }

    // Helper function to compute where lines, going to markers outside the min/max constraints, intersect the boundary box
    bool computeLineIntersection(const Eigen::Vector2f & p1, const Eigen::Vector2f & p2, Eigen::Vector2f & inter1, Eigen::Vector2f & inter2 )
    {
        // We assume that p1[0] < p2[0]
        float m = (p2[1] - p1[1]) / (p2[0] - p1[0]);

        float y1 = p1[1] + m * (mx_min - p1[0]);
        if( y1 < my_min || y1 > my_max )
        {
            float y_temp = ( m>0 ) ? my_min : my_max;
            inter1[0] = (y_temp - p1[1])/m + p1[0];
            inter1[1] = y_temp;
        } else {
            inter1[0] = mx_min;
            inter1[1] = y1;
        }

        float y2 = p1[1] + m * (mx_max - p1[0]);
        if( y2 < my_min || y2 > my_max )
        {
            float y_temp = ( m<0 ) ? my_min : my_max;
            inter2[0] = (y_temp - p1[1])/m + p1[0];
            inter2[1] = y_temp;
        } else {
            inter2[0] = mx_max;
            inter2[1] = y2;
        }

        if ( checkBoundary(p1[0], p1[1]) )
        {
            inter1 = p1;
        } 
        if( checkBoundary(p2[0], p2[1]) )
        {
            inter2 = p2;
        } 

        return checkBoundary( inter1[0], inter1[1] ) || checkBoundary( inter2[0], inter2[1] );
    }
};