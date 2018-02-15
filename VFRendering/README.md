# Vector Field Rendering

**libvfrendering** is a C++ library for rendering vectorfields using OpenGL. Originally developed for [spirit](https://github.com/spirit-code/spirit) and based on [WegGLSpins.js](https://github.com/FlorianRhiem/WebGLSpins.js), it has an extendable architecture and currently offers renderer implementations for:
- glyph-based vector field representations as arrows
- colormapped surface and isosurface rendering
- mapping vector directions onto a sphere

The library is still very much a work-in-progress, so its API is not yet stable and there are still several features missing that will be added in later releases. If you miss a feature or have another idea on how to improve libvfrendering, please open an issue or pull request!

![Demo](https://github.com/FlorianRhiem/VFRendering/raw/master/docs/images/demo.png "Demo")

## Getting Started

To use **libvfrendering**, you need to perform the following steps:

1. Include \<VFRendering/View.hxx\>
2. Create a *VFRendering::Geometry*
3. Read or calculate the vector directions
4. Pass geometry and directions to a *VFRendering::View*
5. Draw the view in an existing OpenGL context

### 1. Include \<VFRendering/View.hxx\>

When using **libvfrendering**, you will mostly interact with View objects, so it should be enough to `#include <VFRendering/View.hxx>`.

### 2. Create a VFRendering::Geometry

The **geometry describes the positions** on which you evaluated the vector field and how they might form a grid (optional, e.g. for isosurface and surface rendering). You can pass the positions directly to the constructor or call one of the class' static methods.

As an example, this is how you could create a simple, cartesian 30x30x30 geometry, with coordinates between -1 and 1:

```c++
auto geometry = VFRendering::Geometry::cartesianGeometry(
    {30, 30, 30},
    {-1.0, -1.0, -1.0},
    {1.0, 1.0, 1.0}
);
```

### 3. Read or calculate the vector directions

This step highly depends on your use case. The **directions are stored as a `std::vector<glm::vec3>`**, so they can be created in a simple loop:

```c++
std::vector<glm::vec3> directions;
for (int iz = 0; iz < 10; iz++) {
    for (int iy = 0; iy < 10; iy++) {
        for (int ix = 0; ix < 10; ix++) {
            // calculate direction for ix, iy, iz
            directions.push_back(glm::normalize({ix-4.5, iy-4.5, iz-4.5}));
        }
    }
}
```

As shown here, the directions should be in **C order** when using the `VFRendering::Geometry` static methods. If you do not know [glm](http://glm.g-truc.net/), think of a `glm::vec3` as a struct containing three floats x, y and z.

### 4. Create a VFRendering::VectorField

This class simply contains geometry and directions.

``` c++
VFRendering::VectorField vf(geometry, directions);
```

To update the VectorField data, use `VectorField::update`.
If the directions changed but the geometry is the same, you can use the `VectorField::updateVectors` method or `VectorField::updateGeometry` vice versa.

### 5. Create a VFRendering::View and a Renderer

The view object is what you will interact most with. It provides an interface to the various renderers and includes functions for handling mouse input.

You can **create a new view** and then **initialize the renderer(s)** (as an example, we use the `VFRendering::ArrowRenderer`):

``` c++
VFRendering::View view;
auto arrow_renderer_ptr = std::make_shared<VFRendering::ArrowRenderer>(view, vf);
view.renderers( {{ arrow_renderer_ptr, {0, 0, 1, 1} }} );
```

### 5. Draw the view in an existing OpenGL context

To actually see something, you need to create an OpenGL context using a toolkit of your choice, e.g. Qt or GLFW. After creating the context, pass the framebuffer size to the **setFramebufferSize method**. You can then call the **draw method** of the view to render the vector field, either in a loop or only when you update the data.

```c++
view.draw();
```

For a complete example, including an interactive camera, see [demo.cxx](demo.cxx).


## Python Package

The Python package has bindings which correspond directly to the C++ class and function names.
To use **pyVFRendering**, you need to perform the following steps:

1. `import pyVFRendering as vfr`
2. Create a `vfr.Geometry`
3. Read or calculate the vector directions
4. Pass geometry and directions to a `vfr.View`
5. Draw the view in an existing OpenGL context


### 1. import

In order to `import pyVFRendering as vfr`, you can either `pip install pyVFRendering` or download and build it yourself.

You can build with `python3 setup.py build`, which will generate a library somewhere in your `build` subfolder, which you can `import` in python. Note that you may need to add the folder to your `PYTHONPATH`.

### 2. Create a pyVFRendering.Geometry

As above:

```python
geometry = vfr.Geometry.cartesianGeometry(
    (30, 30, 30),       # number of lattice points
    (-1.0, -1.0, -1.0), # lower bound
    (1.0, 1.0, 1.0) )   # upper bound
```

### 3. Read or calculate the vector directions

This step highly depends on your use case. Example:

```python
directions = []
for iz in range(n_cells[2]):
    for iy in range(n_cells[1]):
        for ix in range(n_cells[0]):
            # calculate direction for ix, iy, iz
            directions.append( [ix-4.5, iy-4.5, iz-4.5] )
```

### 4. Pass geometry and directions to a pyVFRendering.View

You can **create a new view** and then **pass the geometry and directions by calling the update method**:

```python
view = vfr.View()
view.update(geometry, directions)
```

If the directions changed but the geometry is the same, you can use the **updateVectors method**.

### 5. Draw the view in an existing OpenGL context

To actually see something, you need to create an OpenGL context using a framework of your choice, e.g. Qt or GLFW. After creating the context, pass the framebuffer size to the **setFramebufferSize method**. You can then call the **draw method** of the view to render the vector field, either in a loop or only when you update the data.

```python
view.setFramebufferSize(width*self.window().devicePixelRatio(), height*self.window().devicePixelRatio())
view.draw()
```

For a complete example, including an interactive camera, see [demo.py](demo.py).


## Renderers

**libvfrendering** offers several types of renderers, which all inherit from `VFRendering::RendererBase`.
The most relevant are the `VectorFieldRenderer`s:

- VFRendering::ArrowRenderer, which renders the vectors as colored arrows
- VFRendering::SphereRenderer, which renders the vectors as colored spheres
- VFRendering::SurfaceRenderer, which renders the surface of the geometry using a colormap
- VFRendering::IsosurfaceRenderer, which renders an isosurface of the vectorfield using a colormap
- VFRendering::VectorSphereRenderer, which renders the vectors as dots on a sphere, with the position of each dot representing the direction of the vector

In addition to these, there also the following renderers which do not require a `VectorField`:
- VFRendering::CombinedRenderer, which can be used to create a combination of several renderers, like an isosurface rendering with arrows
- VFRendering::BoundingBoxRenderer, which is used for rendering bounding boxes around the geometry rendered by an VFRendering::ArrorRenderer, VFRendering::SurfaceRenderer or VFRendering::IsosurfaceRenderer
- VFRendering::CoordinateSystemRenderer, which is used for rendering a coordinate system, with the axes colored by using the colormap

To control what renderers are used, you can use `VFRendering::View::renderers`, where you can pass it a `std::vector`s of `std::pair`s of renderers as `std::shared_ptr<VFRendering::RendererBase>` (i.e. shared pointers) and viewports as `glm::vec4`.

## Options

To modify the way the vector field is rendered, **libvfrendering** offers a variety of options. To set these, you can create an **VFRendering::Options** object.

As an example, to adjust the vertical field of view, you would do the following:

```c++
VFRendering::Options options;
options.set<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>(30);
view.updateOptions(options);
```

If you want to set only one option, you can also use **View::setOption**:

```c++
view.setOption<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>(30);
```

If you want to set an option for an individual Renderer, you can use the methods **RendererBase::updateOptions** and **RendererBase::setOption** in the same way.

Whether this way of setting options should be replaced by getters/setters will be evaluated as the API becomes more stable.

Currently, the following options are available:

| Index | Type  | Default value | Header file | Documentation |
|-------|-------|---------------|-------------|---------------|
| View::Option::BOUNDING_BOX_MIN | glm::vec3 | {-1, -1, -1} | View.hxx | VFRendering::Utilities::Options::Option< View::Option::BOUNDING_BOX_MIN > |
| View::Option::BOUNDING_BOX_MAX | glm::vec3 | {1, 1, 1} | View.hxx | VFRendering::Utilities::Options::Option< View::Option::BOUNDING_BOX_MAX > |
| View::Option::SYSTEM_CENTER | glm::vec3 | {0, 0, 0} | View.hxx | VFRendering::Utilities::Options::Option< View::Option::SYSTEM_CENTER > |
| View::Option::VERTICAL_FIELD_OF_VIEW | float | 45.0 | View.hxx | VFRendering::Utilities::Options::Option< View::Option::VERTICAL_FIELD_OF_VIEW > |
| View::Option::BACKGROUND_COLOR |  glm::vec3 | {0, 0, 0} | View.hxx | VFRendering::Utilities::Options::Option< View::Option::BACKGROUND_COLOR > |
| View::Option::COLORMAP_IMPLEMENTATION | std::string | VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::DEFAULT) | View.hxx | VFRendering::Utilities::Options::Option< View::Option::COLORMAP_IMPLEMENTATION > |
| View::Option::IS_VISIBLE_IMPLEMENTATION | std::string | bool is_visible(vec3 position, vec3 direction) { return true; } | View.hxx | VFRendering::Utilities::Options::Option< View::Option::IS_VISIBLE_IMPLEMENTATION > |
| View::Option::CAMERA_POSITION |  glm::vec3 | {14.5, 14.5, 30} | View.hxx | VFRendering::Utilities::Options::Option< View::Option::CAMERA_POSITION > |
| View::Option::CENTER_POSITION |  glm::vec3 | {14.5, 14.5, 0} |  View.hxx | VFRendering::Utilities::Options::Option< View::Option::CENTER_POSITION > |
| View::Option::UP_VECTOR | glm::vec3 | {0, 1, 0}  | View.hxx | VFRendering::Utilities::Options::Option< View::Option::UP_VECTOR > |
| ArrowRenderer::Option::CONE_RADIUS | float | 0.25 | ArrowRenderer.hxx | VFRendering::Utilities::Options::Option< ArrowRenderer::Option::CONE_RADIUS > |
| ArrowRenderer::Option::CONE_HEIGHT | float | 0.6 | ArrowRenderer.hxx | VFRendering::Utilities::Options::Option< ArrowRenderer::Option::CONE_HEIGHT > |
| ArrowRenderer::Option::CYLINDER_RADIUS | float | 0.125 | ArrowRenderer.hxx | VFRendering::Utilities::Options::Option< ArrowRenderer::Option::CYLINDER_RADIUS > |
| ArrowRenderer::Option::CYLINDER_HEIGHT | float | 0.7 | ArrowRenderer.hxx | VFRendering::Utilities::Options::Option< ArrowRenderer::Option::CYLINDER_HEIGHT > |
| ArrowRenderer::Option::LEVEL_OF_DETAIL | unsigned int | 20 | ArrowRenderer.hxx | VFRendering::Utilities::Options::Option< ArrowRenderer::Option::LEVEL_OF_DETAIL > |
| BoundingBoxRenderer::Option::COLOR | glm::vec3 | {1.0, 1.0, 1.0} | BoundingBoxRenderer.hxx | VFRendering::Utilities::Options::Option< BoundingBoxRenderer::Option::COLOR > |
| CoordinateSystemRenderer::Option::AXIS_LENGTH | glm::vec3 | {0.5, 0.5, 0.5} | CoordinateSystemRenderer.hxx | VFRendering::Utilities::Options::Option< CoordinateSystemRenderer::Option::AXIS_LENGTH > |
| CoordinateSystemRenderer::Option::ORIGIN | glm::vec3 | {0.0, 0.0, 0.0} | CoordinateSystemRenderer.hxx | VFRendering::Utilities::Options::Option< CoordinateSystemRenderer::Option::ORIGIN > |
| CoordinateSystemRenderer::Option::NORMALIZE | bool | false | CoordinateSystemRenderer.hxx | VFRendering::Utilities::Options::Option< CoordinateSystemRenderer::Option::NORMALIZE > |
| CoordinateSystemRenderer::Option::LEVEL_OF_DETAIL | unsigned int | 100 | CoordinateSystemRenderer.hxx | VFRendering::Utilities::Options::Option< CoordinateSystemRenderer::Option::LEVEL_OF_DETAIL > |
| CoordinateSystemRenderer::Option::CONE_HEIGHT | float | 0.3 | CoordinateSystemRenderer.hxx | VFRendering::Utilities::Options::Option< CoordinateSystemRenderer::Option::CONE_HEIGHT > |
| CoordinateSystemRenderer::Option::CONE_RADIUS | float | 0.1 | CoordinateSystemRenderer.hxx | VFRendering::Utilities::Options::Option< CoordinateSystemRenderer::Option::CONE_RADIUS > |
| CoordinateSystemRenderer::Option::CYLINDER_HEIGHT | float | 0.7 | CoordinateSystemRenderer.hxx | VFRendering::Utilities::Options::Option< CoordinateSystemRenderer::Option::CYLINDER_HEIGHT > |
| CoordinateSystemRenderer::Option::CYLINDER_RADIUS | float | 0.07 | CoordinateSystemRenderer.hxx | VFRendering::Utilities::Options::Option< CoordinateSystemRenderer::Option::CYLINDER_RADIUS > |
| IsosurfaceRenderer::Option::ISOVALUE | float | 0.0 | IsosurfaceRenderer.hxx | VFRendering::Utilities::Options::Option< IsosurfaceRenderer::Option::ISOVALUE > |
| IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION | std::string | float lighting(vec3 position, vec3 normal) { return 1.0; } | IsosurfaceRenderer.hxx | VFRendering::Utilities::Options::Option< IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION > |
| IsosurfaceRenderer::Option::VALUE_FUNCTION | std::function<isovalue_type(const glm::vec3&, const glm::vec3&)> | [] (const glm::vec3& position, const glm::vec3& direction) { return direction.z; } | IsosurfaceRenderer.hxx | VFRendering::Utilities::Options::Option< IsosurfaceRenderer::Option::VALUE_FUNCTION > |
| VectorSphereRenderer::Option::POINT_SIZE_RANGE |  glm::vec2 | {1.0, 4.0} | VectorSphereRenderer.hxx | VFRendering::Utilities::Options::Option< VectorSphereRenderer::Option::POINT_SIZE_RANGE > |
| VectorSphereRenderer::Option::INNER_SPHERE_RADIUS | float | 0.95 | VectorSphereRenderer.hxx | VFRendering::Utilities::Options::Option< VectorSphereRenderer::Option::INNER_SPHERE_RADIUS > |
| VectorSphereRenderer::Option::USE_SPHERE_FAKE_PERSPECTIVE | bool | true | VectorSphereRenderer.hxx | VFRendering::Utilities::Options::Option< VectorSphereRenderer::Option::USE_SPHERE_FAKE_PERSPECTIVE > |


## ToDo

- A **EGS plugin** for combining **libvfrendering** with existing **EGS** plugins.
- Methods for reading geometry and directions from data files
- Documentation

See the issues for further information and adding your own requests.
