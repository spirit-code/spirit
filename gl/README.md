GL
------

This library is built on top of **core** and encompasses a set of functions which are currently
designed for the visualisation of *atomistic spin systems*.

### Camera
Trackball camera?
Other cameras?

### Render modes
This library has been built to provide different render modes:
* Arrows
* Surface
* Sphere
The arrows simply show the spins' directions and positions.
The surface mode shows a smooth surface on which the color map is smoothly interpolated.
For the surface, the z-filter is applied per-fragment, resulting in comparatively smooth structures.
The sphere mode shows one (correspondingly coloured) dot per spin, at the position the spin points
to on the sphere. In contrast to the other modes, this allows one to see global structures instead
of local ones. 