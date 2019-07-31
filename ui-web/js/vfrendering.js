"use strict";


function VFRendering(Module, canvas) {
    this._canvas = canvas;
    this._options = {};
    this._mergeOptions(this._options, VFRendering.defaultOptions);
    // this._gl = null;
    // this._gl_initialized = false;
    // this._renderers = [];
    // for (var i = 0; i < this._options.renderers.length; i++) {
    //     var renderer = this._options.renderers[i];
    //     var viewport = [0, 0, 1, 1];
    //     if (typeof renderer === typeof []) {
    //         viewport = renderer[1];
    //         renderer = renderer[0];
    //     }
    //     this._renderers.push([new renderer(this), viewport]);
    // }
    // this._initGLContext();
    // this._instancePositionArray = null;
    // this._instanceDirectionArray = null;

    this._currentScale = 1;
    this.isTouchDevice = 'ontouchstart' in document.documentElement;
    // this._options.useTouch = (this._options.useTouch && this.isTouchDevice);
    // if (this.isTouchDevice) {
    //     this._lastPanDeltaX = 0;
    //     this._lastPanDeltaY = 0;
    //     var mc = new Hammer.Manager(canvas, {});
    //     mc.add(new Hammer.Pan({ direction: Hammer.DIRECTION_ALL, threshold: 0, pointers: 1}));
    //     mc.on("pan", this._handlePan.bind(this));
    //     mc.add(new Hammer.Pinch({}));
    //     mc.on("pinchin pinchout pinchmove pinchend", this._handlePinch.bind(this));
    //     mc.on("pinchstart", this._handlePinchStart.bind(this));
    // }
    this._mouseDown = false;
    this._lastMouseX = null;
    this._lastMouseY = null;

    this._createBindings(Module);

    canvas.addEventListener('mousewheel',       this._handleMouseScroll.bind(this));
    canvas.addEventListener('DOMMouseScroll',   this._handleMouseScroll.bind(this));
    canvas.addEventListener('mousedown',        this._handleMouseDown.bind(this));
    canvas.addEventListener('mousemove',        this._handleMouseMove.bind(this));
    document.addEventListener('mouseup',        this._handleMouseUp.bind(this));
}

VFRendering.defaultOptions = {};
VFRendering.defaultOptions.verticalFieldOfView = 45;
VFRendering.defaultOptions.allowCameraMovement = true;
// VFRendering.defaultOptions.colormapImplementation = VFRendering.colormapImplementations['red'];
VFRendering.defaultOptions.cameraLocation = [50, 50, 100];
VFRendering.defaultOptions.centerLocation = [50, 50, 0];
VFRendering.defaultOptions.upVector = [0, 0, 1];
VFRendering.defaultOptions.backgroundColor = [0.5, 0.5, 0.5];
VFRendering.defaultOptions.zRange = [-1, 1];
VFRendering.defaultOptions.boundingBox = null;
VFRendering.defaultOptions.boundingBoxColor = [1, 1, 1];
VFRendering.defaultOptions.useTouch = true;

VFRendering.prototype._mergeOptions = function(options, defaultOptions) {
    this._options = {};
    for (var option in defaultOptions) {
        this._options[option] = defaultOptions[option];
    }
    for (var option in options) {
        if (defaultOptions.hasOwnProperty(option)) {
            this._options[option] = options[option];
        } else {
            console.warn("VFRendering does not recognize option '" + option +"'.");
        }
    }
};

VFRendering.prototype._createBindings = function(Module) {
    Module._initialize();
    // Module._draw();

    Module.draw = Module.cwrap('draw');
    VFRendering.prototype._draw = function() {
        Module.draw();
    };

    // Module.set_camera = Module.cwrap('set_camera');
    VFRendering.prototype.set_camera = function(position, center, up) {
        position = new Float32Array(position);
        var position_ptr = Module._malloc(position.length * position.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(position, position_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        center = new Float32Array(center);
        var center_ptr = Module._malloc(center.length * center.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(center, center_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        up = new Float32Array(up);
        var up_ptr = Module._malloc(up.length * up.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(up, up_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module._set_camera(position_ptr, center_ptr, up_ptr);
    };

    VFRendering.prototype.align_camera = function(direction, up) {
        direction = new Float32Array(direction);
        var direction_ptr = Module._malloc(direction.length * direction.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(direction, direction_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        up = new Float32Array(up);
        var up_ptr = Module._malloc(up.length * up.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(up, up_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module._align_camera(direction_ptr, up_ptr);
    };

    // ----------------------- Functions

    VFRendering.prototype.draw = function() {
        var width = this._canvas.clientWidth;
        var height = this._canvas.clientHeight;
        this._canvas.width = width;
        this._canvas.height = height;
        Module._draw();
    }

    VFRendering.prototype.updateDirections = function(directions_ptr) {
        Module._update_directions(directions_ptr);
        Module._draw();
    }

    // ----------------------- Handlers

    VFRendering.prototype._handleMouseDown = function(event) {
        // if (this._options.useTouch) return;
        if (!this._options.allowCameraMovement) {
            return;
        }
        this._mouseDown = true;
        this._lastMouseX = event.clientX;
        this._lastMouseY = event.clientY;
    };

    VFRendering.prototype._handleMouseUp = function(event) {
        // if (this._options.useTouch) return;
        this._mouseDown = false;
    };

    VFRendering.prototype._handleMouseMove = function(event) {
        // console.log(event);
        // if (this._options.useTouch) return;
        if (!this._options.allowCameraMovement) {
            return;
        }
        if (!this._mouseDown) {
        return;
        }
        var newX = event.clientX;
        var newY = event.clientY;
        var deltaX = newX - this._lastMouseX;
        var deltaY = newY - this._lastMouseY;
        // if (event.shiftKey)
        // {
        //     this.zoom(deltaY > 0 ? 1 : -1);
        // }
        // else
        // {
        //     var forwardVector = VFRendering._difference(this._options.centerLocation, this._options.cameraLocation);
        //     var cameraDistance = VFRendering._length(forwardVector);
        //     forwardVector = VFRendering._normalize(forwardVector);
        //     this._options.upVector = VFRendering._normalize(this._options.upVector);
        //     var rightVector = VFRendering._cross(forwardVector, this._options.upVector);
        //     this._options.upVector = VFRendering._cross(rightVector, forwardVector);
        //     this._options.upVector = VFRendering._normalize(this._options.upVector);
        //     if (event.altKey) {
        //         var translation =  [
        //             (deltaY / 100 * this._options.upVector[0] - deltaX / 100 * rightVector[0])*cameraDistance*0.1,
        //             (deltaY / 100 * this._options.upVector[1] - deltaX / 100 * rightVector[1])*cameraDistance*0.1,
        //             (deltaY / 100 * this._options.upVector[2] - deltaX / 100 * rightVector[2])*cameraDistance*0.1];
        //         this._options.cameraLocation[0] += translation[0];
        //         this._options.cameraLocation[1] += translation[1];
        //         this._options.cameraLocation[2] += translation[2];
        //         this._options.centerLocation[0] += translation[0];
        //         this._options.centerLocation[1] += translation[1];
        //         this._options.centerLocation[2] += translation[2];
        //     } else {
        //         this._rotationHelper(deltaX, deltaY);
        //     }
        // }
        var prev = Module._malloc(2*Module.HEAPF32.BYTES_PER_ELEMENT);
        var na_ptr = prev+0*Module.HEAPF32.BYTES_PER_ELEMENT;
        var nb_ptr = prev+1*Module.HEAPF32.BYTES_PER_ELEMENT;
        Module.HEAPF32[na_ptr/Module.HEAPF32.BYTES_PER_ELEMENT] = this._lastMouseX;
        Module.HEAPF32[nb_ptr/Module.HEAPF32.BYTES_PER_ELEMENT] = this._lastMouseY;
        var current = Module._malloc(2*Module.HEAPF32.BYTES_PER_ELEMENT);
        var na_ptr = current+0*Module.HEAPF32.BYTES_PER_ELEMENT;
        var nb_ptr = current+1*Module.HEAPF32.BYTES_PER_ELEMENT;
        Module.HEAPF32[na_ptr/Module.HEAPF32.BYTES_PER_ELEMENT] = newX;
        Module.HEAPF32[nb_ptr/Module.HEAPF32.BYTES_PER_ELEMENT] = newY;

        Module._mouse_move(prev, current, 1);
        this.draw();
        this._lastMouseX = newX;
        this._lastMouseY = newY;
        // console.log(newX);
        // console.log(newY);
        // this.draw();
    };

    VFRendering.prototype._handleMouseScroll = function(event) {
        // if (this._options.useTouch) return;
        if (!this._options.allowCameraMovement) {
            return;
        }
        var scale = 10;
        if (event.shiftKey)
        {
            scale = 1;
        }
        var delta = Math.max(-1, Math.min(1, (event.wheelDelta || -event.detail)));
        Module._mouse_scroll(delta, scale);
        this.draw();
    };
};

// Module_VFR().then(function(Module) {
// Module_VFR.ready(function() {
    


    // // ----------------------- Linear Algebra Utilities ---------------------------

    // VFRendering._difference = function(a, b) {
    //     return [
    //         a[0]-b[0],
    //         a[1]-b[1],
    //         a[2]-b[2]
    //     ];
    // };

    // VFRendering._length = function(a) {
    //     return Math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
    // };

    // VFRendering._normalize = function(a) {
    //     var length = VFRendering._length(a);
    //     return [a[0]/length, a[1]/length, a[2]/length];
    // };

    // VFRendering._dot = function(a, b) {
    //     return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    // };

    // VFRendering._cross = function(a, b) {
    //     return [
    //         a[1]*b[2]-a[2]*b[1],
    //         a[2]*b[0]-a[0]*b[2],
    //         a[0]*b[1]-a[1]*b[0]
    //     ];
    // };

    // VFRendering._rotationMatrix = function(axis, angle) {
    //     var c = Math.cos(Math.PI * angle / 180);
    //     var s = Math.sin(Math.PI * angle / 180);
    //     var x = axis[0];
    //     var y = axis[1];
    //     var z = axis[2];
    //     return [
    //         [x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0],
    //         [x*y*(1-c)+z*s, y*y*(1-c)+c, z*y*(1-c)-x*s, 0],
    //         [x*z*(1-c)-y*s, z*y*(1-c)+x*s, z*z*(1-c)+c,0],
    //         [0, 0, 0, 1]
    //     ];
    // };

    // VFRendering._matrixMultiply = function(matrix, vector) {
    //     var result = [0, 0, 0];
    //     for(var i = 0; i < 3; i++) {
    //         for(var j = 0; j < 3; j++) {
    //             result[i] += matrix[i][j]*vector[j];
    //         }
    //         result[i] += matrix[i][3];
    //     }
    //     return result;
    // };

    // VFRendering._perspectiveProjectionMatrix = function(verticalFieldOfView, aspectRatio, zNear, zFar) {
    //     var f = 1.0/Math.tan(verticalFieldOfView*Math.PI/180/2);
    //     if (aspectRatio < 1.0) {
    //         f *= aspectRatio;
    //     }
    //     return [
    //         [f/aspectRatio, 0, 0, 0],
    //         [0, f, 0, 0],
    //         [0, 0, (zNear+zFar)/(zNear-zFar), 2*zFar*zNear/(zNear-zFar)],
    //         [0, 0, -1, 0]
    //     ];
    // };

    // VFRendering._orthographicProjectionMatrix = function(left, right, bottom, top, near, far) {
    //   var sx = 2.0/(right-left);
    //   var sy = 2.0/(top-bottom);
    //   var sz = 2.0/(far-near);
    //   var tx = (right+left)/(right-left);
    //   var ty = (top+bottom)/(top-bottom);
    //   var tz = (far+near)/(far-near);
    //   return [
    //         [sx, 0, 0, tx],
    //         [0, sy, 0, ty],
    //         [0, 0, sz, tz],
    //         [0, 0, 0, 1]
    //     ];
    // }

    // VFRendering._lookAtMatrix = function(cameraLocation, centerLocation, upVector) {
    //     var forwardVector = VFRendering._difference(centerLocation, cameraLocation);
    //     forwardVector = VFRendering._normalize(forwardVector);
    //     upVector = VFRendering._normalize(upVector);
    //     var rightVector = VFRendering._cross(forwardVector, upVector);
    //     rightVector = VFRendering._normalize(rightVector);
    //     upVector = VFRendering._cross(rightVector, forwardVector);
    //     var matrix = [
    //         [rightVector[0], rightVector[1], rightVector[2], 0],
    //         [upVector[0], upVector[1], upVector[2], 0],
    //         [-forwardVector[0], -forwardVector[1], -forwardVector[2], 0],
    //         [0, 0, 0, 1]
    //     ];
    //     var translationVector = VFRendering._matrixMultiply(matrix, cameraLocation);
    //     matrix[0][3] = -translationVector[0];
    //     matrix[1][3] = -translationVector[1];
    //     matrix[2][3] = -translationVector[2];
    //     return matrix;
    // };

    // VFRendering._toFloat32Array = function(matrix) {
    //     return new Float32Array([
    //         matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0],
    //         matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1],
    //         matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2],
    //         matrix[0][3], matrix[1][3], matrix[2][3], matrix[3][3]
    //     ]);
    // };
// });

