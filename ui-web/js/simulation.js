"use strict";

var Module = {
    preRun: [],
    postRun: [],
    print: function (text) {
        if (arguments.length > 1) text = Array.prototype.slice.call(arguments).join(' ');
        console.log(text);
    },
    printErr: function (text) {
        if (arguments.length > 1) text = Array.prototype.slice.call(arguments).join(' ');
        console.error(text);
    },
    onRuntimeInitialized: function () {
        this.isReady = true;
        for (var i in this._readyCallbacks) {
            this._readyCallbacks[i]();
        }
    },
    isReady: false,
    _readyCallbacks: [],
    ready: function(readyCallback) {
        if (this.isReady) {
            readyCallback();
        } else {
            this._readyCallbacks.push(readyCallback);
        }
    }
};

Module.ready(function() {
    Module.State_Setup = Module.cwrap('State_Setup', 'number', ['string']);
    window.Simulation = function(options) {
        var defaultOptions = {
        };
        this._options = {};
        this._mergeOptions(options, defaultOptions);
        this._state = Module.State_Setup("");
        this.showBoundingBox = true;
    };

    Module.iterate = Module.cwrap('JS_LLG_Iteration', null, ['number']);
    Simulation.prototype.performIteration = function() {
        Module.iterate(this._state);
        this.update();
    };

    Simulation.prototype._mergeOptions = function(options, defaultOptions) {
        this._options = {};
        for (var option in defaultOptions) {
            this._options[option] = defaultOptions[option];
        }
        for (var option in options) {
            if (defaultOptions.hasOwnProperty(option)) {
                this._options[option] = options[option];
            } else {
                console.warn("Spirit Simulation does not recognize option '" + option +"'.");
            }
        }
    };

    Module.getSpinDirections = Module.cwrap('System_Get_Spin_Directions', 'number', ['number']);
    Simulation.prototype.update = function() {
        var NX = 100;
        var NY = 100;
        var N = NX*NY;
        var result_ptr = Module.getSpinDirections(this._state);
        var double_directions = Module.HEAPF64.subarray(result_ptr/8, result_ptr/8+N*3);
        var spinPositions = [];
        var spinDirections = [];
        for (var i = 0; i < N*3; i++) {
            if (-1 > spinDirections[i] || 1 < spinDirections[i]) {
                alert(spinDirections[i]);
            }
            if (Number.isNaN(spinDirections[i])) {
                alert("NaN!");
            }
        }
        for (var i = 0; i < N; i++) {
          var row = Math.floor(i/NX);
          var column = i % NX;
          var spinPosition = [column, row, 0];
          Array.prototype.push.apply(spinPositions, spinPosition);
          var spinDirection = [double_directions[i], double_directions[i+N], double_directions[i+2*N]];
          Array.prototype.push.apply(spinDirections, spinDirection);
        }
        webglspins.updateSpins(spinPositions, spinDirections);
        var surfaceIndices = WebGLSpins.generateCartesianSurfaceIndices(NX, NY);
        var boundingBox = null;
        if (this. showBoundingBox) {
            boundingBox = this.getBoundingBox();
        }
        webglspins.updateOptions({
            surfaceIndices: surfaceIndices,
            boundingBox: boundingBox
        });
    };

    Module.Configuration_PlusZ = Module.cwrap('Configuration_PlusZ', null, ['number', 'number', 'number']);
    Simulation.prototype.setAllSpinsPlusZ = function() {
        Module.Configuration_PlusZ(this._state, -1, -1);
        this.update();
    };
    Module.Configuration_MinusZ = Module.cwrap('Configuration_MinusZ', null, ['number', 'number', 'number']);
    Simulation.prototype.setAllSpinsMinusZ = function() {
        Module.Configuration_MinusZ(this._state, -1, -1);
        this.update();
    };
    Module.Configuration_Random = Module.cwrap('Configuration_Random', null, ['number', 'number', 'number']);
    Simulation.prototype.setAllSpinsRandom = function() {
        Module.Configuration_Random(this._state, -1, -1);
        this.update();
    };
    Module.Configuration_Skyrmion = Module.cwrap('Configuration_Skyrmion', null, ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.createSkyrmion = function(order, phase, radius, position, updown, rl, achiral, exp) {
        position = new Float64Array(position);
        var position_ptr = Module._malloc(position.length * position.BYTES_PER_ELEMENT);
        Module.HEAPF64.set(position, position_ptr/Module.HEAPF64.BYTES_PER_ELEMENT);
        Module.Configuration_Skyrmion(this._state, position_ptr, radius, order, phase, updown, achiral, rl, exp, -1, -1);
        Module._free(position_ptr);
        this.update();
    };
    Module.Configuration_SpinSpiral = Module.cwrap('Configuration_SpinSpiral', null, ['number', 'string', 'number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.createSpinSpiral = function(direction_type, q, axis, theta) {
        q = new Float64Array(q);
        var q_ptr = Module._malloc(q.length * q.BYTES_PER_ELEMENT);
        Module.HEAPF64.set(q, q_ptr/Module.HEAPF64.BYTES_PER_ELEMENT);
        axis = new Float64Array(axis);
        var axis_ptr = Module._malloc(axis.length * axis.BYTES_PER_ELEMENT);
        Module.HEAPF64.set(axis, axis_ptr/Module.HEAPF64.BYTES_PER_ELEMENT);
        Module.Configuration_SpinSpiral(this._state, direction_type, q_ptr, axis_ptr, theta, -1, -1);
        Module._free(q_ptr);
        Module._free(axis_ptr);
        this.update();
    };
    Module.Configuration_DomainWall = Module.cwrap('Configuration_DomainWall', null, ['number', 'number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.createDomainWall = function(position, direction, greater) {
        position = new Float64Array(position);
        var position_ptr = Module._malloc(position.length * position.BYTES_PER_ELEMENT);
        Module.HEAPF64.set(position, position_ptr/Module.HEAPF64.BYTES_PER_ELEMENT);
        direction = new Float64Array(direction);
        var direction_ptr = Module._malloc(direction.length * direction.BYTES_PER_ELEMENT);
        Module.HEAPF64.set(direction, direction_ptr/Module.HEAPF64.BYTES_PER_ELEMENT);
        Module.Configuration_DomainWall(this._state, position_ptr, direction_ptr, greater, -1, -1);
        Module._free(position_ptr);
        Module._free(direction_ptr);
        this.update();
    };
    Module.Hamiltonian_Set_Boundary_Conditions = Module.cwrap('Hamiltonian_Set_Boundary_Conditions', null, ['number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianBoundaryConditions = function(periodical_a, periodical_b, periodical_c) {
        var periodical = new Int8Array([periodical_a, periodical_b, periodical_c]);
        var periodical_ptr = Module._malloc(periodical.length * periodical.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(periodical, periodical_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_Boundary_Conditions(this._state, periodical_ptr, -1, -1);
        Module._free(periodical_ptr);
        this.update();
    };
    Module.Hamiltonian_Set_mu_s = Module.cwrap('Hamiltonian_Set_mu_s', null, ['number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianMuSpin = function(mu_spin) {
        Module.Hamiltonian_Set_mu_s(this._state, mu_spin, -1, -1);
        this.update();
    };
    Module.Hamiltonian_Set_Field = Module.cwrap('Hamiltonian_Set_Field', null, ['number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianExternalField = function(magnitude, normal_x, normal_y, normal_z) {
        var normal = new Float32Array([normal_x, normal_y, normal_z]);
        var normal_ptr = Module._malloc(normal.length * normal.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(normal, normal_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_Field(this._state, magnitude, normal_ptr, -1, -1);
        Module._free(normal_ptr);
        this.update();
    };
    Module.Hamiltonian_Set_Exchange = Module.cwrap('Hamiltonian_Set_Exchange', null, ['number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianExchange = function(values) {
        values = new Float32Array(values);
        var values_ptr = Module._malloc(values.length * values.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(values, values_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_Exchange(this._state, values.length, values_ptr, -1, -1);
        Module._free(values_ptr);
        this.update();
    };
    Module.Hamiltonian_Set_DMI = Module.cwrap('Hamiltonian_Set_DMI', null, ['number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianDMI = function(dij) {
        Module.Hamiltonian_Set_DMI(this._state, dij, -1, -1);
        this.update();
    };
    Module.Hamiltonian_Set_Anisotropy = Module.cwrap('Hamiltonian_Set_Anisotropy', null, ['number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianAnisotropy = function(magnitude, normal_x, normal_y, normal_z) {
        var normal = new Float32Array([normal_x, normal_y, normal_z]);
        var normal_ptr = Module._malloc(normal.length * normal.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(normal, normal_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_Anisotropy(this._state, magnitude, normal_ptr, -1, -1);
        Module._free(normal_ptr);
        this.update();
    };
    Module.Hamiltonian_Set_STT = Module.cwrap('Hamiltonian_Set_STT', null, ['number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianSpinTorque = function(magnitude, normal_x, normal_y, normal_z) {
        var normal = new Float32Array([normal_x, normal_y, normal_z]);
        var normal_ptr = Module._malloc(normal.length * normal.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(normal, normal_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_STT(this._state, magnitude, normal_ptr, -1, -1);
        Module._free(normal_ptr);
        this.update();
    };
    Module.Hamiltonian_Set_Temperature = Module.cwrap('Hamiltonian_Set_Temperature', null, ['number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianTemperature = function(temperature) {
        Module.Hamiltonian_Set_Temperature(this._state, temperature, -1, -1);
        this.update();
    };
    Module.Parameters_Set_LLG_Time_Step = Module.cwrap('Parameters_Set_LLG_Time_Step', null, ['number', 'number', 'number', 'number']);
    Simulation.prototype.updateLLGTimeStep = function(time_step) {
        Module.Parameters_Set_LLG_Time_Step(this._state, time_step, -1, -1);
        this.update();
    };
    Module.Parameters_Set_LLG_Damping = Module.cwrap('Parameters_Set_LLG_Damping', null, ['number', 'number', 'number', 'number']);
    Simulation.prototype.updateLLGDamping = function(damping) {
        Module.Parameters_Set_LLG_Damping(this._state, damping, -1, -1);
        this.update();
    };
    Module.Parameters_Set_GNEB_Spring_Constant = Module.cwrap('Parameters_Set_GNEB_Spring_Constant', null, ['number', 'number', 'number', 'number']);
    Simulation.prototype.updateGNEBSpringConstant = function(spring_constant) {
        Module.Parameters_Set_GNEB_Spring_Constant(this._state, spring_constant, -1, -1);
        this.update();
    };
    Module.Parameters_Set_GNEB_Climbing_Falling = Module.cwrap('Parameters_Set_GNEB_Climbing_Falling', null, ['number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.updateGNEBClimbingFalling = function(climbing, falling) {
        Module.Parameters_Set_GNEB_Climbing_Falling(this._state, climbing, falling, -1, -1);
        this.update();
    };
    Module.Geometry_Get_Bounds = Module.cwrap('Geometry_Get_Bounds', null, ['number', 'number', 'number']);
    Simulation.prototype.getBoundingBox = function() {
        var bounding_box_ptr = Module._malloc(6*Module.HEAPF32.BYTES_PER_ELEMENT);
        var xmin_ptr = bounding_box_ptr+0*Module.HEAPF32.BYTES_PER_ELEMENT;
        var xmax_ptr = bounding_box_ptr+3*Module.HEAPF32.BYTES_PER_ELEMENT;
        var ymin_ptr = bounding_box_ptr+1*Module.HEAPF32.BYTES_PER_ELEMENT;
        var ymax_ptr = bounding_box_ptr+4*Module.HEAPF32.BYTES_PER_ELEMENT;
        var zmin_ptr = bounding_box_ptr+2*Module.HEAPF32.BYTES_PER_ELEMENT;
        var zmax_ptr = bounding_box_ptr+5*Module.HEAPF32.BYTES_PER_ELEMENT;
        Module.Geometry_Get_Bounds(this._state, xmin_ptr, xmax_ptr);
        var xmin = Module.HEAPF32[xmin_ptr/Module.HEAPF32.BYTES_PER_ELEMENT];
        var xmax = Module.HEAPF32[xmax_ptr/Module.HEAPF32.BYTES_PER_ELEMENT];
        var ymin = Module.HEAPF32[ymin_ptr/Module.HEAPF32.BYTES_PER_ELEMENT];
        var ymax = Module.HEAPF32[ymax_ptr/Module.HEAPF32.BYTES_PER_ELEMENT];
        var zmin = Module.HEAPF32[zmin_ptr/Module.HEAPF32.BYTES_PER_ELEMENT];
        var zmax = Module.HEAPF32[zmax_ptr/Module.HEAPF32.BYTES_PER_ELEMENT];
        Module._free(bounding_box_ptr);
        return [xmin, ymin, zmin, xmax, ymax, zmax];
    };
});
