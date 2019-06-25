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
    window.Simulation = function(finishedCallback, options) {
        var defaultOptions = {
        };
        this._options = {};
        this._mergeOptions(options, defaultOptions);

        // FS.writeFile("/input.cfg", "translation_vectors\n1 0 0 20\n0 1 0 20\n0 0 1 1\n");

        // var cfgfile = "input/skyrmions_2D.cfg";
        // var cfgfile = "input/skyrmions_3D.cfg";
        // var cfgfile = "input/nanostrip_skyrmions.cfg";
        var cfgfile = "";
        this.getConfig(cfgfile, function(config) {
            // FS.writeFile("/input.cfg", config);
            this._state = Module.State_Setup("");
            this.showBoundingBox = true;
            finishedCallback(this);
        }.bind(this));
    };

    Module.iteration = Module.cwrap('Simulation_SingleShot', null, ['number', 'number', 'number']);
    Simulation.prototype.performIteration = function() {
        Module.iteration(this._state);
        this.update();
    };

    Module.startsim = Module.cwrap('Simulation_LLG_Start', null, ['number', 'number', 'number', 'number', Boolean, 'number', 'number']);
    Simulation.prototype.startSimulation = function() {
        Module.startsim(this._state, 1, 1000000, 1000, true);
    }

    Module.stopsim = Module.cwrap('Simulation_Stop', null, ['number', 'number', 'number']);
    Simulation.prototype.stopSimulation = function() {
        Module.stopsim(this._state);
    }

    Simulation.prototype.getConfig = function(cfg_name, callback) {
        if (cfg_name != "")
        {
            $.get( cfg_name, {}, function(res) {
                    // console.log(res);
                    callback(res);
                }
            );
        }
        else
        {
            callback(" ");
        }
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

    Module.Spirit_Version_Full = Module.cwrap('Spirit_Version_Full', 'string', []);
    Simulation.prototype.spiritVersion = function() {
        return Module.Spirit_Version_Full();
    };

    Module.getSpinDirections = Module.cwrap('System_Get_Spin_Directions', 'number', ['number']);
    Module.getSpinPositions = Module.cwrap('Geometry_Get_Positions', 'number', ['number']);
    Simulation.prototype.update = function() {
        var _n = Simulation.prototype.getNCells(this._state);
        var NX = _n[0];
        var NY = _n[1];
        var NZ = _n[2];
        var N = NX*NY*NZ;
        var result_ptr = Module.getSpinDirections(this._state);
        var double_directions = Module.HEAPF32.subarray(result_ptr/4, result_ptr/4+N*3);
        var pos_ptr = Module.getSpinPositions(this._state);
        var double_positions = Module.HEAPF32.subarray(pos_ptr/4, pos_ptr/4+N*3);
        var spinPositions = [];
        var spinDirections = [];
        for (var i = 0; i < N; i++) {
          var spinPosition = [double_positions[3*i], double_positions[3*i+1], double_positions[3*i+2]];
          Array.prototype.push.apply(spinPositions, spinPosition);
          var spinDirection = [double_directions[3*i], double_directions[3*i+1], double_directions[3*i+2]];
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

    Module.Configuration_PlusZ = Module.cwrap('Configuration_PlusZ', null, ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.setAllSpinsPlusZ = function() {
        var pos = new Float32Array([0,0,0]);
        var pos_ptr = Module._malloc(pos.length * pos.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(pos, pos_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        var border = new Float32Array([-1,-1,-1]);
        var border_ptr = Module._malloc(border.length * border.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(border, border_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Configuration_PlusZ(this._state, pos_ptr, border_ptr, -1, -1, 0, 0, -1, -1);
        this.update();
    };
    Module.Configuration_MinusZ = Module.cwrap('Configuration_MinusZ', null, ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.setAllSpinsMinusZ = function() {
        var pos = new Float32Array([0,0,0]);
        var pos_ptr = Module._malloc(pos.length * pos.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(pos, pos_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        var border = new Float32Array([-1,-1,-1]);
        var border_ptr = Module._malloc(border.length * border.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(border, border_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Configuration_MinusZ(this._state, pos_ptr, border_ptr, -1, -1, 0, 0, -1, -1);
        this.update();
    };
    Module.Configuration_Random = Module.cwrap('Configuration_Random', null, ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.setAllSpinsRandom = function() {
        var pos = new Float32Array([0,0,0]);
        var pos_ptr = Module._malloc(pos.length * pos.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(pos, pos_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        var border = new Float32Array([-1,-1,-1]);
        var border_ptr = Module._malloc(border.length * border.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(border, border_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Configuration_Random(this._state, pos_ptr, border_ptr, -1, -1, 0, 0, -1, -1);
        Module._free(pos_ptr);
        Module._free(border_ptr);
        this.update();
    };
    Module.Configuration_Skyrmion = Module.cwrap('Configuration_Skyrmion', null, ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.createSkyrmion = function(order, phase, radius, position, updown, rl, achiral) {
        position = new Float32Array(position);
        var position_ptr = Module._malloc(position.length * position.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(position, position_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        var border = new Float32Array([-1,-1,-1]);
        var border_ptr = Module._malloc(border.length * border.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(border, border_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Configuration_Skyrmion(this._state, radius, order, phase, updown, achiral, rl, position_ptr, border_ptr, -1, -1, 0, 0, -1, -1);
        Module._free(position_ptr);
        this.update();
    };
    Module.Configuration_SpinSpiral = Module.cwrap('Configuration_SpinSpiral', null, ['number', 'string', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.createSpinSpiral = function(direction_type, q, axis, theta) {
        q = new Float32Array(q);
        var q_ptr = Module._malloc(q.length * q.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(q, q_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        axis = new Float32Array(axis);
        var axis_ptr = Module._malloc(axis.length * axis.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(axis, axis_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        var pos = new Float32Array([0,0,0]);
        var pos_ptr = Module._malloc(pos.length * pos.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(pos, pos_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        var border = new Float32Array([-1,-1,-1]);
        var border_ptr = Module._malloc(border.length * border.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(border, border_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Configuration_SpinSpiral(this._state, direction_type, q_ptr, axis_ptr, theta, pos_ptr, border_ptr, -1, -1, 0, 0, -1, -1);
        Module._free(q_ptr);
        Module._free(axis_ptr);
        this.update();
    };
    Module.Configuration_Domain = Module.cwrap('Configuration_Domain', null, ['number', 'number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.createDomain = function(direction, position, border) {
        position = new Float32Array(position);
        var position_ptr = Module._malloc(position.length * position.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(position, position_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        direction = new Float32Array(direction);
        var direction_ptr = Module._malloc(direction.length * direction.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(direction, direction_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        border = new Float32Array(border);
        var border_ptr = Module._malloc(border.length * border.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(border, border_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Configuration_Domain(this._state, direction_ptr, position_ptr, border_ptr, -1, -1, 0, 0, -1, -1);
        Module._free(position_ptr);
        Module._free(direction_ptr);
        this.update();
    };
    Module.Hamiltonian_Set_Boundary_Conditions = Module.cwrap('Hamiltonian_Set_Boundary_Conditions', null, ['number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianBoundaryConditions = function(periodical_a, periodical_b, periodical_c) {
        var periodical = new Int8Array([periodical_a, periodical_b, periodical_c]);
        var periodical_ptr = Module._malloc(periodical.length * periodical.BYTES_PER_ELEMENT);
        Module.HEAP8.set(periodical, periodical_ptr/Module.HEAP8.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_Boundary_Conditions(this._state, periodical_ptr, -1, -1);
        Module._free(periodical_ptr);
        this.update();
    };
    Module.Hamiltonian_Set_mu_s = Module.cwrap('Geometry_Set_mu_s', null, ['number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianMuSpin = function(mu_spin) {
        Module.Hamiltonian_Set_mu_s(this._state, mu_spin, -1, -1);
        this.update();
    };
    Module.Hamiltonian_Set_Field = Module.cwrap('Hamiltonian_Set_Field', null, ['number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianExternalField = function(magnitude, normal_x, normal_y, normal_z) {
        var normal = new Float32Array([normal_x, normal_y, normal_z]);
        var normal_ptr = Module._malloc(normal.length * normal.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(normal, normal_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_Field(this._state, magnitude, normal_ptr, -1, -1);
        Module._free(normal_ptr);
        this.update();
    };
    Module.Hamiltonian_Set_Exchange = Module.cwrap('Hamiltonian_Set_Exchange', null, ['number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianExchange = function(values) {
        values = new Float32Array(values);
        var values_ptr = Module._malloc(values.length * values.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(values, values_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_Exchange(this._state, values.length, values_ptr, -1, -1);
        Module._free(values_ptr);
        this.update();
    };
    Module.Hamiltonian_Set_DMI = Module.cwrap('Hamiltonian_Set_DMI', null, ['number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianDMI = function(values) {
        values = new Float32Array(values);
        var values_ptr = Module._malloc(values.length * values.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(values, values_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_DMI(this._state, values.length, values_ptr, -1, -1);
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
    Module.Hamiltonian_Set_DDI = Module.cwrap('Hamiltonian_Set_DDI', null, ['number', 'number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianDDI = function(method, n_periodical) {
        var periodical = new Int32Array([n_periodical, n_periodical, n_periodical]);
        var periodical_ptr = Module._malloc(periodical.length * periodical.BYTES_PER_ELEMENT);
        Module.HEAP32.set(periodical, periodical_ptr/Module.HEAP32.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_DDI(this._state, method, periodical_ptr, 0);
        Module._free(periodical_ptr);
        this.update();
    };
    Module.Parameters_LLG_Set_Convergence = Module.cwrap('Parameters_LLG_Set_Convergence', null, ['number', 'number', 'number', 'number']);
    Simulation.prototype.updateLLGConvergence = function(convergence) {
        Module.Parameters_LLG_Set_Convergence(this._state, convergence, -1 -1);
        this.update();
    };
    Module.Parameters_LLG_Set_STT = Module.cwrap('Parameters_LLG_Set_STT', null, ['number', 'number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianSpinTorque = function(magnitude, normal_x, normal_y, normal_z) {
        var normal = new Float32Array([normal_x, normal_y, normal_z]);
        var normal_ptr = Module._malloc(normal.length * normal.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(normal, normal_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Parameters_LLG_Set_STT(this._state, false, magnitude, normal_ptr, -1, -1);
        Module._free(normal_ptr);
        this.update();
    };
    Module.Parameters_LLG_Set_Temperature = Module.cwrap('Parameters_LLG_Set_Temperature', null, ['number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianTemperature = function(temperature) {
        Module.Parameters_LLG_Set_Temperature(this._state, temperature, -1, -1);
        this.update();
    };
    Module.Parameters_LLG_Set_Time_Step = Module.cwrap('Parameters_LLG_Set_Time_Step', null, ['number', 'number', 'number', 'number']);
    Simulation.prototype.updateLLGTimeStep = function(time_step) {
        Module.Parameters_LLG_Set_Time_Step(this._state, time_step, -1, -1);
        this.update();
    };
    Module.Parameters_LLG_Set_Damping = Module.cwrap('Parameters_LLG_Set_Damping', null, ['number', 'number', 'number', 'number']);
    Simulation.prototype.updateLLGDamping = function(damping) {
        Module.Parameters_LLG_Set_Damping(this._state, damping);
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
    Module.Geometry_Get_N_Cells = Module.cwrap('Geometry_Get_N_Cells', null, ['number', 'number', 'number']);
    Simulation.prototype.getNCells = function (state) {
        var ncells_ptr = Module._malloc(3*Module.HEAP32.BYTES_PER_ELEMENT);
        var na_ptr = ncells_ptr+0*Module.HEAP32.BYTES_PER_ELEMENT;
        var nb_ptr = ncells_ptr+1*Module.HEAP32.BYTES_PER_ELEMENT;
        var nc_ptr = ncells_ptr+2*Module.HEAP32.BYTES_PER_ELEMENT;
        Module.Geometry_Get_N_Cells(state, ncells_ptr, -1, -1);
        var NX = Module.HEAP32[na_ptr/Module.HEAP32.BYTES_PER_ELEMENT];
        var NY = Module.HEAP32[nb_ptr/Module.HEAP32.BYTES_PER_ELEMENT];
        var NZ = Module.HEAP32[nc_ptr/Module.HEAP32.BYTES_PER_ELEMENT];
        Module._free(ncells_ptr);
        return [NX, NY, NZ];
    }
});
