"use strict";

function Core(Module, finishedCallback)
{
    this._createSpiritBindings(Module);
}

function VFRendering(Module, canvas, finishedCallback)
{
    this._canvas = canvas;

    this._options = {};
    this._mergeOptions(this._options, VFRendering.defaultOptions);

    this._createVFRenderingBindings(Module);

    this.isTouchDevice = 'ontouchstart' in document.documentElement;
    this._options.useTouch = (VFRendering.defaultOptions.useTouch && this.isTouchDevice);
    if (this.isTouchDevice) {
        this._lastPanDeltaX = 0;
        this._lastPanDeltaY = 0;
        var mc = new Hammer.Manager(canvas, {});
        mc.add(new Hammer.Pan({ direction: Hammer.DIRECTION_ALL, threshold: 0, pointers: 1}));
        mc.on("pan", this._handlePan.bind(this));
        mc.add(new Hammer.Pinch({}));
        mc.on("pinchstart pinchin pinchout pinchmove pinchend", this._handlePinch.bind(this));
    }
    this._mouseDown = false;
    this._lastMouseX = null;
    this._lastMouseY = null;

    canvas.addEventListener('mousewheel',       this._handleMouseScroll.bind(this));
    canvas.addEventListener('DOMMouseScroll',   this._handleMouseScroll.bind(this));
    canvas.addEventListener('mousedown',        this._handleMouseDown.bind(this));
    canvas.addEventListener('mousemove',        this._handleMouseMove.bind(this));
    document.addEventListener('mouseup',        this._handleMouseUp.bind(this));
}


VFRendering.defaultOptions = {};
VFRendering.defaultOptions.allowCameraMovement = true;
VFRendering.defaultOptions.useTouch = true;

VFRendering.prototype.updateOptions = function(options)
{
    var changedOptions = [];
    for (var option in options) {
        if (this._options.hasOwnProperty(option)) {
            if (this._options[option] !== options[option]) {
                this._options[option] = options[option];
                changedOptions.push(option);
            }
        } else {
            console.warn("VFRendering does not recognize option '" + option +"'.");
        }
    }
    if (changedOptions.length == 0) {
        return;
    }
};

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

Core.prototype._createSpiritBindings = function(Module)
{
    Module.State_Setup = Module.cwrap('State_Setup', 'number', ['string']);
    Core.prototype.setup = function(config) {
        this._state = Module.State_Setup(config);
    };

    Core.prototype.getConfig = function(cfg_name, callback) {
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

    // ------------------------------------------------

    Module.Geometry_Get_N_Cells = Module.cwrap('Geometry_Get_N_Cells', null, ['number', 'number', 'number']);
    Core.prototype.getNCells = function () {
        var ncells_ptr = Module._malloc(3*Module.HEAP32.BYTES_PER_ELEMENT);
        var na_ptr = ncells_ptr+0*Module.HEAP32.BYTES_PER_ELEMENT;
        var nb_ptr = ncells_ptr+1*Module.HEAP32.BYTES_PER_ELEMENT;
        var nc_ptr = ncells_ptr+2*Module.HEAP32.BYTES_PER_ELEMENT;
        Module.Geometry_Get_N_Cells(this._state, ncells_ptr, -1, -1);
        var NX = Module.HEAP32[na_ptr/Module.HEAP32.BYTES_PER_ELEMENT];
        var NY = Module.HEAP32[nb_ptr/Module.HEAP32.BYTES_PER_ELEMENT];
        var NZ = Module.HEAP32[nc_ptr/Module.HEAP32.BYTES_PER_ELEMENT];
        Module._free(ncells_ptr);
        return [NX, NY, NZ];
    }
    Module.Geometry_Set_N_Cells = Module.cwrap('Geometry_Set_N_Cells', null, ['number', 'number']);
    Core.prototype.setNCells = function (n_cells) {
        var ncells_ptr = Module._malloc(3*Module.HEAP32.BYTES_PER_ELEMENT);
        var na_ptr = ncells_ptr+0*Module.HEAP32.BYTES_PER_ELEMENT;
        var nb_ptr = ncells_ptr+1*Module.HEAP32.BYTES_PER_ELEMENT;
        var nc_ptr = ncells_ptr+2*Module.HEAP32.BYTES_PER_ELEMENT;
        Module.HEAP32[na_ptr/Module.HEAP32.BYTES_PER_ELEMENT] = n_cells[0];
        Module.HEAP32[nb_ptr/Module.HEAP32.BYTES_PER_ELEMENT] = n_cells[1];
        Module.HEAP32[nc_ptr/Module.HEAP32.BYTES_PER_ELEMENT] = n_cells[2];
        Module.Geometry_Set_N_Cells(this._state, ncells_ptr);
        Module._free(ncells_ptr);
    }

    // ------------------------------------------------

    Module.Simulation_SingleShot = Module.cwrap('Simulation_SingleShot', null, ['number', 'number', 'number']);
    Core.prototype.performIteration = function() {
        Module.Simulation_SingleShot(this._state);
        // this.update();
    };

    Module.Simulation_LLG_Start = Module.cwrap('Simulation_LLG_Start', null, ['number', 'number', 'number', 'number', Boolean, 'number', 'number']);
    Core.prototype.startSimulation = function() {
        Module.Simulation_LLG_Start(this._state, 1, 1000000, 1000, true);
    }

    Module.stopsim = Module.cwrap('Simulation_Stop', null, ['number', 'number', 'number']);
    Core.prototype.stopSimulation = function() {
        Module.stopsim(this._state);
    }

    Module.simulation_running = Module.cwrap('Simulation_Running_Anywhere_On_Chain', Boolean, ['number', 'number']);
    Core.prototype.simulationRunning = function() {
        return Module.simulation_running(this._state);
    }

    Module.Spirit_Version_Full = Module.cwrap('Spirit_Version_Full', 'string', []);
    Core.prototype.spiritVersion = function() {
        return Module.Spirit_Version_Full();
    };

    // Module.System_Get_Spin_Directions = Module.cwrap('System_Get_Spin_Directions', 'number', ['number', 'number', 'number']);
    // Core.prototype.getSpinDirections = function() {
    //     return Module.System_Get_Spin_Directions(this._state, -1, -1);
    // };

    Module.Geometry_Get_Positions = Module.cwrap('Geometry_Get_Positions', 'number', ['number']);
    Core.prototype.getSpinPositions = function() {
        return Module.Geometry_Get_Positions(this._state);
    };

    Module.Configuration_PlusZ = Module.cwrap('Configuration_PlusZ', null, ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']);
    Core.prototype.setAllSpinsPlusZ = function() {
        var pos = new Float32Array([0,0,0]);
        var pos_ptr = Module._malloc(pos.length * pos.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(pos, pos_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        var border = new Float32Array([-1,-1,-1]);
        var border_ptr = Module._malloc(border.length * border.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(border, border_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Configuration_PlusZ(this._state, pos_ptr, border_ptr, -1, -1, 0, 0, -1, -1);
        // this.update();
    };
    Module.Configuration_MinusZ = Module.cwrap('Configuration_MinusZ', null, ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']);
    Core.prototype.setAllSpinsMinusZ = function() {
        var pos = new Float32Array([0,0,0]);
        var pos_ptr = Module._malloc(pos.length * pos.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(pos, pos_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        var border = new Float32Array([-1,-1,-1]);
        var border_ptr = Module._malloc(border.length * border.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(border, border_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Configuration_MinusZ(this._state, pos_ptr, border_ptr, -1, -1, 0, 0, -1, -1);
        // this.update();
    };
    Module.Configuration_Random = Module.cwrap('Configuration_Random', null, ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']);
    Core.prototype.setAllSpinsRandom = function() {
        var pos = new Float32Array([0,0,0]);
        var pos_ptr = Module._malloc(pos.length * pos.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(pos, pos_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        var border = new Float32Array([-1,-1,-1]);
        var border_ptr = Module._malloc(border.length * border.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(border, border_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Configuration_Random(this._state, pos_ptr, border_ptr, -1, -1, 0, 0, -1, -1);
        Module._free(pos_ptr);
        Module._free(border_ptr);
        // this.update();
    };
    Module.Configuration_Skyrmion = Module.cwrap('Configuration_Skyrmion', null, ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']);
    Core.prototype.createSkyrmion = function(order, phase, radius, position, updown, rl, achiral) {
        position = new Float32Array(position);
        var position_ptr = Module._malloc(position.length * position.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(position, position_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        var border = new Float32Array([-1,-1,-1]);
        var border_ptr = Module._malloc(border.length * border.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(border, border_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Configuration_Skyrmion(this._state, radius, order, phase, updown, achiral, rl, position_ptr, border_ptr, -1, -1, 0, 0, -1, -1);
        Module._free(position_ptr);
        // this.update();
    };
    Module.Configuration_Hopfion = Module.cwrap('Configuration_Hopfion', null, ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']);
    Core.prototype.createHopfion = function(radius, order, position) {
        position = new Float32Array(position);
        var position_ptr = Module._malloc(position.length * position.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(position, position_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        var border = new Float32Array([-1,-1,-1]);
        var border_ptr = Module._malloc(border.length * border.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(border, border_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Configuration_Hopfion(this._state, radius, order, position_ptr, border_ptr, -1, -1, 0, 0, -1, -1);
        Module._free(position_ptr);
        // this.update();
    };
    Module.Configuration_SpinSpiral = Module.cwrap('Configuration_SpinSpiral', null, ['number', 'string', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']);
    Core.prototype.createSpinSpiral = function(direction_type, q, axis, theta) {
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
        // this.update();
    };
    Module.Configuration_Domain = Module.cwrap('Configuration_Domain', null, ['number', 'number', 'number', 'number', 'number', 'number']);
    Core.prototype.createDomain = function(direction, position, border) {
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
        // this.update();
    };
    Module.Hamiltonian_Set_Boundary_Conditions = Module.cwrap('Hamiltonian_Set_Boundary_Conditions', null, ['number', 'number', 'number']);
    Core.prototype.updateHamiltonianBoundaryConditions = function(periodical_a, periodical_b, periodical_c) {
        var periodical = new Int8Array([periodical_a, periodical_b, periodical_c]);
        var periodical_ptr = Module._malloc(periodical.length * periodical.BYTES_PER_ELEMENT);
        Module.HEAP8.set(periodical, periodical_ptr/Module.HEAP8.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_Boundary_Conditions(this._state, periodical_ptr, -1, -1);
        Module._free(periodical_ptr);
        // this.update();
    };
    Module.Hamiltonian_Set_mu_s = Module.cwrap('Geometry_Set_mu_s', null, ['number', 'number', 'number']);
    Core.prototype.updateHamiltonianMuSpin = function(mu_spin) {
        Module.Hamiltonian_Set_mu_s(this._state, mu_spin, -1, -1);
        // this.update();
    };
    Module.Hamiltonian_Set_Field = Module.cwrap('Hamiltonian_Set_Field', null, ['number', 'number', 'number', 'number']);
    Core.prototype.updateHamiltonianExternalField = function(magnitude, normal_x, normal_y, normal_z) {
        var normal = new Float32Array([normal_x, normal_y, normal_z]);
        var normal_ptr = Module._malloc(normal.length * normal.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(normal, normal_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_Field(this._state, magnitude, normal_ptr, -1, -1);
        Module._free(normal_ptr);
        // this.update();
    };
    Module.Hamiltonian_Set_Exchange = Module.cwrap('Hamiltonian_Set_Exchange', null, ['number', 'number', 'number', 'number']);
    Core.prototype.updateHamiltonianExchange = function(values) {
        values = new Float32Array(values);
        var values_ptr = Module._malloc(values.length * values.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(values, values_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_Exchange(this._state, values.length, values_ptr, -1, -1);
        Module._free(values_ptr);
        // this.update();
    };
    Module.Hamiltonian_Set_DMI = Module.cwrap('Hamiltonian_Set_DMI', null, ['number', 'number', 'number', 'number']);
    Core.prototype.updateHamiltonianDMI = function(values) {
        values = new Float32Array(values);
        var values_ptr = Module._malloc(values.length * values.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(values, values_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_DMI(this._state, values.length, values_ptr, -1, -1);
        // this.update();
    };
    Module.Hamiltonian_Set_Anisotropy = Module.cwrap('Hamiltonian_Set_Anisotropy', null, ['number', 'number', 'number', 'number', 'number']);
    Core.prototype.updateHamiltonianAnisotropy = function(magnitude, normal_x, normal_y, normal_z) {
        var normal = new Float32Array([normal_x, normal_y, normal_z]);
        var normal_ptr = Module._malloc(normal.length * normal.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(normal, normal_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_Anisotropy(this._state, magnitude, normal_ptr, -1, -1);
        Module._free(normal_ptr);
        // this.update();
    };
    Module.Hamiltonian_Set_DDI = Module.cwrap('Hamiltonian_Set_DDI', null, ['number', 'number', 'number', 'number', 'number', 'number']);
    Core.prototype.updateHamiltonianDDI = function(method, n_periodical) {
        var periodical = new Int32Array([n_periodical, n_periodical, n_periodical]);
        var periodical_ptr = Module._malloc(periodical.length * periodical.BYTES_PER_ELEMENT);
        Module.HEAP32.set(periodical, periodical_ptr/Module.HEAP32.BYTES_PER_ELEMENT);
        Module.Hamiltonian_Set_DDI(this._state, method, periodical_ptr, 0);
        Module._free(periodical_ptr);
        // this.update();
    };
    Module.Parameters_LLG_Set_Convergence = Module.cwrap('Parameters_LLG_Set_Convergence', null, ['number', 'number', 'number', 'number']);
    Core.prototype.updateLLGConvergence = function(convergence) {
        Module.Parameters_LLG_Set_Convergence(this._state, convergence, -1 -1);
        // this.update();
    };
    Module.Parameters_LLG_Set_STT = Module.cwrap('Parameters_LLG_Set_STT', null, ['number', 'number', 'number', 'number', 'number', 'number']);
    Core.prototype.updateHamiltonianSpinTorque = function(magnitude, normal_x, normal_y, normal_z) {
        var normal = new Float32Array([normal_x, normal_y, normal_z]);
        var normal_ptr = Module._malloc(normal.length * normal.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(normal, normal_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.Parameters_LLG_Set_STT(this._state, false, magnitude, normal_ptr, -1, -1);
        Module._free(normal_ptr);
        // this.update();
    };
    Module.Parameters_LLG_Set_Temperature = Module.cwrap('Parameters_LLG_Set_Temperature', null, ['number', 'number', 'number', 'number']);
    Core.prototype.updateHamiltonianTemperature = function(temperature) {
        Module.Parameters_LLG_Set_Temperature(this._state, temperature, -1, -1);
        // this.update();
    };
    Module.Parameters_LLG_Set_Time_Step = Module.cwrap('Parameters_LLG_Set_Time_Step', null, ['number', 'number', 'number', 'number']);
    Core.prototype.updateLLGTimeStep = function(time_step) {
        Module.Parameters_LLG_Set_Time_Step(this._state, time_step, -1, -1);
        // this.update();
    };
    Module.Parameters_LLG_Set_Damping = Module.cwrap('Parameters_LLG_Set_Damping', null, ['number', 'number', 'number', 'number']);
    Core.prototype.updateLLGDamping = function(damping) {
        Module.Parameters_LLG_Set_Damping(this._state, damping);
        // this.update();
    };
    Module.Geometry_Get_Bounds = Module.cwrap('Geometry_Get_Bounds', null, ['number', 'number', 'number']);
    Core.prototype.getBoundingBox = function() {
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
    Module.IO_Image_Write = Module.cwrap('IO_Image_Write', null, ['number', 'string', 'number', 'string', 'number', 'number']);
    Core.prototype.exportOVFDataURI = function () {
        Module.IO_Image_Write(this._state, "/export.ovf", 3, 'Generated with Spirit Web UI', -1, -1);
        var ovf_data = FS.readFile("/export.ovf");
        function uint8ArrayToString_chunked(data){
          var CHUNK_SIZE = 32768;
          var chunks = [];
          for (var i=0; i < data.length; i += CHUNK_SIZE) {
            chunks.push(String.fromCharCode.apply(null, data.subarray(i, i+CHUNK_SIZE)));
          }
          return chunks.join("");
        }
        return "data:application/octet-stream;base64," + btoa(uint8ArrayToString_chunked(ovf_data));
    }
    Module.IO_Image_Read = Module.cwrap('IO_Image_Read', null, ['number', 'string', 'number', 'number', 'number']);
    Core.prototype.importOVFData = function (ovf_data) {
        var stream = FS.open('/import.ovf', 'w');
        FS.write(stream, ovf_data, 0, ovf_data.length, 0);
        FS.close(stream);
        Module.IO_Image_Read(this._state, "/import.ovf", 0, -1, -1);
        // this.update();
    }
    Module.System_Update_Data = Module.cwrap('System_Update_Data', null, ['number', 'number', 'number']);
    Core.prototype.System_Update_Data = function () {
        Module.System_Update_Data(this._state, -1, -1);
    }
    Module.IO_Image_Write_Energy_per_Spin = Module.cwrap('IO_Image_Write_Energy_per_Spin', null, ['number', 'string', 'number', 'number', 'number']);
    Core.prototype.exportEnergyDataURI = function () {
        Module.IO_Image_Write_Energy_per_Spin(this._state, "/energy.txt", 3, -1, -1);
        var energy_data = FS.readFile("/energy.txt");
        function uint8ArrayToString_chunked(data){
          var CHUNK_SIZE = 32768;
          var chunks = [];
          for (var i=0; i < data.length; i += CHUNK_SIZE) {
            chunks.push(String.fromCharCode.apply(null, data.subarray(i, i+CHUNK_SIZE)));
          }
          return chunks.join("");
        }
        return "data:application/octet-stream;base64," + btoa(uint8ArrayToString_chunked(energy_data));
    }
};

// ------------------------------------------------

VFRendering.prototype._createVFRenderingBindings = function(Module)
{
    Module.initialize = Module.cwrap('initialize', null, ['number']);
    VFRendering.prototype.initialize = function() {
        Module.initialize(this._state);
    };

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

    VFRendering.prototype.recenter_camera = function() {
        Module._recenter_camera();
    };

    Module.set_background = Module.cwrap('set_background', null, ['number']);
    VFRendering.prototype.set_background = function(colour) {
        colour = new Float32Array(colour);
        var colour_ptr = Module._malloc(colour.length * colour.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(colour, colour_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.set_background(colour_ptr);
    };

    Module.set_coordinate_system = Module.cwrap('set_coordinate_system', null, ['number', 'number']);
    VFRendering.prototype.set_coordinatesystem = function(show, position) {
        Module.set_coordinate_system(this._state, show, position);
    };

    Module.set_miniview = Module.cwrap('set_miniview', null, ['number', 'number']);
    VFRendering.prototype.set_miniview = function(show, position) {
        Module.set_miniview(this._state, show, position);
        Module._draw();
    }

    Module.set_boundingbox_colour = Module.cwrap('set_boundingbox_colour', null, ['number', 'number', 'number']);
    VFRendering.prototype.set_boundingbox_colour = function(colour) {
        colour = new Float32Array(colour);
        var colour_ptr = Module._malloc(colour.length * colour.BYTES_PER_ELEMENT);
        Module.HEAPF32.set(colour, colour_ptr/Module.HEAPF32.BYTES_PER_ELEMENT);
        Module.set_boundingbox_colour(colour_ptr);
    };

    Module.set_boundingbox = Module.cwrap('set_boundingbox', null, ['number', 'number', 'number']);
    VFRendering.prototype.set_boundingbox = function(show, line_width) {
        Module.set_boundingbox(this._state, show, line_width);
    };

    Module.set_dots = Module.cwrap('set_dots', null, ['number', 'number']);
    VFRendering.prototype.set_dots = function(show) {
        Module.set_dots(this._state, show);
    };

    Module.set_arrows = Module.cwrap('set_arrows', null, ['number', 'number']);
    VFRendering.prototype.set_arrows = function(show) {
        Module.set_arrows(this._state, show);
    };

    Module.set_spheres = Module.cwrap('set_spheres', null, ['number', 'number']);
    VFRendering.prototype.set_spheres = function(show) {
        Module.set_spheres(this._state, show);
    };

    Module.set_boxes = Module.cwrap('set_boxes', null, ['number', 'number']);
    VFRendering.prototype.set_boxes = function(show) {
        Module.set_boxes(this._state, show);
    };

    Module.set_surface = Module.cwrap('set_surface', null, ['number', 'number']);
    VFRendering.prototype.set_surface = function(show) {
        Module.set_surface(this._state, show);
    };

    Module.set_isosurface = Module.cwrap('set_isosurface', null, ['number', 'number']);
    VFRendering.prototype.set_isosurface = function(show) {
        Module.set_isosurface(this._state, show);
    };

    // ----------------------- Functions

    VFRendering.prototype.draw = function() {
        var width = this._canvas.clientWidth;
        var height = this._canvas.clientHeight;
        this._canvas.width = width;
        this._canvas.height = height;
        Module._draw();
    }

    VFRendering.prototype.updateDirections = function() {
        Module._update_directions(this._state);
        Module._draw();
    }

    VFRendering.prototype.updateGeometry = function() {
        Module._update_geometry(this._state);
        Module._draw();
    }

    VFRendering.prototype.set_rendermode = function(mode) {
        Module._set_rendermode(mode);
        Module._draw();
    }

    VFRendering.prototype.updateVisibility = function(zRange) {
        Module._set_visibility(zRange[0], zRange[1]);
        Module._draw();
    }

    VFRendering.prototype.setVectorSphere = function(pointSizeRange) {
        Module._set_vectorsphere(pointSizeRange[0], pointSizeRange[1]);
        Module._draw();
    }

    VFRendering.prototype.setColormap = function(colormap) {
        var idx_cmap = 0;
        if( colormap == "hsv" )
        {
            idx_cmap = 0;
        }
        else if( colormap == "hue" )
        {
            idx_cmap = 1;
        }
        else if( colormap == "bluered" )
        {
            idx_cmap = 2;
        }
        else if( colormap == "bluegreenred" )
        {
            idx_cmap = 3;
        }
        else if( colormap == "bluewhitered" )
        {
            idx_cmap = 4;
        }
        else if( colormap == "red" )
        {
            idx_cmap = 5;
        }
        else if( colormap == "white" )
        {
            idx_cmap = 6;
        }
        else if( colormap == "gray" )
        {
            idx_cmap = 7;
        }
        else if( colormap == "black" )
        {
            idx_cmap = 8;
        }
        else if( colormap == "black" )
        {
            idx_cmap = 8;
        }
        Module._set_colormap(idx_cmap);
        Module._draw();
    }

    // ----------------------- Handlers

    VFRendering.prototype._handlePinch = function(event) {
        if (!this._options.useTouch) return;
        if (event.scale > 1) {
            Module._mouse_scroll(-0.3*event.scale);
        } else {
            Module._mouse_scroll(0.3/event.scale);
        }
        this.draw();
    };

    VFRendering.prototype._handlePan = function(event) {
        if (!this._options.useTouch) return;
        var deltaX = event.deltaX;
        var deltaY = event.deltaY;

        var prev = Module._malloc(2*Module.HEAPF32.BYTES_PER_ELEMENT);
        var na_ptr = prev+0*Module.HEAPF32.BYTES_PER_ELEMENT;
        var nb_ptr = prev+1*Module.HEAPF32.BYTES_PER_ELEMENT;
        Module.HEAPF32[na_ptr/Module.HEAPF32.BYTES_PER_ELEMENT] = this._lastPanDeltaX;
        Module.HEAPF32[nb_ptr/Module.HEAPF32.BYTES_PER_ELEMENT] = this._lastPanDeltaY;
        var current = Module._malloc(2*Module.HEAPF32.BYTES_PER_ELEMENT);
        var na_ptr = current+0*Module.HEAPF32.BYTES_PER_ELEMENT;
        var nb_ptr = current+1*Module.HEAPF32.BYTES_PER_ELEMENT;
        Module.HEAPF32[na_ptr/Module.HEAPF32.BYTES_PER_ELEMENT] = deltaX;
        Module.HEAPF32[nb_ptr/Module.HEAPF32.BYTES_PER_ELEMENT] = deltaY;

        if (event.isFinal) {
            this._lastPanDeltaX = 0;
            this._lastPanDeltaY = 0;
        } else {
            this._lastPanDeltaX = event.deltaX;
            this._lastPanDeltaY = event.deltaY;
        }

        Module._mouse_move(prev, current, 1);
        this.draw();
    };

    VFRendering.prototype._handleMouseDown = function(event) {
        if (this._options.useTouch) return;
        if (!this._options.allowCameraMovement) {
            return;
        }
        this._mouseDown = true;
        this._lastMouseX = event.clientX;
        this._lastMouseY = event.clientY;
    };

    VFRendering.prototype._handleMouseUp = function(event) {
        if (this._options.useTouch) return;
        this._mouseDown = false;
    };

    VFRendering.prototype._handleMouseMove = function(event) {
        if (this._options.useTouch) return;
        if (!this._options.allowCameraMovement) {
            return;
        }
        if (!this._mouseDown) {
            return;
        }
        var newX = event.clientX;
        var newY = event.clientY;

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
    };

    VFRendering.prototype._handleMouseScroll = function(event) {
        if (this._options.useTouch) return;
        if (!this._options.allowCameraMovement) {
            return;
        }
        var scale = 1;
        if (event.shiftKey)
        {
            scale = 0.1;
        }
        var delta = Math.max(-1, Math.min(1, (event.wheelDelta || -event.detail)));
        Module._mouse_scroll(-delta*scale);
        this.draw();
    };
};

// --------------------------------------

function Spirit(Module, canvas)
{
    this.core = new Core(Module);
    this.vfr = new VFRendering(Module, canvas);

    // ---------------------------------------

    this.core.setup("");

    // Default geometry
    this.core.setNCells([100, 100, 1]);

    // Default Hamiltonian
    this.core.updateHamiltonianMuSpin(2);
    this.core.updateHamiltonianExternalField(25, 0, 0, 1);
    this.core.updateHamiltonianExchange([10]);
    this.core.updateHamiltonianDMI([6]);

    // Default configuration
    // this.core.setAllSpinsPlusZ();
    this.core.setAllSpinsRandom();

    // ---------------------------------------

    this.vfr._state = this.core._state;
    this.vfr.initialize();

    // console.log("ready VFR");
    this.vfr.updateGeometry();
    this.vfr.updateDirections();
    this.vfr.recenter_camera();
    this.vfr.draw();
};