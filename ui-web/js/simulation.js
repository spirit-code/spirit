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

$.getScript("core.js");

Module.ready(function() {
    Module.createSimulation = Module.cwrap('createSimulation', 'number', []);
    window.Simulation = function(options) {
        var defaultOptions = {
        };
        this._options = {};
        this._mergeOptions(options, defaultOptions);
        this._state = Module.createSimulation();
    };

    Module.performIteration = Module.cwrap('performIteration', 'number', ['number']);
    Simulation.prototype.performIteration = function() {
        Module.performIteration(this._state);
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
                console.warn("JuSpin Simulation does not recognize option '" + option +"'.");
            }
        }
    };

    Module.getSpinDirections = Module.cwrap('getSpinDirections', 'number', ['number']);
    Simulation.prototype.update = function() {
        var NX = 100;
        var NY = 100;
        var N = NX*NY;
        var result_ptr = Module.getSpinDirections(this._state);
        var double_directions = Module.HEAPF64.subarray(result_ptr/8, result_ptr/8+N*3);
        var spinPositions = [];
        var spinDirections = [];
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
        webglspins.updateOptions({surfaceIndices: surfaceIndices});
    };

    Module.PlusZ = Module.cwrap('PlusZ', null, ['number']);
    Simulation.prototype.setAllSpinsPlusZ = function() {
        Module.PlusZ(this._state);
        this.update();
    };
    Module.MinusZ = Module.cwrap('MinusZ', null, ['number']);
    Simulation.prototype.setAllSpinsMinusZ = function() {
        Module.MinusZ(this._state);
        this.update();
    };
    Module.Random = Module.cwrap('Random', null, ['number']);
    Simulation.prototype.setAllSpinsRandom = function() {
        Module.Random(this._state);
        this.update();
    };
    Module.Hamiltonian_Field = Module.cwrap('Hamiltonian_Field', null, ['number', 'number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianExternalField = function(magnitude, normal_x, normal_y, normal_z) {
        Module.Hamiltonian_Field(this._state, magnitude, normal_x, normal_y, normal_z);
        this.update();
    };
    Module.Hamiltonian_mu_s = Module.cwrap('Hamiltonian_mu_s', null, ['number', 'number']);
    Simulation.prototype.updateHamiltonianMuSpin = function(mu_spin) {
        Module.Hamiltonian_mu_s(this._state, mu_spin);
        this.update();
    };
    Module.Hamiltonian_Boundary_Conditions = Module.cwrap('Hamiltonian_Boundary_Conditions', null, ['number', 'number', 'number', 'number']);
    Simulation.prototype.updateHamiltonianBoundaryConditions = function(periodical_a, periodical_b, periodical_c) {
        Module.Hamiltonian_Boundary_Conditions(this._state, periodical_a, periodical_b, periodical_c);
        this.update();
    };
});
