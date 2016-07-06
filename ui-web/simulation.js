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
    Module.c_test = Module.cwrap('test', 'number', ['number']);
});

function Simulation(options) {
    var defaultOptions = {
    };
    this._options = {};
    this._mergeOptions(options, defaultOptions);
}

Simulation.prototype.performIteration = function() {
    var NX = 100;
    var NY = 100;
    var N = NX*NY;
    var result_ptr = Module.c_test(N);
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
}
;
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