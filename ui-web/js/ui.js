
$(document).ready(function()
{
    $('form').attr('onsubmit', 'return false;');
    var isSimulating = false;

    window.addEventListener('keydown', function(event) {
    //     var key = event.keyCode || event.which;
    //     if (key == 32) { // space
    //         // var sim = window.currentSimulation;
    //         // isSimulating = sim.simulationRunning();
    //         // // isSimulating = !isSimulating; // module.simulation_running()
    //         // $("#btn-play").toggleClass("fa-play fa-pause");
    //         // if (!isSimulating) {
    //         // sim.startSimulation();
    //         // window.requestAnimationFrame(function () {
    //         //     update(sim);
    //         // });
    //         // }
    //         // else {
    //         // sim.stopSimulation();
    //         // }
    //         // isSimulating = sim.simulationRunning();
    //     }
    //     else{
    //         console.log('key code ' + key + ' was pressed!');
    //     }
    });

    // Module_VFR.ready(function() {
    //     console.log("ready VFR");
    //     vfr = new VFRendering(document.getElementById("webgl-canvas"));
    //     vfr.draw();
    //     console.log("ready VFR done");
    // });


    function updateFromSpirit(spirit){
    }

    // Module_Spirit.ready(function() {
        
    //     console.log("ready Spirit");
    //     new Spirit(updateFromSpirit);
    //     console.log("ready Spirit done");

    //     // var version = window.currentSimulation.spiritVersion();
    //     // document.getElementById("spirit-version").textContent="Version" + version;
    // });


    Module_Spirit().then(function(Module) {
        // this is reached when everything is ready, and you can call methods on Module
        // console.log("ready Spirit");
        window.spirit = new Spirit(Module, document.getElementById("webgl-canvas"), updateFromSpirit);
        // console.log(spirit.core);
        // spirit.core.setcells();
        // this._state = Module._State_Setup("");

        // x = function(finishedCallback) {
        //     // var defaultOptions = {
        //     // };
        //     // this._options = {};
        //     // this._mergeOptions(options, defaultOptions);
    
        //     // FS.writeFile("/input.cfg", "translation_vectors\n1 0 0 20\n0 1 0 20\n0 0 1 1\n");
    
        //     // var cfgfile = "input/skyrmions_2D.cfg";
        //     // var cfgfile = "input/skyrmions_3D.cfg";
        //     // var cfgfile = "input/nanostrip_skyrmions.cfg";
        //     // var cfgfile = "";
        //     // this.getConfig(cfgfile, function(config) {
        //     //     // FS.writeFile("/input.cfg", config);
        //     //     this._state = Module.State_Setup("");
        //     //     this.showBoundingBox = true;
        //     //     finishedCallback(this);
        //     // }.bind(this));
        //     this._state = Module._State_Setup("");
        //     finishedCallback(this);
        // }

        // Default geometry
        spirit.setNCells([21, 21, 21]);

        // Default Hamiltonian
        spirit.core.updateHamiltonianMuSpin(2);
        spirit.core.updateHamiltonianExternalField(25, 0, 0, 1);
        spirit.core.updateHamiltonianExchange([10]);
        spirit.core.updateHamiltonianDMI([6]);

        // Default configuration
        spirit.core.setAllSpinsPlusZ();
        spirit.core.setAllSpinsRandom();
        // spirit.core.createSkyrmion(1, -90, 5, [0, 0, 0], false, false, false);
        
        // console.log("ready Spirit done");
        
        // ----------------------------------
        // console.log("ready VFR");
        spirit.vfr.draw();

        // console.log("ready VFR done");

        // console.log("before directions update");
        // var _n = spirit.getNCells();
        // var NX = _n[0];
        // var NY = _n[1];
        // var NZ = _n[2];
        // var N = NX*NY*NZ;

        // var directions_ptr = spirit.core.getSpinDirections();
        spirit.vfr.updateDirections();

    }
    ).then(function()
    {
        $( window ).resize(function() {
            if (!isSimulating) {
                spirit.vfr.draw();
            }
        });
        $('.collapse').on('hidden.bs.collapse', function () {
            if (!isSimulating) {
                spirit.vfr.draw();
            }
        });
        $('.collapse').on('shown.bs.collapse', function () {
            if (!isSimulating) {
                spirit.vfr.draw();
            }
        });
        $('#input-show-settings').on('change', function () {
            if (!isSimulating) {
                spirit.vfr.draw();
            }
        });

        $("#btn-play").click(function() {
            isSimulating = !isSimulating;
            $("#btn-play").toggleClass("fa-play fa-pause");
            if (isSimulating) {
                spirit.core.startSimulation();
                window.requestAnimationFrame(function () {
                    update();
                });
            }
            else {
                spirit.core.stopSimulation();
            }
        });

        window.addEventListener('keydown', function(event) {
            var key = event.keyCode || event.which;
            if (key == 88) { // x
                spirit.vfr.align_camera([-1, 0, 0], [0, 0, 1]);
                spirit.vfr.draw();
            }
            else if (key == 89) { // y
                // webglspins.alignCamera([0, 1, 0], [0, 0, 1]);
                spirit.vfr.align_camera([0, 1, 0], [0, 0, 1]);
                spirit.vfr.draw();
            }
            else if (key == 90) { // z
                // webglspins.alignCamera([0, 0, -1], [0, 1, 0]);
                spirit.vfr.align_camera([0, 0, -1], [0, 1, 0]);
                spirit.vfr.draw();
            }
            // else{
            //     console.log('key code ' + key + ' was pressed!');
            // }
        });

        function update() {
            if (isSimulating) {
                spirit.core.performIteration();
                // var directions_ptr = spirit.core.getSpinDirections();

                spirit.vfr.updateDirections();
                window.requestAnimationFrame(function () {
                    update()
                });
            }
        };
    }
    ).then(function()
    {
        $('#div-load').hide();
    });
});