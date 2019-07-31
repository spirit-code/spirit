
$(document).ready(function()
{
    $('form').attr('onsubmit', 'return false;');
    var isSimulating = false;
    var canvas = document.getElementById("webgl-canvas");

    Module_Spirit().then(function(Module) {
        window.FS = Module.FS
        window.spirit = new Spirit(Module, canvas);
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

        function simulation_start_stop()
        {
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
        }

        $("#btn-play").click(function() {
            simulation_start_stop();
        });

        canvas.addEventListener('keydown', function(event) {
            var key = event.keyCode || event.which;
            if (key == 32) { // space
                simulation_start_stop();
            }
            // else{
            //     console.log('key code ' + key + ' was pressed!');
            // }
        });

        window.addEventListener('keydown', function(event) {
            var key = event.keyCode || event.which;
            if (key == 88) { // x
                spirit.vfr.align_camera([-1, 0, 0], [0, 0, 1]);
                spirit.vfr.draw();
            }
            else if (key == 89) { // y
                spirit.vfr.align_camera([0, 1, 0], [0, 0, 1]);
                spirit.vfr.draw();
            }
            else if (key == 90) { // z
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
                spirit.vfr.updateDirections();
                spirit.vfr.draw();
                window.requestAnimationFrame(function () {
                    update()
                });
            }
        };

        // --------------------------

        function updateGridSize() {
            var x = Number($('#input-gridsize-x').val());
            var y = Number($('#input-gridsize-y').val());
            var z = Number($('#input-gridsize-z').val());
            spirit.core.setNCells([x, y, z]);
            spirit.vfr.updateGeometry();
            spirit.vfr.updateDirections();
            spirit.vfr.recenter_camera();
            spirit.vfr.draw();
        }
        $('#button-gridsize-update').on('click', function(e) {
            updateGridSize();
        });

        // --------------------------

        $('#button-plusz').on('click', function(e) {
            spirit.core.setAllSpinsPlusZ();
            spirit.vfr.updateDirections();
        });
        $('#button-minusz').on('click', function(e) {
            spirit.core.setAllSpinsMinusZ();
            spirit.vfr.updateDirections();
        });
        $('#button-random').on('click', function(e) {
            spirit.core.setAllSpinsRandom();
            spirit.vfr.updateDirections();
        });

        $('#button-skyrmion-create').on('click', function(e) {
            var order = Number($('#input-skyrmion-order').val());
            var phase = Number($('#input-skyrmion-phase').val());
            var radius = Number($('#input-skyrmion-radius').val());
            var positionx = Number($('#input-skyrmion-positionx').val());
            var positiony = Number($('#input-skyrmion-positiony').val());
            var positionz = Number($('#input-skyrmion-positionz').val());
            var updown = $('#input-skyrmion-updown')[0].checked;
            var rl = false;//$('#input-skyrmion-rl')[0].checked;
            var achiral = $('#input-skyrmion-achiral')[0].checked;
            // var exp = $('#input-skyrmion-exp')[0].checked;
            var valid = true;
            if (Number.isNaN(order)) {
                valid = false;
                $('#input-skyrmion-order').parent().addClass('has-error');
            } else {
                $('#input-skyrmion-order').parent().removeClass('has-error');
            }
            if (Number.isNaN(phase)) {
                valid = false;
                $('#input-skyrmion-phase').parent().addClass('has-error');
            } else {
                $('#input-skyrmion-phase').parent().removeClass('has-error');
            }
            if (Number.isNaN(radius)) {
                valid = false;
                $('#input-skyrmion-radius').parent().addClass('has-error');
            } else {
                $('#input-skyrmion-radius').parent().removeClass('has-error');
            }
            if (Number.isNaN(positionx)) {
                valid = false;
                $('#input-skyrmion-positionx').parent().addClass('has-error');
            } else {
                $('#input-skyrmion-positionx').parent().removeClass('has-error');
            }
            if (Number.isNaN(positiony)) {
                valid = false;
                $('#input-skyrmion-positiony').parent().addClass('has-error');
            } else {
                $('#input-skyrmion-positiony').parent().removeClass('has-error');
            }
            if (Number.isNaN(positionz)) {
                valid = false;
                $('#input-skyrmion-positionz').parent().addClass('has-error');
            } else {
                $('#input-skyrmion-positionz').parent().removeClass('has-error');
            }
            if (valid) {
                var position = [positionx, positiony, positionz];
                spirit.core.createSkyrmion(order, phase, radius, position, updown, rl, achiral);
                spirit.vfr.updateDirections();
            }
        });

        $('#button-hopfion-create').on('click', function(e) {
            var order = Number($('#input-hopfion-order').val());
            var radius = Number($('#input-hopfion-radius').val());
            var positionx = Number($('#input-hopfion-positionx').val());
            var positiony = Number($('#input-hopfion-positiony').val());
            var positionz = Number($('#input-hopfion-positionz').val());
            // var exp = $('#input-hopfion-exp')[0].checked;
            var valid = true;
            if (Number.isNaN(order)) {
                valid = false;
                $('#input-hopfion-order').parent().addClass('has-error');
            } else {
                $('#input-hopfion-order').parent().removeClass('has-error');
            }
            if (Number.isNaN(radius)) {
                valid = false;
                $('#input-hopfion-radius').parent().addClass('has-error');
            } else {
                $('#input-hopfion-radius').parent().removeClass('has-error');
            }
            if (Number.isNaN(positionx)) {
                valid = false;
                $('#input-hopfion-positionx').parent().addClass('has-error');
            } else {
                $('#input-hopfion-positionx').parent().removeClass('has-error');
            }
            if (Number.isNaN(positiony)) {
                valid = false;
                $('#input-hopfion-positiony').parent().addClass('has-error');
            } else {
                $('#input-hopfion-positiony').parent().removeClass('has-error');
            }
            if (Number.isNaN(positionz)) {
                valid = false;
                $('#input-hopfion-positionz').parent().addClass('has-error');
            } else {
                $('#input-hopfion-positionz').parent().removeClass('has-error');
            }
            if (valid) {
                var position = [positionx, positiony, positionz];
                spirit.core.createHopfion(radius, order, position);
                spirit.vfr.updateDirections();
            }
        });

        $('#button-spinspiral-create').on('click', function(e) {
            var qx = Number($('#input-spinspiral-qx').val());
            var qy = Number($('#input-spinspiral-qy').val());
            var qz = Number($('#input-spinspiral-qz').val());
            var axisx = Number($('#input-spinspiral-axisx').val());
            var axisy = Number($('#input-spinspiral-axisy').val());
            var axisz = Number($('#input-spinspiral-axisz').val());
            var angle = Number($('#input-spinspiral-angle').val());
            var wrt = $("option:selected", $('#select-spinspiral-wrt'))[0].value;
            var valid = true;
            if (Number.isNaN(qx)) {
                valid = false;
                $('#input-spinspiral-qx').parent().addClass('has-error');
            } else {
                $('#input-spinspiral-qx').parent().removeClass('has-error');
            }
            if (Number.isNaN(qy)) {
                valid = false;
                $('#input-spinspiral-qy').parent().addClass('has-error');
            } else {
                $('#input-spinspiral-qy').parent().removeClass('has-error');
            }
            if (Number.isNaN(qz)) {
                valid = false;
                $('#input-spinspiral-qz').parent().addClass('has-error');
            } else {
                $('#input-spinspiral-qz').parent().removeClass('has-error');
            }
            if (Number.isNaN(axisx)) {
                valid = false;
                $('#input-spinspiral-axisx').parent().addClass('has-error');
            } else {
                $('#input-spinspiral-axisx').parent().removeClass('has-error');
            }
            if (Number.isNaN(axisy)) {
                valid = false;
                $('#input-spinspiral-axisy').parent().addClass('has-error');
            } else {
                $('#input-spinspiral-axisy').parent().removeClass('has-error');
            }
            if (Number.isNaN(axisz)) {
                valid = false;
                $('#input-spinspiral-axisz').parent().addClass('has-error');
            } else {
                $('#input-spinspiral-axisz').parent().removeClass('has-error');
            }
            if (Number.isNaN(angle)) {
                valid = false;
                $('#input-spinspiral-angle').parent().addClass('has-error');
            } else {
                $('#input-spinspiral-angle').parent().removeClass('has-error');
            }
            if (valid) {
                var q = [qx, qy, qz];
                var axis = [axisx, axisy, axisz];
                spirit.core.createSpinSpiral(wrt, q, axis, angle);
                spirit.vfr.updateDirections();
            }
        });

        $('#button-domain-create').on('click', function(e) {
            var positionx = Number($('#input-domain-positionx').val());
            var positiony = Number($('#input-domain-positiony').val());
            var positionz = Number($('#input-domain-positionz').val());
            var borderx = Number($('#input-domain-borderx').val());
            var bordery = Number($('#input-domain-bordery').val());
            var borderz = Number($('#input-domain-borderz').val());
            var directionx = Number($('#input-domain-directionx').val());
            var directiony = Number($('#input-domain-directiony').val());
            var directionz = Number($('#input-domain-directionz').val());
            // var greater = $('#input-domain-greater')[0].checked;
            var valid = true;
            if (Number.isNaN(positionx)) {
                valid = false;
                $('#input-domain-positionx').parent().addClass('has-error');
            } else {
                $('#input-domain-positionx').parent().removeClass('has-error');
            }
            if (Number.isNaN(positiony)) {
                valid = false;
                $('#input-domain-positiony').parent().addClass('has-error');
            } else {
                $('#input-domain-positiony').parent().removeClass('has-error');
            }
            if (Number.isNaN(positionz)) {
                valid = false;
                $('#input-domain-positionz').parent().addClass('has-error');
            } else {
                $('#input-domain-positionz').parent().removeClass('has-error');
            }
            if (Number.isNaN(borderx)) {
                valid = false;
                $('#input-domain-borderx').parent().addClass('has-error');
            } else {
                $('#input-domain-borderx').parent().removeClass('has-error');
            }
            if (Number.isNaN(bordery)) {
                valid = false;
                $('#input-domain-bordery').parent().addClass('has-error');
            } else {
                $('#input-domain-bordery').parent().removeClass('has-error');
            }
            if (Number.isNaN(borderz)) {
                valid = false;
                $('#input-domain-borderz').parent().addClass('has-error');
            } else {
                $('#input-domain-borderz').parent().removeClass('has-error');
            }
            if (Number.isNaN(directionx)) {
                valid = false;
                $('#input-domain-directionx').parent().addClass('has-error');
            } else {
                $('#input-domain-directionx').parent().removeClass('has-error');
            }
            if (Number.isNaN(directiony)) {
                valid = false;
                $('#input-domain-directiony').parent().addClass('has-error');
            } else {
                $('#input-domain-directiony').parent().removeClass('has-error');
            }
            if (Number.isNaN(directionz)) {
                valid = false;
                $('#input-domain-directionz').parent().addClass('has-error');
            } else {
                $('#input-domain-directionz').parent().removeClass('has-error');
            }
            if (valid) {
                var position = [positionx, positiony, positionz];
                var border = [borderx, bordery, borderz];
                var direction = [directionx, directiony, directionz];
                spirit.core.createDomain(direction, position, border);
                spirit.vfr.updateDirections();
            }
        });

        // --------------------------

    function updateColormap() {
        var colormap = $("option:selected", $('#select-colormap'))[0].value;
            spirit.vfr.setColormap(colormap);
        }
        $('#select-colormap').on('change', updateColormap);
        updateColormap();

        function updateBackgroundColor() {
            var backgroundColor = $("option:selected", $('#select-backgroundcolor'))[0].value;
            var colors = {
                'white': [1.0, 1.0, 1.0],
                'gray': [0.5, 0.5, 0.5],
                'black': [0.0, 0.0, 0.0]
            };
            var boundingBoxColors = {
                'white': [0.0, 0.0, 0.0],
                'gray': [1.0, 1.0, 1.0],
                'black': [1.0, 1.0, 1.0]
            };
            spirit.vfr.set_background(colors[backgroundColor]);
            spirit.vfr.set_boundingbox_colour(boundingBoxColors[backgroundColor]);
            spirit.vfr.draw();
            // console.log("background update", colors[backgroundColor]);
        }
        $('#select-backgroundcolor').on('change', updateBackgroundColor);
        updateBackgroundColor();

        function updateRenderers() {
            var rendermode = $("option:selected", $('#select-rendermode'))[0].value;
            var show_miniview           = $('#input-show-miniview').is(':checked');
            var show_coordinatesystem   = $('#input-show-coordinatesystem').is(':checked');
            var show_boundingbox        = $('#input-show-boundingbox').is(':checked');
            var boundingbox_line_width  = 0;
            var show_dots               = $('#input-show-dots').is(':checked');
            var show_arrows             = $('#input-show-arrows').is(':checked');
            var show_spheres            = $('#input-show-spheres').is(':checked');
            var show_boxes              = $('#input-show-boxes').is(':checked');
            var show_isosurface         = $('#input-show-isosurface').is(':checked');
            var show_surface            = $('#input-show-surface').is(':checked');
            // console.log(rendermode);

            // var coordinateSystemPosition = JSON.parse($("option:selected", $('#select-coordinatesystem-position'))[0].value);
            // var miniViewPosition = JSON.parse($("option:selected", $('#select-miniview-position'))[0].value);
            var coordinateSystemPosition = $('#select-coordinatesystem-position')[0].selectedIndex;
            var miniViewPosition = $('#select-miniview-position')[0].selectedIndex;

            if( rendermode == "SYSTEM" )
            {
                spirit.vfr.set_rendermode(0);
            }
            else
            {
                spirit.vfr.set_rendermode(1);
            }

            spirit.vfr.set_miniview(show_miniview, miniViewPosition);
            spirit.vfr.set_coordinatesystem(show_coordinatesystem, coordinateSystemPosition);
            spirit.vfr.set_boundingbox(show_boundingbox, boundingbox_line_width);
            spirit.vfr.set_dots(show_dots);
            spirit.vfr.set_arrows(show_arrows);
            spirit.vfr.set_spheres(show_spheres);
            spirit.vfr.set_boxes(show_boxes);
            spirit.vfr.set_surface(show_surface);
            spirit.vfr.set_isosurface(show_isosurface);

            spirit.vfr.draw();
        }
        $('#select-rendermode').on('change', updateRenderers);
        $('#select-coordinatesystem-position').on('change', updateRenderers);
        $('#select-miniview-position').on('change', updateRenderers);
        $('#input-show-miniview').on('change', updateRenderers);
        $('#input-show-coordinatesystem').on('change', updateRenderers);
        $('#input-show-boundingbox').on('change', updateRenderers);
        $('#input-show-dots').on('change', updateRenderers);
        $('#input-show-arrows').on('change', updateRenderers);
        $('#input-show-spheres').on('change', updateRenderers);
        $('#input-show-boxes').on('change', updateRenderers);
        $('#input-show-surface').on('change', updateRenderers);
        $('#input-show-isosurface').on('change', updateRenderers);
        updateRenderers();

        $("#input-zrange-filter").slider();
        function updateZRangeFilter() {
            var zRange = $("#input-zrange-filter").slider('getValue');
            spirit.vfr.updateVisibility(zRange);
        }
        $('#input-zrange-filter').on('change', updateZRangeFilter);
        updateZRangeFilter();

        $("#input-spinspherewidget-pointsize").slider();
        function updateSpherePointSize() {
            var pointSizeRange = $("#input-spinspherewidget-pointsize").slider('getValue');
            spirit.vfr.setVectorSphere(pointSizeRange);
        }
        $('#input-spinspherewidget-pointsize').on('change', updateSpherePointSize);
        updateSpherePointSize();

        if (!VFRendering.isTouchDevice) {
            $('#input-use-touch')[0].disabled="disabled";
            $('#input-use-touch')[0].checked = false;
        }

        function updateHamiltonianBoundaryConditions() {
            var periodical_a = 0;
            var periodical_b = 0;
            var periodical_c = 0;
            if ($('#input-periodical-a')[0].checked) {
                periodical_a = 1;
            }
            if ($('#input-periodical-b')[0].checked) {
                periodical_b = 1;
            }
            if ($('#input-periodical-c')[0].checked) {
                periodical_c = 1;
            }
            spirit.core.updateHamiltonianBoundaryConditions(periodical_a, periodical_b, periodical_c);
            var show_boundingbox = $('#input-show-boundingbox').is(':checked');
            var boundingbox_line_width  = 0;
            spirit.vfr.set_boundingbox(show_boundingbox, boundingbox_line_width);
            spirit.vfr.draw();
        }
        $('#input-periodical-a').on('change', updateHamiltonianBoundaryConditions);
        $('#input-periodical-b').on('change', updateHamiltonianBoundaryConditions);
        $('#input-periodical-c').on('change', updateHamiltonianBoundaryConditions);

        function updateHamiltonianMuSpin() {
            var muspin = Number($('#input-externalfield-muspin').val());
            var valid = true;
            if (Number.isNaN(muspin)) {
                valid = false;
                $('#input-externalfield-muspin').parent().addClass('has-error');
            } else {
                $('#input-externalfield-muspin').parent().removeClass('has-error');
            }
            if (valid) {
                spirit.core.updateHamiltonianMuSpin(muspin);
            }
        }
        $('#input-externalfield-muspin').on('change', updateHamiltonianMuSpin);

        function updateHamiltonianExternalField() {
            if ($('#input-externalfield')[0].checked) {
            var magnitude = Number($('#input-externalfield-magnitude').val());
            var normalx = Number($('#input-externalfield-directionx').val());
            var normaly = Number($('#input-externalfield-directiony').val());
            var normalz = Number($('#input-externalfield-directionz').val());
            var valid = true;
            if (Number.isNaN(magnitude)) {
                valid = false;
                $('#input-externalfield-magnitude').parent().addClass('has-error');
            } else {
                $('#input-externalfield-magnitude').parent().removeClass('has-error');
            }
            if (Number.isNaN(normalx)) {
                valid = false;
                $('#input-externalfield-directionx').parent().addClass('has-error');
            } else {
                $('#input-externalfield-directionx').parent().removeClass('has-error');
            }
            if (Number.isNaN(normaly)) {
                valid = false;
                $('#input-externalfield-directiony').parent().addClass('has-error');
            } else {
                $('#input-externalfield-directiony').parent().removeClass('has-error');
            }
            if (Number.isNaN(normalz)) {
                valid = false;
                $('#input-externalfield-directionz').parent().addClass('has-error');
            } else {
                $('#input-externalfield-directionz').parent().removeClass('has-error');
            }
            if (valid) {
                spirit.core.updateHamiltonianExternalField(magnitude, normalx, normaly, normalz);
            }
            } else {
                spirit.core.updateHamiltonianExternalField(0, 0, 0, 1);
            }
        }
        $('#input-externalfield').on('change', updateHamiltonianExternalField);
        $('#input-externalfield-magnitude').on('change', updateHamiltonianExternalField);
        $('#input-externalfield-directionx').on('change', updateHamiltonianExternalField);
        $('#input-externalfield-directiony').on('change', updateHamiltonianExternalField);
        $('#input-externalfield-directionz').on('change', updateHamiltonianExternalField);

        function updateHamiltonianExchange() {
            if ($('#input-exchange')[0].checked) {
            var value1 = Number($('#input-exchangemagnitudes1').val());
            var value2 = Number($('#input-exchangemagnitudes2').val());
            var value3 = Number($('#input-exchangemagnitudes3').val());
            var value4 = Number($('#input-exchangemagnitudes4').val());
            var valid = true;
            if (Number.isNaN(value1)) {
                valid = false;
                $('#input-exchangemagnitudes1').parent().addClass('has-error');
            } else {
                $('#input-exchangemagnitudes1').parent().removeClass('has-error');
            }
            if (Number.isNaN(value2)) {
                valid = false;
                $('#input-exchangemagnitudes2').parent().addClass('has-error');
            } else {
                $('#input-exchangemagnitudes2').parent().removeClass('has-error');
            }
            if (Number.isNaN(value3)) {
                valid = false;
                $('#input-exchangemagnitudes3').parent().addClass('has-error');
            } else {
                $('#input-exchangemagnitudes3').parent().removeClass('has-error');
            }
            if (Number.isNaN(value4)) {
                valid = false;
                $('#input-exchangemagnitudes4').parent().addClass('has-error');
            } else {
                $('#input-exchangemagnitudes4').parent().removeClass('has-error');
            }
            if (valid) {
                spirit.core.updateHamiltonianExchange([value1, value2, value3, value4]);
            }
            } else {
                spirit.core.updateHamiltonianExchange([0]);
            }
        }
        $('#input-exchange').on('change', updateHamiltonianExchange);
        $('#input-exchangemagnitudes1').on('change', updateHamiltonianExchange);
        $('#input-exchangemagnitudes2').on('change', updateHamiltonianExchange);
        $('#input-exchangemagnitudes3').on('change', updateHamiltonianExchange);
        $('#input-exchangemagnitudes4').on('change', updateHamiltonianExchange);

        function updateHamiltonianDMI() {
            if ($('#input-dmi')[0].checked) {
            var dij = Number($('#input-dmi-magnitude').val());
            var valid = true;
            if (Number.isNaN(dij)) {
                valid = false;
                $('#input-dmi-magnitude').parent().addClass('has-error');
            } else {
                $('#input-dmi-magnitude').parent().removeClass('has-error');
            }
            if (valid) {
                spirit.core.updateHamiltonianDMI([dij]);
            }
            } else {
                spirit.core.updateHamiltonianDMI([0]);
            }
        }
        $('#input-dmi').on('change', updateHamiltonianDMI);
        $('#input-dmi-magnitude').on('change', updateHamiltonianDMI);

        function updateHamiltonianAnisotropy() {
            if ($('#input-anisotropy')[0].checked) {
            var magnitude = Number($('#input-anisotropy-magnitude').val());
            var normalx = Number($('#input-anisotropy-directionx').val());
            var normaly = Number($('#input-anisotropy-directiony').val());
            var normalz = Number($('#input-anisotropy-directionz').val());
            var valid = true;
            if (Number.isNaN(magnitude)) {
                valid = false;
                $('#input-anisotropy-magnitude').parent().addClass('has-error');
            } else {
                $('#input-anisotropy-magnitude').parent().removeClass('has-error');
            }
            if (Number.isNaN(normalx)) {
                valid = false;
                $('#input-anisotropy-directionx').parent().addClass('has-error');
            } else {
                $('#input-anisotropy-directionx').parent().removeClass('has-error');
            }
            if (Number.isNaN(normaly)) {
                valid = false;
                $('#input-anisotropy-directiony').parent().addClass('has-error');
            } else {
                $('#input-anisotropy-directiony').parent().removeClass('has-error');
            }
            if (Number.isNaN(normalz)) {
                valid = false;
                $('#input-anisotropy-directionz').parent().addClass('has-error');
            } else {
                $('#input-anisotropy-directionz').parent().removeClass('has-error');
            }
            if (valid) {
                spirit.core.updateHamiltonianAnisotropy(magnitude, normalx, normaly, normalz);
            }
            } else {
                spirit.core.updateHamiltonianAnisotropy(0, 0, 0, 1);
            }
        }
        $('#input-anisotropy').on('change', updateHamiltonianAnisotropy);
        $('#input-anisotropy-magnitude').on('change', updateHamiltonianAnisotropy);
        $('#input-anisotropy-directionx').on('change', updateHamiltonianAnisotropy);
        $('#input-anisotropy-directiony').on('change', updateHamiltonianAnisotropy);
        $('#input-anisotropy-directionz').on('change', updateHamiltonianAnisotropy);

        function updateHamiltonianDDI() {
            if ($('#input-ddi')[0].checked) {
                spirit.core.updateHamiltonianDDI(1, 4);
            }
            else {
                spirit.core.updateHamiltonianDDI(0, 4);
            }
        }
        $('#input-ddi').on('change', updateHamiltonianDDI);

        function updateHamiltonianSpinTorque() {
            if ($('#input-spintorque')[0].checked) {
            var magnitude = Number($('#input-spintorque-magnitude').val());
            var normalx = Number($('#input-spintorque-directionx').val());
            var normaly = Number($('#input-spintorque-directiony').val());
            var normalz = Number($('#input-spintorque-directionz').val());
            var valid = true;
            if (Number.isNaN(magnitude)) {
                valid = false;
                $('#input-spintorque-magnitude').parent().addClass('has-error');
            } else {
                $('#input-spintorque-magnitude').parent().removeClass('has-error');
            }
            if (Number.isNaN(normalx)) {
                valid = false;
                $('#input-spintorque-directionx').parent().addClass('has-error');
            } else {
                $('#input-spintorque-directionx').parent().removeClass('has-error');
            }
            if (Number.isNaN(normaly)) {
                valid = false;
                $('#input-spintorque-directiony').parent().addClass('has-error');
            } else {
                $('#input-spintorque-directiony').parent().removeClass('has-error');
            }
            if (Number.isNaN(normalz)) {
                valid = false;
                $('#input-spintorque-directionz').parent().addClass('has-error');
            } else {
                $('#input-spintorque-directionz').parent().removeClass('has-error');
            }
            if (valid) {
                spirit.core.updateHamiltonianSpinTorque(magnitude, normalx, normaly, normalz);
            }
            } else {
                spirit.core.updateHamiltonianSpinTorque(0, 0, 0, 1);
            }
        }
        $('#input-spintorque').on('change', updateHamiltonianSpinTorque);
        $('#input-spintorque-magnitude').on('change', updateHamiltonianSpinTorque);
        $('#input-spintorque-directionx').on('change', updateHamiltonianSpinTorque);
        $('#input-spintorque-directiony').on('change', updateHamiltonianSpinTorque);
        $('#input-spintorque-directionz').on('change', updateHamiltonianSpinTorque);

        function updateHamiltonianTemperature() {
            if ($('#input-temperature')[0].checked) {
            var temperature = Number($('#input-temperature-value').val());
            var valid = true;
            if (Number.isNaN(temperature)) {
                valid = false;
                $('#input-temperature-value').parent().addClass('has-error');
            } else {
                $('#input-temperature-value').parent().removeClass('has-error');
            }
            if (valid) {
                spirit.core.updateHamiltonianTemperature(temperature);
            }
            } else {
                spirit.core.updateHamiltonianTemperature(0);
            }
        }
        $('#input-temperature').on('change', updateHamiltonianTemperature);
        $('#input-temperature-value').on('change', updateHamiltonianTemperature);

        function updateLLGDamping() {
            var damping = Number($('#input-llg-damping').val());
            var valid = true;
            if (Number.isNaN(damping)) {
                valid = false;
                $('#input-llg-damping').parent().addClass('has-error');
            } else {
                $('#input-llg-damping').parent().removeClass('has-error');
            }
            if (valid) {
                spirit.core.updateLLGDamping(damping);
            }
        }
        $('#input-llg-damping').on('change', updateLLGDamping);

        function updateLLGTimeStep() {
            var time_step = Number($('#input-llg-timestep').val());
            var valid = true;
            if (Number.isNaN(time_step)) {
                valid = false;
                $('#input-llg-timestep').parent().addClass('has-error');
            } else {
                $('#input-llg-timestep').parent().removeClass('has-error');
            }
            if (valid) {
                spirit.core.updateLLGTimeStep(time_step);
            }
        }
        $('#input-llg-timestep').on('change', updateLLGTimeStep);

        // function updateGNEBSpringConstant() {
        //   var spring_constant = Number($('#input-gneb-springconst').val());
        //   var valid = true;
        //   if (Number.isNaN(spring_constant)) {
        //     valid = false;
        //     $('#input-gneb-springconst').parent().addClass('has-error');
        //   } else {
        //     $('#input-gneb-springconst').parent().removeClass('has-error');
        //   }
        //   if (valid) {
        //     spirit.core.updateGNEBSpringConstant(spring_constant);
        //   }
        // }
        // $('#input-gneb-springconst').on('change', updateGNEBSpringConstant);

        // function updateGNEBClimbingFalling() {
        //   var climbing = $('#input-gneb-radio-climbing')[0].checked;
        //   var falling = $('#input-gneb-radio-falling')[0].checked;
        //   spirit.core.updateGNEBClimbingFalling(climbing, falling);
        // }
        // $('#input-gneb-radio-normal').on('change', updateGNEBClimbingFalling);
        // $('#input-gneb-radio-climbing').on('change', updateGNEBClimbingFalling);
        // $('#input-gneb-radio-falling').on('change', updateGNEBClimbingFalling);

        function updateUseTouch() {
            var useTouch = $('#input-use-touch')[0].checked;
            VFRendering.updateOptions({useTouch: useTouch});
        }
        $('#input-use-touch').on('change', updateUseTouch);


        $('#button-camera-x').on('click', function(e) {
            spirit.vfr.align_camera([-1, 0, 0], [0, 0, 1]);
            spirit.vfr.draw();
        });
        $('#button-camera-y').on('click', function(e) {
            spirit.vfr.align_camera([0, 1, 0], [0, 0, 1]);
            spirit.vfr.draw();
        });
        $('#button-camera-z').on('click', function(e) {
            spirit.vfr.align_camera([0, 0, -1], [0, 1, 0]);
            spirit.vfr.draw();
        });

        function downloadURI(uri, name) {
            var link = document.createElement("a");
            link.download = name;
            link.href = uri;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            delete link;
        }

        $('#button-screenshot').on('click', function(e) {
            spirit.vfr.draw();
            downloadURI(canvas.toDataURL(), "spirit_screenshot.png");
        });
        if (window.File && window.FileReader && window.FileList && window.Blob) {
            $('#input-import-ovf').on('change', function (e) {
                if (e.target.files.length < 1) {
                    return;
                }
                var reader = new FileReader();
                reader.onload = function() {
                    spirit.core.importOVFData(new Uint8Array(reader.result));
                    spirit.vfr.updateDirections();
                };
                reader.readAsArrayBuffer(e.target.files[0]);
            });
        } else {
            $('#input-import-ovf').parent().hide();
        }

        $('#button-export-ovf').on('click', function(e) {
            downloadURI(spirit.core.exportOVFDataURI(), $('#input-export-spins').val()+'.ovf');
        });

        $('#button-export-energy').on('click', function(e) {
            spirit.core.System_Update_Data();
            downloadURI(spirit.core.exportEnergyDataURI(), $('#input-export-energy').val()+'.txt');
        });

        // ---------------------------------------------------------------------

        var url_string = window.location.href;
        var url = new URL(url_string);
        var example = String(url.searchParams.get("example"));

        // console.log(example);
        // console.log(example.localeCompare("racetrack"));
        // console.log(example.localeCompare("hopfion"));

        // Default camera
        spirit.vfr.set_camera([50, 50, 100], [50, 50, 0], [0, 1, 0]);

        if( example.localeCompare("racetrack") == 0 )
        {
            // Geometry
            spirit.core.setNCells([120, 30, 1]);
            document.getElementById('input-gridsize-x').value = 120;
            document.getElementById('input-gridsize-y').value = 30;
            document.getElementById('input-gridsize-z').value = 1;

            // Configuration
            spirit.core.setAllSpinsPlusZ();
            spirit.core.createSkyrmion(1, -90, 5, [-15, 0, 0], false, false, false);
            spirit.core.createSkyrmion(1, -90, 5, [15, 0, 0], false, false, false);

            // Hamiltonian
            document.getElementById('input-periodical-a').checked = true;
            document.getElementById('input-periodical-b').checked = false;
            document.getElementById('input-periodical-c').checked = false;

            // Parameters
            document.getElementById('input-spintorque').checked = true;
            document.getElementById('input-spintorque-magnitude').value = 0.5;
            document.getElementById('input-spintorque-directionx').value = -0.5;
            document.getElementById('input-spintorque-directiony').value = 1;
            document.getElementById('input-spintorque-directionz').value = 0;

            // Visualisation
            document.getElementById('input-show-arrows').checked = false;
            document.getElementById('input-show-surface').checked = true;
            document.getElementById('select-colormap').value = "bluewhitered";
            spirit.vfr.set_camera([60, 15, 80], [60, 15, 0], [0, 1, 0]);
        }
        else if( example.localeCompare("hopfion") == 0 )
        {
            // Geometry
            spirit.core.setNCells([30, 30, 30]);
            document.getElementById('input-gridsize-x').value = 30;
            document.getElementById('input-gridsize-y').value = 30;
            document.getElementById('input-gridsize-z').value = 30;

            // Configuration
            spirit.core.setAllSpinsPlusZ();
            spirit.core.createHopfion(5, 1, [0, 0, 0]);

            // Hamiltonian (TODO)
            // spirit.core.updateHamiltonianExternalField(5, 0, 0, 1);
            // spirit.core.updateHamiltonianAnisotropy(0, 0, 0, 1);
            // spirit.core.updateHamiltonianExchange([1, 0, 0, -0.1]);
            // spirit.core.updateHamiltonianDMI([0.1]);
            document.getElementById('input-periodical-a').checked = true;
            document.getElementById('input-periodical-b').checked = true;
            document.getElementById('input-periodical-c').checked = true;
            document.getElementById('input-externalfield').checked = false;
            document.getElementById('input-externalfield-magnitude').value = 0;
            document.getElementById('input-externalfield-directionx').value = 0;
            document.getElementById('input-externalfield-directiony').value = 0;
            document.getElementById('input-externalfield-directionz').value = 1;
            document.getElementById('input-anisotropy').checked = false;
            document.getElementById('input-anisotropy-magnitude').value = 0;
            document.getElementById('input-anisotropy-directionx').value = 0;
            document.getElementById('input-anisotropy-directiony').value = 0;
            document.getElementById('input-anisotropy-directionz').value = 1;
            document.getElementById('input-exchangemagnitudes1').value = 1;
            document.getElementById('input-exchangemagnitudes2').value = 0;
            document.getElementById('input-exchangemagnitudes3').value = 0;
            document.getElementById('input-exchangemagnitudes4').value = -0.25;
            document.getElementById('input-dmi').checked = false;
            document.getElementById('input-dmi-magnitude').value = 0;

            // Visualisation
            document.getElementById('input-show-arrows').checked = false;
            document.getElementById('input-show-isosurface').checked = true;
            spirit.vfr.set_camera([-25, -25, 50], [15, 15, 15], [0, 0, 1]);
        }
        else
        {
            console.log("unknown example: ", example)
        }

        // Grid
        updateGridSize();

        // Hamiltonian
        updateHamiltonianBoundaryConditions();
        updateHamiltonianMuSpin();
        updateHamiltonianExternalField();
        updateHamiltonianExchange();
        updateHamiltonianDMI();
        updateHamiltonianAnisotropy();
        updateHamiltonianDDI();

        // Parameters
        updateHamiltonianSpinTorque();
        updateHamiltonianTemperature();
        updateLLGDamping();
        updateLLGTimeStep();

        // Visualisation
        updateRenderers();
        updateColormap();
    }
    ).then(function()
    {
        var version = spirit.core.spiritVersion();
        document.getElementById("spirit-version").textContent="Version " + version;
        $('#div-load').hide();
    });
});