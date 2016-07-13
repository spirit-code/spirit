$(document).ready(function() {
  webglspins = new WebGLSpins(document.getElementById("webgl-canvas"), {
    cameraLocation: [50, 50, 100],
    centerLocation: [50, 50, 0],
    upVector: [0, 1, 0],
    backgroundColor: [0.5, 0.5, 0.5]
  });
  WebGLSpins.colormapImplementations['hsv'] = `
    float atan2(float y, float x) {
        return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
    }
    vec3 hsv2rgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }
    vec3 colormap(vec3 direction) {
        vec2 xy = normalize(direction.xy);
        float hue = atan2(xy.x, xy.y) / 3.14159 / 2.0;
        if (direction.z > 0.0) {
        return hsv2rgb(vec3(hue, 1.0-direction.z, 1.0));
        } else {
        return hsv2rgb(vec3(hue, 1.0, 1.0+direction.z));
        }
    }`;
  function updateColormap() {
    var colormap = $("option:selected", $('#select-colormap'))[0].value;
    webglspins.updateOptions({
      colormapImplementation: WebGLSpins.colormapImplementations[colormap]
    });
  }
  $('#select-colormap').on('change', function (e) {
    updateColormap();
  });
  updateColormap();

  function updateBackgroundColor() {
    var backgroundColor = $("option:selected", $('#select-backgroundcolor'))[0].value;
    var colors = {
      'white': [1.0, 1.0, 1.0],
      'gray': [0.5, 0.5, 0.5],
      'black': [0.0, 0.0, 0.0]
    };
    webglspins.updateOptions({
      backgroundColor: colors[backgroundColor]
    });
  }
  $('#select-backgroundcolor').on('change', function (e) {
    updateBackgroundColor();
  });
  updateBackgroundColor();

  function updateRenderers() {
    var rendermode = $("option:selected", $('#select-rendermode'))[0].value;
    var renderers = [WebGLSpins.renderers[rendermode]];
    var showCoordinateSystemWidget = $('#input-show-coordinatesystem').is(':checked');
    if (showCoordinateSystemWidget) {
      var coordinateSystemWidgetPosition = JSON.parse($("option:selected", $('#select-coordinatesystemwidget-position'))[0].value);
      renderers.push([WebGLSpins.renderers.COORDINATESYSTEM, coordinateSystemWidgetPosition]);
    }
    var showSphereWidget = $('#input-show-spinspherewidget').is(':checked');
    var showSphereWidgetBackground = $('#input-show-spinspherewidget-background').is(':checked');
    var innerSphereRadius = 0.95;
    if (!showSphereWidgetBackground) {
      innerSphereRadius = 0.0;
    }
    if (showSphereWidget) {
      var sphereWidgetPosition = JSON.parse($("option:selected", $('#select-spinspherewidget-position'))[0].value);
      renderers.push([WebGLSpins.renderers.SPHERE, sphereWidgetPosition]);
    }
    webglspins.updateOptions({
      renderers: renderers,
      innerSphereRadius: innerSphereRadius
    });
  }
  $('#select-rendermode').on('change', function (e) {
    updateRenderers();
  });
  $('#input-show-coordinatesystem').on('change', function (e) {
    updateRenderers();
  });
  $('#select-coordinatesystemwidget-position').on('change', function (e) {
    updateRenderers();
  });
  $('#input-show-spinspherewidget').on('change', function (e) {
    updateRenderers();
  });
  $('#input-show-spinspherewidget-background').on('change', function (e) {
    updateRenderers();
  });
  $('#select-spinspherewidget-position').on('change', function (e) {
    updateRenderers();
  });
  updateRenderers();

  $('#button-plusz').on('click', function(e) {
    var sim = window.currentSimulation;
    sim.setAllSpinsPlusZ();
  });
  $('#button-minusz').on('click', function(e) {
    var sim = window.currentSimulation;
    sim.setAllSpinsMinusZ();
  });
  $('#button-random').on('click', function(e) {
    var sim = window.currentSimulation;
    sim.setAllSpinsRandom();
  });

  $("#input-zrange-filter").slider();
  function updateZRangeFilter() {
    var zRange = $("#input-zrange-filter").slider('getValue');
    webglspins.updateOptions({
      zRange: zRange
    });
  }
  $('#input-zrange-filter').on('change', function (e) {
    updateZRangeFilter();
  });
  updateZRangeFilter();

  $("#input-spinspherewidget-pointsize").slider();
  function updateSpherePointSize() {
    var pointSizeRange = $("#input-spinspherewidget-pointsize").slider('getValue');
    webglspins.updateOptions({
      pointSizeRange: pointSizeRange
    });
  }
  $('#input-spinspherewidget-pointsize').on('change', function (e) {
    updateSpherePointSize();
  });
  updateSpherePointSize();

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
      window.currentSimulation.updateHamiltonianMuSpin(muspin);
    }
  }

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
        window.currentSimulation.updateHamiltonianExternalField(magnitude, normalx, normaly, normalz);
      }
    } else {
      window.currentSimulation.updateHamiltonianExternalField(0, 0, 0, 1);
    }
  }
  $('#input-externalfield').on('change', function (e) {
    updateHamiltonianExternalField();
  });
  $('#input-externalfield-muspin').on('change', function (e) {
    updateHamiltonianMuSpin();
  });
  $('#input-externalfield-magnitude').on('change', function (e) {
    updateHamiltonianExternalField();
  });
  $('#input-externalfield-directionx').on('change', function (e) {
    updateHamiltonianExternalField();
  });
  $('#input-externalfield-directiony').on('change', function (e) {
    updateHamiltonianExternalField();
  });
  $('#input-externalfield-directionz').on('change', function (e) {
    updateHamiltonianExternalField();
  });

  var isSimulating = false;

  Module.ready(function() {
    var sim = new Simulation();
    window.currentSimulation = sim;
    sim.update();
    updateHamiltonianExternalField();
    $('#div-load').hide();
    function update(sim) {
      sim.performIteration();
      if (isSimulating) {
        window.requestAnimationFrame(function () {
          update(sim)
        });
      }
    }
    $("#btn-play").click(function() {
      isSimulating = !isSimulating;
      $("#btn-play").toggleClass("fa-play fa-pause");
      if (isSimulating) {
        window.requestAnimationFrame(function () {
          update(sim);
        });
      }
    });
    update(sim);
  });
  });
  $("#btn-extended-controls").click(function() {
  $("#webgl-extended-controls").toggleClass("hidden");
});