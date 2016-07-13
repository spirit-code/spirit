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
  $('#select-colormap').on('change', updateColormap);
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
  $('#select-backgroundcolor').on('change', updateBackgroundColor);
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
  $('#select-rendermode').on('change', updateRenderers);
  $('#input-show-coordinatesystem').on('change', updateRenderers);
  $('#select-coordinatesystemwidget-position').on('change', updateRenderers);
  $('#input-show-spinspherewidget').on('change', updateRenderers);
  $('#input-show-spinspherewidget-background').on('change', updateRenderers);
  $('#select-spinspherewidget-position').on('change', updateRenderers);
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
  $('#input-zrange-filter').on('change', updateZRangeFilter);
  updateZRangeFilter();

  $("#input-spinspherewidget-pointsize").slider();
  function updateSpherePointSize() {
    var pointSizeRange = $("#input-spinspherewidget-pointsize").slider('getValue');
    webglspins.updateOptions({
      pointSizeRange: pointSizeRange
    });
  }
  $('#input-spinspherewidget-pointsize').on('change', updateSpherePointSize);
  updateSpherePointSize();

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
      window.currentSimulation.updateHamiltonianBoundaryConditions(periodical_a, periodical_b, periodical_c);
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
      window.currentSimulation.updateHamiltonianMuSpin(muspin);
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
        window.currentSimulation.updateHamiltonianExternalField(magnitude, normalx, normaly, normalz);
      }
    } else {
      window.currentSimulation.updateHamiltonianExternalField(0, 0, 0, 1);
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
      if (valid) {
        window.currentSimulation.updateHamiltonianExchange([value1, value2]);
      }
    } else {
      window.currentSimulation.updateHamiltonianExchange([0, 0]);
    }
  }
  $('#input-exchange').on('change', updateHamiltonianExchange);
  $('#input-exchangemagnitudes1').on('change', updateHamiltonianExchange);
  $('#input-exchangemagnitudes2').on('change', updateHamiltonianExchange);

  function updateHamiltonianDMI() {
    var dij = Number($('#input-dmi-magnitude').val());
    var valid = true;
    if (Number.isNaN(dij)) {
      valid = false;
      $('#input-dmi-magnitude').parent().addClass('has-error');
    } else {
      $('#input-dmi-magnitude').parent().removeClass('has-error');
    }
    if (valid) {
      window.currentSimulation.updateHamiltonianDMI(dij);
    }
  }
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
        window.currentSimulation.updateHamiltonianAnisotropy(magnitude, normalx, normaly, normalz);
      }
    } else {
      window.currentSimulation.updateHamiltonianAnisotropy(0, 0, 0, 1);
    }
  }
  $('#input-anisotropy').on('change', updateHamiltonianAnisotropy);
  $('#input-anisotropy-magnitude').on('change', updateHamiltonianAnisotropy);
  $('#input-anisotropy-directionx').on('change', updateHamiltonianAnisotropy);
  $('#input-anisotropy-directiony').on('change', updateHamiltonianAnisotropy);
  $('#input-anisotropy-directionz').on('change', updateHamiltonianAnisotropy);

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
        window.currentSimulation.updateHamiltonianSpinTorque(magnitude, normalx, normaly, normalz);
      }
    } else {
      window.currentSimulation.updateHamiltonianSpinTorque(0, 0, 0, 1);
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
        window.currentSimulation.updateHamiltonianTemperature(temperature);
      }
    } else {
        window.currentSimulation.updateHamiltonianTemperature(0);
    }
  }
  $('#input-temperature').on('change', updateHamiltonianTemperature);
  $('#input-temperature-value').on('change', updateHamiltonianTemperature);


  var isSimulating = false;

  Module.ready(function() {
    var sim = new Simulation();
    window.currentSimulation = sim;
    sim.update();
    updateHamiltonianBoundaryConditions();
    updateHamiltonianMuSpin();
    updateHamiltonianExternalField();
    updateHamiltonianExchange();
    updateHamiltonianDMI();
    updateHamiltonianAnisotropy();
    updateHamiltonianSpinTorque();
    updateHamiltonianTemperature();
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