$(document).ready(function() {
  $('form').attr('onsubmit', 'return false;');
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
    var boundingBoxColors = {
      'white': [0.0, 0.0, 0.0],
      'gray': [1.0, 1.0, 1.0],
      'black': [1.0, 1.0, 1.0]
    };
    webglspins.updateOptions({
      backgroundColor: colors[backgroundColor],
      boundingBoxColor: boundingBoxColors[backgroundColor]
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
    var innerSphereRadius = 0.95;
    if (showSphereWidget) {
      var sphereWidgetPosition = JSON.parse($("option:selected", $('#select-spinspherewidget-position'))[0].value);
      if (rendermode == 'SPHERE') {
        renderers.push([WebGLSpins.renderers.SURFACE, sphereWidgetPosition]);
      } else {
        renderers.push([WebGLSpins.renderers.SPHERE, sphereWidgetPosition]);
      }
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
  $('#select-spinspherewidget-position').on('change', updateRenderers);
  updateRenderers();

  function updateShowBoundingBox() {
    var showBoundingBox = $('#input-show-boundingbox')[0].checked;
    window.currentSimulation.showBoundingBox = showBoundingBox;
    window.currentSimulation.update();
  }

  $('#input-show-boundingbox').on('change', updateShowBoundingBox);

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
      window.currentSimulation.createSkyrmion(order, phase, radius, position, updown, rl, achiral);
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
      window.currentSimulation.createSpinSpiral(wrt, q, axis, angle);
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
      window.currentSimulation.createDomain(direction, position, border);
    }
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
        window.currentSimulation.updateHamiltonianDMI([dij]);
      }
    } else {
      window.currentSimulation.updateHamiltonianDMI([0]);
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

  function updateHamiltonianDDI() {
    if ($('#input-ddi')[0].checked) {
      window.currentSimulation.updateHamiltonianDDI(1, 4);
    }
    else {
      window.currentSimulation.updateHamiltonianDDI(0, 4);
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
      window.currentSimulation.updateLLGDamping(damping);
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
      window.currentSimulation.updateLLGTimeStep(time_step);
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
  //     window.currentSimulation.updateGNEBSpringConstant(spring_constant);
  //   }
  // }
  // $('#input-gneb-springconst').on('change', updateGNEBSpringConstant);

  // function updateGNEBClimbingFalling() {
  //   var climbing = $('#input-gneb-radio-climbing')[0].checked;
  //   var falling = $('#input-gneb-radio-falling')[0].checked;
  //   window.currentSimulation.updateGNEBClimbingFalling(climbing, falling);
  // }
  // $('#input-gneb-radio-normal').on('change', updateGNEBClimbingFalling);
  // $('#input-gneb-radio-climbing').on('change', updateGNEBClimbingFalling);
  // $('#input-gneb-radio-falling').on('change', updateGNEBClimbingFalling);

  function updateUseTouch() {
    var useTouch = $('#input-use-touch')[0].checked;
    webglspins.updateOptions({useTouch: useTouch});
  }
  $('#input-use-touch').on('change', updateUseTouch);


  $('#button-camera-x').on('click', function(e) {
    webglspins.alignCamera([-1, 0, 0], [0, 0, 1]);
  });
  $('#button-camera-y').on('click', function(e) {
    webglspins.alignCamera([0, 1, 0], [0, 0, 1]);
  });
  $('#button-camera-z').on('click', function(e) {
    webglspins.alignCamera([0, 0, -1], [0, 1, 0]);
  });

  var isSimulating = false;
  
  function updateSimulation(sim){  
    window.currentSimulation = sim;
    if (!webglspins.isTouchDevice) {
      $('#input-use-touch')[0].disabled="disabled";
      $('#input-use-touch')[0].checked = false;
    }
    sim.update();
    updateShowBoundingBox();
    updateHamiltonianBoundaryConditions();
    updateHamiltonianMuSpin();
    updateHamiltonianExternalField();
    updateHamiltonianExchange();
    updateHamiltonianDMI();
    updateHamiltonianAnisotropy();
    updateHamiltonianDDI();
    updateHamiltonianSpinTorque();
    updateHamiltonianTemperature();
    updateLLGDamping();
    updateLLGTimeStep();
    // updateGNEBSpringConstant();
    // updateGNEBClimbingFalling();
    $('#div-load').hide();
    $( window ).resize(function() {
      if (!isSimulating) {
        webglspins.draw();
      }
    });
    $('.collapse').on('hidden.bs.collapse', function () {
      if (!isSimulating) {
        webglspins.draw();
      }
    });
    $('.collapse').on('shown.bs.collapse', function () {
      if (!isSimulating) {
        webglspins.draw();
      }
    });
    $('#input-show-settings').on('change', function () {
      if (!isSimulating) {
        webglspins.draw();
      }
    });
    function update(sim) {
      if (isSimulating) {
        sim.performIteration();
        window.requestAnimationFrame(function () {
          update(sim)
        });
      }
    };
    $("#btn-play").click(function() {
      isSimulating = !isSimulating;
      $("#btn-play").toggleClass("fa-play fa-pause");
      if (isSimulating) {
        sim.startSimulation();
        window.requestAnimationFrame(function () {
          update(sim);
        });
      }
      else {
        sim.stopSimulation();
      }
    });
    sim.updateLLGConvergence(-1);
  };
  Module.ready(function() {
    new Simulation(updateSimulation);

    var version = window.currentSimulation.spiritVersion();
    document.getElementById("spirit-version").textContent="Version" + version;
  });
  $("#btn-extended-controls").click(function() {
    $("#webgl-extended-controls").toggleClass("hidden");
  });
});