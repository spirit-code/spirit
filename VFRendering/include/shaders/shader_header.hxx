#ifndef SHADER_HEADER

#ifdef __EMSCRIPTEN__
#define VERT_SHADER_HEADER std::string("#version 100\nprecision highp float;\n#define in attribute\n#define out varying\n")
#define FRAG_SHADER_HEADER std::string("#version 100\nprecision highp float;\n#define in varying\n#define fo_FragColor gl_FragColor")
#else
#define VERT_SHADER_HEADER std::string("#version 330\n")
#define FRAG_SHADER_HEADER std::string("#version 330\nout vec4 fo_FragColor;\n")
#endif

#endif