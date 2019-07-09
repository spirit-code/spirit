#ifndef SHADER_HEADER

#ifdef __EMSCRIPTEN__
#define SHADER_HEADER std::string("#version 300 es\nprecision highp float;\n")
#else
#define SHADER_HEADER std::string("#version 330\n")
#endif

#endif