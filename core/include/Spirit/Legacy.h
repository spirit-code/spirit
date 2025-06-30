#pragma once
#ifndef SPIRIT_CORE_UTILITY_H
#define SPIRIT_CORE_UTILITY_H
#include "Spirit_Defines.h"

#include "DLL_Define_Export.h"

struct State;

/*
Utility
====================================================================

```C
#include "Spirit/Legacy.h"
```

Collection of functions provided for backwards compatibility. These might be removed in future releases.
*/

// Convert a config (input) file in the old format to the new TOML based config format.
PREFIX void Legacy_Convert_Config_to_TOML( const char * config, const char * toml_config ) SUFFIX;

#endif
