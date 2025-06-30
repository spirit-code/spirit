Utility
====================================================================

```C
#include "Spirit/Legacy.h"
```

Collection of functions provided for backwards compatibility. These might be removed in future releases.

### Legacy_Convert_Config_to_TOML

```c
void Legacy_Convert_Config_to_TOML( const char * config, const char * toml_config );
```

Convert a config (input) file in the old format to the new TOML based config format.
