# ChaiScript Extras

User contributed wrappers and modules for ChaiScript.

## Modules

- [Math](#math): Adds common math methods to ChaiScript.
- [String ID](#string-id): String hashing with [string_id](https://github.com/foonathan/string_id)
- [String](#string): Adds some extra string methods to ChaiScript strings

## Math

The Math module adds some standard math functions to ChaiScript.

### Install
``` cpp
#include "chaiscript/extras/math.hpp"
```
``` cpp
chaiscript::ChaiScript chai;
auto mathlib = chaiscript::extras::math::bootstrap();
chai.add(mathlib);
```

### Usage

``` chaiscript
var result = cos(0.5f)
```

### Options

Compile with one of the following flags to enable or disable features...
- `CHAISCRIPT_EXTRAS_MATH_SKIP_ADVANCED` When enabled, will skip some of the advanced math functions.

## String ID

Adds [String ID](https://github.com/foonathan/string_id) support to ChaiScript.

### Install

``` cpp
#include "chaiscript/extras/string_id.hpp"
```

``` cpp
auto string_idlib = chaiscript::extras::string_id::bootstrap();
chai.add(string_idlib);
```

## String

Adds various string methods to extend how strings can be used in ChaiScript:
- `string::replace(string, string)`
- `string::trim()`
- `string::trimStart()`
- `string::trimEnd()`
- `string::split(string)`
- `string::toLowerCase()`
- `string::toUpperCase()`
- `string::includes()`

### Install

``` cpp
#include "chaiscript/extras/string_methods.hpp"
```

``` cpp
auto stringmethods = chaiscript::extras::string_methods::bootstrap();
chai.add(stringmethods);
```

### Usage

``` chaiscript
var input = "Hello, World!"
var output = input.replace("Hello", "Goodbye")
// => "Goodbye, World!"
```
