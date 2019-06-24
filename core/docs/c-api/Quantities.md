

Quantities
====================================================================

```C
#include "Spirit/Quantities.h"
```



### Quantity_Get_Magnetization

```C
void Quantity_Get_Magnetization(State * state, float m[3], int idx_image=-1, int idx_chain=-1)
```

Total Magnetization



### Quantity_Get_Topological_Charge

```C
float Quantity_Get_Topological_Charge(State * state, int idx_image=-1, int idx_chain=-1)
```

Topological Charge



### Quantity_Get_Grad_Force_MinimumMode

```C
void Quantity_Get_Grad_Force_MinimumMode(State * state, float * gradient, float * eval, float * mode, float * forces, int idx_image=-1, int idx_chain=-1)
```

Minimum mode following information

