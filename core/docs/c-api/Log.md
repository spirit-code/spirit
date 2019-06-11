

Logging
====================================================================

```C
#include "Spirit/Log.h"
```



Definition of log levels and senders
--------------------------------------------------------------------



### Spirit_Log_Level

```C
typedef enum { Log_Level_All       = 0, Log_Level_Severe    = 1, Log_Level_Error     = 2, Log_Level_Warning   = 3, Log_Level_Parameter = 4, Log_Level_Info      = 5, Log_Level_Debug     = 6 } Spirit_Log_Level
```

Levels



### Spirit_Log_Sender

```C
typedef enum { Log_Sender_All  = 0, Log_Sender_IO   = 1, Log_Sender_GNEB = 2, Log_Sender_LLG  = 3, Log_Sender_MC   = 4, Log_Sender_MMF  = 5, Log_Sender_EMA  = 6, Log_Sender_API  = 7, Log_Sender_UI   = 8, Log_Sender_HTST = 9 } Spirit_Log_Sender
```

Senders



Logging functions
--------------------------------------------------------------------



### Log_Send

```C
void Log_Send(State *state, Spirit_Log_Level level, Spirit_Log_Sender sender, const char * message, int idx_image=-1, int idx_chain=-1)
```

Send a Log message



### Log_Append

```C
void Log_Append(State *state)
```

Append the Log to it's file



### Log_Dump

```C
void Log_Dump(State *state)
```

Dump the Log into it's file



### Log_Get_N_Entries

```C
int Log_Get_N_Entries(State *state)
```

Get the number of Log entries



### Log_Get_N_Errors

```C
int Log_Get_N_Errors(State *state)
```

Get the number of errors in the Log



### Log_Get_N_Warnings

```C
int Log_Get_N_Warnings(State *state)
```

Get the number of warnings in the Log



Set Log parameters
--------------------------------------------------------------------



### Log_Set_Output_File_Tag

```C
void Log_Set_Output_File_Tag(State *state, const char * tag)
```

The tag in front of the log file



### Log_Set_Output_Folder

```C
void Log_Set_Output_Folder(State *state, const char * folder)
```

The output folder for the log file



### Log_Set_Output_To_Console

```C
void Log_Set_Output_To_Console(State *state, bool output, int level)
```

Whether to write log messages to the console and corresponding level



### Log_Set_Output_To_File

```C
void Log_Set_Output_To_File(State *state, bool output, int level)
```

Whether to write log messages to the log file and corresponding level



Get Log parameters
--------------------------------------------------------------------



### Log_Get_Output_File_Tag

```C
const char * Log_Get_Output_File_Tag(State *state)
```

Returns the tag in front of the log file



### Log_Get_Output_Folder

```C
const char * Log_Get_Output_Folder(State *state)
```

Returns the output folder for the log file



### Log_Get_Output_To_Console

```C
bool Log_Get_Output_To_Console(State *state)
```

Returns whether to write log messages to the console



### Log_Get_Output_Console_Level

```C
int Log_Get_Output_Console_Level(State *state)
```

Returns the console logging level



### Log_Get_Output_To_File

```C
bool Log_Get_Output_To_File(State *state)
```

Returns whether to write log messages to the log file



### Log_Get_Output_File_Level

```C
int Log_Get_Output_File_Level(State *state)
```

Returns the file logging level

